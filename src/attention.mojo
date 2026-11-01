# Fused multi-head attention (flash-attention style) for the training engine.
#
# The forward pass never materializes the T x T attention matrix: each block streams key/value
# tiles through shared memory with an online softmax, and keeps only one log-sum-exp per query
# row. The backward pass recomputes the probabilities from that log-sum-exp, tile by tile.
#
# Layouts (rows are batch-major, r = b * T + t):
#   qkv  [R, 3D]  head h's Q, K, V are columns h*HD, D + h*HD, 2D + h*HD
#   att  [R, D]   head h's output is columns h*HD
#   lse  [B*H*T]  log-sum-exp of each query row's scaled scores, indexed (b*H + h)*T + t
#   dvec [B*H*T]  sum_d dO*O per query row (the softmax-backward correction)
# T here (nt in the code) is this batch's tokens per canvas, a multiple of BQ: the model's T,
# or fewer when the engine runs only some tokens (CROP, see token_order in src/model.mojo).
#
# Each GPU kernel has a plain per-row b_* counterpart computing the same thing, used when the
# engine runs without a GPU.

from std.math import exp, log
from std.memory import stack_allocation
from max.gpu import thread_idx, block_idx, barrier
from max.gpu.primitives import warp
from max.gpu.memory import AddressSpace
from src.model import in_band, run_len

comptime FP = Pointer[Float32, MutAnyOrigin]
comptime BQ = 64  # query rows per block (4 threads per row)
comptime BK = 32  # key rows per shared-memory tile
comptime NEG_INF: Float32 = -3.0e38


# ============================================================================ CPU bodies


def b_attn_fwd(att: FP, lse: FP, qkv: FP, D: Int, H: Int, hd: Int, scale: Float32, nt: Int, i: Int):
    """Query row i = (b*H + h)*nt + t: two passes over the keys (max, then exp-sum and P V)."""
    var bh = i // nt
    var t = i - bh * nt
    var b = bh // H
    var h = bh - b * H
    var q = qkv.unsafe_offset((b * nt + t) * 3 * D + h * hd)
    var kb = qkv.unsafe_offset(b * nt * 3 * D + D + h * hd)
    var vb = qkv.unsafe_offset(b * nt * 3 * D + 2 * D + h * hd)
    var o = att.unsafe_offset((b * nt + t) * D + h * hd)
    var mx = NEG_INF
    for j in range(nt):
        if not in_band(t, j):
            continue
        var s: Float32 = 0.0
        for d in range(hd):
            s += q[unsafe_offset=d] * kb[unsafe_offset=j * 3 * D + d]
        mx = max(mx, s * scale)
    for d in range(hd):
        o[unsafe_offset=d] = 0.0
    var l: Float32 = 0.0
    for j in range(nt):
        if not in_band(t, j):
            continue
        var s: Float32 = 0.0
        for d in range(hd):
            s += q[unsafe_offset=d] * kb[unsafe_offset=j * 3 * D + d]
        var p = exp(s * scale - mx)
        l += p
        for d in range(hd):
            o[unsafe_offset=d] += p * vb[unsafe_offset=j * 3 * D + d]
    for d in range(hd):
        o[unsafe_offset=d] /= l
    lse[unsafe_offset=i] = mx + log(l)


def b_attn_dvec(dvec: FP, att: FP, datt: FP, D: Int, H: Int, hd: Int, nt: Int, i: Int):
    var bh = i // nt
    var t = i - bh * nt
    var b = bh // H
    var h = bh - b * H
    var off = (b * nt + t) * D + h * hd
    var s: Float32 = 0.0
    for d in range(hd):
        s += att[unsafe_offset=off + d] * datt[unsafe_offset=off + d]
    dvec[unsafe_offset=i] = s


def b_attn_bwd_kv(dqkv: FP, qkv: FP, datt: FP, lse: FP, dvec: FP, D: Int, H: Int, hd: Int,
                  scale: Float32, nt: Int, i: Int):
    """Key row i = (b*H + h)*nt + j: dK_j = scale * sum_t dS_tj Q_t, dV_j = sum_t P_tj dO_t."""
    var bh = i // nt
    var j = i - bh * nt
    var b = bh // H
    var h = bh - b * H
    var k = qkv.unsafe_offset((b * nt + j) * 3 * D + D + h * hd)
    var v = qkv.unsafe_offset((b * nt + j) * 3 * D + 2 * D + h * hd)
    var dk = dqkv.unsafe_offset((b * nt + j) * 3 * D + D + h * hd)
    var dv = dqkv.unsafe_offset((b * nt + j) * 3 * D + 2 * D + h * hd)
    for d in range(hd):
        dk[unsafe_offset=d] = 0.0
        dv[unsafe_offset=d] = 0.0
    for t in range(nt):
        if not in_band(t, j):
            continue
        var q = qkv.unsafe_offset((b * nt + t) * 3 * D + h * hd)
        var do = datt.unsafe_offset((b * nt + t) * D + h * hd)
        var s: Float32 = 0.0
        var dp: Float32 = 0.0
        for d in range(hd):
            s += q[unsafe_offset=d] * k[unsafe_offset=d]
            dp += do[unsafe_offset=d] * v[unsafe_offset=d]
        var p = exp(s * scale - lse[unsafe_offset=bh * nt + t])
        var ds = p * (dp - dvec[unsafe_offset=bh * nt + t]) * scale
        for d in range(hd):
            dv[unsafe_offset=d] += p * do[unsafe_offset=d]
            dk[unsafe_offset=d] += ds * q[unsafe_offset=d]


def b_attn_bwd_q(dqkv: FP, qkv: FP, datt: FP, lse: FP, dvec: FP, D: Int, H: Int, hd: Int,
                 scale: Float32, nt: Int, i: Int):
    """Query row i = (b*H + h)*nt + t: dQ_t = scale * sum_j dS_tj K_j."""
    var bh = i // nt
    var t = i - bh * nt
    var b = bh // H
    var h = bh - b * H
    var q = qkv.unsafe_offset((b * nt + t) * 3 * D + h * hd)
    var do = datt.unsafe_offset((b * nt + t) * D + h * hd)
    var dq = dqkv.unsafe_offset((b * nt + t) * 3 * D + h * hd)
    var l = lse[unsafe_offset=i]
    var dv = dvec[unsafe_offset=i]
    for d in range(hd):
        dq[unsafe_offset=d] = 0.0
    for j in range(nt):
        if not in_band(t, j):
            continue
        var k = qkv.unsafe_offset((b * nt + j) * 3 * D + D + h * hd)
        var v = qkv.unsafe_offset((b * nt + j) * 3 * D + 2 * D + h * hd)
        var s: Float32 = 0.0
        var dp: Float32 = 0.0
        for d in range(hd):
            s += q[unsafe_offset=d] * k[unsafe_offset=d]
            dp += do[unsafe_offset=d] * v[unsafe_offset=d]
        var ds = exp(s * scale - l) * (dp - dv) * scale
        for d in range(hd):
            dq[unsafe_offset=d] += ds * k[unsafe_offset=d]


# ============================================================================ GPU kernels
# 256 threads per block. Thread tid owns row tid // 4 of its block's tile and, of that row,
# every 4th score column (tid % 4 + 4c) and one HD/4-wide slice of the head dimensions.
# Shared tiles have rows padded to a multiple of 4 floats, so rows are read as float4 vectors.

comptime SharedFP = Pointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
comptime LP = BK + 4  # row stride of the score tiles (Ps, dSs)


def ld[HD: Int]() -> Int:
    """Row stride of a K/V/Q/dO tile."""
    return HD + 4


def dot_row[HD: Int](x: SIMD[DType.float32, HD], tile: SharedFP, off: Int) -> Float32:
    """x . tile[off : off + HD], read 4 floats at a time."""
    var acc = SIMD[DType.float32, 4](0.0)
    comptime for d in range(0, HD, 4):
        acc += x.slice[4, offset=d]() * tile.unsafe_load[width=4, alignment=16](off + d)
    return acc.reduce_add()


def row_reduce4(red: SharedFP, v: Float32, tid: Int, is_max: Bool) -> Float32:
    """Max or sum of v over the 4 threads that share a row (every thread must call it)."""
    red[unsafe_offset=tid] = v
    barrier()
    var b = tid - tid % 4
    var r0 = red[unsafe_offset=b]
    var r1 = red[unsafe_offset=b + 1]
    var r2 = red[unsafe_offset=b + 2]
    var r3 = red[unsafe_offset=b + 3]
    barrier()
    if is_max:
        return max(max(r0, r1), max(r2, r3))
    return (r0 + r1) + (r2 + r3)


def load_tile[HD: Int](dst: SharedFP, src: FP, row0: Int, stride: Int, tid: Int):
    """dst[BK][ld] = src rows row0..row0+BK (row stride `stride`), HD columns each."""
    comptime for e in range((BK * HD) // 256):
        var idx = tid + e * 256
        var r = idx // HD
        var d = idx - r * HD
        dst[unsafe_offset=r * ld[HD]() + d] = src[unsafe_offset=(row0 + r) * stride + d]


def quad_max(v: Float32) -> Float32:
    """Max over the 4 adjacent lanes that share a row (tid % 4), by shuffles: no barrier."""
    var m = max(v, warp.shuffle_xor(v, UInt32(1)))
    return max(m, warp.shuffle_xor(m, UInt32(2)))


def quad_sum[W: Int](v: SIMD[DType.float32, W]) -> SIMD[DType.float32, W]:
    var s = v
    comptime for e in range(W):
        s[e] += warp.shuffle_xor(s[e], UInt32(1))
        s[e] += warp.shuffle_xor(s[e], UInt32(2))
    return s


def store_slice[HD: Int](dst: FP, v: SIMD[DType.float32, HD], q4: Int, mul: Float32):
    """dst[0:HD/4] = mul * this lane's quarter (q4) of v."""
    comptime SL = HD // 4
    comptime for q in range(4):
        if q4 == q:
            comptime for e in range(SL):
                dst[unsafe_offset=e] = v[q * SL + e] * mul


def k_attn_fwd[HD: Int](att: FP, lse: FP, qkv: FP, D: Int32, H: Int32, scale: Float32, TN: Int32):
    """One block = BQ query rows of one (sprite, head). grid = B * H * (T / BQ).
    Each lane accumulates P V over its own score columns for all HD dims; the row's 4 lanes
    share the running max by shuffles and sum their outputs once at the end."""
    var nt = run_len(TN)
    comptime LD = ld[HD]()
    comptime SL = HD // 4  # head dims each lane writes
    comptime NC = BK // 4  # score columns per thread per tile
    var Ks = stack_allocation[BK * LD, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var Vs = stack_allocation[BK * LD, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var tid = Int(thread_idx.x)
    var nd = Int(D)
    var nh = Int(H)
    var tiles = nt // BQ
    var bh = Int(block_idx.x) // tiles
    var t0 = (Int(block_idx.x) - bh * tiles) * BQ
    var b = bh // nh
    var h = bh - b * nh
    var r = tid // 4
    var q4 = tid % 4
    var t = t0 + r
    var base = qkv.unsafe_offset(b * nt * 3 * nd + h * HD)

    var qreg = SIMD[DType.float32, HD](0.0)
    comptime for d in range(HD):
        qreg[d] = base[unsafe_offset=t * 3 * nd + d] * scale
    var o = SIMD[DType.float32, HD](0.0)
    var m = NEG_INF
    var l: Float32 = 0.0  # this lane's share of the row's exp-sum

    for k0 in range(0, nt, BK):
        if not in_band(t0, k0):  # the same for the whole block, so barriers stay uniform
            continue
        load_tile[HD](Ks, base.unsafe_offset(nd), k0, 3 * nd, tid)
        load_tile[HD](Vs, base.unsafe_offset(2 * nd), k0, 3 * nd, tid)
        barrier()
        var s = SIMD[DType.float32, NC](0.0)
        var tmax = NEG_INF
        comptime for c in range(NC):
            var acc = dot_row[HD](qreg, Ks, (q4 + 4 * c) * LD)
            s[c] = acc
            tmax = max(tmax, acc)
        var m_new = max(m, quad_max(tmax))
        var alpha = exp(m - m_new)
        l *= alpha
        o *= alpha
        comptime for c in range(NC):
            var p = exp(s[c] - m_new)
            l += p
            o += p * Vs.unsafe_load[width=HD, alignment=16]((q4 + 4 * c) * LD)
        m = m_new
        barrier()  # Ks and Vs are overwritten by the next tile

    o = quad_sum[HD](o)
    var lt = quad_sum[1](SIMD[DType.float32, 1](l))[0]
    store_slice[HD](att.unsafe_offset((b * nt + t) * nd + h * HD + q4 * SL), o, q4, 1.0 / lt)
    if q4 == 0:
        lse[unsafe_offset=bh * nt + t] = m + log(lt)


def k_attn_dvec(dvec: FP, att: FP, datt: FP, D: Int32, H: Int32, hd: Int32, n: Int32, TN: Int32):
    """One thread per element of att (grid covers n * hd = R * D, coalesced); each head's hd
    adjacent lanes sum their products by shuffles (hd divides the warp, 16 or 32)."""
    var nt = run_len(TN)
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    var nh = Int(hd)
    var v: Float32 = 0.0
    if i < Int(n) * nh:
        v = att[unsafe_offset=i] * datt[unsafe_offset=i]
    var off = nh // 2
    while off > 0:
        v += warp.shuffle_down(v, UInt32(off))
        off //= 2
    if i < Int(n) * nh and i % nh == 0:
        var r = i // Int(D)
        var h = (i - r * Int(D)) // nh
        var b = r // nt
        dvec[unsafe_offset=(b * Int(H) + h) * nt + r - b * nt] = v


def k_attn_bwd_kv[HD: Int](dqkv: FP, qkv: FP, datt: FP, lse: FP, dvec: FP, D: Int32, H: Int32,
                           scale: Float32, TN: Int32):
    """One block = BQ key rows of one (sprite, head), streaming query tiles of BK rows.
    grid = B * H * (T / BQ)."""
    var nt = run_len(TN)
    comptime LD = ld[HD]()
    comptime SL = HD // 4
    comptime NC = BK // 4
    var Qs = stack_allocation[BK * LD, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var dOs = stack_allocation[BK * LD, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var Ps = stack_allocation[BQ * LP, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var dSs = stack_allocation[BQ * LP, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var Ls = stack_allocation[BK, Float32, address_space=AddressSpace.SHARED]()
    var Dv = stack_allocation[BK, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var nd = Int(D)
    var nh = Int(H)
    var tiles = nt // BQ
    var bh = Int(block_idx.x) // tiles
    var j0 = (Int(block_idx.x) - bh * tiles) * BQ
    var b = bh // nh
    var h = bh - b * nh
    var r = tid // 4
    var q4 = tid % 4
    var j = j0 + r
    var base = qkv.unsafe_offset(b * nt * 3 * nd + h * HD)
    var dbase = datt.unsafe_offset(b * nt * nd + h * HD)

    var kreg = SIMD[DType.float32, HD](0.0)
    var vreg = SIMD[DType.float32, HD](0.0)
    comptime for d in range(HD):
        kreg[d] = base[unsafe_offset=j * 3 * nd + nd + d] * scale
        vreg[d] = base[unsafe_offset=j * 3 * nd + 2 * nd + d]
    var dk = SIMD[DType.float32, SL](0.0)
    var dv = SIMD[DType.float32, SL](0.0)

    for t0 in range(0, nt, BK):
        if not in_band(t0, j0):
            continue
        load_tile[HD](Qs, base, t0, 3 * nd, tid)
        load_tile[HD](dOs, dbase, t0, nd, tid)
        if tid < BK:
            Ls[unsafe_offset=tid] = lse[unsafe_offset=bh * nt + t0 + tid]
            Dv[unsafe_offset=tid] = dvec[unsafe_offset=bh * nt + t0 + tid]
        barrier()
        comptime for c in range(NC):
            var i = q4 + 4 * c
            var s = dot_row[HD](kreg, Qs, i * LD)
            var dp = dot_row[HD](vreg, dOs, i * LD)
            var p = exp(s - Ls[unsafe_offset=i])
            Ps[unsafe_offset=r * LP + i] = p
            dSs[unsafe_offset=r * LP + i] = p * (dp - Dv[unsafe_offset=i])
        barrier()
        for i0 in range(0, BK, 4):
            var p4 = Ps.unsafe_load[width=4, alignment=16](r * LP + i0)
            var ds4 = dSs.unsafe_load[width=4, alignment=16](r * LP + i0)
            comptime for ii in range(4):
                var o = (i0 + ii) * LD + q4 * SL
                dv += p4[ii] * dOs.unsafe_load[width=SL, alignment=16](o)
                dk += ds4[ii] * Qs.unsafe_load[width=SL, alignment=16](o)
        barrier()

    var row = dqkv.unsafe_offset((b * nt + j) * 3 * nd + h * HD + q4 * SL)
    comptime for e in range(SL):
        row[unsafe_offset=nd + e] = dk[e] * scale
        row[unsafe_offset=2 * nd + e] = dv[e]


def k_attn_bwd_q[HD: Int](dqkv: FP, qkv: FP, datt: FP, lse: FP, dvec: FP, D: Int32, H: Int32,
                          scale: Float32, TN: Int32):
    """One block = BQ query rows of one (sprite, head), streaming key tiles. grid as k_attn_fwd."""
    var nt = run_len(TN)
    comptime LD = ld[HD]()
    comptime SL = HD // 4
    comptime NC = BK // 4
    var Ks = stack_allocation[BK * LD, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var Vs = stack_allocation[BK * LD, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var dSs = stack_allocation[BQ * LP, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var tid = Int(thread_idx.x)
    var nd = Int(D)
    var nh = Int(H)
    var tiles = nt // BQ
    var bh = Int(block_idx.x) // tiles
    var t0 = (Int(block_idx.x) - bh * tiles) * BQ
    var b = bh // nh
    var h = bh - b * nh
    var r = tid // 4
    var q4 = tid % 4
    var t = t0 + r
    var base = qkv.unsafe_offset(b * nt * 3 * nd + h * HD)

    var qreg = SIMD[DType.float32, HD](0.0)
    var doreg = SIMD[DType.float32, HD](0.0)
    comptime for d in range(HD):
        qreg[d] = base[unsafe_offset=t * 3 * nd + d] * scale
        doreg[d] = datt[unsafe_offset=(b * nt + t) * nd + h * HD + d]
    var l = lse[unsafe_offset=bh * nt + t]
    var dvi = dvec[unsafe_offset=bh * nt + t]
    var dq = SIMD[DType.float32, SL](0.0)

    for k0 in range(0, nt, BK):
        if not in_band(t0, k0):  # the same for the whole block, so barriers stay uniform
            continue
        load_tile[HD](Ks, base.unsafe_offset(nd), k0, 3 * nd, tid)
        load_tile[HD](Vs, base.unsafe_offset(2 * nd), k0, 3 * nd, tid)
        barrier()
        comptime for c in range(NC):
            var j = q4 + 4 * c
            var s = dot_row[HD](qreg, Ks, j * LD)
            var dp = dot_row[HD](doreg, Vs, j * LD)
            dSs[unsafe_offset=r * LP + j] = exp(s - l) * (dp - dvi)
        barrier()
        for j0 in range(0, BK, 4):
            var ds4 = dSs.unsafe_load[width=4, alignment=16](r * LP + j0)
            comptime for jj in range(4):
                dq += ds4[jj] * Ks.unsafe_load[width=SL, alignment=16]((j0 + jj) * LD + q4 * SL)
        barrier()

    var row = dqkv.unsafe_offset((b * nt + t) * 3 * nd + h * HD + q4 * SL)
    comptime for e in range(SL):
        row[unsafe_offset=e] = dq[e] * scale
