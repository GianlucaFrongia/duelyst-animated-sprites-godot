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
#
# Each GPU kernel has a plain per-row b_* counterpart computing the same thing, used when the
# engine runs without a GPU.

from std.math import exp, log
from std.memory import stack_allocation
from max.gpu import thread_idx, block_idx, barrier
from max.gpu.memory import AddressSpace
from arch.model import T

comptime FP = Pointer[Float32, MutAnyOrigin]
comptime BQ = 64  # query rows per block (4 threads per row)
comptime BK = 32  # key rows per shared-memory tile
comptime NEG_INF: Float32 = -3.0e38


# ============================================================================ CPU bodies


def b_attn_fwd(att: FP, lse: FP, qkv: FP, D: Int, H: Int, hd: Int, scale: Float32, i: Int):
    """Query row i = (b*H + h)*T + t: two passes over the keys (max, then exp-sum and P V)."""
    var bh = i // T
    var t = i - bh * T
    var b = bh // H
    var h = bh - b * H
    var q = qkv.unsafe_offset((b * T + t) * 3 * D + h * hd)
    var kb = qkv.unsafe_offset(b * T * 3 * D + D + h * hd)
    var vb = qkv.unsafe_offset(b * T * 3 * D + 2 * D + h * hd)
    var o = att.unsafe_offset((b * T + t) * D + h * hd)
    var mx = NEG_INF
    for j in range(T):
        var s: Float32 = 0.0
        for d in range(hd):
            s += q[unsafe_offset=d] * kb[unsafe_offset=j * 3 * D + d]
        mx = max(mx, s * scale)
    for d in range(hd):
        o[unsafe_offset=d] = 0.0
    var l: Float32 = 0.0
    for j in range(T):
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


def b_attn_dvec(dvec: FP, att: FP, datt: FP, D: Int, H: Int, hd: Int, i: Int):
    var bh = i // T
    var t = i - bh * T
    var b = bh // H
    var h = bh - b * H
    var off = (b * T + t) * D + h * hd
    var s: Float32 = 0.0
    for d in range(hd):
        s += att[unsafe_offset=off + d] * datt[unsafe_offset=off + d]
    dvec[unsafe_offset=i] = s


def b_attn_bwd_kv(dqkv: FP, qkv: FP, datt: FP, lse: FP, dvec: FP, D: Int, H: Int, hd: Int,
                  scale: Float32, i: Int):
    """Key row i = (b*H + h)*T + j: dK_j = scale * sum_t dS_tj Q_t, dV_j = sum_t P_tj dO_t."""
    var bh = i // T
    var j = i - bh * T
    var b = bh // H
    var h = bh - b * H
    var k = qkv.unsafe_offset((b * T + j) * 3 * D + D + h * hd)
    var v = qkv.unsafe_offset((b * T + j) * 3 * D + 2 * D + h * hd)
    var dk = dqkv.unsafe_offset((b * T + j) * 3 * D + D + h * hd)
    var dv = dqkv.unsafe_offset((b * T + j) * 3 * D + 2 * D + h * hd)
    for d in range(hd):
        dk[unsafe_offset=d] = 0.0
        dv[unsafe_offset=d] = 0.0
    for t in range(T):
        var q = qkv.unsafe_offset((b * T + t) * 3 * D + h * hd)
        var do = datt.unsafe_offset((b * T + t) * D + h * hd)
        var s: Float32 = 0.0
        var dp: Float32 = 0.0
        for d in range(hd):
            s += q[unsafe_offset=d] * k[unsafe_offset=d]
            dp += do[unsafe_offset=d] * v[unsafe_offset=d]
        var p = exp(s * scale - lse[unsafe_offset=bh * T + t])
        var ds = p * (dp - dvec[unsafe_offset=bh * T + t]) * scale
        for d in range(hd):
            dv[unsafe_offset=d] += p * do[unsafe_offset=d]
            dk[unsafe_offset=d] += ds * q[unsafe_offset=d]


def b_attn_bwd_q(dqkv: FP, qkv: FP, datt: FP, lse: FP, dvec: FP, D: Int, H: Int, hd: Int,
                 scale: Float32, i: Int):
    """Query row i = (b*H + h)*T + t: dQ_t = scale * sum_j dS_tj K_j."""
    var bh = i // T
    var t = i - bh * T
    var b = bh // H
    var h = bh - b * H
    var q = qkv.unsafe_offset((b * T + t) * 3 * D + h * hd)
    var do = datt.unsafe_offset((b * T + t) * D + h * hd)
    var dq = dqkv.unsafe_offset((b * T + t) * 3 * D + h * hd)
    var l = lse[unsafe_offset=i]
    var dv = dvec[unsafe_offset=i]
    for d in range(hd):
        dq[unsafe_offset=d] = 0.0
    for j in range(T):
        var k = qkv.unsafe_offset((b * T + j) * 3 * D + D + h * hd)
        var v = qkv.unsafe_offset((b * T + j) * 3 * D + 2 * D + h * hd)
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

comptime SharedFP = Pointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]


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
    """dst[BK][HD+1] = src rows row0..row0+BK (row stride `stride`), HD columns each."""
    comptime for e in range((BK * HD) // 256):
        var idx = tid + e * 256
        var r = idx // HD
        var d = idx - r * HD
        dst[unsafe_offset=r * (HD + 1) + d] = src[unsafe_offset=(row0 + r) * stride + d]


def k_attn_fwd[HD: Int](att: FP, lse: FP, qkv: FP, D: Int32, H: Int32, scale: Float32):
    """One block = BQ query rows of one (sprite, head). grid = B * H * (T / BQ)."""
    comptime LD = HD + 1
    comptime SL = HD // 4  # head dims per thread
    comptime NC = BK // 4  # score columns per thread per tile
    var Ks = stack_allocation[BK * LD, Float32, address_space=AddressSpace.SHARED]()
    var Vs = stack_allocation[BK * LD, Float32, address_space=AddressSpace.SHARED]()
    var Ps = stack_allocation[BQ * (BK + 1), Float32, address_space=AddressSpace.SHARED]()
    var red = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var nd = Int(D)
    var nh = Int(H)
    var tiles = T // BQ
    var bh = Int(block_idx.x) // tiles
    var t0 = (Int(block_idx.x) - bh * tiles) * BQ
    var b = bh // nh
    var h = bh - b * nh
    var r = tid // 4
    var q4 = tid % 4
    var t = t0 + r
    var base = qkv.unsafe_offset(b * T * 3 * nd + h * HD)

    var qreg = SIMD[DType.float32, HD](0.0)
    comptime for d in range(HD):
        qreg[d] = base[unsafe_offset=t * 3 * nd + d] * scale
    var o = SIMD[DType.float32, SL](0.0)
    var m = NEG_INF
    var l: Float32 = 0.0

    for k0 in range(0, T, BK):
        load_tile[HD](Ks, base.unsafe_offset(nd), k0, 3 * nd, tid)
        load_tile[HD](Vs, base.unsafe_offset(2 * nd), k0, 3 * nd, tid)
        barrier()
        var s = SIMD[DType.float32, NC](0.0)
        var tmax = NEG_INF
        comptime for c in range(NC):
            var j = q4 + 4 * c
            var acc: Float32 = 0.0
            comptime for d in range(HD):
                acc += qreg[d] * Ks[unsafe_offset=j * LD + d]
            s[c] = acc
            tmax = max(tmax, acc)
        var m_new = max(m, row_reduce4(red, tmax, tid, True))
        var tsum: Float32 = 0.0
        comptime for c in range(NC):
            var p = exp(s[c] - m_new)
            Ps[unsafe_offset=r * (BK + 1) + q4 + 4 * c] = p
            tsum += p
        var alpha = exp(m - m_new)
        l = l * alpha + row_reduce4(red, tsum, tid, False)  # its barriers also publish Ps
        m = m_new
        o *= alpha
        for j in range(BK):
            var p = Ps[unsafe_offset=r * (BK + 1) + j]
            comptime for e in range(SL):
                o[e] += p * Vs[unsafe_offset=j * LD + q4 * SL + e]
        barrier()  # Ks, Vs and Ps are overwritten by the next tile

    var out = att.unsafe_offset((b * T + t) * nd + h * HD + q4 * SL)
    var inv = 1.0 / l
    comptime for e in range(SL):
        out[unsafe_offset=e] = o[e] * inv
    if q4 == 0:
        lse[unsafe_offset=bh * T + t] = m + log(l)


def k_attn_dvec(dvec: FP, att: FP, datt: FP, D: Int32, H: Int32, hd: Int32, n: Int32):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        b_attn_dvec(dvec, att, datt, Int(D), Int(H), Int(hd), i)


def k_attn_bwd_kv[HD: Int](dqkv: FP, qkv: FP, datt: FP, lse: FP, dvec: FP, D: Int32, H: Int32,
                           scale: Float32):
    """One block = BQ key rows of one (sprite, head), streaming query tiles of BK rows.
    grid = B * H * (T / BQ)."""
    comptime LD = HD + 1
    comptime SL = HD // 4
    comptime NC = BK // 4
    var Qs = stack_allocation[BK * LD, Float32, address_space=AddressSpace.SHARED]()
    var dOs = stack_allocation[BK * LD, Float32, address_space=AddressSpace.SHARED]()
    var Ps = stack_allocation[BQ * (BK + 1), Float32, address_space=AddressSpace.SHARED]()
    var dSs = stack_allocation[BQ * (BK + 1), Float32, address_space=AddressSpace.SHARED]()
    var Ls = stack_allocation[BK, Float32, address_space=AddressSpace.SHARED]()
    var Dv = stack_allocation[BK, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var nd = Int(D)
    var nh = Int(H)
    var tiles = T // BQ
    var bh = Int(block_idx.x) // tiles
    var j0 = (Int(block_idx.x) - bh * tiles) * BQ
    var b = bh // nh
    var h = bh - b * nh
    var r = tid // 4
    var q4 = tid % 4
    var j = j0 + r
    var base = qkv.unsafe_offset(b * T * 3 * nd + h * HD)
    var dbase = datt.unsafe_offset(b * T * nd + h * HD)

    var kreg = SIMD[DType.float32, HD](0.0)
    var vreg = SIMD[DType.float32, HD](0.0)
    comptime for d in range(HD):
        kreg[d] = base[unsafe_offset=j * 3 * nd + nd + d] * scale
        vreg[d] = base[unsafe_offset=j * 3 * nd + 2 * nd + d]
    var dk = SIMD[DType.float32, SL](0.0)
    var dv = SIMD[DType.float32, SL](0.0)

    for t0 in range(0, T, BK):
        load_tile[HD](Qs, base, t0, 3 * nd, tid)
        load_tile[HD](dOs, dbase, t0, nd, tid)
        if tid < BK:
            Ls[unsafe_offset=tid] = lse[unsafe_offset=bh * T + t0 + tid]
            Dv[unsafe_offset=tid] = dvec[unsafe_offset=bh * T + t0 + tid]
        barrier()
        comptime for c in range(NC):
            var i = q4 + 4 * c
            var s: Float32 = 0.0
            var dp: Float32 = 0.0
            comptime for d in range(HD):
                s += Qs[unsafe_offset=i * LD + d] * kreg[d]
                dp += dOs[unsafe_offset=i * LD + d] * vreg[d]
            var p = exp(s - Ls[unsafe_offset=i])
            Ps[unsafe_offset=r * (BK + 1) + i] = p
            dSs[unsafe_offset=r * (BK + 1) + i] = p * (dp - Dv[unsafe_offset=i])
        barrier()
        for i in range(BK):
            var p = Ps[unsafe_offset=r * (BK + 1) + i]
            var ds = dSs[unsafe_offset=r * (BK + 1) + i]
            comptime for e in range(SL):
                dv[e] += p * dOs[unsafe_offset=i * LD + q4 * SL + e]
                dk[e] += ds * Qs[unsafe_offset=i * LD + q4 * SL + e]
        barrier()

    var row = dqkv.unsafe_offset((b * T + j) * 3 * nd + h * HD + q4 * SL)
    comptime for e in range(SL):
        row[unsafe_offset=nd + e] = dk[e] * scale
        row[unsafe_offset=2 * nd + e] = dv[e]


def k_attn_bwd_q[HD: Int](dqkv: FP, qkv: FP, datt: FP, lse: FP, dvec: FP, D: Int32, H: Int32,
                          scale: Float32):
    """One block = BQ query rows of one (sprite, head), streaming key tiles. grid as k_attn_fwd."""
    comptime LD = HD + 1
    comptime SL = HD // 4
    comptime NC = BK // 4
    var Ks = stack_allocation[BK * LD, Float32, address_space=AddressSpace.SHARED]()
    var Vs = stack_allocation[BK * LD, Float32, address_space=AddressSpace.SHARED]()
    var dSs = stack_allocation[BQ * (BK + 1), Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var nd = Int(D)
    var nh = Int(H)
    var tiles = T // BQ
    var bh = Int(block_idx.x) // tiles
    var t0 = (Int(block_idx.x) - bh * tiles) * BQ
    var b = bh // nh
    var h = bh - b * nh
    var r = tid // 4
    var q4 = tid % 4
    var t = t0 + r
    var base = qkv.unsafe_offset(b * T * 3 * nd + h * HD)

    var qreg = SIMD[DType.float32, HD](0.0)
    var doreg = SIMD[DType.float32, HD](0.0)
    comptime for d in range(HD):
        qreg[d] = base[unsafe_offset=t * 3 * nd + d] * scale
        doreg[d] = datt[unsafe_offset=(b * T + t) * nd + h * HD + d]
    var l = lse[unsafe_offset=bh * T + t]
    var dvi = dvec[unsafe_offset=bh * T + t]
    var dq = SIMD[DType.float32, SL](0.0)

    for k0 in range(0, T, BK):
        load_tile[HD](Ks, base.unsafe_offset(nd), k0, 3 * nd, tid)
        load_tile[HD](Vs, base.unsafe_offset(2 * nd), k0, 3 * nd, tid)
        barrier()
        comptime for c in range(NC):
            var j = q4 + 4 * c
            var s: Float32 = 0.0
            var dp: Float32 = 0.0
            comptime for d in range(HD):
                s += qreg[d] * Ks[unsafe_offset=j * LD + d]
                dp += doreg[d] * Vs[unsafe_offset=j * LD + d]
            dSs[unsafe_offset=r * (BK + 1) + j] = exp(s - l) * (dp - dvi)
        barrier()
        for j in range(BK):
            var ds = dSs[unsafe_offset=r * (BK + 1) + j]
            comptime for e in range(SL):
                dq[e] += ds * Ks[unsafe_offset=j * LD + q4 * SL + e]
        barrier()

    var row = dqkv.unsafe_offset((b * T + t) * 3 * nd + h * HD + q4 * SL)
    comptime for e in range(SL):
        row[unsafe_offset=e] = dq[e] * scale
