# GPU kernels shared by every architecture's engine (src/gpu.mojo launches them).
#
# Each op is written once as a per-thread "body" function b_*(..., i). On a GPU build the
# body is launched as a kernel k_* (one thread per output element); otherwise the same body
# runs in a plain CPU loop. That CPU path is slow, but it lets us test the exact kernel logic
# against the gradient-checked reference model (gpu_check.mojo).

from std.math import sqrt, exp, tanh, log
from std.memory import stack_allocation
from max.gpu import global_idx, thread_idx, block_idx, barrier, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.memory import AddressSpace

comptime FP = Pointer[Float32, MutAnyOrigin]
comptime IP = Pointer[Int32, MutAnyOrigin]
comptime BLOCK = 256
comptime NCHUNK = 1024  # partial sums for gradient norm and loss
comptime TK = 16  # K-slice held in shared memory by the tiled matmul
comptime SPLITK = 32  # max K-splits for weight-gradient matmuls (K = rows, few output tiles)
comptime GELU_C: Float32 = 0.7978845608


def blocks(n: Int) -> Int:
    return (n + BLOCK - 1) // BLOCK


def b_fill(p: FP, v: Float32, i: Int):
    p[unsafe_offset=i] = v


def b_axpy(y: FP, x: FP, a: Float32, i: Int):
    y[unsafe_offset=i] += a * x[unsafe_offset=i]


def b_gemm(c: FP, a: FP, b: FP, M: Int, N: Int, K: Int, H: Int,
           sAi: Int, sAk: Int, sBk: Int, sBj: Int, sCi: Int, sCj: Int,
           zbA: Int, zhA: Int, zbB: Int, zhB: Int, zbC: Int, zhC: Int,
           alpha: Float32, acc: Int, i: Int):
    """Strided, batched C = alpha * A @ B. Batch index z splits into (z // H, z % H)
    so one launch covers every (sprite, head) pair in attention."""
    var mn = M * N
    var z = i // mn
    var rem = i - z * mn
    var row = rem // N
    var col = rem - row * N
    var zb = z // H
    var zh = z - zb * H
    var pa = a.unsafe_offset(zb * zbA + zh * zhA + row * sAi)
    var pb = b.unsafe_offset(zb * zbB + zh * zhB + col * sBj)
    var s: Float32 = 0.0
    for k in range(K):
        s += pa[unsafe_offset=k * sAk] * pb[unsafe_offset=k * sBk]
    var o = zb * zbC + zh * zhC + row * sCi + col * sCj
    if acc != 0:
        c[unsafe_offset=o] += alpha * s
    else:
        c[unsafe_offset=o] = alpha * s


def b_add_bias(y: FP, bias: FP, O: Int, i: Int):
    y[unsafe_offset=i] += bias[unsafe_offset=i % O]


def b_bias_grad(db: FP, dy: FP, R: Int, O: Int, i: Int):
    var s: Float32 = 0.0
    for r in range(R):
        s += dy[unsafe_offset=r * O + i]
    db[unsafe_offset=i] += s


def b_ln_fwd(y: FP, xhat: FP, rstd: FP, x: FP, g: FP, b: FP, D: Int, i: Int):
    var xr = x.unsafe_offset(i * D)
    var mean: Float32 = 0.0
    for d in range(D):
        mean += xr[unsafe_offset=d]
    mean /= Float32(D)
    var v: Float32 = 0.0
    for d in range(D):
        var e = xr[unsafe_offset=d] - mean
        v += e * e
    v /= Float32(D)
    var r = 1.0 / sqrt(v + 1e-5)
    rstd[unsafe_offset=i] = r
    for d in range(D):
        var xh = (xr[unsafe_offset=d] - mean) * r
        xhat[unsafe_offset=i * D + d] = xh
        y[unsafe_offset=i * D + d] = xh * g[unsafe_offset=d] + b[unsafe_offset=d]


def b_ln_bwd_dx(dx: FP, dy: FP, xhat: FP, rstd: FP, g: FP, D: Int, i: Int):
    var m1: Float32 = 0.0
    var m2: Float32 = 0.0
    for d in range(D):
        var dxh = dy[unsafe_offset=i * D + d] * g[unsafe_offset=d]
        m1 += dxh
        m2 += dxh * xhat[unsafe_offset=i * D + d]
    m1 /= Float32(D)
    m2 /= Float32(D)
    var r = rstd[unsafe_offset=i]
    for d in range(D):
        var o = i * D + d
        dx[unsafe_offset=o] += r * (dy[unsafe_offset=o] * g[unsafe_offset=d] - m1 - xhat[unsafe_offset=o] * m2)


def b_ln_bwd_dgb(dg: FP, db: FP, dy: FP, xhat: FP, R: Int, D: Int, i: Int):
    var sg: Float32 = 0.0
    var sb: Float32 = 0.0
    for r in range(R):
        var d = dy[unsafe_offset=r * D + i]
        sg += d * xhat[unsafe_offset=r * D + i]
        sb += d
    dg[unsafe_offset=i] += sg
    db[unsafe_offset=i] += sb


def b_gelu_fwd(y: FP, x: FP, i: Int):
    var v = x[unsafe_offset=i]
    y[unsafe_offset=i] = 0.5 * v * (1.0 + tanh(GELU_C * (v + 0.044715 * v * v * v)))


def b_gelu_bwd(dx: FP, dy: FP, x: FP, i: Int):
    var v = x[unsafe_offset=i]
    var th = tanh(GELU_C * (v + 0.044715 * v * v * v))
    var d = 0.5 * (1.0 + th) + 0.5 * v * (1.0 - th * th) * GELU_C * (1.0 + 3.0 * 0.044715 * v * v)
    dx[unsafe_offset=i] = dy[unsafe_offset=i] * d


def b_chunk_sum(dst: FP, x: FP, n: Int, square: Int, i: Int):
    var per = (n + NCHUNK - 1) // NCHUNK
    var s: Float32 = 0.0
    for j in range(i * per, min(n, (i + 1) * per)):
        var v = x[unsafe_offset=j]
        s += v * v if square != 0 else v
    dst[unsafe_offset=i] = s


def b_adamw(w: FP, g: FP, m: FP, v: FP, dec: FP, gmul: Float32, lr: Float32,
            c1: Float32, c2: Float32, wd: Float32, i: Int):
    var b1: Float32 = 0.9
    var b2: Float32 = 0.95
    var gi = g[unsafe_offset=i] * gmul
    var mi = b1 * m[unsafe_offset=i] + (1.0 - b1) * gi
    var vi = b2 * v[unsafe_offset=i] + (1.0 - b2) * gi * gi
    m[unsafe_offset=i] = mi
    v[unsafe_offset=i] = vi
    var upd = (mi / c1) / (sqrt(vi / c2) + 1e-8) + wd * dec[unsafe_offset=i] * w[unsafe_offset=i]
    w[unsafe_offset=i] -= lr * upd


def k_fill(p: FP, v: Float32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_fill(p, v, i)


def k_axpy(y: FP, x: FP, a: Float32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_axpy(y, x, a, i)


def k_gemm(c: FP, a: FP, b: FP, M: Int32, N: Int32, K: Int32, H: Int32,
           sAi: Int32, sAk: Int32, sBk: Int32, sBj: Int32, sCi: Int32, sCj: Int32,
           zbA: Int32, zhA: Int32, zbB: Int32, zhB: Int32, zbC: Int32, zhC: Int32,
           alpha: Float32, acc: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_gemm(c, a, b, Int(M), Int(N), Int(K), Int(H), Int(sAi), Int(sAk), Int(sBk), Int(sBj),
               Int(sCi), Int(sCj), Int(zbA), Int(zhA), Int(zbB), Int(zhB), Int(zbC), Int(zhC),
               alpha, Int(acc), i)


def k_gemm_tiled[TM: Int, TN: Int, RM: Int, RN: Int](
        c: FP, a: FP, b: FP, M: Int32, N: Int32, K: Int32, H: Int32,
        sAi: Int32, sAk: Int32, sBk: Int32, sBj: Int32, sCi: Int32, sCj: Int32,
        zbA: Int32, zhA: Int32, zbB: Int32, zhB: Int32, zbC: Int32, zhC: Int32,
        alpha: Float32, acc: Int32, tiles_m: Int32, tiles_n: Int32, bias: FP, has_bias: Int32):
    """Same contract as b_gemm (plus bias[col] when has_bias), but one 256-thread block computes a TM x TN tile of C.
    A and B are staged through shared memory TK columns at a time, and each thread keeps
    an RM x RN block of outputs in registers. Rows/columns of a thread's block are strided
    by the thread-grid size, so shared-memory reads and global writes stay contiguous."""
    comptime TX = TN // RN
    comptime TY = TM // RM
    comptime LDA = TM + 1  # +1 padding avoids bank conflicts on transposed stores
    comptime LDB = TN + 1
    var As = stack_allocation[TK * LDA, Float32, address_space=AddressSpace.SHARED]()
    var Bs = stack_allocation[TK * LDB, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var nm = Int(M)
    var nn = Int(N)
    var nk = Int(K)
    var per_z = Int(tiles_m) * Int(tiles_n)
    var z = blk // per_z
    var rem = blk - z * per_z
    var tm = rem // Int(tiles_n)
    var tn = rem - tm * Int(tiles_n)
    var zb = z // Int(H)
    var zh = z - zb * Int(H)
    var pa = a.unsafe_offset(zb * Int(zbA) + zh * Int(zhA))
    var pb = b.unsafe_offset(zb * Int(zbB) + zh * Int(zhB))
    var ai = Int(sAi)
    var ak = Int(sAk)
    var bk = Int(sBk)
    var bj = Int(sBj)
    var row0 = tm * TM
    var col0 = tn * TN
    var ty = tid // TX
    var tx = tid - ty * TX
    var accv = SIMD[DType.float32, RM * RN](0.0)

    for k0 in range(0, nk, TK):
        # Stage A[row0:row0+TM, k0:k0+TK]; neighbouring threads follow A's unit stride.
        comptime for e in range((TK * TM) // 256):
            var idx = tid + e * 256
            var r: Int
            var kk: Int
            if ak == 1:
                r = idx // TK
                kk = idx - r * TK
            else:
                kk = idx // TM
                r = idx - kk * TM
            var v: Float32 = 0.0
            if row0 + r < nm and k0 + kk < nk:
                v = pa[unsafe_offset=(row0 + r) * ai + (k0 + kk) * ak]
            As[unsafe_offset=kk * LDA + r] = v
        # Stage B[k0:k0+TK, col0:col0+TN].
        comptime for e in range((TK * TN) // 256):
            var idx = tid + e * 256
            var cc: Int
            var kk: Int
            if bj == 1:
                kk = idx // TN
                cc = idx - kk * TN
            else:
                cc = idx // TK
                kk = idx - cc * TK
            var v: Float32 = 0.0
            if col0 + cc < nn and k0 + kk < nk:
                v = pb[unsafe_offset=(k0 + kk) * bk + (col0 + cc) * bj]
            Bs[unsafe_offset=kk * LDB + cc] = v
        barrier()
        for kk in range(TK):
            var av = SIMD[DType.float32, RM](0.0)
            comptime for i in range(RM):
                av[i] = As[unsafe_offset=kk * LDA + ty + i * TY]
            comptime for j in range(RN):
                var bv = Bs[unsafe_offset=kk * LDB + tx + j * TX]
                comptime for i in range(RM):
                    accv[i * RN + j] += av[i] * bv
        barrier()

    var cb = zb * Int(zbC) + zh * Int(zhC)
    comptime for i in range(RM):
        var r = row0 + ty + i * TY
        if r < nm:
            comptime for j in range(RN):
                var cc = col0 + tx + j * TX
                if cc < nn:
                    var o = cb + r * Int(sCi) + cc * Int(sCj)
                    var v = alpha * accv[i * RN + j]
                    if has_bias != 0:
                        v += bias[unsafe_offset=cc]
                    if acc != 0:
                        c[unsafe_offset=o] += v
                    else:
                        c[unsafe_offset=o] = v


def k_splitk_reduce(c: FP, part: FP, S: Int32, M: Int32, N: Int32, sCi: Int32, sCj: Int32,
                    alpha: Float32, acc: Int32):
    """C (+)= alpha * sum over the S partial products written by a split-K gemm."""
    var i = Int(global_idx.x)
    var mn = Int(M) * Int(N)
    if i >= mn:
        return
    var s: Float32 = 0.0
    for k in range(Int(S)):
        s += part[unsafe_offset=k * mn + i]
    var row = i // Int(N)
    var col = i - row * Int(N)
    var o = row * Int(sCi) + col * Int(sCj)
    if acc != 0:
        c[unsafe_offset=o] += alpha * s
    else:
        c[unsafe_offset=o] = alpha * s


def k_add_bias(y: FP, bias: FP, O: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_add_bias(y, bias, Int(O), i)


def k_colsum_partial(part: FP, dy: FP, xhat: FP, R: Int32, O: Int32, S: Int32, with_xhat: Int32):
    """Column sums of dy over one of S row-chunks; thread i = (chunk, column). With xhat it
    also sums dy * xhat (the layer-norm gain gradient) into the second half of the chunk."""
    var i = Int(global_idx.x)
    var o = Int(O)
    if i >= Int(S) * o:
        return
    var s = i // o
    var col = i - s * o
    var per = (Int(R) + Int(S) - 1) // Int(S)
    var sb: Float32 = 0.0
    var sg: Float32 = 0.0
    for r in range(s * per, min(Int(R), (s + 1) * per)):
        var d = dy[unsafe_offset=r * o + col]
        sb += d
        if with_xhat != 0:
            sg += d * xhat[unsafe_offset=r * o + col]
    part[unsafe_offset=s * 2 * o + col] = sb
    part[unsafe_offset=s * 2 * o + o + col] = sg


def k_colsum_reduce(db: FP, dg: FP, part: FP, O: Int32, S: Int32, with_xhat: Int32):
    var i = Int(global_idx.x)
    var o = Int(O)
    if i >= o:
        return
    var sb: Float32 = 0.0
    var sg: Float32 = 0.0
    for s in range(Int(S)):
        sb += part[unsafe_offset=s * 2 * o + i]
        sg += part[unsafe_offset=s * 2 * o + o + i]
    db[unsafe_offset=i] += sb
    if with_xhat != 0:
        dg[unsafe_offset=i] += sg


def k_bias_grad(db: FP, dy: FP, R: Int32, O: Int32):
    var i = Int(global_idx.x)
    if i < Int(O):
        b_bias_grad(db, dy, Int(R), Int(O), i)


def b_dyt_fwd(y: FP, xs: FP, x: FP, g: FP, b: FP, a: FP, D: Int, i: Int):
    var d = i % D
    var v = x[unsafe_offset=i]
    xs[unsafe_offset=i] = v
    y[unsafe_offset=i] = g[unsafe_offset=d] * tanh(a[unsafe_offset=0] * v) + b[unsafe_offset=d]


def b_dyt_bwd_dx(dx: FP, dy: FP, xs: FP, g: FP, a: FP, D: Int, i: Int):
    var a0 = a[unsafe_offset=0]
    var t = tanh(a0 * xs[unsafe_offset=i])
    dx[unsafe_offset=i] += dy[unsafe_offset=i] * g[unsafe_offset=i % D] * a0 * (1.0 - t * t)


def k_dyt_fwd(y: FP, xs: FP, x: FP, g: FP, b: FP, a: FP, D: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_dyt_fwd(y, xs, x, g, b, a, Int(D), i)


def k_dyt_bwd_dx(dx: FP, dy: FP, xs: FP, g: FP, a: FP, D: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_dyt_bwd_dx(dx, dy, xs, g, a, Int(D), i)


def k_dyt_colsum_partial(part: FP, dy: FP, xs: FP, a: FP, R: Int32, O: Int32, S: Int32):
    """Like k_colsum_partial, the column sums of dy, dy * t and dy * (1 - t^2) * x (t = tanh(a x))
    over one of S row chunks: the gradients of DyT's b, g and (after weighting by g) a."""
    var i = Int(global_idx.x)
    var o = Int(O)
    if i >= Int(S) * o:
        return
    var s = i // o
    var col = i - s * o
    var a0 = a[unsafe_offset=0]
    var per = (Int(R) + Int(S) - 1) // Int(S)
    var sb: Float32 = 0.0
    var sg: Float32 = 0.0
    var sa: Float32 = 0.0
    for r in range(s * per, min(Int(R), (s + 1) * per)):
        var e = dy[unsafe_offset=r * o + col]
        var v = xs[unsafe_offset=r * o + col]
        var t = tanh(a0 * v)
        sb += e
        sg += e * t
        sa += e * (1.0 - t * t) * v
    part[unsafe_offset=s * 3 * o + col] = sb
    part[unsafe_offset=s * 3 * o + o + col] = sg
    part[unsafe_offset=s * 3 * o + 2 * o + col] = sa


def k_dyt_colsum_reduce(db: FP, dg: FP, da: FP, part: FP, g: FP, O: Int32, S: Int32):
    """db, dg += the partials' column sums; da[0] += sum over columns of g * (the third sum). One block."""
    var red = stack_allocation[BLOCK, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var o = Int(O)
    var acc: Float32 = 0.0
    for col in range(tid, o, BLOCK):
        var sb: Float32 = 0.0
        var sg: Float32 = 0.0
        var sa: Float32 = 0.0
        for s in range(Int(S)):
            sb += part[unsafe_offset=s * 3 * o + col]
            sg += part[unsafe_offset=s * 3 * o + o + col]
            sa += part[unsafe_offset=s * 3 * o + 2 * o + col]
        db[unsafe_offset=col] += sb
        dg[unsafe_offset=col] += sg
        acc += g[unsafe_offset=col] * sa
    red[unsafe_offset=tid] = acc
    barrier()
    var half = BLOCK // 2
    while half > 0:
        if tid < half:
            red[unsafe_offset=tid] += red[unsafe_offset=tid + half]
        barrier()
        half //= 2
    if tid == 0:
        da[unsafe_offset=0] += red[unsafe_offset=0]


comptime LN_ROWS = BLOCK // 64  # rows per block the host launches for: enough for a warp of 32 or 64


def k_ln_fwd_warp(y: FP, xhat: FP, rstd: FP, x: FP, g: FP, b: FP, D: Int32, n: Int32):
    """b_ln_fwd with one warp per row, so the row's reads are coalesced across lanes."""
    var lane = Int(thread_idx.x) % WARP_SIZE
    var i = Int(block_idx.x) * (BLOCK // WARP_SIZE) + Int(thread_idx.x) // WARP_SIZE
    if i >= Int(n):
        return
    var nd = Int(D)
    var xr = x.unsafe_offset(i * nd)
    var s: Float32 = 0.0
    for d in range(lane, nd, WARP_SIZE):
        s += xr[unsafe_offset=d]
    var mean = warp.sum(s) / Float32(nd)
    var v: Float32 = 0.0
    for d in range(lane, nd, WARP_SIZE):
        var e = xr[unsafe_offset=d] - mean
        v += e * e
    var r = 1.0 / sqrt(warp.sum(v) / Float32(nd) + 1e-5)
    if lane == 0:
        rstd[unsafe_offset=i] = r
    for d in range(lane, nd, WARP_SIZE):
        var xh = (xr[unsafe_offset=d] - mean) * r
        xhat[unsafe_offset=i * nd + d] = xh
        y[unsafe_offset=i * nd + d] = xh * g[unsafe_offset=d] + b[unsafe_offset=d]


def k_ln_bwd_dx_warp(dx: FP, dy: FP, xhat: FP, rstd: FP, g: FP, D: Int32, n: Int32):
    """b_ln_bwd_dx with one warp per row."""
    var lane = Int(thread_idx.x) % WARP_SIZE
    var i = Int(block_idx.x) * (BLOCK // WARP_SIZE) + Int(thread_idx.x) // WARP_SIZE
    if i >= Int(n):
        return
    var nd = Int(D)
    var m1: Float32 = 0.0
    var m2: Float32 = 0.0
    for d in range(lane, nd, WARP_SIZE):
        var dxh = dy[unsafe_offset=i * nd + d] * g[unsafe_offset=d]
        m1 += dxh
        m2 += dxh * xhat[unsafe_offset=i * nd + d]
    m1 = warp.sum(m1) / Float32(nd)
    m2 = warp.sum(m2) / Float32(nd)
    var r = rstd[unsafe_offset=i]
    for d in range(lane, nd, WARP_SIZE):
        var o = i * nd + d
        dx[unsafe_offset=o] += r * (dy[unsafe_offset=o] * g[unsafe_offset=d] - m1 - xhat[unsafe_offset=o] * m2)


def k_ln_bwd_dgb(dg: FP, db: FP, dy: FP, xhat: FP, R: Int32, D: Int32):
    var i = Int(global_idx.x)
    if i < Int(D):
        b_ln_bwd_dgb(dg, db, dy, xhat, Int(R), Int(D), i)


def k_gelu_fwd(y: FP, x: FP, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_gelu_fwd(y, x, i)


def k_gelu_bwd(dx: FP, dy: FP, x: FP, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_gelu_bwd(dx, dy, x, i)


def k_chunk_sum(dst: FP, x: FP, n: Int32, square: Int32):
    var i = Int(global_idx.x)
    if i < NCHUNK:
        b_chunk_sum(dst, x, Int(n), Int(square), i)


def k_adamw(w: FP, g: FP, m: FP, v: FP, dec: FP, gmul: Float32, lr: Float32,
            c1: Float32, c2: Float32, wd: Float32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_adamw(w, g, m, v, dec, gmul, lr, c1, c2, wd, i)


def k_muon_prep(x: FP, m: FP, g: FP, gmul: Float32, mu: Float32, n: Int32):
    """Momentum buffer m = mu m + g, and x = g + mu m (Nesterov direction) for one matrix."""
    var i = Int(global_idx.x)
    if i < Int(n):
        var gi = g[unsafe_offset=i] * gmul
        var mi = mu * m[unsafe_offset=i] + gi
        m[unsafe_offset=i] = mi
        x[unsafe_offset=i] = gi + mu * mi


def k_sumsq_blocks(part: FP, x: FP, per: Int32, n: Int32):
    """part[block] = sum of x^2 over the block's per * BLOCK elements."""
    var red = stack_allocation[BLOCK, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var base = Int(block_idx.x) * BLOCK * Int(per)
    var s: Float32 = 0.0
    for e in range(Int(per)):
        var j = base + e * BLOCK + tid
        if j < Int(n):
            var v = x[unsafe_offset=j]
            s += v * v
    red[unsafe_offset=tid] = s
    barrier()
    var half = BLOCK // 2
    while half > 0:
        if tid < half:
            red[unsafe_offset=tid] += red[unsafe_offset=tid + half]
        barrier()
        half //= 2
    if tid == 0:
        part[unsafe_offset=Int(block_idx.x)] = red[unsafe_offset=0]


def k_scale_inv_norm(x: FP, part: FP, nb: Int32, n: Int32):
    """x /= ||x||_F from the nb partials of k_sumsq_blocks, which every block sums itself (no host
    round trip, and many blocks instead of one: the matrices are up to F x D)."""
    var red = stack_allocation[BLOCK, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var s: Float32 = 0.0
    for j in range(tid, Int(nb), BLOCK):
        s += part[unsafe_offset=j]
    red[unsafe_offset=tid] = s
    barrier()
    var half = BLOCK // 2
    while half > 0:
        if tid < half:
            red[unsafe_offset=tid] += red[unsafe_offset=tid + half]
        barrier()
        half //= 2
    var i = Int(global_idx.x)
    if i < Int(n):
        x[unsafe_offset=i] *= 1.0 / (sqrt(red[unsafe_offset=0]) + 1e-7)


def k_muon_apply(w: FP, x: FP, dec: FP, lr: Float32, wd: Float32, scale: Float32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        w[unsafe_offset=i] -= lr * (scale * x[unsafe_offset=i] + wd * dec[unsafe_offset=i] * w[unsafe_offset=i])
