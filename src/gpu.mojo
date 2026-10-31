# Shared GPU plumbing for every architecture's training engine.
#
# Each op is written once as a per-thread "body" function b_*(..., i). On a GPU build
# the body is launched as a kernel k_* (one thread per output element); otherwise the
# same body runs in a plain CPU loop. That CPU path is slow, but it lets us test the
# exact kernel logic against the gradient-checked reference model (gpu_check.mojo).
#
# Device owns one float and one int arena in device memory, laid out as
#   [architecture activations | parameters, gradients, Adam moments, decay mask | scratch]
# An architecture's Engine computes its own activation offsets from 0, then creates the
# Device, which appends the parameter and scratch regions. Activations are [R, D] row-major
# with R = batch * tokens.

from std.math import sqrt, exp, tanh, log
from std.sys import has_accelerator
from std.memory import stack_allocation
from max.gpu import global_idx, thread_idx, block_idx, barrier, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host.device_context import DeviceExternalFunction
from max.gpu.host._amdgpu_hip import HIP
from std.ffi import OwnedDLHandle
from src.metal import MSL, gemm_stub, attn_fwd_stub, attn_bwd_stub

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


def metal_kernels_available(use_gpu: Bool) raises -> Bool:
    """Whether Device will load the Metal kernels (src/metal.mojo): an Apple GPU in use. Lets an
    engine size its buffers for the fused attention before creating the Device."""
    comptime if has_accelerator():
        if use_gpu:
            return DeviceContext().api() == "metal"
    return False


def _fp(l: List[Float32]) -> FP:
    return l.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


def _ip(l: List[Int32]) -> IP:
    return l.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


# ============================================================================ device


struct Device(Movable):
    var R: Int  # activation rows
    var D: Int  # activation width (layer norm size)
    var gpu: Bool
    var ctx: Optional[DeviceContext]
    var fdev: Optional[DeviceBuffer[DType.float32]]
    var idev: Optional[DeviceBuffer[DType.int32]]
    var fhost: List[Float32]  # CPU-emulation storage
    var ihost: List[Int32]
    var nparam: Int
    var o_w: Int
    var o_g: Int
    var o_m: Int
    var o_v: Int
    var o_dec: Int
    var o_partial: Int
    var o_splitk: Int
    var splitk_cap: Int
    var o_mu: Int  # Muon scratch: X, Y (largest matrix each), then Gram A, B (<= largest matrix each)
    var muon_mm: Int
    var muon: Bool
    var mats: List[Int]  # (offset, rows, cols) of each Muon weight matrix, in ascending offset order
    var muon_mu: Float32
    var nf: Int  # floats in the arena
    # Metal simdgroup-matrix gemms (src/metal.mojo), by A/B orientation; None off Apple GPUs.
    var g_nn: Optional[DeviceExternalFunction[]]
    var g_nt: Optional[DeviceExternalFunction[]]
    var g_tn: Optional[DeviceExternalFunction[]]
    var g_tt: Optional[DeviceExternalFunction[]]
    var a_fwd: Optional[DeviceExternalFunction[]]  # fused attention, head size 32
    var a_bkv: Optional[DeviceExternalFunction[]]
    var a_bq: Optional[DeviceExternalFunction[]]
    # rocBLAS on AMD GPUs (when librocblas.so loads): its sgemm replaces the tiled matmul.
    var rocblas: Optional[OwnedDLHandle]
    var rb_handle: Int  # rocblas_handle, an opaque C pointer

    def __init__(out self, R: Int, D: Int, nparam: Int, splitk_cap: Int, arch_floats: Int,
                 arch_ints: Int, use_gpu: Bool) raises:
        """splitk_cap: scratch floats for split-K partials (SPLITK * the largest weight matrix)."""
        self.R = R
        self.D = D
        self.nparam = nparam
        self.splitk_cap = splitk_cap
        var o = (arch_floats + 63) // 64 * 64  # aligned, for the Metal gemm's float4 loads
        self.o_w = o
        self.o_g = o + nparam
        self.o_m = o + 2 * nparam
        self.o_v = o + 3 * nparam
        self.o_dec = o + 4 * nparam
        self.o_partial = o + 5 * nparam
        self.o_splitk = self.o_partial + NCHUNK
        self.muon_mm = splitk_cap // SPLITK
        self.o_mu = self.o_splitk + splitk_cap
        self.muon = False
        self.mats = List[Int]()
        self.muon_mu = 0.95
        self.nf = self.o_mu + 4 * self.muon_mm
        var ni = max(arch_ints, 1)
        self.gpu = False
        self.ctx = None
        self.fdev = None
        self.idev = None
        self.fhost = List[Float32]()
        self.ihost = List[Int32]()
        self.g_nn = None
        self.g_nt = None
        self.g_tn = None
        self.g_tt = None
        self.a_fwd = None
        self.a_bkv = None
        self.a_bq = None
        self.rocblas = None
        self.rb_handle = 0
        comptime if has_accelerator():
            if use_gpu:
                var c = DeviceContext()
                var fb = c.enqueue_create_buffer[DType.float32](self.nf)
                var ib = c.enqueue_create_buffer[DType.int32](ni)
                fb.enqueue_fill(0.0)
                ib.enqueue_fill(0)
                c.synchronize()
                if c.api() == "hip":
                    try:
                        var lib = OwnedDLHandle("librocblas.so")
                        var h: Int = 0
                        if lib.call["rocblas_create_handle", Int32](Pointer(to=h)) == 0 and \
                                lib.call["rocblas_set_stream", Int32](h, HIP(c.stream())) == 0:
                            self.rocblas = lib^
                            self.rb_handle = h
                    except:
                        pass  # no rocBLAS installed: the Mojo kernels below do the matmuls
                if c.api() == "metal":
                    self.g_nn = c.load_function[gemm_stub](function_name="gemm_nn", asm=MSL)
                    self.g_nt = c.load_function[gemm_stub](function_name="gemm_nt", asm=MSL)
                    self.g_tn = c.load_function[gemm_stub](function_name="gemm_tn", asm=MSL)
                    self.g_tt = c.load_function[gemm_stub](function_name="gemm_tt", asm=MSL)
                    self.a_fwd = c.load_function[attn_fwd_stub](function_name="attn_fwd32", asm=MSL)
                    self.a_bkv = c.load_function[attn_bwd_stub](function_name="attn_bwd_kv32", asm=MSL)
                    self.a_bq = c.load_function[attn_bwd_stub](function_name="attn_bwd_q32", asm=MSL)
                self.fdev = fb^
                self.idev = ib^
                self.ctx = c^
                self.gpu = True
        if not self.gpu:
            self.fhost = List[Float32](length=self.nf, fill=0.0)
            self.ihost = List[Int32](length=ni, fill=0)

    def device_name(self) raises -> String:
        comptime if has_accelerator():
            if self.gpu:
                return self.ctx.value().name()
        return "CPU (kernel emulation)"

    def F(self, off: Int) -> FP:
        if self.gpu:
            return self.fdev.value().unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[
                MutAnyOrigin]().unsafe_offset(off)
        return _fp(self.fhost).unsafe_offset(off)

    def I(self, off: Int) -> IP:
        if self.gpu:
            return self.idev.value().unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[
                MutAnyOrigin]().unsafe_offset(off)
        return _ip(self.ihost).unsafe_offset(off)

    def upload_f(self, off: Int, src: List[Float32]) raises:
        comptime if has_accelerator():
            if self.gpu:
                var ctx = self.ctx.value()
                var h = ctx.enqueue_create_host_buffer[DType.float32](len(src))
                ctx.synchronize()
                for i in range(len(src)):
                    h[i] = src[i]
                ctx.enqueue_copy(dst_ptr=self.F(off), src_buf=h)
                ctx.synchronize()
                return
        var p = self.F(off)
        for i in range(len(src)):
            p[unsafe_offset=i] = src[i]

    def upload_i(self, off: Int, src: List[Int32]) raises:
        comptime if has_accelerator():
            if self.gpu:
                var ctx = self.ctx.value()
                var h = ctx.enqueue_create_host_buffer[DType.int32](len(src))
                ctx.synchronize()
                for i in range(len(src)):
                    h[i] = src[i]
                ctx.enqueue_copy(dst_ptr=self.I(off), src_buf=h)
                ctx.synchronize()
                return
        var p = self.I(off)
        for i in range(len(src)):
            p[unsafe_offset=i] = src[i]

    def download_f(self, off: Int, n: Int, mut dst: List[Float32]) raises:
        comptime if has_accelerator():
            if self.gpu:
                var ctx = self.ctx.value()
                var h = ctx.enqueue_create_host_buffer[DType.float32](n)
                ctx.enqueue_copy(dst_buf=h, src_ptr=self.F(off))
                ctx.synchronize()
                for i in range(n):
                    dst[i] = h[i]
                return
        var p = self.F(off)
        for i in range(n):
            dst[i] = p[unsafe_offset=i]

    def sync(self) raises:
        comptime if has_accelerator():
            if self.gpu:
                self.ctx.value().synchronize()

    def fill(self, p: FP, n: Int, v: Float32) raises:
        comptime if has_accelerator():
            if self.gpu:
                self.ctx.value().enqueue_function[k_fill](p, v, Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
                return
        for i in range(n):
            b_fill(p, v, i)

    def axpy(self, y: FP, x: FP, a: Float32, n: Int) raises:
        comptime if has_accelerator():
            if self.gpu:
                self.ctx.value().enqueue_function[k_axpy](y, x, a, Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
                return
        for i in range(n):
            b_axpy(y, x, a, i)

    def gemm(self, c: FP, a: FP, b: FP, M: Int, N: Int, K: Int, Z: Int, H: Int,
             sAi: Int, sAk: Int, sBk: Int, sBj: Int, sCi: Int, sCj: Int,
             zbA: Int, zhA: Int, zbB: Int, zhB: Int, zbC: Int, zhC: Int,
             alpha: Float32, acc: Bool) raises:
        var n = M * N * Z
        var ac = 1 if acc else 0
        comptime if has_accelerator():
            if self.rocblas and Z == 1 and sCj == 1 and (sAi == 1 or sAk == 1) and (sBk == 1 or sBj == 1):
                self.rocblas_gemm(c, a, b, M, N, K, sAi, sAk, sBk, sBj, sCi, alpha, acc)
                return
            if self.gpu:
                # Weight gradients have a huge K (every row of the batch) but only a few
                # output tiles: split K across blocks into scratch, then reduce.
                var tiles = Z * ((M + 63) // 64) * ((N + 63) // 64)
                var S = 1
                if Z == 1 and tiles < 128:
                    while S < SPLITK and K % (2 * S) == 0 and K // (2 * S) >= 256 and M * N * 2 * S <= self.splitk_cap:
                        S *= 2
                if S > 1:
                    var part = self.F(self.o_splitk)
                    var kc = K // S
                    self.gemm_tiled(part, a, b, M, N, kc, S, 1, sAi, sAk, sBk, sBj, N, 1,
                                    kc * sAk, 0, kc * sBk, 0, M * N, 0, 1.0, 0)
                    self.ctx.value().enqueue_function[k_splitk_reduce](
                        c, part, Int32(S), Int32(M), Int32(N), Int32(sCi), Int32(sCj), alpha, Int32(ac),
                        grid_dim=blocks(M * N), block_dim=BLOCK)
                else:
                    self.gemm_tiled(c, a, b, M, N, K, Z, H, sAi, sAk, sBk, sBj, sCi, sCj,
                                    zbA, zhA, zbB, zhB, zbC, zhC, alpha, ac)
                return
        for i in range(n):
            b_gemm(c, a, b, M, N, K, H, sAi, sAk, sBk, sBj, sCi, sCj,
                   zbA, zhA, zbB, zhB, zbC, zhC, alpha, ac, i)

    def rocblas_gemm(self, c: FP, a: FP, b: FP, M: Int, N: Int, K: Int, sAi: Int, sAk: Int,
                     sBk: Int, sBj: Int, sCi: Int, alpha: Float32, acc: Bool) raises:
        """Row-major C[M,N] (+)= alpha A B as rocBLAS's column-major C^T = B^T A^T: a row-major
        operand with unit column stride is its own transpose there, otherwise it is transposed."""
        var op_a = Int32(111) if sAk == 1 else Int32(112)  # rocblas_operation_none / _transpose
        var op_b = Int32(111) if sBj == 1 else Int32(112)
        var lda = sAi if sAk == 1 else sAk
        var ldb = sBk if sBj == 1 else sBj
        var beta: Float32 = 1.0 if acc else 0.0
        # rocBLAS 7.2 on gfx1030 picks a 5-8x slower kernel for weight gradients once K (the batch
        # rows) passes 65536, so long K is summed in equal chunks of at most that.
        var nchunks = (K + 65535) // 65536
        var kc = (K + nchunks - 1) // nchunks
        for k0 in range(0, K, kc):
            var rc = self.rocblas.value().call["rocblas_sgemm", Int32](
                self.rb_handle, op_b, op_a, Int32(N), Int32(M), Int32(min(kc, K - k0)), Pointer(to=alpha),
                b.unsafe_offset(k0 * sBk), Int32(ldb), a.unsafe_offset(k0 * sAk), Int32(lda), Pointer(to=beta),
                c, Int32(sCi))
            if rc != 0:
                raise Error("rocblas_sgemm failed with status " + String(rc))
            beta = 1.0

    def gemm_tiled(self, c: FP, a: FP, b: FP, M: Int, N: Int, K: Int, Z: Int, H: Int,
                   sAi: Int, sAk: Int, sBk: Int, sBj: Int, sCi: Int, sCj: Int,
                   zbA: Int, zhA: Int, zbB: Int, zhB: Int, zbC: Int, zhC: Int,
                   alpha: Float32, ac: Int, bias: Optional[FP] = None) raises:
        """bias: added per output column (only without split-K, i.e. here)."""
        comptime if has_accelerator():
            var ctx = self.ctx.value()
            var bp = bias.value() if bias else c
            var hb = Int32(1) if bias else Int32(0)
            if self.g_nn and self.metal_gemm_ok(c, a, b, M, N, K, sAi, sAk, sBk, sBj, sCj,
                                                 zbA, zhA, zbB, zhB):
                self.metal_gemm(c, a, b, M, N, K, Z, H, sAi, sAk, sBk, sBj, sCi,
                                zbA, zhA, zbB, zhB, zbC, zhC, alpha, ac, 0)
                return
            if N <= 32:  # narrow outputs (small test configs): tall, narrow tiles
                var tm = (M + 127) // 128
                var tn = (N + 15) // 16
                ctx.enqueue_function[k_gemm_tiled[128, 16, 8, 1]](
                    c, a, b, Int32(M), Int32(N), Int32(K), Int32(H),
                    Int32(sAi), Int32(sAk), Int32(sBk), Int32(sBj), Int32(sCi), Int32(sCj),
                    Int32(zbA), Int32(zhA), Int32(zbB), Int32(zhB), Int32(zbC), Int32(zhC),
                    alpha, Int32(ac), Int32(tm), Int32(tn), bp, hb, grid_dim=Z * tm * tn, block_dim=256)
            else:
                var tm = (M + 63) // 64
                var tn = (N + 63) // 64
                ctx.enqueue_function[k_gemm_tiled[64, 64, 4, 4]](
                    c, a, b, Int32(M), Int32(N), Int32(K), Int32(H),
                    Int32(sAi), Int32(sAk), Int32(sBk), Int32(sBj), Int32(sCi), Int32(sCj),
                    Int32(zbA), Int32(zhA), Int32(zbB), Int32(zhB), Int32(zbC), Int32(zhC),
                    alpha, Int32(ac), Int32(tm), Int32(tn), bp, hb, grid_dim=Z * tm * tn, block_dim=256)

    def metal_gemm(self, c: FP, a: FP, b: FP, M: Int, N: Int, K: Int,
                   Z: Int, H: Int, sAi: Int, sAk: Int, sBk: Int, sBj: Int, sCi: Int,
                   zbA: Int, zhA: Int, zbB: Int, zhB: Int, zbC: Int, zhC: Int,
                   alpha: Float32, ac: Int, bias: Int) raises:
        """The Metal gemm for A and B's orientation; bias is an address (0 for none)."""
        var ta = sAi == 1 and sAk != 1
        var tb = sBk == 1 and sBj != 1
        if ta and tb:
            self.launch_gemm(self.g_tt.value(), c, a, b, M, N, K, Z, H, sAi, sAk, sBk, sBj, sCi,
                             zbA, zhA, zbB, zhB, zbC, zhC, alpha, ac, bias)
        elif ta:
            self.launch_gemm(self.g_tn.value(), c, a, b, M, N, K, Z, H, sAi, sAk, sBk, sBj, sCi,
                             zbA, zhA, zbB, zhB, zbC, zhC, alpha, ac, bias)
        elif tb:
            self.launch_gemm(self.g_nt.value(), c, a, b, M, N, K, Z, H, sAi, sAk, sBk, sBj, sCi,
                             zbA, zhA, zbB, zhB, zbC, zhC, alpha, ac, bias)
        else:
            self.launch_gemm(self.g_nn.value(), c, a, b, M, N, K, Z, H, sAi, sAk, sBk, sBj, sCi,
                             zbA, zhA, zbB, zhB, zbC, zhC, alpha, ac, bias)

    def launch_gemm(self, f: DeviceExternalFunction[], c: FP, a: FP, b: FP, M: Int, N: Int, K: Int,
                    Z: Int, H: Int, sAi: Int, sAk: Int, sBk: Int, sBj: Int, sCi: Int,
                    zbA: Int, zhA: Int, zbB: Int, zhB: Int, zbC: Int, zhC: Int,
                    alpha: Float32, ac: Int, bias: Int) raises:
        comptime if has_accelerator():
            self.ctx.value().enqueue_function(
                f, c, a, b, Int32(M), Int32(N), Int32(K), Int32(H),
                Int32(sAi), Int32(sAk), Int32(sBk), Int32(sBj), Int32(sCi),
                Int32(zbA), Int32(zhA), Int32(zbB), Int32(zhB), Int32(zbC), Int32(zhC),
                alpha, Int32(ac), Int32(N // 64), UInt64(bias), grid_dim=Z * (M // 64) * (N // 64), block_dim=256)

    def metal_gemm_ok(self, c: FP, a: FP, b: FP, M: Int, N: Int, K: Int, sAi: Int, sAk: Int,
                      sBk: Int, sBj: Int, sCj: Int, zbA: Int, zhA: Int, zbB: Int, zhB: Int) -> Bool:
        """Whether the Metal gemm handles this shape: whole 64x64 tiles, K % 16 == 0, unit stride
        along one axis of A and of B, and float4-aligned A and B rows."""
        if M % 64 != 0 or N % 64 != 0 or K % 16 != 0 or sCj != 1:
            return False
        if (sAi != 1 and sAk != 1) or (sBk != 1 and sBj != 1):
            return False
        var sa = sAk if sAi == 1 and sAk != 1 else sAi
        var sb = sBj if sBk == 1 and sBj != 1 else sBk
        for v in [sa, sb, zbA, zhA, zbB, zhB]:
            if v % 4 != 0:
                return False
        return Int(a) % 16 == 0 and Int(b) % 16 == 0

    def linear_fwd(self, y: FP, x: FP, w: FP, b: FP, I: Int, O: Int, acc: Bool = False) raises:
        """y[R,O] = x[R,I] @ W[O,I]^T + b, or y += that with acc (a residual add)."""
        var R = self.R
        comptime if has_accelerator():
            if self.g_nn and self.metal_gemm_ok(y, x, w, R, O, I, I, 1, 1, I, 1, 0, 0, 0, 0):
                self.metal_gemm(y, x, w, R, O, I, 1, 1, I, 1, 1, I, O, 0, 0, 0, 0, 0, 0, 1.0, 1 if acc else 0, Int(b))
                return
            if self.gpu and not self.rocblas:  # the tiled matmul adds the bias itself
                self.gemm_tiled(y, x, w, R, O, I, 1, 1, I, 1, 1, I, O, 1, 0, 0, 0, 0, 0, 0, 1.0, 1 if acc else 0, b)
                return
        self.gemm(y, x, w, R, O, I, 1, 1, I, 1, 1, I, O, 1, 0, 0, 0, 0, 0, 0, 1.0, acc)
        comptime if has_accelerator():
            if self.gpu:
                self.ctx.value().enqueue_function[k_add_bias](y, b, Int32(O), Int32(R * O),
                                                              grid_dim=blocks(R * O), block_dim=BLOCK)
                return
        for i in range(R * O):
            b_add_bias(y, b, O, i)

    def linear_bwd(self, dx: FP, dw: FP, db: FP, dy: FP, x: FP, w: FP, I: Int, O: Int, dx_acc: Bool = True) raises:
        """Accumulates dx += dy @ W (dx = dy @ W without dx_acc), dW += dy^T @ x, db += sum(dy)."""
        var R = self.R
        self.gemm(dx, dy, w, R, I, O, 1, 1, O, 1, I, 1, I, 1, 0, 0, 0, 0, 0, 0, 1.0, dx_acc)
        self.gemm(dw, dy, x, O, I, R, 1, 1, 1, O, I, 1, I, 1, 0, 0, 0, 0, 0, 0, 1.0, True)
        comptime if has_accelerator():
            if self.gpu:
                self.colsum(db, db, dy, dy, O, False)
                return
        for i in range(O):
            b_bias_grad(db, dy, R, O, i)

    def colsum(self, db: FP, dg: FP, dy: FP, xhat: FP, O: Int, with_xhat: Bool) raises:
        """db += column sums of dy[R,O] (and dg += column sums of dy * xhat), split over row
        chunks so thousands of threads share the R = B*T rows instead of O threads."""
        comptime if has_accelerator():
            var ctx = self.ctx.value()
            var S = 64
            var part = self.F(self.o_splitk)
            var wx = Int32(1 if with_xhat else 0)
            ctx.enqueue_function[k_colsum_partial](part, dy, xhat, Int32(self.R), Int32(O), Int32(S), wx,
                                                   grid_dim=blocks(S * O), block_dim=BLOCK)
            ctx.enqueue_function[k_colsum_reduce](db, dg, part, Int32(O), Int32(S), wx,
                                                  grid_dim=blocks(O), block_dim=BLOCK)

    def ln_fwd(self, y: FP, xhat: FP, rstd: FP, x: FP, g: FP, b: FP) raises:
        var R = self.R
        var D = self.D
        comptime if has_accelerator():
            if self.gpu:
                self.ctx.value().enqueue_function[k_ln_fwd_warp](y, xhat, rstd, x, g, b, Int32(D), Int32(R),
                                                                 grid_dim=(R + LN_ROWS - 1) // LN_ROWS, block_dim=BLOCK)
                return
        for i in range(R):
            b_ln_fwd(y, xhat, rstd, x, g, b, D, i)

    def ln_bwd(self, dx: FP, dg: FP, db: FP, dy: FP, xhat: FP, rstd: FP, g: FP) raises:
        var R = self.R
        var D = self.D
        comptime if has_accelerator():
            if self.gpu:
                var ctx = self.ctx.value()
                ctx.enqueue_function[k_ln_bwd_dx_warp](dx, dy, xhat, rstd, g, Int32(D), Int32(R),
                                                       grid_dim=(R + LN_ROWS - 1) // LN_ROWS, block_dim=BLOCK)
                self.colsum(db, dg, dy, xhat, D, True)
                return
        for i in range(R):
            b_ln_bwd_dx(dx, dy, xhat, rstd, g, D, i)
        for i in range(D):
            b_ln_bwd_dgb(dg, db, dy, xhat, R, D, i)

    def dyt_fwd(self, y: FP, xs: FP, x: FP, g: FP, b: FP, a: FP) raises:
        """DyT in place of ln_fwd (src/kernels.mojo dyt_fwd); a points at the norm's scalar."""
        var n = self.R * self.D
        comptime if has_accelerator():
            if self.gpu:
                self.ctx.value().enqueue_function[k_dyt_fwd](y, xs, x, g, b, a, Int32(self.D), Int32(n),
                                                             grid_dim=blocks(n), block_dim=BLOCK)
                return
        for i in range(n):
            b_dyt_fwd(y, xs, x, g, b, a, self.D, i)

    def dyt_bwd(self, dx: FP, dg: FP, db: FP, da: FP, dy: FP, xs: FP, g: FP, a: FP) raises:
        var R = self.R
        var D = self.D
        comptime if has_accelerator():
            if self.gpu:
                var ctx = self.ctx.value()
                var S = 64
                var part = self.F(self.o_splitk)
                ctx.enqueue_function[k_dyt_bwd_dx](dx, dy, xs, g, a, Int32(D), Int32(R * D),
                                                   grid_dim=blocks(R * D), block_dim=BLOCK)
                ctx.enqueue_function[k_dyt_colsum_partial](part, dy, xs, a, Int32(R), Int32(D), Int32(S),
                                                           grid_dim=blocks(S * D), block_dim=BLOCK)
                ctx.enqueue_function[k_dyt_colsum_reduce](db, dg, da, part, g, Int32(D), Int32(S),
                                                          grid_dim=1, block_dim=BLOCK)
                return
        for i in range(R * D):
            b_dyt_bwd_dx(dx, dy, xs, g, a, D, i)
        var a0 = a[unsafe_offset=0]
        for i in range(R * D):
            var d = i % D
            var v = xs[unsafe_offset=i]
            var t = tanh(a0 * v)
            var e = dy[unsafe_offset=i]
            db[unsafe_offset=d] += e
            dg[unsafe_offset=d] += e * t
            da[unsafe_offset=0] += e * g[unsafe_offset=d] * (1.0 - t * t) * v

    def gelu_fwd(self, y: FP, x: FP, n: Int) raises:
        comptime if has_accelerator():
            if self.gpu:
                self.ctx.value().enqueue_function[k_gelu_fwd](y, x, Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
                return
        for i in range(n):
            b_gelu_fwd(y, x, i)

    def gelu_bwd(self, dx: FP, dy: FP, x: FP, n: Int) raises:
        comptime if has_accelerator():
            if self.gpu:
                self.ctx.value().enqueue_function[k_gelu_bwd](dx, dy, x, Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
                return
        for i in range(n):
            b_gelu_bwd(dx, dy, x, i)

    def chunk_sum(self, x: FP, n: Int, square: Bool) raises -> Float32:
        var sq = 1 if square else 0
        var part = self.F(self.o_partial)
        var done = False
        comptime if has_accelerator():
            if self.gpu:
                self.ctx.value().enqueue_function[k_chunk_sum](part, x, Int32(n), Int32(sq),
                                                               grid_dim=blocks(NCHUNK), block_dim=BLOCK)
                done = True
        if not done:
            for i in range(NCHUNK):
                b_chunk_sum(part, x, n, sq, i)
        var h = List[Float32](length=NCHUNK, fill=0.0)
        self.download_f(self.o_partial, NCHUNK, h)
        var s: Float64 = 0.0
        for i in range(NCHUNK):
            s += Float64(h[i])
        return Float32(s)

    def load_params(self, w: List[Float32], decay: List[UInt8]) raises:
        """Uploads weights and the weight-decay mask, and resets the Adam moments."""
        var n = len(w)
        self.upload_f(self.o_w, w)
        var dec = List[Float32](length=n, fill=0.0)
        for i in range(n):
            dec[i] = Float32(Int(decay[i]))
        self.upload_f(self.o_dec, dec)
        self.fill(self.F(self.o_m), n, 0.0)
        self.fill(self.F(self.o_v), n, 0.0)
        self.sync()

    def store_params(self, mut w: List[Float32]) raises:
        self.download_f(self.o_w, len(w), w)

    def optimizer_step(self, lr: Float32, step: Int, wd: Float32, clip: Float32) raises -> Float32:
        """Global-norm gradient clipping + AdamW. Returns the gradient norm before clipping."""
        var n = self.nparam
        var norm = sqrt(self.chunk_sum(self.F(self.o_g), n, True))
        var gmul: Float32 = 1.0
        if norm > clip:
            gmul = clip / norm
        comptime if has_accelerator():
            if self.gpu and self.muon:
                self.muon_step(lr, step, wd, gmul)
                return norm
        var c1 = 1.0 - Float32(0.9 ** Float64(step))
        var c2 = 1.0 - Float32(0.95 ** Float64(step))
        var w = self.F(self.o_w)
        var g = self.F(self.o_g)
        var mm = self.F(self.o_m)
        var vv = self.F(self.o_v)
        var dec = self.F(self.o_dec)
        comptime if has_accelerator():
            if self.gpu:
                self.ctx.value().enqueue_function[k_adamw](w, g, mm, vv, dec, gmul, lr, c1, c2, wd, Int32(n),
                                                           grid_dim=blocks(n), block_dim=BLOCK)
                return norm
        for i in range(n):
            b_adamw(w, g, mm, vv, dec, gmul, lr, c1, c2, wd, i)
        return norm

    def muon_step(self, lr: Float32, step: Int, wd: Float32, gmul: Float32) raises:
        """AdamW on everything outside self.mats; Muon (Newton-Schulz orthogonalized momentum)
        on each matrix. The update is scaled by 0.2 sqrt(max(rows, cols)) so its RMS matches
        AdamW's and the same learning rate and weight decay apply."""
        comptime if has_accelerator():
            var ctx = self.ctx.value()
            var c1 = 1.0 - Float32(0.9 ** Float64(step))
            var c2 = 1.0 - Float32(0.95 ** Float64(step))
            var w = self.F(self.o_w)
            var g = self.F(self.o_g)
            var mm = self.F(self.o_m)
            var vv = self.F(self.o_v)
            var dec = self.F(self.o_dec)
            var pos = 0
            for k in range(len(self.mats) // 3 + 1):
                var end = self.nparam if 3 * k >= len(self.mats) else self.mats[3 * k]
                if end > pos:
                    ctx.enqueue_function[k_adamw](
                        w.unsafe_offset(pos), g.unsafe_offset(pos), mm.unsafe_offset(pos),
                        vv.unsafe_offset(pos), dec.unsafe_offset(pos), gmul, lr, c1, c2, wd, Int32(end - pos),
                        grid_dim=blocks(end - pos), block_dim=BLOCK)
                if 3 * k < len(self.mats):
                    pos = end + self.mats[3 * k + 1] * self.mats[3 * k + 2]
            for k in range(len(self.mats) // 3):
                var off = self.mats[3 * k]
                var R = self.mats[3 * k + 1]
                var C = self.mats[3 * k + 2]
                var n = R * C
                var xa = self.F(self.o_mu)
                var xb = self.F(self.o_mu + self.muon_mm)
                var ga = self.F(self.o_mu + 2 * self.muon_mm)
                var gb = self.F(self.o_mu + 3 * self.muon_mm)
                ctx.enqueue_function[k_muon_prep](xa, mm.unsafe_offset(off), g.unsafe_offset(off), gmul,
                                                  self.muon_mu, Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
                var per = max(8, (n + BLOCK * NCHUNK - 1) // (BLOCK * NCHUNK))  # at most NCHUNK partials
                var nb = (n + BLOCK * per - 1) // (BLOCK * per)
                var part = self.F(self.o_partial)
                ctx.enqueue_function[k_sumsq_blocks](part, xa, Int32(per), Int32(n), grid_dim=nb, block_dim=BLOCK)
                ctx.enqueue_function[k_scale_inv_norm](xa, part, Int32(nb), Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
                var wide = R <= C
                var q = R if wide else C  # side of the Gram matrix
                for _ in range(5):
                    if wide:  # A = X X^T, B = c A A + b A, Y = a X + B X
                        self.gemm(ga, xa, xa, R, R, C, 1, 1, C, 1, 1, C, R, 1, 0, 0, 0, 0, 0, 0, 1.0, False)
                    else:  # A = X^T X, B = c A A + b A, Y = a X + X B
                        self.gemm(ga, xa, xa, C, C, R, 1, 1, 1, C, C, 1, C, 1, 0, 0, 0, 0, 0, 0, 1.0, False)
                    self.gemm(gb, ga, ga, q, q, q, 1, 1, q, 1, q, 1, q, 1, 0, 0, 0, 0, 0, 0, 2.0315, False)
                    self.axpy(gb, ga, -4.775, q * q)
                    if wide:
                        self.gemm(xb, gb, xa, R, C, R, 1, 1, R, 1, C, 1, C, 1, 0, 0, 0, 0, 0, 0, 1.0, False)
                    else:
                        self.gemm(xb, xa, gb, R, C, C, 1, 1, C, 1, C, 1, C, 1, 0, 0, 0, 0, 0, 0, 1.0, False)
                    self.axpy(xb, xa, 3.4445, n)
                    var t = xa
                    xa = xb
                    xb = t
                var scale = Float32(0.2 * sqrt(Float64(max(R, C))))
                ctx.enqueue_function[k_muon_apply](w.unsafe_offset(off), xa, dec.unsafe_offset(off), lr, wd, scale,
                                                   Int32(n), grid_dim=blocks(n), block_dim=BLOCK)

    def zero_grad(self) raises:
        self.fill(self.F(self.o_g), self.nparam, 0.0)
