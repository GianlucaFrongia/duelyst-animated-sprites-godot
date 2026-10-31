# Low-level math: matrix multiplies, layer norm, GELU, softmax.
# Everything works on flat row-major Float32 buffers addressed by raw pointers,
# so the same code can be dropped into a GPU kernel later.

from std.sys import simd_width_of
from std.math import sqrt, exp, tanh

comptime FP = Pointer[Float32, MutAnyOrigin]
comptime NW = simd_width_of[DType.float32]()


def ptr(mut l: List[Float32]) -> FP:
    return l.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def fill(p: FP, n: Int, v: Float32):
    for i in range(n):
        p[unsafe_offset=i] = v


def copy(dst: FP, src: FP, n: Int):
    for i in range(n):
        dst[unsafe_offset=i] = src[unsafe_offset=i]


def dot(a: FP, b: FP, n: Int) -> Float32:
    var acc = SIMD[DType.float32, NW](0)
    var i = 0
    while i + NW <= n:
        acc += a.unsafe_load[width=NW](i) * b.unsafe_load[width=NW](i)
        i += NW
    var s = acc.reduce_add()
    while i < n:
        s += a[unsafe_offset=i] * b[unsafe_offset=i]
        i += 1
    return s


def axpy(y: FP, x: FP, a: Float32, n: Int):
    """y += a * x."""
    var av = SIMD[DType.float32, NW](a)
    var i = 0
    while i + NW <= n:
        y.unsafe_store(i, y.unsafe_load[width=NW](i) + av * x.unsafe_load[width=NW](i))
        i += NW
    while i < n:
        y[unsafe_offset=i] += a * x[unsafe_offset=i]
        i += 1


# ---------------------------------------------------------------- matmuls
# Three layouts cover every forward and backward product we need.


def mm_nt(c: FP, a: FP, b: FP, M: Int, N: Int, K: Int, accumulate: Bool):
    """C[M,N] (=|+=) A[M,K] @ B[N,K]^T."""
    for i in range(M):
        var ai = a.unsafe_offset(i * K)
        for j in range(N):
            var s = dot(ai, b.unsafe_offset(j * K), K)
            if accumulate:
                c[unsafe_offset=i * N + j] += s
            else:
                c[unsafe_offset=i * N + j] = s


def mm_nn(c: FP, a: FP, b: FP, M: Int, N: Int, K: Int):
    """C[M,N] += A[M,K] @ B[K,N]."""
    for i in range(M):
        var ci = c.unsafe_offset(i * N)
        for k in range(K):
            var aik = a[unsafe_offset=i * K + k]
            if aik != 0.0:
                axpy(ci, b.unsafe_offset(k * N), aik, N)


def mm_tn(c: FP, a: FP, b: FP, M: Int, N: Int, K: Int):
    """C[M,N] += A[K,M]^T @ B[K,N]."""
    for k in range(K):
        var bk = b.unsafe_offset(k * N)
        for i in range(M):
            var aki = a[unsafe_offset=k * M + i]
            if aki != 0.0:
                axpy(c.unsafe_offset(i * N), bk, aki, N)


# ---------------------------------------------------------------- linear
# y[T,O] = x[T,I] @ W[O,I]^T + b[O]


def linear_fwd(y: FP, x: FP, w: FP, b: FP, T: Int, I: Int, O: Int):
    mm_nt(y, x, w, T, O, I, False)
    for t in range(T):
        axpy(y.unsafe_offset(t * O), b, 1.0, O)


def linear_bwd(dx: FP, dw: FP, db: FP, dy: FP, x: FP, w: FP, T: Int, I: Int, O: Int):
    """Accumulates into dx, dw, db."""
    mm_nn(dx, dy, w, T, I, O)  # dx[T,I] += dy[T,O] @ W[O,I]
    mm_tn(dw, dy, x, O, I, T)  # dW[O,I] += dy^T @ x
    for t in range(T):
        axpy(db, dy.unsafe_offset(t * O), 1.0, O)


# ---------------------------------------------------------------- layer norm


def ln_fwd(y: FP, xhat: FP, rstd: FP, x: FP, g: FP, b: FP, T: Int, D: Int):
    for t in range(T):
        var xt = x.unsafe_offset(t * D)
        var mean: Float32 = 0.0
        for i in range(D):
            mean += xt[unsafe_offset=i]
        mean /= Float32(D)
        var v: Float32 = 0.0
        for i in range(D):
            var d = xt[unsafe_offset=i] - mean
            v += d * d
        v /= Float32(D)
        var r = 1.0 / sqrt(v + 1e-5)
        rstd[unsafe_offset=t] = r
        for i in range(D):
            var xh = (xt[unsafe_offset=i] - mean) * r
            xhat[unsafe_offset=t * D + i] = xh
            y[unsafe_offset=t * D + i] = xh * g[unsafe_offset=i] + b[unsafe_offset=i]


def ln_bwd(dx: FP, dg: FP, db: FP, dy: FP, xhat: FP, rstd: FP, g: FP, T: Int, D: Int):
    """Accumulates into dx, dg, db."""
    for t in range(T):
        var m1: Float32 = 0.0
        var m2: Float32 = 0.0
        for i in range(D):
            var dxh = dy[unsafe_offset=t * D + i] * g[unsafe_offset=i]
            m1 += dxh
            m2 += dxh * xhat[unsafe_offset=t * D + i]
        m1 /= Float32(D)
        m2 /= Float32(D)
        var r = rstd[unsafe_offset=t]
        for i in range(D):
            var o = t * D + i
            var d = dy[unsafe_offset=o]
            var xh = xhat[unsafe_offset=o]
            dx[unsafe_offset=o] += r * (d * g[unsafe_offset=i] - m1 - xh * m2)
            dg[unsafe_offset=i] += d * xh
            db[unsafe_offset=i] += d


def dyt_fwd(y: FP, xs: FP, x: FP, g: FP, b: FP, a: Float32, T: Int, D: Int):
    """Dynamic Tanh ("Transformers without Normalization"), in place of layer norm:
    y = g * tanh(a x) + b, a one scalar per norm. Keeps a copy of x in xs for the backward."""
    for i in range(T * D):
        var v = x[unsafe_offset=i]
        xs[unsafe_offset=i] = v
        y[unsafe_offset=i] = g[unsafe_offset=i % D] * tanh(a * v) + b[unsafe_offset=i % D]


def dyt_bwd(dx: FP, dg: FP, db: FP, da: FP, dy: FP, xs: FP, g: FP, a: Float32, T: Int, D: Int):
    """Accumulates into dx, dg, db and da[0]."""
    var sa: Float64 = 0.0
    for i in range(T * D):
        var d = i % D
        var v = xs[unsafe_offset=i]
        var t = tanh(a * v)
        var e = dy[unsafe_offset=i]
        dx[unsafe_offset=i] += e * g[unsafe_offset=d] * a * (1.0 - t * t)
        dg[unsafe_offset=d] += e * t
        db[unsafe_offset=d] += e
        sa += Float64(e * g[unsafe_offset=d] * (1.0 - t * t) * v)
    da[unsafe_offset=0] += Float32(sa)


# ---------------------------------------------------------------- GELU (tanh approximation)

comptime GELU_C: Float32 = 0.7978845608  # sqrt(2/pi)


def gelu_fwd(y: FP, x: FP, n: Int):
    for i in range(n):
        var v = x[unsafe_offset=i]
        y[unsafe_offset=i] = 0.5 * v * (1.0 + tanh(GELU_C * (v + 0.044715 * v * v * v)))


def gelu_bwd(dx: FP, dy: FP, x: FP, n: Int):
    """dx = dy * gelu'(x) (overwrites dx)."""
    for i in range(n):
        var v = x[unsafe_offset=i]
        var th = tanh(GELU_C * (v + 0.044715 * v * v * v))
        var d = 0.5 * (1.0 + th) + 0.5 * v * (1.0 - th * th) * GELU_C * (1.0 + 3.0 * 0.044715 * v * v)
        dx[unsafe_offset=i] = dy[unsafe_offset=i] * d


# ---------------------------------------------------------------- softmax


def softmax_rows(p: FP, rows: Int, cols: Int):
    for r in range(rows):
        var row = p.unsafe_offset(r * cols)
        var mx = row[unsafe_offset=0]
        for j in range(1, cols):
            mx = max(mx, row[unsafe_offset=j])
        var s: Float32 = 0.0
        for j in range(cols):
            var e = exp(row[unsafe_offset=j] - mx)
            row[unsafe_offset=j] = e
            s += e
        var inv = 1.0 / s
        for j in range(cols):
            row[unsafe_offset=j] *= inv
