# The GPU device shared by every architecture's engine: one float and one int arena in device
# memory, and the generic ops (matmul, linear layers, norms, GELU, reductions) launched on it.
# Without a GPU the same kernel bodies run as CPU loops (src/gpu_kernels.mojo).
#
# The float arena is laid out as
#   [architecture activations | parameters, gradients, Adam moments, decay mask | scratch]
# An architecture's Engine computes its own activation offsets from 0, then creates the
# Device, which appends the parameter and scratch regions. Activations are [R, D] row-major
# with R = batch * tokens. The optimizers that update the parameters are in src/optim.mojo.

from std.math import sqrt, tanh
from std.sys import has_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host._amdgpu_hip import HIP
from std.ffi import OwnedDLHandle
from src.gpu_kernels import *


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
    var nf: Int  # floats in the arena
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
        var o = (arch_floats + 63) // 64 * 64  # 64-float aligned parameter region
        self.o_w = o
        self.o_g = o + nparam
        self.o_m = o + 2 * nparam
        self.o_v = o + 3 * nparam
        self.o_dec = o + 4 * nparam
        self.o_partial = o + 5 * nparam
        self.o_splitk = self.o_partial + NCHUNK
        self.muon_mm = splitk_cap // SPLITK
        self.o_mu = self.o_splitk + splitk_cap
        self.nf = self.o_mu + 4 * self.muon_mm
        var ni = max(arch_ints, 1)
        self.gpu = False
        self.ctx = None
        self.fdev = None
        self.idev = None
        self.fhost = List[Float32]()
        self.ihost = List[Int32]()
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

    def linear_fwd(self, y: FP, x: FP, w: FP, b: FP, I: Int, O: Int, acc: Bool = False) raises:
        """y[R,O] = x[R,I] @ W[O,I]^T + b, or y += that with acc (a residual add)."""
        var R = self.R
        comptime if has_accelerator():
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

    def zero_grad(self) raises:
        self.fill(self.F(self.o_g), self.nparam, 0.0)
