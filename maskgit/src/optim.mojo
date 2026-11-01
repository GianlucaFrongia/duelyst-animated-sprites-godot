# Optimizers over the Device's parameter region (src/gpu.mojo): global-norm gradient clipping,
# then AdamW on every parameter, or Muon on the weight matrices and AdamW on the rest.

from std.math import sqrt, cos
from std.sys import has_accelerator
from src.gpu import Device, FP, BLOCK, NCHUNK, blocks
from src.gpu_kernels import k_adamw, b_adamw, k_muon_prep, k_sumsq_blocks, k_scale_inv_norm, k_muon_apply
from src.model import Matrix


def lr_at(step: Int, steps: Int, warmup: Int, lr_max: Float32, decay: Float64) -> Float32:
    """Linear warmup, then cosine decay to a tenth; or with decay > 0, warmup-stable-decay
    (DeepSeek LLM): hold lr_max, then fall linearly to a tenth over the last decay * steps."""
    if step <= warmup:
        return lr_max * Float32(step) / Float32(warmup)
    if decay > 0.0:
        var d0 = steps - Int(decay * Float64(steps))
        return lr_max if step <= d0 else lr_max * Float32(1.0 - 0.9 * Float64(step - d0) / Float64(max(1, steps - d0)))
    var prog = Float64(step - warmup) / Float64(max(1, steps - warmup))
    return lr_max * Float32(0.1 + 0.9 * 0.5 * (1.0 + cos(3.141592653589793 * prog)))


struct Optimizer(Movable):
    var mats: List[Matrix]  # Muon's weight matrices, in ascending offset order; empty: AdamW only
    var mu: Float32  # Muon momentum

    def __init__(out self):
        """AdamW on every parameter."""
        self.mats = List[Matrix]()
        self.mu = 0.95

    def __init__(out self, var mats: List[Matrix], mu: Float32):
        """Muon (needs the GPU) on mats, AdamW on everything else."""
        self.mats = mats^
        self.mu = mu

    def step(self, dev: Device, lr: Float32, step: Int, wd: Float32, clip: Float32) raises -> Float32:
        """Global-norm gradient clipping + the update. Returns the gradient norm before clipping."""
        var n = dev.nparam
        var norm = sqrt(dev.chunk_sum(dev.F(dev.o_g), n, True))
        var gmul: Float32 = 1.0
        if norm > clip:
            gmul = clip / norm
        comptime if has_accelerator():
            if dev.gpu and len(self.mats) > 0:
                self._muon(dev, lr, step, wd, gmul)
                return norm
        self._adamw(dev, 0, n, lr, step, wd, gmul)
        return norm

    def _adamw(self, dev: Device, start: Int, end: Int, lr: Float32, step: Int, wd: Float32, gmul: Float32) raises:
        """AdamW on parameters start..end."""
        var n = end - start
        if n <= 0:
            return
        var c1 = 1.0 - Float32(0.9 ** Float64(step))
        var c2 = 1.0 - Float32(0.95 ** Float64(step))
        var w = dev.F(dev.o_w + start)
        var g = dev.F(dev.o_g + start)
        var mm = dev.F(dev.o_m + start)
        var vv = dev.F(dev.o_v + start)
        var dec = dev.F(dev.o_dec + start)
        comptime if has_accelerator():
            if dev.gpu:
                dev.ctx.value().enqueue_function[k_adamw](w, g, mm, vv, dec, gmul, lr, c1, c2, wd, Int32(n),
                                                          grid_dim=blocks(n), block_dim=BLOCK)
                return
        for i in range(n):
            b_adamw(w, g, mm, vv, dec, gmul, lr, c1, c2, wd, i)

    def _muon(self, dev: Device, lr: Float32, step: Int, wd: Float32, gmul: Float32) raises:
        """AdamW between the matrices; Muon (Newton-Schulz orthogonalized momentum) on each matrix.
        The update is scaled by 0.2 sqrt(max(rows, cols)) so its RMS matches AdamW's and the same
        learning rate and weight decay apply."""
        comptime if has_accelerator():
            var pos = 0
            for mat in self.mats:
                self._adamw(dev, pos, mat.off, lr, step, wd, gmul)
                pos = mat.off + mat.rows * mat.cols
            self._adamw(dev, pos, dev.nparam, lr, step, wd, gmul)
            var ctx = dev.ctx.value()
            var w = dev.F(dev.o_w)
            var g = dev.F(dev.o_g)
            var mm = dev.F(dev.o_m)
            var dec = dev.F(dev.o_dec)
            for mat in self.mats:
                var off = mat.off
                var R = mat.rows
                var C = mat.cols
                var n = R * C
                var xa = dev.F(dev.o_mu)
                var xb = dev.F(dev.o_mu + dev.muon_mm)
                var ga = dev.F(dev.o_mu + 2 * dev.muon_mm)
                var gb = dev.F(dev.o_mu + 3 * dev.muon_mm)
                ctx.enqueue_function[k_muon_prep](xa, mm.unsafe_offset(off), g.unsafe_offset(off), gmul,
                                                  self.mu, Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
                var per = max(8, (n + BLOCK * NCHUNK - 1) // (BLOCK * NCHUNK))  # at most NCHUNK partials
                var nb = (n + BLOCK * per - 1) // (BLOCK * per)
                var part = dev.F(dev.o_partial)
                ctx.enqueue_function[k_sumsq_blocks](part, xa, Int32(per), Int32(n), grid_dim=nb, block_dim=BLOCK)
                ctx.enqueue_function[k_scale_inv_norm](xa, part, Int32(nb), Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
                var wide = R <= C
                var q = R if wide else C  # side of the Gram matrix
                for _ in range(5):
                    if wide:  # A = X X^T, B = c A A + b A, Y = a X + B X
                        dev.gemm(ga, xa, xa, R, R, C, 1, 1, C, 1, 1, C, R, 1, 0, 0, 0, 0, 0, 0, 1.0, False)
                    else:  # A = X^T X, B = c A A + b A, Y = a X + X B
                        dev.gemm(ga, xa, xa, C, C, R, 1, 1, 1, C, C, 1, C, 1, 0, 0, 0, 0, 0, 0, 1.0, False)
                    dev.gemm(gb, ga, ga, q, q, q, 1, 1, q, 1, q, 1, q, 1, 0, 0, 0, 0, 0, 0, 2.0315, False)
                    dev.axpy(gb, ga, -4.775, q * q)
                    if wide:
                        dev.gemm(xb, gb, xa, R, C, R, 1, 1, R, 1, C, 1, C, 1, 0, 0, 0, 0, 0, 0, 1.0, False)
                    else:
                        dev.gemm(xb, xa, gb, R, C, C, 1, 1, C, 1, C, 1, C, 1, 0, 0, 0, 0, 0, 0, 1.0, False)
                    dev.axpy(xb, xa, 3.4445, n)
                    var t = xa
                    xa = xb
                    xb = t
                var scale = Float32(0.2 * sqrt(Float64(max(R, C))))
                ctx.enqueue_function[k_muon_apply](w.unsafe_offset(off), xa, dec.unsafe_offset(off), lr, wd, scale,
                                                   Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
