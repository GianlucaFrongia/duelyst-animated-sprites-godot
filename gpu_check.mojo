# Checks the GPU engine against the CPU reference model (which gradcheck.mojo verifies).
# Run this on your Mac before training on the GPU:
#
#     mojo run gpu_check.mojo            # uses the GPU if Mojo finds one
#     mojo run gpu_check.mojo --cpu      # runs the same kernels as plain CPU loops
#
# It compares logits, loss, every gradient, and the weights after one optimizer step, for a
# small config (the generic kernels) and a wide one whose sizes are multiples of 64 (the
# Metal matmuls and the chunked embedding gradient).

from std.sys import argv
from std.random import seed, random_ui64, random_float64
from arch.model import Config, Model, Work, forward, backward, masked_loss, T, PP, NPIX, tgt_pix
from arch.engine import Engine


def max_rel_diff(a: List[Float32], b: List[Float32]) -> Float32:
    var num: Float32 = 0.0
    var den: Float32 = 1e-6
    for i in range(len(a)):
        num = max(num, abs(a[i] - b[i]))
        den = max(den, abs(a[i]))
    return num / den


def main() raises:
    var a = argv()
    var use_gpu = True
    for i in range(len(a)):
        if String(a[i]) == "--cpu":
            use_gpu = False
    check(Config(V=6, C=3, D=32, L=2, H=2, F=64), 3, use_gpu)
    check(Config(V=16, C=3, D=64, L=2, H=2, F=128), 3, use_gpu)


def check(cfg: Config, B: Int, use_gpu: Bool) raises:
    seed(7)
    print("config: D", cfg.D, "L", cfg.L, "H", cfg.H, "F", cfg.F, "V", cfg.V, "| batch", B)
    var m = Model(cfg)
    m.init_weights()
    for i in range(m.n):
        if m.decay[i] == 1:
            m.w[i] *= 5.0
    var wk = Work(cfg)

    # random batch
    var ids_all = List[Int32](length=B * NPIX, fill=0)
    var tgt_all = List[Int32](length=B * NPIX, fill=0)
    var mask_all = List[Int32](length=B * T, fill=0)
    var cls_all = List[Int32](length=B, fill=0)
    var gscale = List[Float32](length=B, fill=0.0)
    var e = Engine(cfg, B, use_gpu)
    print("device:", e.dev.device_name())

    m.zero_grad()
    var ref_loss: Float64 = 0.0
    var ref_logits = List[Float32]()
    for b in range(B):
        var tgt = List[Int](length=NPIX, fill=0)
        for i in range(NPIX):
            tgt[i] = Int(random_ui64(0, UInt64(cfg.V - 1)))
        var mask = List[Bool](length=T, fill=False)
        var nm = 0
        for t in range(T):
            mask[t] = random_float64() < 0.4
            if mask[t]:
                nm += 1
        var ids = tgt.copy()
        for t in range(T):
            if mask[t]:
                for k in range(PP):
                    ids[tgt_pix(t, k)] = cfg.V
        var cls = Int(random_ui64(0, UInt64(cfg.C)))  # includes the "no class" id
        var gs = Float32(1.0 / Float64(nm * PP * B))
        forward(m, wk, ids, cls)
        for i in range(len(wk.logits)):
            ref_logits.append(wk.logits[i])
        ref_loss += Float64(masked_loss(wk, cfg.V, tgt, mask, gs)) / Float64(B)
        backward(m, wk, ids, cls)
        for i in range(NPIX):
            ids_all[b * NPIX + i] = Int32(ids[i])
            tgt_all[b * NPIX + i] = Int32(tgt[i])
        for t in range(T):
            mask_all[b * T + t] = 1 if mask[t] else 0
        cls_all[b] = Int32(cls)
        gscale[b] = gs

    e.dev.load_params(m.w, m.decay)
    e.upload_batch(ids_all, tgt_all, mask_all, cls_all, gscale)
    e.dev.zero_grad()
    e.forward(m)
    var loss = e.loss()
    e.backward(m)
    e.dev.sync()

    var logits = List[Float32](length=len(ref_logits), fill=0.0)
    e.dev.download_f(e.o_logits, len(ref_logits), logits)
    var grads = List[Float32](length=m.n, fill=0.0)
    e.dev.download_f(e.dev.o_g, m.n, grads)

    var d_logits = max_rel_diff(ref_logits, logits)
    var d_loss = abs(Float32(ref_loss) - loss) / Float32(ref_loss)
    var d_grad = max_rel_diff(m.g, grads)

    # one optimizer step on both sides
    var norm_gpu = e.dev.optimizer_step(1e-3, 1, 0.05, 1.0)
    var norm_cpu = m.clip_grad(1.0)
    m.adamw_step(1e-3, 1, 0.05)
    var w_gpu = List[Float32](length=m.n, fill=0.0)
    e.dev.download_f(e.dev.o_w, m.n, w_gpu)
    var d_w = max_rel_diff(m.w, w_gpu)

    print("logits    max rel diff:", d_logits)
    print("loss      cpu", Float32(ref_loss), "engine", loss, "rel diff", d_loss)
    print("gradients max rel diff:", d_grad)
    print("grad norm cpu", norm_cpu, "engine", norm_gpu)
    print("weights after AdamW step, max rel diff:", d_w)
    var ok = d_logits < 1e-3 and d_loss < 1e-3 and d_grad < 1e-3 and d_w < 1e-3
    if not ok:
        raise Error("engine does not match the CPU reference")
    print("engine matches the CPU reference")
