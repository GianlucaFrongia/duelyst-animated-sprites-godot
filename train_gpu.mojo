# Train on the GPU (Apple M-series via Metal, or NVIDIA/AMD) with the batched engine.
#
#     mojo build -O3 train_gpu.mojo -o train_gpu
#     ./train_gpu --data data --out checkpoints/sprites --steps 2000 --batch 8
#     ./train_gpu --data data --out checkpoints/sprites --minutes 30 --warmup 300   # time budget
#
# Checkpoints are the same format as train.mojo, so sample.mojo works unchanged.
# Pass --cpu to run the identical kernels as CPU loops (slow; for debugging only).
#
# If <data>/val exists (prepare_animations.py --val-frac), every --val-every steps it logs
# the loss on held-out units twice: with the context frames, and with them hidden (drawing
# a first frame from nothing). Masks are fixed, so the numbers compare across evals and runs.

from std.sys import argv
from std.math import cos
from std.os import makedirs
from std.os.path import dirname, exists
from std.random import seed, random_float64, random_ui64
from std.time import perf_counter_ns
from arch.model import Config, Model, T, PP, NPIX
from src.data import Dataset, CF, random_mask, apply_mask, blank_frames, arg_str, arg_int, arg_float
from arch.engine import Engine


def val_loss(e: Engine, m: Model, ids: List[Int32], tgt: List[Int32], mask: List[Int32],
             cls: List[Int32], gscale: List[Float32], n: Int, batch: Int) raises -> Float64:
    """Mean masked loss over n held-out examples (a multiple of batch), forward pass only."""
    var total: Float64 = 0.0
    for b0 in range(0, n, batch):
        var px = Slice(b0 * NPIX, (b0 + batch) * NPIX)
        e.upload_batch(List[Int32](ids[px]), List[Int32](tgt[px]), List[Int32](mask[b0 * T:(b0 + batch) * T]),
                       List[Int32](cls[b0:b0 + batch]), List[Float32](gscale[b0:b0 + batch]))
        e.forward(m)
        total += Float64(e.loss())
    return total / Float64(n // batch)


def build_batch(ds: Dataset, C: Int, V: Int, batch: Int, flip_prob: Float64, cfg_drop: Float64,
                context_drop: Float64, history_drop: Float64, mut ids: List[Int32], mut tgt: List[Int32],
                mut mask_i: List[Int32], mut cls: List[Int32], mut gscale: List[Float32]):
    """A random training batch, masked, into the upload lists."""
    var target = List[Int](length=NPIX, fill=0)
    var masked_ids = List[Int](length=NPIX, fill=0)
    var mask = List[Bool](length=T, fill=False)
    var order = List[Int](length=T, fill=0)
    for b in range(batch):
        var i = Int(random_ui64(0, UInt64(ds.N - 1)))
        ds.pair(i, random_float64() < flip_prob, target)
        var c = Int(ds.labels[i])
        if random_float64() < cfg_drop:
            c = C
        var n_masked = random_mask(mask, order)
        apply_mask(masked_ids, target, mask, V, blank_frames(context_drop, history_drop))
        for p in range(NPIX):
            tgt[b * NPIX + p] = Int32(target[p])
            ids[b * NPIX + p] = Int32(masked_ids[p])
        for t in range(T):
            mask_i[b * T + t] = 1 if mask[t] else 0
        cls[b] = Int32(c)
        gscale[b] = Float32(1.0 / Float64(n_masked * PP * batch))


def main() raises:
    var a = argv()
    var args = List[String]()
    var use_gpu = True
    for i in range(len(a)):
        args.append(String(a[i]))
        if String(a[i]) == "--cpu":
            use_gpu = False

    var data_dir = arg_str(args, "--data", "data")
    var out = arg_str(args, "--out", "checkpoints/sprites")
    var steps = arg_int(args, "--steps", 2000)
    var batch = arg_int(args, "--batch", 8)
    var lr_max = Float32(arg_float(args, "--lr", 1e-3))
    var wd = Float32(arg_float(args, "--weight-decay", 0.05))
    var warmup = arg_int(args, "--warmup", 100)
    # --minutes: a wall-clock budget. --steps is re-set from the speed of steps 11..PROBE, which
    # are still in warmup (their learning rate doesn't depend on --steps).
    var minutes = arg_float(args, "--minutes", 0.0)
    comptime PROBE = 50
    if minutes > 0.0 and warmup <= PROBE:
        raise Error("--minutes needs --warmup above " + String(PROBE))
    # --decay F: warmup-stable-decay (DeepSeek LLM) instead of cosine: hold --lr, then decay linearly
    # to a tenth of it over the last fraction F of the steps.
    var decay = arg_float(args, "--decay", 0.0)
    var cfg_drop = arg_float(args, "--cfg-drop", 0.1)
    var context_drop = arg_float(args, "--context-drop", 0.15)  # chance to train first-frame generation
    var history_drop = arg_float(args, "--history-drop", 0.1)  # chance to hide all but the newest context frame
    var flip_prob = arg_float(args, "--flip", 0.0)  # chance to mirror a pair (flips which way units face)
    var log_every = arg_int(args, "--log-every", 10)
    var save_every = arg_int(args, "--save-every", 250)
    var val_every = arg_int(args, "--val-every", save_every)
    var resume = arg_str(args, "--resume", "")
    var use_muon = False  # --muon: Muon on the weight matrices, AdamW on the rest
    for i in range(len(args)):
        if args[i] == "--muon":
            use_muon = True
    var ds = Dataset(data_dir)
    print("dataset:", ds.N, "frame pairs,", ds.V, "colors,", ds.C, "classes")

    # Held-out examples, masked once: each with its context (v_ids) and with it hidden (v_ids0).
    var n_val = 0
    var v_ids = List[Int32]()
    var v_ids0 = List[Int32]()
    var v_tgt = List[Int32]()
    var v_mask = List[Int32]()
    var v_cls = List[Int32]()
    var v_gs = List[Float32]()
    if val_every > 0 and exists(data_dir + "/val/meta.txt"):
        var vs = Dataset(data_dir + "/val")
        n_val = (min(vs.N, arg_int(args, "--val-n", 504)) // batch) * batch
        if n_val == 0:
            print("warning:", vs.N, "validation examples, fewer than one batch: no validation")
        seed(1234)  # the same masks whatever --seed and model size
        var vt = List[Int](length=NPIX, fill=0)
        var vm = List[Int](length=NPIX, fill=0)
        var vmask = List[Bool](length=T, fill=False)
        var vorder = List[Int](length=T, fill=0)
        for i in range(n_val):
            var j = i * vs.N // n_val  # spread over the whole held-out set
            vs.pair(j, False, vt)
            var n_masked = random_mask(vmask, vorder)
            apply_mask(vm, vt, vmask, ds.V, 0)
            for p in range(NPIX):
                v_ids.append(Int32(vm[p]))
            apply_mask(vm, vt, vmask, ds.V, CF)
            for p in range(NPIX):
                v_ids0.append(Int32(vm[p]))
                v_tgt.append(Int32(vt[p]))
            for t in range(T):
                v_mask.append(Int32(1) if vmask[t] else Int32(0))
            v_cls.append(Int32(vs.labels[j]))
            v_gs.append(Float32(1.0 / Float64(n_masked * PP * batch)))
        print("validation:", n_val, "of", vs.N, "held-out examples")
    seed(arg_int(args, "--seed", 0))
    var dim = arg_int(args, "--dim", 64)
    var cfg = Config(
        V=ds.V, C=ds.C, D=dim,
        L=arg_int(args, "--layers", 4),
        H=arg_int(args, "--heads", 4),
        F=arg_int(args, "--mlp", 4 * dim),
    )
    if cfg.D % cfg.H != 0:
        raise Error("--dim must be divisible by --heads")
    var m = Model(cfg)
    m.init_weights()
    if resume.byte_length() > 0:
        m.load_weights(resume)
        print("resumed weights from", resume)
    print("model:", m.n, "parameters (D", cfg.D, "L", cfg.L, "H", cfg.H, "F", cfg.F, ") | batch", batch)

    var e = Engine(cfg, batch, use_gpu)
    print("device:", e.dev.device_name(), "|", Float32(Float64(e.dev.nf) * 4.0 / 1e9), "GB of buffers")
    if use_gpu and not e.dev.gpu:
        print("warning: no GPU found by Mojo, falling back to slow CPU kernel emulation")
    e.dev.load_params(m.w, m.decay)
    if use_muon:
        if not e.dev.gpu:
            raise Error("--muon needs the GPU")
        e.dev.muon = True
        e.dev.muon_mu = Float32(arg_float(args, "--muon-mu", 0.95))
        var D = cfg.D
        for l in range(cfg.L):
            for mat in [(m.r_wqkv, 3 * D, D), (m.r_wo, D, D), (m.r_w1, cfg.F, D), (m.r_w2, D, cfg.F)]:
                e.dev.mats.append(m.lo(l, mat[0]))
                e.dev.mats.append(mat[1])
                e.dev.mats.append(mat[2])
        e.dev.mats.append(m.o_wout)
        e.dev.mats.append(PP * cfg.V)
        e.dev.mats.append(D)
    var d = dirname(out)
    if d.byte_length() > 0:
        makedirs(d, exist_ok=True)

    var ids = List[Int32](length=batch * NPIX, fill=0)
    var tgt = List[Int32](length=batch * NPIX, fill=0)
    var mask_i = List[Int32](length=batch * T, fill=0)
    var cls = List[Int32](length=batch, fill=0)
    var gscale = List[Float32](length=batch, fill=0.0)
    build_batch(ds, cfg.C, cfg.V, batch, flip_prob, cfg_drop, context_drop, history_drop, ids, tgt, mask_i, cls, gscale)

    var t0 = perf_counter_ns()
    var t_probe = t0
    var loss_acc: Float64 = 0.0
    var norm_acc: Float64 = 0.0
    var step = 0
    while step < steps:
        step += 1
        if minutes > 0.0 and (step == 11 or step == PROBE + 1):
            e.dev.sync()
            if step == 11:
                t_probe = perf_counter_ns()
            else:
                var per_step = Float64(perf_counter_ns() - t_probe) / 1e9 / Float64(PROBE - 10)
                var left = minutes * 60.0 - Float64(perf_counter_ns() - t0) / 1e9
                steps = PROBE + max(1, Int(left / per_step))
                print("time budget", minutes, "min:", steps, "steps at", Float32(per_step), "s/step", flush=True)
        var lr: Float32
        if step <= warmup:
            lr = lr_max * Float32(step) / Float32(warmup)
        elif decay > 0.0:
            var d0 = steps - Int(decay * Float64(steps))
            lr = lr_max if step <= d0 else lr_max * Float32(1.0 - 0.9 * Float64(step - d0) / Float64(max(1, steps - d0)))
        else:
            var prog = Float64(step - warmup) / Float64(max(1, steps - warmup))
            lr = lr_max * Float32(0.1 + 0.9 * 0.5 * (1.0 + cos(3.141592653589793 * prog)))

        e.upload_batch(ids, tgt, mask_i, cls, gscale)
        e.dev.zero_grad()
        e.forward(m)
        # The next batch is built on the CPU while the GPU runs this forward pass.
        build_batch(ds, cfg.C, cfg.V, batch, flip_prob, cfg_drop, context_drop, history_drop, ids, tgt, mask_i, cls, gscale)
        var loss = e.loss()
        e.backward(m)
        var norm = e.dev.optimizer_step(lr, step, wd, 1.0)
        loss_acc += Float64(loss)
        norm_acc += Float64(norm)

        if step % log_every == 0 or step == steps:
            e.dev.sync()
            var n = Float64(log_every if step % log_every == 0 else step % log_every)
            var secs = Float64(perf_counter_ns() - t0) / 1e9
            var sps = Float64(step * batch) / secs
            var eta = Float64((steps - step) * batch) / sps
            print("step", step, "/", steps, "| loss", Float32(loss_acc / n), "| grad norm",
                  Float32(norm_acc / n), "| lr", lr, "|", Float32(sps), "pairs/s | eta",
                  Int(eta / 60.0), "min", flush=True)
            loss_acc = 0.0
            norm_acc = 0.0
        if n_val > 0 and (step % val_every == 0 or step == steps):
            var lc = val_loss(e, m, v_ids, v_tgt, v_mask, v_cls, v_gs, n_val, batch)
            var l0 = val_loss(e, m, v_ids0, v_tgt, v_mask, v_cls, v_gs, n_val, batch)
            print("val step", step, "| loss with context", Float32(lc), "| loss from scratch", Float32(l0), flush=True)
        if step % save_every == 0 or step == steps:
            e.dev.store_params(m.w)
            m.save(out)
            print("saved", out + ".bin", flush=True)
