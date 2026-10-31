# Train the masked pixel transformer.
#
#     mojo run train.mojo --data data --out checkpoints/sprites --steps 2000
#
# For speed, build once with optimizations and run the binary:
#     mojo build -O3 train.mojo -o train && ./train --data data ...

from std.sys import argv
from std.math import cos
from std.os import makedirs
from std.os.path import dirname
from std.random import seed, random_float64, random_ui64
from std.time import perf_counter_ns
from arch.model import Config, Model, Work, forward, backward, masked_loss, T, PP, NPIX
from src.data import Dataset, CF, random_mask, apply_mask, blank_frames, arg_str, arg_int, arg_float


def main() raises:
    var a = argv()
    var args = List[String]()
    for i in range(len(a)):
        args.append(String(a[i]))

    var data_dir = arg_str(args, "--data", "data")
    var out = arg_str(args, "--out", "checkpoints/sprites")
    var steps = arg_int(args, "--steps", 2000)
    var batch = arg_int(args, "--batch", 16)
    var lr_max = Float32(arg_float(args, "--lr", 1e-3))
    var wd = Float32(arg_float(args, "--weight-decay", 0.05))
    var warmup = arg_int(args, "--warmup", 100)
    var cfg_drop = arg_float(args, "--cfg-drop", 0.1)
    var context_drop = arg_float(args, "--context-drop", 0.15)  # chance to train first-frame generation
    var history_drop = arg_float(args, "--history-drop", 0.1)  # chance to hide all but the newest context frame
    var flip_prob = arg_float(args, "--flip", 0.0)  # chance to mirror a pair (flips which way units face)  # chance to train without the class
    var log_every = arg_int(args, "--log-every", 10)
    var save_every = arg_int(args, "--save-every", 250)
    var resume = arg_str(args, "--resume", "")
    seed(arg_int(args, "--seed", 0))

    var ds = Dataset(data_dir)
    print("dataset:", ds.N, "frame pairs,", ds.V, "colors,", ds.C, "classes")

    var cfg = Config(
        V=ds.V, C=ds.C,
        D=arg_int(args, "--dim", 64),
        L=arg_int(args, "--layers", 4),
        H=arg_int(args, "--heads", 4),
        F=arg_int(args, "--mlp", 4 * arg_int(args, "--dim", 64)),
    )
    if cfg.D % cfg.H != 0:
        raise Error("--dim must be divisible by --heads")
    var m = Model(cfg)
    m.init_weights()
    if resume.byte_length() > 0:
        m.load_weights(resume)
        print("resumed weights from", resume)
    print("model:", m.n, "parameters (D", cfg.D, "L", cfg.L, "H", cfg.H, "F", cfg.F, ")")
    var wk = Work(cfg)
    var d = dirname(out)
    if d.byte_length() > 0:
        makedirs(d, exist_ok=True)

    var target = List[Int](length=NPIX, fill=0)
    var ids = List[Int](length=NPIX, fill=0)
    var mask = List[Bool](length=T, fill=False)
    var order = List[Int](length=T, fill=0)

    var t0 = perf_counter_ns()
    var loss_acc: Float64 = 0.0
    var norm_acc: Float64 = 0.0
    for step in range(1, steps + 1):
        # learning rate: linear warmup, then cosine decay to 10%
        var lr: Float32
        if step <= warmup:
            lr = lr_max * Float32(step) / Float32(warmup)
        else:
            var prog = Float64(step - warmup) / Float64(max(1, steps - warmup))
            lr = lr_max * Float32(0.1 + 0.9 * 0.5 * (1.0 + cos(3.141592653589793 * prog)))

        m.zero_grad()
        var batch_loss: Float64 = 0.0
        for _ in range(batch):
            var i = Int(random_ui64(0, UInt64(ds.N - 1)))
            ds.pair(i, random_float64() < flip_prob, target)
            var cls = Int(ds.labels[i])
            if random_float64() < cfg_drop:
                cls = cfg.C  # "no class": lets us use classifier-free guidance when sampling
            var n_masked = random_mask(mask, order)
            apply_mask(ids, target, mask, cfg.V, blank_frames(context_drop, history_drop))
            forward(m, wk, ids, cls)
            var gs = Float32(1.0 / Float64(n_masked * PP * batch))
            batch_loss += Float64(masked_loss(wk, cfg.V, target, mask, gs))
            backward(m, wk, ids, cls)
        var norm = m.clip_grad(1.0)
        m.adamw_step(lr, step, wd)
        loss_acc += batch_loss / Float64(batch)
        norm_acc += Float64(norm)

        if step % log_every == 0 or step == steps:
            var n = Float64(log_every if step % log_every == 0 else step % log_every)
            var secs = Float64(perf_counter_ns() - t0) / 1e9
            var sps = Float64(step * batch) / secs
            var eta = Float64((steps - step) * batch) / sps
            print("step", step, "/", steps, "| loss", Float32(loss_acc / n), "| grad norm",
                  Float32(norm_acc / n), "| lr", lr, "|", Float32(sps), "pairs/s | eta",
                  Int(eta / 60.0), "min", flush=True)
            loss_acc = 0.0
            norm_acc = 0.0
        if step % save_every == 0 or step == steps:
            m.save(out)
            print("saved", out + ".bin", flush=True)
