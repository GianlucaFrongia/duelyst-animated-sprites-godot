# Train on the GPU (AMD via ROCm) with the batched engine (src/engine.mojo).
#
#     sh tools/build.sh train_gpu      # ./train_gpu-ctx; ARCH=<name> for another architecture
#     ./train_gpu-ctx --data data --out checkpoints/sprites --steps 2000 --batch 8
#     ./train_gpu-ctx --data data --out checkpoints/sprites --minutes 30 --warmup 300   # time budget
#
# Pass --cpu to run the identical kernels as CPU loops (slow; for debugging only).
#
# If <data>/val exists (prepare_animations.py --val-frac), every --val-every steps it logs
# the loss on held-out units twice: with the context frames, and with them hidden (drawing
# a first frame from nothing). Masks are fixed, so the numbers compare across evals and runs.
#
# CROP architectures run only the target tokens near the keyframes (token_order in
# src/model.mojo); batches are drawn from examples that need about as many tokens, so a batch
# runs few more than it needs. The validation loss still averages over every masked pixel of the
# frame, counting the ones outside the crop as free, so it compares with uncropped runs; how many
# opaque target pixels the crop drops is printed once. Uncropped in-betweening architectures also
# log that same quantity (--val-crop, default 3: the loss over just the pixels inside the crop
# inbetween_crop would run, normalized by all masked pixels), for a like-for-like comparison.

from std.sys import argv
from std.os import makedirs
from std.os.path import dirname, exists
from std.random import seed, random_float64, random_ui64
from std.time import perf_counter_ns
from src.model import (
    Config, Model, T, PP, NPIX, FRAME_PIX, CROP, STACKED, INBETWEEN, tgt_pix, full_tokens, token_order, crop_len,
    take_tokens,
)
from src.data import (
    Dataset, CF, TGT0, random_mask, random_mask_rows, apply_mask, apply_mask_rows, blank_frames,
    arg_str, arg_int, arg_float,
)
from src.engine import Engine
from src.optim import Optimizer, lr_at


def val_loss(mut e: Engine, m: Model, ids: List[Int32], tgt: List[Int32], vb: ValBatches,
             cls: List[Int32], gscale: List[Float32], n: Int, batch: Int) raises -> Float64:
    """Mean masked loss over n held-out examples (a multiple of batch), forward pass only."""
    var total: Float64 = 0.0
    for k in range(n // batch):
        var b0 = k * batch
        var px = Slice(b0 * NPIX, (b0 + batch) * NPIX)
        var rows = Slice(vb.start[k], vb.start[k] + batch * vb.n[k])
        e.set_tokens(List[Int32](vb.toks[rows]), vb.n[k])
        e.upload_batch(List[Int32](ids[px]), List[Int32](tgt[px]), List[Int32](vb.mask[rows]),
                       List[Int32](cls[b0:b0 + batch]), List[Float32](gscale[b0:b0 + batch]))
        e.forward(m)
        total += Float64(e.loss())
    return total / Float64(n // batch)


struct ValBatches(Movable):
    """The tokens each held-out batch runs, and its mask per row (rows of batch k start at start[k])."""
    var toks: List[Int32]
    var mask: List[Int32]
    var start: List[Int]
    var n: List[Int]

    def __init__(out self):
        self.toks = List[Int32]()
        self.mask = List[Int32]()
        self.start = List[Int]()
        self.n = List[Int]()

    def add(mut self, canvases: List[Int32], targets: List[Int32], masks: List[Bool], batch: Int, V: Int,
            margin: Int, run_all: Bool) raises -> Int:
        """Appends one batch of canvases (batch * NPIX, masked as the model sees them) with their
        targets and target-token masks (batch * T), cropped like a CROP = margin architecture
        (margin < 0: every token). run_all: run every token anyway, but mask only the cropped ones.
        Returns the opaque target pixels the crop drops."""
        var orders = List[List[Int]]()
        var need = 0
        var canvas = List[Int](length=NPIX, fill=0)
        for b in range(batch):
            var order = full_tokens()
            if margin >= 0:
                for i in range(NPIX):
                    canvas[i] = Int(canvases[b * NPIX + i])
                need = max(need, token_order(canvas, 0, V, order, margin))
            orders.append(order^)
        var n = crop_len(need) if margin >= 0 else T
        self.start.append(len(self.toks))
        self.n.append(T if run_all else n)
        var toks = List[Int](capacity=n)
        var dropped = 0
        for b in range(batch):
            take_tokens(orders[b], n, toks)
            var run = List[Bool](length=T, fill=False)
            for r in range(n):
                run[toks[r]] = True
            if run_all:
                for t in range(T):
                    self.toks.append(Int32(t))
                    self.mask.append(Int32(1) if masks[b * T + t] and run[t] else Int32(0))
            else:
                for r in range(n):
                    self.toks.append(Int32(toks[r]))
                    self.mask.append(Int32(1) if masks[b * T + toks[r]] else Int32(0))
            for t in range(T):
                if not run[t]:
                    for k in range(PP):
                        if targets[b * NPIX + tgt_pix(t, k)] != 0:
                            dropped += 1
        return dropped


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


def crop_buckets(ds: Dataset, V: Int) -> List[List[Int]]:
    """Examples grouped by the tokens they run with their keyframes visible (crop_len // 64)."""
    var buckets = List[List[Int]]()
    for _ in range(T // 64 + 1):
        buckets.append(List[Int]())
    var canvas = List[Int](length=NPIX, fill=0)
    var order = List[Int](length=T, fill=0)
    for i in range(ds.N):
        ds.pair(i, False, canvas)
        buckets[crop_len(token_order(canvas, 0, V, order)) // 64].append(i)
    return buckets^


def build_batch_crop(ds: Dataset, buckets: List[List[Int]], C: Int, V: Int, batch: Int, flip_prob: Float64,
                     cfg_drop: Float64, context_drop: Float64, history_drop: Float64, mut ids: List[Int32],
                     mut tgt: List[Int32], mut mask_i: List[Int32], mut cls: List[Int32], mut gscale: List[Float32],
                     mut toks_i: List[Int32]) -> Int:
    """build_batch for CROP architectures: examples from one bucket (or, when the batch hides
    context frames, any), each running the same number of tokens n, which it returns."""
    var blank = blank_frames(context_drop, history_drop)  # one draw per batch
    var i0 = Int(random_ui64(0, UInt64(ds.N - 1)))
    var bucket = -1
    if blank == 0:  # i0 = the example at a random position of the buckets laid end to end
        var pos = Int(random_ui64(0, UInt64(ds.N - 1)))
        bucket = 0
        while pos >= len(buckets[bucket]):
            pos -= len(buckets[bucket])
            bucket += 1
        i0 = buckets[bucket][pos]
    var targets = List[List[Int]]()
    var orders = List[List[Int]]()
    var labels = List[Int]()
    var need = 0
    var masked_ids = List[Int](length=NPIX, fill=0)
    for b in range(batch):
        var i = i0
        if b > 0:
            if bucket >= 0:
                i = buckets[bucket][Int(random_ui64(0, UInt64(len(buckets[bucket]) - 1)))]
            else:
                i = Int(random_ui64(0, UInt64(ds.N - 1)))
        var target = List[Int](length=NPIX, fill=0)
        ds.pair(i, random_float64() < flip_prob, target)
        for p in range(NPIX):
            masked_ids[p] = V if p < blank * FRAME_PIX else target[p]
        var order = List[Int](length=T, fill=0)
        need = max(need, token_order(masked_ids, 0, V, order))
        targets.append(target^)
        orders.append(order^)
        labels.append(Int(ds.labels[i]))
    var n = crop_len(need)
    toks_i.clear()
    var toks = List[Int](capacity=n)
    var mask = List[Bool](length=T, fill=False)
    var order = List[Int](length=T, fill=0)
    for b in range(batch):
        take_tokens(orders[b], n, toks)
        var c = labels[b]
        if random_float64() < cfg_drop:
            c = C
        var n_masked = random_mask_rows(mask, order, toks)
        apply_mask_rows(masked_ids, targets[b], mask, V, blank, toks)
        for p in range(NPIX):
            tgt[b * NPIX + p] = Int32(targets[b][p])
            ids[b * NPIX + p] = Int32(masked_ids[p])
        for r in range(n):
            mask_i[b * n + r] = 1 if mask[r] else 0
            toks_i.append(Int32(toks[r]))
        cls[b] = Int32(c)
        gscale[b] = Float32(1.0 / Float64(n_masked * PP * batch))
    return n


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
    var v_mask = List[Bool]()
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
                v_mask.append(vmask[t])
            v_cls.append(Int32(vs.labels[j]))
            v_gs.append(Float32(1.0 / Float64(n_masked * PP * batch)))
        print("validation:", n_val, "of", vs.N, "held-out examples")
    # The tokens each validation batch runs: all of them, or with CROP those near the keyframes
    # (all when they are hidden).
    var vb_c = ValBatches()
    var vb_0 = ValBatches()
    var vb_m = ValBatches()  # uncropped archs: masked only where a CROP = val_crop arch would run
    var val_crop = arg_int(args, "--val-crop", 3)
    var matched = CROP < 0 and not STACKED and INBETWEEN and val_crop >= 0
    var dropped = 0
    var opaque = 0
    for b0 in range(0, n_val, batch):
        var px = Slice(b0 * NPIX, (b0 + batch) * NPIX)
        var mk = List[Bool](v_mask[b0 * T:(b0 + batch) * T])
        dropped += vb_c.add(List[Int32](v_ids[px]), List[Int32](v_tgt[px]), mk, batch, ds.V, CROP, False)
        _ = vb_0.add(List[Int32](v_ids0[px]), List[Int32](v_tgt[px]), mk, batch, ds.V, CROP, False)
        if matched:
            _ = vb_m.add(List[Int32](v_ids[px]), List[Int32](v_tgt[px]), mk, batch, ds.V, val_crop, True)
    for i in range(n_val):
        for p in range(TGT0, NPIX):
            if v_tgt[i * NPIX + p] != 0:
                opaque += 1
    comptime if CROP >= 0:
        if n_val > 0:
            var tot = 0
            for k in range(len(vb_c.n)):
                tot += vb_c.n[k]
            print("validation crop: runs", Float32(Float64(tot) / Float64(len(vb_c.n))), "of", T,
                  "tokens per canvas, drops", dropped, "of", opaque, "opaque target pixels (",
                  Float32(100.0 * Float64(dropped) / Float64(max(opaque, 1))), "% )")
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
    var buckets = List[List[Int]]()
    comptime if CROP >= 0:
        buckets = crop_buckets(ds, cfg.V)
        var line = String("crop buckets (tokens: examples):")
        for k in range(len(buckets)):
            if len(buckets[k]) > 0:
                line += " " + String(k * 64) + ":" + String(len(buckets[k]))
        print(line)
    print("device:", e.dev.device_name(), "|", Float32(Float64(e.dev.nf) * 4.0 / 1e9), "GB of buffers")
    if use_gpu and not e.dev.gpu:
        print("warning: no GPU found by Mojo, falling back to slow CPU kernel emulation")
    e.dev.load_params(m.w, m.decay)
    var opt = Optimizer()
    if use_muon:
        if not e.dev.gpu:
            raise Error("--muon needs the GPU")
        opt = Optimizer(m.matrices(), Float32(arg_float(args, "--muon-mu", 0.95)))
    var d = dirname(out)
    if d.byte_length() > 0:
        makedirs(d, exist_ok=True)

    var ids = List[Int32](length=batch * NPIX, fill=0)
    var tgt = List[Int32](length=batch * NPIX, fill=0)
    var mask_i = List[Int32](length=batch * T, fill=0)
    var cls = List[Int32](length=batch, fill=0)
    var gscale = List[Float32](length=batch, fill=0.0)
    var toks_i = List[Int32]()
    var n_tok = T
    var tok_acc: Float64 = 0.0
    comptime if CROP >= 0:
        n_tok = build_batch_crop(ds, buckets, cfg.C, cfg.V, batch, flip_prob, cfg_drop, context_drop, history_drop,
                                 ids, tgt, mask_i, cls, gscale, toks_i)
    else:
        build_batch(ds, cfg.C, cfg.V, batch, flip_prob, cfg_drop, context_drop, history_drop, ids, tgt, mask_i, cls,
                    gscale)

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
        var lr = lr_at(step, steps, warmup, lr_max, decay)

        comptime if CROP >= 0:
            e.set_tokens(toks_i, n_tok)
            tok_acc += Float64(n_tok)
            e.upload_batch(ids, tgt, List[Int32](mask_i[0:batch * n_tok]), cls, gscale)
        else:
            e.upload_batch(ids, tgt, mask_i, cls, gscale)
        e.dev.zero_grad()
        e.forward(m)
        # The next batch is built on the CPU while the GPU runs this forward pass.
        comptime if CROP >= 0:
            n_tok = build_batch_crop(ds, buckets, cfg.C, cfg.V, batch, flip_prob, cfg_drop, context_drop,
                                     history_drop, ids, tgt, mask_i, cls, gscale, toks_i)
        else:
            build_batch(ds, cfg.C, cfg.V, batch, flip_prob, cfg_drop, context_drop, history_drop, ids, tgt, mask_i,
                        cls, gscale)
        var loss = e.loss()
        e.backward(m)
        var norm = opt.step(e.dev, lr, step, wd, 1.0)
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
                  Int(eta / 60.0), "min", end="")
            comptime if CROP >= 0:
                print(" | tokens", Int(tok_acc / n), end="")
            print(flush=True)
            loss_acc = 0.0
            tok_acc = 0.0
            norm_acc = 0.0
        if n_val > 0 and (step % val_every == 0 or step == steps):
            var lc = val_loss(e, m, v_ids, v_tgt, vb_c, v_cls, v_gs, n_val, batch)
            var l0 = val_loss(e, m, v_ids0, v_tgt, vb_0, v_cls, v_gs, n_val, batch)
            print("val step", step, "| loss with context", Float32(lc), "| loss from scratch", Float32(l0), end="")
            if matched:
                var lm = val_loss(e, m, v_ids, v_tgt, vb_m, v_cls, v_gs, n_val, batch)
                print(" | loss with context inside crop", val_crop, Float32(lm), end="")
            print(flush=True)
        if step % save_every == 0 or step == steps:
            e.dev.store_params(m.w)
            m.save(out)
            print("saved", out + ".bin", flush=True)
