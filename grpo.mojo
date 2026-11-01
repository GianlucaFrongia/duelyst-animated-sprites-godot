# Fine-tune a trained model with GRPO (DeepSeekMath / R1): reinforcement learning without a
# value model. Each iteration draws --group samples of the target frame (the next frame, or the
# middle frame for inbetween) for each of --prompts training examples with the real sampler
# (temperature, guidance, confidence-ordered unmasking), scores each against the real frame,
# and normalizes the scores within each example's group:
# the advantage. The policy gradient raises the log-probability of every token a sample
# committed, at the canvas state it was committed in, by its advantage. The engine's masked loss
# computes exactly that: one pass per unmasking step, the sample as target, the step's tokens as
# the mask and the advantage as the per-example loss scale.
#
#     ./grpo-ctx --ckpt checkpoints/cmp_ctx --data data_duelyst_hr_soc_val --out checkpoints/cmp_ctx_grpo
#
# Reward: foreground pixel accuracy, the share of pixels where the sample or the real frame is
# not transparent on which the two agree. Every --eval-every iterations it logs the mean reward
# on held-out units (<data>/val, the same prompts and seed each time), which is the number to watch.
# ponytail: no KL penalty to the starting model (GRPO's beta) and no clipped ratio (each sample
# is used for one update, so the ratio is 1); add them if the held-out reward rises but the
# samples degrade.

from std.sys import argv
from std.math import sqrt
from std.os import makedirs
from std.os.path import dirname, exists
from std.random import seed, random_ui64
from std.time import perf_counter_ns
from src.model import Config, Model, Work, T, NT, PP, NPIX, FRAME_PIX, tgt_pix, is_target
from src.data import Dataset, TGT0, arg_str, arg_int, arg_float
from src.engine import Engine
from src.optim import Optimizer
from sample import generate_frames


def reward(ids: List[Int], n: Int, real: List[UInt8], r0: Int) -> Float64:
    """Foreground accuracy of the target frame of canvas n against real[r0 : r0 + FRAME_PIX]."""
    var fg = 0
    var same = 0
    for i in range(FRAME_PIX):
        var g = ids[n * NPIX + TGT0 + i]
        var r = Int(real[r0 + i])
        if g != 0 or r != 0:
            fg += 1
            if g == r:
                same += 1
    return Float64(same) / Float64(fg) if fg > 0 else 1.0


def draw(mut m: Model, mut wk: Work, mut gen: Optional[Engine], ds: Dataset, ex: List[Int], reps: Int,
         mut ids: List[Int], mut commit: List[Int], rows: Int, cls: Int, use_cfg: Bool, n_steps: Int,
         temp: Float32, guidance: Float32, choice_noise: Float64) raises:
    """reps samples of the target frame of each example in ex (one class), canvas n = i * reps + j."""
    for i in range(len(ex)):
        for j in range(reps):
            var n = i * reps + j
            for p in range(TGT0):
                ids[n * NPIX + p] = Int(ds.pixels[ex[i] * NPIX + p])
    generate_frames(m, wk, gen, ids, len(ex) * reps, rows, cls, use_cfg, n_steps, temp, guidance, choice_noise,
                    commit)


def held_out(mut m: Model, mut wk: Work, mut gen: Optional[Engine], v: Dataset, groups: List[List[Int]],
             cls: List[Int], mut ids: List[Int], mut commit: List[Int], rows: Int, use_cfg: Bool, n_steps: Int,
             temp: Float32, guidance: Float32, choice_noise: Float64) raises -> Float64:
    """Mean reward of one sample per held-out prompt, with the same draws every time."""
    seed(4321)
    var total: Float64 = 0.0
    var count = 0
    for gi in range(len(groups)):
        draw(m, wk, gen, v, groups[gi], 1, ids, commit, rows, cls[gi], use_cfg, n_steps, temp, guidance, choice_noise)
        for i in range(len(groups[gi])):
            total += reward(ids, i, v.pixels, groups[gi][i] * NPIX + TGT0)
            count += 1
    return total / Float64(count)


def main() raises:
    var a = argv()
    var args = List[String]()
    for i in range(len(a)):
        args.append(String(a[i]))
    var ckpt = arg_str(args, "--ckpt", "checkpoints/sprites")
    var data_dir = arg_str(args, "--data", "data")
    var out = arg_str(args, "--out", ckpt + "_grpo")
    var iters = arg_int(args, "--iters", 200)
    var P = arg_int(args, "--prompts", 8)
    var G = arg_int(args, "--group", 4)
    var lr = Float32(arg_float(args, "--lr", 1e-5))
    var n_steps = arg_int(args, "--steps", 12)
    var temp = Float32(arg_float(args, "--temperature", 0.7))
    var guidance = Float32(arg_float(args, "--guidance", 2.0))
    var choice_noise = arg_float(args, "--choice-noise", 2.0)
    var eval_n = arg_int(args, "--eval-n", 64)
    var eval_every = arg_int(args, "--eval-every", 25)
    var save_every = arg_int(args, "--save-every", 50)
    seed(arg_int(args, "--seed", 0))

    var cfg = Config.load(ckpt + ".cfg")
    var m = Model(cfg)
    m.load_weights(ckpt)
    var wk = Work(cfg)
    var ds = Dataset(data_dir)
    var B = P * G
    var use_cfg = guidance != 1.0
    var rows = B * (2 if use_cfg else 1)
    var gen: Optional[Engine] = Engine(cfg, rows, True)  # sampling, conditional + unconditional rows
    var upd = Engine(cfg, B, True)  # policy-gradient passes
    if not upd.dev.gpu:
        raise Error("grpo needs the GPU")
    gen.value().dev.load_params(m.w, m.decay)
    upd.dev.load_params(m.w, m.decay)
    print("model:", m.n, "parameters | ", P, "prompts x", G, "samples | lr", lr, "| device", upd.dev.device_name())

    # Examples by class: an iteration's prompts share one class (the sampler takes one).
    var by_cls = List[List[Int]]()
    for _ in range(cfg.C):
        by_cls.append(List[Int]())
    for i in range(ds.N):
        by_cls[Int(ds.labels[i])].append(i)

    # Held-out prompts: eval_n examples spread over <data>/val, drawn in class groups of up to B.
    var vs: Optional[Dataset] = None
    var v_groups = List[List[Int]]()
    var v_cls = List[Int]()
    if eval_every > 0 and exists(data_dir + "/val/meta.txt"):
        vs = Dataset(data_dir + "/val")
        ref v = vs.value()
        var n = min(eval_n, v.N)
        for c in range(cfg.C):
            var g = List[Int]()
            for i in range(n):
                var j = i * v.N // n
                if Int(v.labels[j]) == c:
                    g.append(j)
                    if len(g) == B:
                        v_groups.append(g.copy())
                        v_cls.append(c)
                        g.clear()
            if len(g) > 0:
                v_groups.append(g.copy())
                v_cls.append(c)
        print("held-out prompts:", n, "of", v.N)

    var ids = List[Int](length=rows * NPIX, fill=cfg.V)
    var commit = List[Int](length=rows * T, fill=0)

    var u_ids = List[Int32](length=B * NPIX, fill=0)
    var u_tgt = List[Int32](length=B * NPIX, fill=0)
    var u_mask = List[Int32](length=B * T, fill=0)
    var u_cls = List[Int32](length=B, fill=0)
    var u_gs = List[Float32](length=B, fill=0.0)
    var adv = List[Float64](length=B, fill=0.0)
    var t0 = perf_counter_ns()
    var opt = Optimizer()  # AdamW
    for it in range(iters + 1):
        if vs and (it % eval_every == 0 or it == iters):
            var r = held_out(m, wk, gen, vs.value(), v_groups, v_cls, ids, commit, rows, use_cfg, n_steps, temp,
                             guidance, choice_noise)
            print("eval iter", it, "| held-out reward", Float32(r), flush=True)
            seed(arg_int(args, "--seed", 0) * 100003 + it)
        if it == iters:
            break

        var c = Int(random_ui64(0, UInt64(cfg.C - 1)))
        while len(by_cls[c]) == 0:
            c = Int(random_ui64(0, UInt64(cfg.C - 1)))
        var ex = List[Int]()
        for _ in range(P):
            ex.append(by_cls[c][Int(random_ui64(0, UInt64(len(by_cls[c]) - 1)))])
        draw(m, wk, gen, ds, ex, G, ids, commit, rows, c, use_cfg, n_steps, temp, guidance, choice_noise)

        # Group-normalized advantages.
        var r_sum: Float64 = 0.0
        var rew = List[Float64](length=B, fill=0.0)
        for n in range(B):
            rew[n] = reward(ids, n, ds.pixels, ex[n // G] * NPIX + TGT0)
            r_sum += rew[n]
        for i in range(P):
            var mu: Float64 = 0.0
            for j in range(G):
                mu += rew[i * G + j]
            mu /= Float64(G)
            var var_: Float64 = 0.0
            for j in range(G):
                var_ += (rew[i * G + j] - mu) ** 2
            var sd = sqrt(var_ / Float64(G))
            for j in range(G):
                adv[i * G + j] = (rew[i * G + j] - mu) / (sd + 1e-4)

        # Policy gradient: for each unmasking step, the canvases as the sampler saw them then,
        # scored on the tokens it committed; gradients add up over the steps.
        for n in range(B):
            for p in range(NPIX):
                u_tgt[n * NPIX + p] = Int32(ids[n * NPIX + p])
            u_cls[n] = Int32(c)
            u_gs[n] = Float32(adv[n] / Float64(NT * PP * B))
        upd.dev.zero_grad()
        for s in range(n_steps):
            for n in range(B):
                for p in range(NPIX):
                    u_ids[n * NPIX + p] = Int32(ids[n * NPIX + p])
                for t in range(T):
                    var ct = commit[n * T + t]
                    u_mask[n * T + t] = 1 if ct == s else 0
                    if ct >= s:
                        for k in range(PP):
                            u_ids[n * NPIX + tgt_pix(t, k)] = Int32(cfg.V)
            upd.upload_batch(u_ids, u_tgt, u_mask, u_cls, u_gs)
            upd.forward(m)
            _ = upd.loss()
            upd.backward(m)
        var norm = opt.step(upd.dev, lr, it + 1, 0.0, 1.0)
        upd.dev.store_params(m.w)
        gen.value().dev.load_params(m.w, m.decay)

        var secs = Float64(perf_counter_ns() - t0) / 1e9
        print("iter", it + 1, "/", iters, "| class", c, "| reward", Float32(r_sum / Float64(B)), "| grad norm",
              Float32(norm), "|", Float32(secs / Float64(it + 1)), "s/iter", flush=True)
        if (it + 1) % save_every == 0 or it + 1 == iters:
            var d = dirname(out)
            if d.byte_length() > 0:
                makedirs(d, exist_ok=True)
            m.save(out)
            print("saved", out + ".bin", flush=True)
