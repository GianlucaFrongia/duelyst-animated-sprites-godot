# Generate animated sprite sheets with iterative unmasking (MaskGIT) and classifier-free guidance.
#
# The first frame is drawn from an empty context, or the sheet opens with a dataset example's
# context frames (--init). Every later frame is drawn with the previous frame(s) as context,
# so a sheet of any length is built one frame at a time.
#
#     sh tools/build.sh sample      # ./sample-ctx; ARCH=<name> for another architecture
#     ./sample-ctx --ckpt checkpoints/sprites --data data --class idle --n 4 --frames 8
#     ./sample-ctx --ckpt checkpoints/sprites --data data --class idle --init 120   # animate a real frame
#     python tools/render.py --samples out/samples.bin --palette data/palette.txt --out out
#
# Several sheets are drawn in parallel on the GPU (conditional and unconditional passes in
# one batch). Without a GPU, or with --cpu, it falls back to the CPU reference model.

from std.sys import argv, has_accelerator
from std.math import cos, exp, log
from std.os import makedirs
from std.os.path import dirname
from std.random import seed, random_float64
from std.time import perf_counter_ns
from src.model import (
    Config, Model, Work, forward_tokens, T, PP, NPIX, FRAME_PIX, IMG, INBETWEEN, CROP, tgt_pix, is_target,
    full_tokens, token_order, crop_len, take_tokens,
)
from src.data import CF, TGT0, read_context, arg_str, arg_int, arg_float
from src.engine import Engine


def class_id(name: String, data_dir: String, C: Int) raises -> Int:
    """Accepts a class name from <data>/classes.txt, a number, or 'none'."""
    if name == "none":
        return C
    var f = open(data_dir + "/classes.txt", "r")
    var txt = f.read()
    f.close()
    var i = 0
    for line in txt.splitlines():
        if line == name:
            return i
        i += 1
    return Int(name)


def batch_logits(mut m: Model, mut wk: Work, eng: Optional[Engine], ids: List[Int32], cls: List[Int32],
                 rows: Int, rtoks: List[List[Int]], mut out: List[Float32]) raises:
    """Logits [rows, n, PP, V] for a batch of canvases, row b running the n tokens rtoks[b]
    (the engine must already run them: set_tokens)."""
    var tpv = len(rtoks[0]) * PP * m.cfg.V
    if eng:
        ref e = eng.value()
        e.dev.upload_i(e.i_ids, ids)
        e.dev.upload_i(e.i_cls, cls)
        e.forward(m)
        e.dev.sync()
        e.dev.download_f(e.o_logits, rows * tpv, out)
        return
    var row = List[Int](length=NPIX, fill=0)
    for b in range(rows):
        for i in range(NPIX):
            row[i] = Int(ids[b * NPIX + i])
        forward_tokens(m, wk, row, Int(cls[b]), rtoks[b])
        for i in range(tpv):
            out[b * tpv + i] = wk.logits[i]


def generate_frames(mut m: Model, mut wk: Work, mut eng: Optional[Engine], mut ids: List[Int], ns: Int,
                    rows: Int, cls: Int, use_cfg: Bool, n_steps: Int, temp: Float32, guidance: Float32,
                    choice_noise: Float64, mut commit: List[Int]) raises:
    """Draws the target frame (bottom half) of each of the ns canvases in ids, given the
    context frame in the top half. Batch rows [0, rows/2) are conditional, the rest unconditional.
    commit[n * T + t] gets the step that fixed token t of canvas n (-1: context, never masked).
    With CROP only the target tokens near the context frames' sprites are drawn (the same count n
    for every canvas, see token_order); the rest of the frame stays transparent."""
    var V = m.cfg.V
    var nr = T  # tokens run per canvas
    var rtoks = List[List[Int]]()  # per batch row
    comptime if CROP >= 0:
        var orders = List[List[Int]]()
        var need = 0
        for n in range(ns):
            var order = List[Int](length=T, fill=0)
            need = max(need, token_order(ids, n * NPIX, V, order))
            orders.append(order^)
        nr = crop_len(need)
        for b in range(rows):
            var toks = List[Int]()
            take_tokens(orders[b % (rows // 2) % ns if use_cfg else b % ns], nr, toks)
            rtoks.append(toks^)
    else:
        for _ in range(rows):
            rtoks.append(full_tokens())
    if eng:
        var flat = List[Int32](capacity=rows * nr)
        for b in range(rows):
            for r in range(nr):
                flat.append(Int32(rtoks[b][r]))
        eng.value().set_tokens(flat, nr)
    var tpv = nr * PP * V
    var half = rows // 2 if use_cfg else rows
    var lg = List[Float32](length=rows * tpv, fill=0.0)
    var ids32 = List[Int32](length=rows * NPIX, fill=Int32(V))
    var cls32 = List[Int32](length=rows, fill=Int32(cls))
    if use_cfg:
        for b in range(half, rows):
            cls32[b] = Int32(m.cfg.C)
    var proposal = List[Int](length=NPIX, fill=0)
    var masked = List[Bool](length=ns * T, fill=False)
    var conf = List[Float64](length=T, fill=0.0)
    var probs = List[Float32](length=V, fill=0.0)
    var n_masked = 0  # target tokens run per canvas (the same for every canvas)
    for r in range(nr):
        if is_target(rtoks[0][r]):
            n_masked += 1
    for n in range(ns):
        for i in range(TGT0, NPIX):
            ids[n * NPIX + i] = 0 if CROP >= 0 else V  # what isn't drawn stays transparent
        for t in range(T):
            masked[n * T + t] = False
            commit[n * T + t] = -1
        for r in range(nr):
            var t = rtoks[n][r]
            masked[n * T + t] = is_target(t)
            if is_target(t):
                for k in range(PP):
                    ids[n * NPIX + tgt_pix(t, k)] = V
    var n_sel = n_masked

    for s in range(n_steps):
        for n in range(ns):
            for i in range(NPIX):
                ids32[n * NPIX + i] = Int32(ids[n * NPIX + i])
                if use_cfg:
                    ids32[(half + n) * NPIX + i] = Int32(ids[n * NPIX + i])
        batch_logits(m, wk, eng, ids32, cls32, rows, rtoks, lg)
        var progress = Float64(s + 1) / Float64(n_steps)

        # How many target tokens stay masked after this step (cosine schedule).
        var keep = Int(cos(1.5707963267948966 * progress) * Float64(n_sel))
        if s == n_steps - 1:
            keep = 0
        keep = min(keep, n_masked - 1)
        keep = max(keep, 0)

        for n in range(ns):
            var oc = n * tpv
            var ou = (half + n) * tpv
            # Propose pixels for every masked token and score the token's confidence.
            for r in range(nr):
                var t = rtoks[n][r]
                if not masked[n * T + t]:
                    continue
                var c: Float64 = 0.0
                for k in range(PP):
                    var o = (r * PP + k) * V
                    var mx: Float32 = -1e30
                    for j in range(V):
                        var l = lg[oc + o + j]
                        if use_cfg:
                            var lu = lg[ou + o + j]
                            l = lu + guidance * (l - lu)
                        l /= temp
                        probs[j] = l
                        mx = max(mx, l)
                    var z: Float32 = 0.0
                    for j in range(V):
                        probs[j] = exp(probs[j] - mx)
                        z += probs[j]
                    var r = Float32(random_float64()) * z
                    var pick = V - 1
                    for j in range(V):
                        r -= probs[j]
                        if r <= 0.0:
                            pick = j
                            break
                    proposal[tgt_pix(t, k)] = pick
                    c += log(Float64(probs[pick] / z) + 1e-12)
                # Gumbel noise on confidence, annealed to zero: explores early, commits late.
                var u = max(random_float64(), 1e-12)
                conf[t] = c + choice_noise * (1.0 - progress) * (-log(-log(u) + 1e-12))

            # Commit every masked token except the `keep` least confident ones.
            var still = List[Bool](length=T, fill=False)
            for _ in range(keep):
                var best = -1
                for t in range(T):
                    if masked[n * T + t] and not still[t] and (best < 0 or conf[t] < conf[best]):
                        best = t
                still[best] = True
            for t in range(T):
                if masked[n * T + t] and not still[t]:
                    for k in range(PP):
                        var p = tgt_pix(t, k)
                        ids[n * NPIX + p] = proposal[p]
                    masked[n * T + t] = False
                    commit[n * T + t] = s
        n_masked = keep


def fill_sequential(mut m: Model, mut wk: Work, mut eng: Optional[Engine], mut ids: List[Int],
                    mut sheets: List[UInt8], init_px: List[Int], ns: Int, n_frames: Int, rows: Int, cls: Int,
                    use_cfg: Bool, n_steps: Int, temp: Float32, guidance: Float32, choice_noise: Float64) raises:
    """Draws every sheet's frames one at a time, each from the frame(s) before it. With init_px
    (--init), the example's context frames open the sheet in place of drawn frames."""
    var commit = List[Int](length=ns * T, fill=0)
    var seeds = CF if len(init_px) > 0 else 0
    for f in range(n_frames):
        if f < seeds:  # a given frame goes where a drawn frame would land
            for n in range(ns):
                for i in range(FRAME_PIX):
                    ids[n * NPIX + TGT0 + i] = init_px[f * FRAME_PIX + i]
        else:
            generate_frames(m, wk, eng, ids, ns, rows, cls, use_cfg, n_steps, temp, guidance, choice_noise, commit)
        for n in range(ns):
            var base = n * NPIX
            for i in range(FRAME_PIX):
                sheets[(n * n_frames + f) * FRAME_PIX + i] = UInt8(ids[base + TGT0 + i])
            for i in range(TGT0):  # shift the history: the newest frame becomes context
                ids[base + i] = ids[base + FRAME_PIX + i]


def fill_inbetween(mut m: Model, mut wk: Work, mut eng: Optional[Engine], mut ids: List[Int],
                   mut sheets: List[UInt8], keys: List[Int], ns: Int, n_frames: Int, rows: Int, cls: Int,
                   use_cfg: Bool, n_steps: Int, temp: Float32, guidance: Float32, choice_noise: Float64) raises:
    """Keyframes A and B become the first and last frame of every sheet; the frames between are
    drawn by halving gaps (n_frames - 1 must be a power of two)."""
    for n in range(ns):
        for i in range(FRAME_PIX):
            sheets[n * n_frames * FRAME_PIX + i] = UInt8(keys[i])
            sheets[(n * n_frames + n_frames - 1) * FRAME_PIX + i] = UInt8(keys[FRAME_PIX + i])
    var commit = List[Int](length=ns * T, fill=0)
    var gap = n_frames - 1
    while gap >= 2:
        for a in range(0, n_frames - 1, gap):
            for n in range(ns):
                for i in range(FRAME_PIX):
                    ids[n * NPIX + i] = Int(sheets[(n * n_frames + a) * FRAME_PIX + i])
                    ids[n * NPIX + FRAME_PIX + i] = Int(sheets[(n * n_frames + a + gap) * FRAME_PIX + i])
            generate_frames(m, wk, eng, ids, ns, rows, cls, use_cfg, n_steps, temp, guidance, choice_noise, commit)
            for n in range(ns):
                for i in range(FRAME_PIX):
                    sheets[(n * n_frames + a + gap // 2) * FRAME_PIX + i] = UInt8(ids[n * NPIX + TGT0 + i])
        gap //= 2


def main() raises:
    var a = argv()
    var args = List[String]()
    var use_gpu = True
    for i in range(len(a)):
        args.append(String(a[i]))
        if String(a[i]) == "--cpu":
            use_gpu = False

    var ckpt = arg_str(args, "--ckpt", "checkpoints/sprites")
    var data_dir = arg_str(args, "--data", "data")
    var n_samples = arg_int(args, "--n", 4)
    var n_frames = arg_int(args, "--frames", 8)
    var group = arg_int(args, "--batch", 4)  # sheets drawn in parallel
    var n_steps = arg_int(args, "--steps", 12)
    var temp = Float32(arg_float(args, "--temperature", 0.7))
    var guidance = Float32(arg_float(args, "--guidance", 2.0))
    var choice_noise = arg_float(args, "--choice-noise", 2.0)
    var out = arg_str(args, "--out", "out/samples.bin")
    var init = arg_int(args, "--init", -1)  # dataset example whose context frames open every sheet
    seed(arg_int(args, "--seed", 0))

    var cfg = Config.load(ckpt + ".cfg")
    var m = Model(cfg)
    m.load_weights(ckpt)
    var wk = Work(cfg)
    var cls = class_id(arg_str(args, "--class", "0"), data_dir, cfg.C)
    var use_cfg = guidance != 1.0 and cls != cfg.C
    group = max(1, min(group, n_samples))
    var rows = group * (2 if use_cfg else 1)

    var eng: Optional[Engine] = None
    comptime if has_accelerator():
        if use_gpu:
            var e = Engine(cfg, rows, True)
            if e.dev.gpu:
                e.dev.load_params(m.w, m.decay)
                print("device:", e.dev.device_name())
                eng = e^
    if not eng:
        print("device: CPU")
    print("sampling", n_samples, "sheets of", n_frames, "frames, class", cls, "| guidance", guidance,
          "| steps", n_steps)

    if init < -1:
        raise Error("--init must be an example index >= 0")
    var init_px = read_context(data_dir, init) if init >= 0 else List[Int]()
    comptime if INBETWEEN:
        if init < 0:
            raise Error("in-betweening needs keyframes: --init <example> uses that example's two keyframes")
        var gap0 = n_frames - 1
        if gap0 < 2 or (gap0 & (gap0 - 1)) != 0:
            raise Error("in-betweening fills gaps by halving: --frames must be 3, 5, 9, 17, ...")

    var ids = List[Int](length=group * NPIX, fill=cfg.V)
    var result = List[UInt8]()
    var done = 0
    while done < n_samples:
        var ns = min(group, n_samples - done)
        var t0 = perf_counter_ns()
        var sheets = List[UInt8](length=ns * n_frames * FRAME_PIX, fill=0)
        for i in range(group * NPIX):
            ids[i] = cfg.V  # empty context: the first frame is drawn from nothing
        comptime if INBETWEEN:
            fill_inbetween(m, wk, eng, ids, sheets, init_px, ns, n_frames, rows, cls, use_cfg, n_steps,
                           temp, guidance, choice_noise)
        else:
            fill_sequential(m, wk, eng, ids, sheets, init_px, ns, n_frames, rows, cls, use_cfg, n_steps,
                            temp, guidance, choice_noise)
        result.extend(sheets^)
        done += ns
        print("sheets", done, "/", n_samples, "in", Float32(Float64(perf_counter_ns() - t0) / 1e9), "s", flush=True)

    var d = dirname(out)
    if d.byte_length() > 0:
        makedirs(d, exist_ok=True)
    var f = open(out, "w")
    f.write_bytes(Span(result))
    f.close()
    var mf = open(out + ".txt", "w")
    mf.write("frames ", n_frames, "\nsize ", IMG, "\n")
    mf.close()
    print("wrote", out)
