# Loads the frame-pair dataset written by tools/prepare_animations.py, plus a tiny CLI arg helper.

from std.math import cos
from std.random import random_float64, random_ui64
from arch.model import T, NT, PP, NPIX, IMG, FRAME_PIX, INBETWEEN, tgt_pix, is_target

comptime CF = NPIX // FRAME_PIX - 1  # context frames per example; the target is the last frame
comptime TGT0 = CF * FRAME_PIX  # first pixel of the target frame


struct Dataset(Movable):
    var pixels: List[UInt8]  # N * NPIX (context frames on top, target frame last)
    var labels: List[UInt8]  # N
    var classes: List[String]
    var N: Int
    var V: Int
    var C: Int

    def __init__(out self, dir: String) raises:
        self.N = 0
        self.V = 0
        self.C = 0
        var size = 0
        var frames = 2  # datasets written before K existed hold pairs
        var order = String("next")
        var f = open(dir + "/meta.txt", "r")
        var txt = f.read()
        f.close()
        for line in txt.splitlines():
            var kv = line.split(" ")
            if len(kv) < 2:
                continue
            if kv[0] == "N":
                self.N = Int(kv[1])
            elif kv[0] == "V":
                self.V = Int(kv[1])
            elif kv[0] == "C":
                self.C = Int(kv[1])
            elif kv[0] == "S":
                size = Int(kv[1])
            elif kv[0] == "K":
                frames = Int(kv[1])
            elif kv[0] == "O":
                order = String(kv[1])
        if size != IMG:
            raise Error("dataset frames are " + String(size) + "px, the model expects " + String(IMG)
                        + "px: re-run tools/prepare_animations.py")
        if frames != CF + 1:
            raise Error(dir + " has " + String(frames) + " frames per example, this architecture needs "
                        + String(CF + 1) + ": prepare it with tools/prepare_animations.py --context " + String(CF))
        if (order == "inbetween") != INBETWEEN:
            raise Error(dir + " holds '" + order + "' examples; this architecture needs "
                        + ("keyframe triples: prepare_animations.py --inbetween" if INBETWEEN
                           else "consecutive frames: prepare_animations.py without --inbetween"))
        var pf = open(dir + "/pixels.bin", "r")
        self.pixels = pf.read_bytes()
        pf.close()
        var lf = open(dir + "/labels.bin", "r")
        self.labels = lf.read_bytes()
        lf.close()
        if len(self.pixels) != self.N * NPIX or len(self.labels) != self.N:
            raise Error("dataset files do not match meta.txt")
        self.classes = List[String]()
        var cf = open(dir + "/classes.txt", "r")
        var ctxt = cf.read()
        cf.close()
        for line in ctxt.splitlines():
            if line.byte_length() > 0:
                self.classes.append(String(line))

    def pair(self, i: Int, flip: Bool, mut out: List[Int]):
        """Copies example i into out (NPIX palette indices), optionally mirrored horizontally."""
        var base = i * NPIX
        for y in range(NPIX // IMG):
            for x in range(IMG):
                var sx = IMG - 1 - x if flip else x
                out[y * IMG + x] = Int(self.pixels[base + y * IMG + sx])


def read_context(dir: String, i: Int) raises -> List[Int]:
    """The CF context frames of example i (CF * FRAME_PIX indices, oldest first), read
    without loading the whole dataset."""
    if i < 0:
        raise Error("example index must be >= 0, got " + String(i))
    var f = open(dir + "/pixels.bin", "r")
    _ = f.seek(i * NPIX)
    var b = f.read_bytes(TGT0)
    f.close()
    if len(b) != TGT0:
        raise Error("example " + String(i) + " is past the end of " + dir + "/pixels.bin")
    var out = List[Int](length=TGT0, fill=0)
    for p in range(TGT0):
        out[p] = Int(b[p])
    return out^


def random_mask(mut mask: List[Bool], mut order: List[Int]) -> Int:
    """MaskGIT cosine schedule: mask a random fraction cos(pi/2 * u) of the target tokens
    (is_target); other tokens are never masked or scored. Returns count."""
    var u = random_float64()
    var ratio = cos(1.5707963267948966 * u)
    var n = Int(ratio * Float64(NT) + 0.999)
    n = max(1, min(NT, n))
    var nt = 0
    for t in range(T):
        mask[t] = False
        if is_target(t):
            order[nt] = t
            nt += 1
    for i in range(n):  # partial Fisher-Yates shuffle
        var j = i + Int(random_ui64(0, UInt64(NT - 1 - i)))
        var tmp = order[i]
        order[i] = order[j]
        order[j] = tmp
        mask[order[i]] = True
    return n


def blank_frames(context_drop: Float64, history_drop: Float64) -> Int:
    """How many context frames apply_mask hides for one training example (random)."""
    var r = random_float64()
    if r < context_drop:
        return CF
    if CF > 1 and r < context_drop + history_drop:
        return CF - 1
    return 0


def apply_mask(mut ids: List[Int], target: List[Int], mask: List[Bool], mask_id: Int, blank_frames: Int):
    """Masks the selected tokens, and hides the oldest blank_frames context frames: all CF of
    them teaches drawing a first frame from nothing, fewer teaches the frames that follow
    while the history is still short."""
    for i in range(NPIX):
        ids[i] = target[i]
    for i in range(blank_frames * FRAME_PIX):
        ids[i] = mask_id
    for t in range(T):
        if mask[t]:
            for k in range(PP):
                ids[tgt_pix(t, k)] = mask_id


# ------------------------------------------------------------------ CLI args


def arg_str(args: List[String], name: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return args[i + 1]
    return default


def arg_int(args: List[String], name: String, default: Int) raises -> Int:
    var s = arg_str(args, name, "")
    return Int(s) if s.byte_length() > 0 else default


def arg_float(args: List[String], name: String, default: Float64) raises -> Float64:
    var s = arg_str(args, name, "")
    return Float64(s) if s.byte_length() > 0 else default
