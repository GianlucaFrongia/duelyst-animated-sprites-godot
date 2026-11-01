# A small bidirectional transformer that predicts masked pixels (MaskGIT-style), shared by
# every architecture in models/. An architecture is only its arch/variant.mojo: how many
# context frames the canvas holds, whether they are extra tokens or embedding slots, the
# dataset's frame order, the attention band, the norm and the token order.
#
# The canvas is CF context frames (oldest first, MASK when not available) and then the target
# frame, each 64x64. Frames are split into 32x32 = 1024 tokens of 2x2 pixel patches, and each
# target token predicts its 4 pixels. With STACKED, every frame's patches are tokens (the
# position embedding tells the frames apart) and only the target frame's are masked and
# predicted. Otherwise the transformer only sees the target frame's 1024 tokens, and each
# token's embedding also adds the context frames' pixels at its own position, so "what was
# here last frame" is in every token.
#
# All parameters live in one flat buffer, so the optimizer, gradient clipping and
# checkpointing are single loops over that buffer.

from std.math import sqrt, log, cos, exp
from std.random import random_float64
from arch.variant import ARCH, CF, STACKED, INBETWEEN, BAND, DYT, HILBERT, CROP
from src.kernels import (
    FP, ptr, fill, copy, axpy, mm_nt, mm_nn, mm_tn,
    linear_fwd, linear_bwd, ln_fwd, ln_bwd, dyt_fwd, dyt_bwd, gelu_fwd, gelu_bwd, softmax_rows,
)

comptime IMG = 64  # frame width and height (also the canvas row stride)
comptime GRID = 32  # tokens per row
comptime FRAME_T = GRID * GRID  # tokens per frame
comptime PP = 4  # pixels per token
comptime FRAME_PIX = IMG * IMG
comptime NPIX = (CF + 1) * FRAME_PIX  # canvas: the context frames, then the target frame
comptime T = (CF + 1) * FRAME_T if STACKED else FRAME_T  # transformer tokens per canvas
comptime NT = FRAME_T  # tokens that are masked and predicted (see is_target)
comptime T0 = T - FRAME_T  # first target token
comptime SLOTS = PP if STACKED else (CF + 1) * PP  # embedded pixels per token (see slot_pix)


def pix_index(t: Int, k: Int) -> Int:
    """Canvas index of sub-pixel k (0..3) of canvas token t (frame t // FRAME_T)."""
    var f = t // FRAME_T
    var tf = t - f * FRAME_T
    var py: Int
    var px: Int
    comptime if not HILBERT:
        py = tf // GRID
        px = tf % GRID
    else:
        # token tf is step tf along a Hilbert curve over the 32x32 token grid
        py = 0
        px = 0
        var d = tf
        var s = 1
        while s < GRID:
            var rx = (d // 2) & 1
            var ry = (d ^ rx) & 1
            if ry == 0:
                if rx == 1:
                    px = s - 1 - px
                    py = s - 1 - py
                var tmp = px
                px = py
                py = tmp
            px += s * rx
            py += s * ry
            d //= 4
            s *= 2
    return f * FRAME_PIX + (2 * py + k // 2) * IMG + 2 * px + k % 2


def in_band(q: Int, k: Int) -> Bool:
    """Whether query token q attends to key token k: their 64-token blocks (2 token rows, or an
    8x8 square with HILBERT) are at most BAND apart within their frames."""
    return abs((q % FRAME_T) // 64 - (k % FRAME_T) // 64) <= BAND


def is_target(t: Int) -> Bool:
    """Whether token t belongs to the target frame (is masked and predicted)."""
    return t >= T0


def tgt_pix(t: Int, k: Int) -> Int:
    """Canvas index of target sub-pixel k of token t."""
    return pix_index(CF * FRAME_T + t - T0, k)


def slot_pix(t: Int, s: Int) -> Int:
    """Canvas index of embedding slot s of token t. STACKED: the token's own 4 pixels.
    Otherwise 4 target pixels, then 4 from each context frame, newest first."""
    comptime if STACKED:
        return pix_index(t, s)
    var g = s // PP
    var frame = CF - g if g > 0 else CF
    return pix_index(frame * FRAME_T + t, s - g * PP)


# ======================================================================== token selection
# With CROP >= 0 (in-betweening) only the target tokens near the keyframes' sprites run: the
# frame halfway between two keyframes lies almost entirely within a few tokens of either. Each
# canvas runs n tokens, n a multiple of the attention tile and the same for a whole batch: every
# token within CROP of an opaque context pixel, padded with the next nearest ones. The tokens
# that don't run are transparent.

comptime CROP_TILE = 64  # n is a multiple of this (the attention kernels' query tile)


@always_inline
def run_len(TN: Int32) -> Int:
    """Tokens per canvas inside a kernel: the batch's count with CROP, else the constant T, which
    the compiler folds into the index arithmetic (a runtime count costs uncropped archs ~5%)."""
    comptime if CROP >= 0:
        return Int(TN)
    return T


def full_tokens() -> List[Int]:
    """Every canvas token, in order: what runs without cropping."""
    var toks = List[Int](capacity=T)
    for t in range(T):
        toks.append(t)
    return toks^


def crop_check() raises:
    """Cropping needs every token to attend to every other (the band is by row index) and the
    context frames in the embedding slots (only target tokens are selected)."""
    comptime if CROP >= 0:
        if STACKED or BAND * 64 < FRAME_T:
            raise Error("CROP needs STACKED = False and full attention (BAND = 16)")


def token_order(ids: List[Int], base: Int, V: Int, mut order: List[Int], margin: Int = CROP) -> Int:
    """Fills order (T entries) with the target tokens of canvas ids[base : base + NPIX] by
    chessboard distance, in tokens, to the nearest opaque pixel of its context frames (ties in
    token order), and returns how many lie within margin (CROP). Without an opaque context pixel (empty
    or masked keyframes) all T tokens are needed and order is the identity."""
    comptime INF = 1 << 20
    var dist = List[Int](length=FRAME_T, fill=INF)  # per cell of the token grid, row-major
    var occupied = False
    for f in range(CF):
        var fb = base + f * FRAME_PIX
        for c in range(FRAME_T):
            var cy = c // GRID
            var cx = c - cy * GRID
            for k in range(PP):
                var v = ids[fb + (2 * cy + k // 2) * IMG + 2 * cx + k % 2]
                if v != 0 and v != V:
                    dist[c] = 0
                    occupied = True
    if not occupied:
        for t in range(T):
            order[t] = t
        return T
    # Two-pass chessboard distance transform (exact for the 8-neighbourhood).
    for cy in range(GRID):
        for cx in range(GRID):
            var d = dist[cy * GRID + cx]
            if cx > 0:
                d = min(d, dist[cy * GRID + cx - 1] + 1)
            if cy > 0:
                for dx in range(max(0, cx - 1), min(GRID, cx + 2)):
                    d = min(d, dist[(cy - 1) * GRID + dx] + 1)
            dist[cy * GRID + cx] = d
    for cy in range(GRID - 1, -1, -1):
        for cx in range(GRID - 1, -1, -1):
            var d = dist[cy * GRID + cx]
            if cx < GRID - 1:
                d = min(d, dist[cy * GRID + cx + 1] + 1)
            if cy < GRID - 1:
                for dx in range(max(0, cx - 1), min(GRID, cx + 2)):
                    d = min(d, dist[(cy + 1) * GRID + dx] + 1)
            dist[cy * GRID + cx] = d
    # Counting sort of the tokens by their cell's distance (stable, so ties stay in token order).
    var td = List[Int](length=T, fill=0)
    var cnt = List[Int](length=GRID + 1, fill=0)
    for t in range(T):
        var p = tgt_pix(t, 0) - CF * FRAME_PIX  # the token's top-left pixel within its frame
        td[t] = min(dist[(p // IMG // 2) * GRID + (p % IMG) // 2], GRID)
        cnt[td[t]] += 1
    var need = 0
    for d in range(min(margin, GRID) + 1):
        need += cnt[d]
    var start = 0
    for d in range(GRID + 1):
        var c = cnt[d]
        cnt[d] = start
        start += c
    for t in range(T):
        order[cnt[td[t]]] = t
        cnt[td[t]] += 1
    return need


def crop_len(need: Int) -> Int:
    """Tokens a canvas that needs `need` runs: rounded up to the attention tile."""
    return min(T, max(CROP_TILE, (need + CROP_TILE - 1) // CROP_TILE * CROP_TILE))


def take_tokens(order: List[Int], n: Int, mut toks: List[Int]):
    """toks = the first n tokens of order, ascending."""
    var chosen = List[Bool](length=T, fill=False)
    for i in range(n):
        chosen[order[i]] = True
    toks.clear()
    for t in range(T):
        if chosen[t]:
            toks.append(t)


def randn() -> Float32:
    var u1 = random_float64()
    if u1 < 1e-12:
        u1 = 1e-12
    var u2 = random_float64()
    return Float32(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2))


# ======================================================================== config


struct Config(Copyable, Movable):
    var V: Int  # palette size including transparent (index V is the MASK token)
    var C: Int  # number of classes (index C is the "no class" token for guidance)
    var D: Int  # model width
    var L: Int  # layers
    var H: Int  # attention heads
    var F: Int  # MLP hidden width

    def __init__(out self, V: Int, C: Int, D: Int, L: Int, H: Int, F: Int):
        self.V = V
        self.C = C
        self.D = D
        self.L = L
        self.H = H
        self.F = F

    def hd(self) -> Int:
        return self.D // self.H

    def save(self, path: String) raises:
        var f = open(path, "w")
        f.write("A ", ARCH, "\nV ", self.V, "\nC ", self.C, "\nD ", self.D, "\nL ", self.L,
                "\nH ", self.H, "\nF ", self.F, "\n")
        f.close()

    @staticmethod
    def load(path: String) raises -> Config:
        """Reads a checkpoint's .cfg and checks it was written by this architecture."""
        var c = Config(0, 0, 0, 0, 0, 0)
        var f = open(path, "r")
        var txt = f.read()
        f.close()
        var arch = String("")
        for line in txt.splitlines():
            var kv = line.split(" ")
            if len(kv) < 2:
                continue
            var k = String(kv[0])
            if k == "A":
                arch = String(kv[1])
                continue
            var v = Int(kv[1])
            if k == "V":
                c.V = v
            elif k == "C":
                c.C = v
            elif k == "D":
                c.D = v
            elif k == "L":
                c.L = v
            elif k == "H":
                c.H = v
            elif k == "F":
                c.F = v
        if arch != ARCH:
            if arch.byte_length() == 0:
                raise Error(path + " has no architecture line: add 'A <name>' (a folder in models/)")
            raise Error(path + " is a '" + arch + "' checkpoint, but this binary is built for '" + ARCH
                        + "': build with ARCH=" + arch + " sh tools/build.sh")
        return c^


# ======================================================================== parameters


@fieldwise_init
struct Matrix(Copyable, Movable):
    """A weight matrix in the flat parameter buffer: rows x cols floats at offset off."""
    var off: Int
    var rows: Int
    var cols: Int


struct Model(Movable):
    var cfg: Config
    var w: List[Float32]  # weights
    var g: List[Float32]  # gradients
    var m: List[Float32]  # Adam first moment
    var v: List[Float32]  # Adam second moment
    var decay: List[UInt8]  # 1 = apply weight decay
    var n: Int
    # global offsets
    var o_tok: Int
    var o_pos: Int
    var o_cls: Int
    var o_layers: Int
    var per_layer: Int
    var o_lnfg: Int
    var o_lnfb: Int
    var o_wout: Int
    var o_bout: Int
    var o_alpha: Int  # DyT only: one scalar per norm (2 per layer, then the final one)
    # offsets inside one layer block
    var r_ln1g: Int
    var r_ln1b: Int
    var r_wqkv: Int
    var r_bqkv: Int
    var r_wo: Int
    var r_bo: Int
    var r_ln2g: Int
    var r_ln2b: Int
    var r_w1: Int
    var r_b1: Int
    var r_w2: Int
    var r_b2: Int

    def __init__(out self, cfg: Config):
        var D = cfg.D
        var F = cfg.F
        var V = cfg.V
        self.cfg = cfg.copy()
        # layer block
        self.r_ln1g = 0
        self.r_ln1b = D
        self.r_wqkv = 2 * D
        self.r_bqkv = self.r_wqkv + 3 * D * D
        self.r_wo = self.r_bqkv + 3 * D
        self.r_bo = self.r_wo + D * D
        self.r_ln2g = self.r_bo + D
        self.r_ln2b = self.r_ln2g + D
        self.r_w1 = self.r_ln2b + D
        self.r_b1 = self.r_w1 + F * D
        self.r_w2 = self.r_b1 + F
        self.r_b2 = self.r_w2 + D * F
        self.per_layer = self.r_b2 + D
        # global
        self.o_tok = 0
        self.o_pos = SLOTS * (V + 1) * D
        self.o_cls = self.o_pos + T * D
        self.o_layers = self.o_cls + (cfg.C + 1) * D
        self.o_lnfg = self.o_layers + cfg.L * self.per_layer
        self.o_lnfb = self.o_lnfg + D
        self.o_wout = self.o_lnfb + D
        self.o_bout = self.o_wout + PP * V * D
        self.o_alpha = self.o_bout + PP * V
        self.n = self.o_alpha + (2 * cfg.L + 1 if DYT else 0)
        self.w = List[Float32](length=self.n, fill=0.0)
        self.g = List[Float32](length=self.n, fill=0.0)
        self.m = List[Float32](length=self.n, fill=0.0)
        self.v = List[Float32](length=self.n, fill=0.0)
        self.decay = List[UInt8](length=self.n, fill=0)

    def lo(self, l: Int, rel: Int) -> Int:
        return self.o_layers + l * self.per_layer + rel

    def matrices(self) -> List[Matrix]:
        """The weight matrices (what Muon orthogonalizes), in ascending offset order."""
        var D = self.cfg.D
        var F = self.cfg.F
        var out = List[Matrix]()
        for l in range(self.cfg.L):
            out.append(Matrix(self.lo(l, self.r_wqkv), 3 * D, D))
            out.append(Matrix(self.lo(l, self.r_wo), D, D))
            out.append(Matrix(self.lo(l, self.r_w1), F, D))
            out.append(Matrix(self.lo(l, self.r_w2), D, F))
        out.append(Matrix(self.o_wout, PP * self.cfg.V, D))
        return out^

    def _init_range(mut self, start: Int, count: Int, std: Float32, decay: Bool):
        for i in range(start, start + count):
            self.w[i] = randn() * std
            self.decay[i] = 1 if decay else 0

    def _const_range(mut self, start: Int, count: Int, value: Float32):
        for i in range(start, start + count):
            self.w[i] = value
            self.decay[i] = 0

    def init_weights(mut self):
        var D = self.cfg.D
        var F = self.cfg.F
        var V = self.cfg.V
        var std: Float32 = 0.02
        var proj_std = std / Float32(sqrt(2.0 * Float64(self.cfg.L)))
        self._init_range(self.o_tok, self.o_pos - self.o_tok, std, True)
        self._init_range(self.o_pos, T * D, std, True)
        self._init_range(self.o_cls, (self.cfg.C + 1) * D, std, True)
        for l in range(self.cfg.L):
            self._const_range(self.lo(l, self.r_ln1g), D, 1.0)
            self._const_range(self.lo(l, self.r_ln1b), D, 0.0)
            self._init_range(self.lo(l, self.r_wqkv), 3 * D * D, std, True)
            self._const_range(self.lo(l, self.r_bqkv), 3 * D, 0.0)
            self._init_range(self.lo(l, self.r_wo), D * D, proj_std, True)
            self._const_range(self.lo(l, self.r_bo), D, 0.0)
            self._const_range(self.lo(l, self.r_ln2g), D, 1.0)
            self._const_range(self.lo(l, self.r_ln2b), D, 0.0)
            self._init_range(self.lo(l, self.r_w1), F * D, std, True)
            self._const_range(self.lo(l, self.r_b1), F, 0.0)
            self._init_range(self.lo(l, self.r_w2), D * F, proj_std, True)
            self._const_range(self.lo(l, self.r_b2), D, 0.0)
        self._const_range(self.o_lnfg, D, 1.0)
        self._const_range(self.o_lnfb, D, 0.0)
        self._init_range(self.o_wout, PP * V * D, std, True)
        self._const_range(self.o_bout, PP * V, 0.0)
        self._const_range(self.o_alpha, self.n - self.o_alpha, 0.5)  # DyT's alpha_0

    def zero_grad(mut self):
        fill(ptr(self.g), self.n, 0.0)

    def clip_grad(mut self, max_norm: Float32) -> Float32:
        var s: Float64 = 0.0
        for i in range(self.n):
            s += Float64(self.g[i]) * Float64(self.g[i])
        var norm = Float32(sqrt(s))
        if norm > max_norm:
            var k = max_norm / norm
            for i in range(self.n):
                self.g[i] *= k
        return norm

    def adamw_step(mut self, lr: Float32, step: Int, wd: Float32):
        var b1: Float32 = 0.9
        var b2: Float32 = 0.95
        var eps: Float32 = 1e-8
        var c1 = 1.0 - Float32(Float64(b1) ** Float64(step))
        var c2 = 1.0 - Float32(Float64(b2) ** Float64(step))
        for i in range(self.n):
            var gi = self.g[i]
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi
            var mh = self.m[i] / c1
            var vh = self.v[i] / c2
            var upd = mh / (sqrt(vh) + eps)
            if self.decay[i] == 1:
                upd += wd * self.w[i]
            self.w[i] -= lr * upd

    def save(self, prefix: String) raises:
        self.cfg.save(prefix + ".cfg")
        var bp = self.w.unsafe_ptr().bitcast[UInt8]()
        var f = open(prefix + ".bin", "w")
        f.write_bytes(Span(unsafe_ptr=bp, length=self.n * 4))
        f.close()

    def load_weights(mut self, prefix: String) raises:
        _ = Config.load(prefix + ".cfg")  # architecture check
        var f = open(prefix + ".bin", "r")
        var b = f.read_bytes()
        f.close()
        if len(b) != self.n * 4:
            raise Error("checkpoint size mismatch: " + String(len(b)) + " vs " + String(self.n * 4))
        var src = b.unsafe_ptr().bitcast[Float32]()
        for i in range(self.n):
            self.w[i] = src[unsafe_offset=i]


# ======================================================================== activations cache


struct Work(Movable):
    """Activations saved by the forward pass for one canvas, plus backward scratch."""

    var x: List[Float32]  # residual stream [T,D]
    var h1: List[Float32]  # per layer: LN1 output [T,D]
    var xhat1: List[Float32]
    var rstd1: List[Float32]
    var qkv: List[Float32]  # per layer [T,3D]
    var probs: List[Float32]  # per layer, per head [T,T]
    var att: List[Float32]  # per layer: concatenated head outputs [T,D]
    var h2: List[Float32]
    var xhat2: List[Float32]
    var rstd2: List[Float32]
    var u: List[Float32]  # per layer: MLP pre-activation [T,F]
    var act: List[Float32]  # per layer: GELU output [T,F]
    var hf: List[Float32]
    var xhatf: List[Float32]
    var rstdf: List[Float32]
    var logits: List[Float32]  # [T, 4V]
    # scratch
    var tmp: List[Float32]
    var qh: List[Float32]
    var kh: List[Float32]
    var vh: List[Float32]
    var oh: List[Float32]
    var dx: List[Float32]
    var dh: List[Float32]
    var dqkv: List[Float32]
    var datt: List[Float32]
    var du: List[Float32]
    var dact: List[Float32]
    var dp: List[Float32]
    var dqh: List[Float32]
    var dkh: List[Float32]
    var dvh: List[Float32]
    var doh: List[Float32]
    var dlogits: List[Float32]
    var toks: List[Int]  # the canvas token of each row of the last forward
    var n: Int  # rows of the last forward

    def __init__(out self, cfg: Config):
        var D = cfg.D
        var L = cfg.L
        var F = cfg.F
        var hd = cfg.hd()

        def z(n: Int) -> List[Float32]:
            return List[Float32](length=n, fill=0.0)

        self.x = z(T * D)
        self.h1 = z(L * T * D)
        self.xhat1 = z(L * T * D)
        self.rstd1 = z(L * T)
        self.qkv = z(L * T * 3 * D)
        self.probs = z(L * cfg.H * T * T)
        self.att = z(L * T * D)
        self.h2 = z(L * T * D)
        self.xhat2 = z(L * T * D)
        self.rstd2 = z(L * T)
        self.u = z(L * T * F)
        self.act = z(L * T * F)
        self.hf = z(T * D)
        self.xhatf = z(T * D)
        self.rstdf = z(T)
        self.logits = z(T * PP * cfg.V)
        self.tmp = z(T * D)
        self.qh = z(T * hd)
        self.kh = z(T * hd)
        self.vh = z(T * hd)
        self.oh = z(T * hd)
        self.dx = z(T * D)
        self.dh = z(T * D)
        self.dqkv = z(T * 3 * D)
        self.datt = z(T * D)
        self.du = z(T * F)
        self.dact = z(T * F)
        self.dp = z(T * T)
        self.dqh = z(T * hd)
        self.dkh = z(T * hd)
        self.dvh = z(T * hd)
        self.doh = z(T * hd)
        self.dlogits = z(T * PP * cfg.V)
        self.toks = full_tokens()
        self.n = T


def gather_cols(dst: FP, src: FP, rows: Int, src_cols: Int, col0: Int, n: Int):
    for r in range(rows):
        copy(dst.unsafe_offset(r * n), src.unsafe_offset(r * src_cols + col0), n)


def scatter_cols(dst: FP, src: FP, rows: Int, dst_cols: Int, col0: Int, n: Int):
    for r in range(rows):
        copy(dst.unsafe_offset(r * dst_cols + col0), src.unsafe_offset(r * n), n)


# ======================================================================== forward


def norm_fwd(y: FP, xhat: FP, rstd: FP, x: FP, g: FP, b: FP, T: Int, D: Int, a: FP):
    """Layer norm, or DyT (a: its scalar) in the variants that set DYT; xhat then holds x."""
    comptime if DYT:
        dyt_fwd(y, xhat, x, g, b, a[unsafe_offset=0], T, D)
    else:
        ln_fwd(y, xhat, rstd, x, g, b, T, D)


def norm_bwd(dx: FP, dg: FP, db: FP, dy: FP, xhat: FP, rstd: FP, g: FP, T: Int, D: Int, da: FP, a: FP):
    comptime if DYT:
        dyt_bwd(dx, dg, db, da, dy, xhat, g, a[unsafe_offset=0], T, D)
    else:
        ln_bwd(dx, dg, db, dy, xhat, rstd, g, T, D)


def forward(mut m: Model, mut wk: Work, ids: List[Int], cls: Int):
    """ids: NPIX palette indices (masked pixels hold cfg.V). Fills wk.logits [T, 4V]."""
    forward_tokens(m, wk, ids, cls, full_tokens())


def forward_tokens(mut m: Model, mut wk: Work, ids: List[Int], cls: Int, toks: List[Int]):
    """forward on only the tokens toks (ascending canvas token indices, see token_order): row r
    of every activation and of wk.logits [len(toks), 4V] is token toks[r]. Attention runs over
    these rows only (the band, if any, by row index, as in the engine)."""
    var n = len(toks)
    wk.toks = toks.copy()
    wk.n = n
    var D = m.cfg.D
    var F = m.cfg.F
    var V = m.cfg.V
    var H = m.cfg.H
    var hd = m.cfg.hd()
    var scale = Float32(1.0 / sqrt(Float64(hd)))
    var W = ptr(m.w)
    var x = ptr(wk.x)

    # embeddings: one table per slot (target and context sub-pixels) + position + class
    for r in range(n):
        var t = toks[r]
        var xt = x.unsafe_offset(r * D)
        copy(xt, W.unsafe_offset(m.o_pos + t * D), D)
        axpy(xt, W.unsafe_offset(m.o_cls + cls * D), 1.0, D)
        for s in range(SLOTS):
            var id = ids[slot_pix(t, s)]
            axpy(xt, W.unsafe_offset(m.o_tok + (s * (V + 1) + id) * D), 1.0, D)

    for l in range(m.cfg.L):
        var h1 = ptr(wk.h1).unsafe_offset(l * T * D)
        var qkv = ptr(wk.qkv).unsafe_offset(l * T * 3 * D)
        var att = ptr(wk.att).unsafe_offset(l * T * D)
        var h2 = ptr(wk.h2).unsafe_offset(l * T * D)
        var u = ptr(wk.u).unsafe_offset(l * T * F)
        var act = ptr(wk.act).unsafe_offset(l * T * F)
        var tmp = ptr(wk.tmp)

        # --- attention block
        norm_fwd(h1, ptr(wk.xhat1).unsafe_offset(l * T * D), ptr(wk.rstd1).unsafe_offset(l * T),
               x, W.unsafe_offset(m.lo(l, m.r_ln1g)), W.unsafe_offset(m.lo(l, m.r_ln1b)), n, D, W.unsafe_offset(m.o_alpha + 2 * l))
        linear_fwd(qkv, h1, W.unsafe_offset(m.lo(l, m.r_wqkv)), W.unsafe_offset(m.lo(l, m.r_bqkv)), n, D, 3 * D)
        for h in range(H):
            gather_cols(ptr(wk.qh), qkv, n, 3 * D, h * hd, hd)
            gather_cols(ptr(wk.kh), qkv, n, 3 * D, D + h * hd, hd)
            gather_cols(ptr(wk.vh), qkv, n, 3 * D, 2 * D + h * hd, hd)
            var p = ptr(wk.probs).unsafe_offset((l * H + h) * T * T)
            mm_nt(p, ptr(wk.qh), ptr(wk.kh), n, n, hd, False)
            for i in range(n * n):
                p[unsafe_offset=i] = p[unsafe_offset=i] * scale if in_band(i // n, i % n) else -3.0e38
            softmax_rows(p, n, n)
            fill(ptr(wk.oh), n * hd, 0.0)
            mm_nn(ptr(wk.oh), p, ptr(wk.vh), n, hd, n)
            scatter_cols(att, ptr(wk.oh), n, D, h * hd, hd)
        linear_fwd(tmp, att, W.unsafe_offset(m.lo(l, m.r_wo)), W.unsafe_offset(m.lo(l, m.r_bo)), n, D, D)
        axpy(x, tmp, 1.0, n * D)

        # --- MLP block
        norm_fwd(h2, ptr(wk.xhat2).unsafe_offset(l * T * D), ptr(wk.rstd2).unsafe_offset(l * T),
               x, W.unsafe_offset(m.lo(l, m.r_ln2g)), W.unsafe_offset(m.lo(l, m.r_ln2b)), n, D, W.unsafe_offset(m.o_alpha + 2 * l + 1))
        linear_fwd(u, h2, W.unsafe_offset(m.lo(l, m.r_w1)), W.unsafe_offset(m.lo(l, m.r_b1)), n, D, F)
        gelu_fwd(act, u, n * F)
        linear_fwd(tmp, act, W.unsafe_offset(m.lo(l, m.r_w2)), W.unsafe_offset(m.lo(l, m.r_b2)), n, F, D)
        axpy(x, tmp, 1.0, n * D)

    norm_fwd(ptr(wk.hf), ptr(wk.xhatf), ptr(wk.rstdf), x,
           W.unsafe_offset(m.o_lnfg), W.unsafe_offset(m.o_lnfb), n, D, W.unsafe_offset(m.o_alpha + 2 * m.cfg.L))
    linear_fwd(ptr(wk.logits), ptr(wk.hf), W.unsafe_offset(m.o_wout), W.unsafe_offset(m.o_bout), n, D, PP * V)


# ======================================================================== loss


def masked_loss(mut wk: Work, V: Int, target: List[Int], mask: List[Bool], grad_scale: Float32) -> Float32:
    """Cross-entropy over the pixels of masked tokens. Returns the mean loss for this canvas
    and writes dlogits = grad_scale * dLoss_sum/dlogits (zero for unmasked tokens).
    mask is per row of the last forward (wk.toks)."""
    var n = wk.n
    var lg = ptr(wk.logits)
    var dl = ptr(wk.dlogits)
    fill(dl, n * PP * V, 0.0)
    var total: Float64 = 0.0
    var count = 0
    for r in range(n):
        if not mask[r]:
            continue
        var t = wk.toks[r]
        for k in range(PP):
            var row = lg.unsafe_offset((r * PP + k) * V)
            var drow = dl.unsafe_offset((r * PP + k) * V)
            var mx = row[unsafe_offset=0]
            for j in range(1, V):
                mx = max(mx, row[unsafe_offset=j])
            var s: Float64 = 0.0
            for j in range(V):
                var e = exp(row[unsafe_offset=j] - mx)
                drow[unsafe_offset=j] = e  # stash exp, normalized below
                s += Float64(e)
            var tgt = target[tgt_pix(t, k)]
            var logp = Float64(row[unsafe_offset=tgt] - mx) - log(s)
            total -= logp
            count += 1
            for j in range(V):
                var p = Float32(Float64(drow[unsafe_offset=j]) / s)
                drow[unsafe_offset=j] = grad_scale * (p - (Float32(1.0) if j == tgt else Float32(0.0)))
    if count == 0:
        return 0.0
    return Float32(total / Float64(count))


# ======================================================================== backward


def backward(mut m: Model, mut wk: Work, ids: List[Int], cls: Int):
    """Backpropagates wk.dlogits through the network, accumulating into m.g."""
    var n = wk.n
    var D = m.cfg.D
    var F = m.cfg.F
    var V = m.cfg.V
    var H = m.cfg.H
    var hd = m.cfg.hd()
    var scale = Float32(1.0 / sqrt(Float64(hd)))
    var W = ptr(m.w)
    var G = ptr(m.g)
    var dx = ptr(wk.dx)
    var dh = ptr(wk.dh)

    # output head + final layer norm
    fill(dh, n * D, 0.0)
    linear_bwd(dh, G.unsafe_offset(m.o_wout), G.unsafe_offset(m.o_bout), ptr(wk.dlogits),
               ptr(wk.hf), W.unsafe_offset(m.o_wout), n, D, PP * V)
    fill(dx, n * D, 0.0)
    norm_bwd(dx, G.unsafe_offset(m.o_lnfg), G.unsafe_offset(m.o_lnfb), dh, ptr(wk.xhatf), ptr(wk.rstdf),
           W.unsafe_offset(m.o_lnfg), n, D, G.unsafe_offset(m.o_alpha + 2 * m.cfg.L), W.unsafe_offset(m.o_alpha + 2 * m.cfg.L))

    for l in range(m.cfg.L - 1, -1, -1):
        var h1 = ptr(wk.h1).unsafe_offset(l * T * D)
        var qkv = ptr(wk.qkv).unsafe_offset(l * T * 3 * D)
        var att = ptr(wk.att).unsafe_offset(l * T * D)
        var h2 = ptr(wk.h2).unsafe_offset(l * T * D)
        var u = ptr(wk.u).unsafe_offset(l * T * F)
        var act = ptr(wk.act).unsafe_offset(l * T * F)

        # --- MLP block: x_out = x_mid + W2 gelu(W1 ln2(x_mid))
        fill(ptr(wk.dact), n * F, 0.0)
        linear_bwd(ptr(wk.dact), G.unsafe_offset(m.lo(l, m.r_w2)), G.unsafe_offset(m.lo(l, m.r_b2)), dx,
                   act, W.unsafe_offset(m.lo(l, m.r_w2)), n, F, D)
        gelu_bwd(ptr(wk.du), ptr(wk.dact), u, n * F)
        fill(dh, n * D, 0.0)
        linear_bwd(dh, G.unsafe_offset(m.lo(l, m.r_w1)), G.unsafe_offset(m.lo(l, m.r_b1)), ptr(wk.du),
                   h2, W.unsafe_offset(m.lo(l, m.r_w1)), n, D, F)
        norm_bwd(dx, G.unsafe_offset(m.lo(l, m.r_ln2g)), G.unsafe_offset(m.lo(l, m.r_ln2b)), dh,
               ptr(wk.xhat2).unsafe_offset(l * T * D), ptr(wk.rstd2).unsafe_offset(l * T),
               W.unsafe_offset(m.lo(l, m.r_ln2g)), n, D, G.unsafe_offset(m.o_alpha + 2 * l + 1), W.unsafe_offset(m.o_alpha + 2 * l + 1))

        # --- attention block: x_mid = x_in + Wo attn(ln1(x_in))
        fill(ptr(wk.datt), n * D, 0.0)
        linear_bwd(ptr(wk.datt), G.unsafe_offset(m.lo(l, m.r_wo)), G.unsafe_offset(m.lo(l, m.r_bo)), dx,
                   att, W.unsafe_offset(m.lo(l, m.r_wo)), n, D, D)
        fill(ptr(wk.dqkv), n * 3 * D, 0.0)
        for h in range(H):
            gather_cols(ptr(wk.qh), qkv, n, 3 * D, h * hd, hd)
            gather_cols(ptr(wk.kh), qkv, n, 3 * D, D + h * hd, hd)
            gather_cols(ptr(wk.vh), qkv, n, 3 * D, 2 * D + h * hd, hd)
            gather_cols(ptr(wk.doh), ptr(wk.datt), n, D, h * hd, hd)
            var p = ptr(wk.probs).unsafe_offset((l * H + h) * T * T)
            var dp = ptr(wk.dp)
            mm_nt(dp, ptr(wk.doh), ptr(wk.vh), n, n, hd, False)  # dP = dO V^T
            fill(ptr(wk.dvh), n * hd, 0.0)
            mm_tn(ptr(wk.dvh), p, ptr(wk.doh), n, hd, n)  # dV = P^T dO
            for i in range(n):  # softmax backward, then the 1/sqrt(hd) scale
                var s: Float32 = 0.0
                for j in range(n):
                    s += p[unsafe_offset=i * n + j] * dp[unsafe_offset=i * n + j]
                for j in range(n):
                    var o = i * n + j
                    dp[unsafe_offset=o] = p[unsafe_offset=o] * (dp[unsafe_offset=o] - s) * scale
            fill(ptr(wk.dqh), n * hd, 0.0)
            mm_nn(ptr(wk.dqh), dp, ptr(wk.kh), n, hd, n)  # dQ = dS K
            fill(ptr(wk.dkh), n * hd, 0.0)
            mm_tn(ptr(wk.dkh), dp, ptr(wk.qh), n, hd, n)  # dK = dS^T Q
            scatter_cols(ptr(wk.dqkv), ptr(wk.dqh), n, 3 * D, h * hd, hd)
            scatter_cols(ptr(wk.dqkv), ptr(wk.dkh), n, 3 * D, D + h * hd, hd)
            scatter_cols(ptr(wk.dqkv), ptr(wk.dvh), n, 3 * D, 2 * D + h * hd, hd)
        fill(dh, n * D, 0.0)
        linear_bwd(dh, G.unsafe_offset(m.lo(l, m.r_wqkv)), G.unsafe_offset(m.lo(l, m.r_bqkv)), ptr(wk.dqkv),
                   h1, W.unsafe_offset(m.lo(l, m.r_wqkv)), n, D, 3 * D)
        norm_bwd(dx, G.unsafe_offset(m.lo(l, m.r_ln1g)), G.unsafe_offset(m.lo(l, m.r_ln1b)), dh,
               ptr(wk.xhat1).unsafe_offset(l * T * D), ptr(wk.rstd1).unsafe_offset(l * T),
               W.unsafe_offset(m.lo(l, m.r_ln1g)), n, D, G.unsafe_offset(m.o_alpha + 2 * l), W.unsafe_offset(m.o_alpha + 2 * l))

    # embeddings
    for r in range(n):
        var t = wk.toks[r]
        var dxt = dx.unsafe_offset(r * D)
        axpy(G.unsafe_offset(m.o_pos + t * D), dxt, 1.0, D)
        axpy(G.unsafe_offset(m.o_cls + cls * D), dxt, 1.0, D)
        for s in range(SLOTS):
            var id = ids[slot_pix(t, s)]
            axpy(G.unsafe_offset(m.o_tok + (s * (V + 1) + id) * D), dxt, 1.0, D)
