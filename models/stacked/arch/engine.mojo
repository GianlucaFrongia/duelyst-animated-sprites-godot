# Training engine for the stacked architecture (both frames as 2,048 canvas tokens):
# embeddings, loss, softmax attention, buffer layout and the forward/backward schedule.
# Generic ops (matmul, layer norm, AdamW, ...) live in src/gpu.mojo.
# Rows are batch-major: row r = b * T + t.

from std.math import sqrt, exp, log
from std.sys import has_accelerator
from std.memory import stack_allocation
from max.gpu import global_idx, thread_idx, block_idx, barrier
from max.gpu.memory import AddressSpace
from src.gpu import FP, IP, BLOCK, NCHUNK, SPLITK, blocks, Device, metal_kernels_available
from arch.model import Config, Model, T, PP, NPIX, FRAME_T, pix_index, in_band
from arch.variant import BAND


def b_softmax(p: FP, n: Int, i: Int):
    var row = p.unsafe_offset(i * n)
    var mx = row[unsafe_offset=0]
    for j in range(1, n):
        mx = max(mx, row[unsafe_offset=j])
    var s: Float32 = 0.0
    for j in range(n):
        var e = exp(row[unsafe_offset=j] - mx)
        row[unsafe_offset=j] = e
        s += e
    var inv = 1.0 / s
    for j in range(n):
        row[unsafe_offset=j] *= inv


def b_softmax_bwd(dp: FP, p: FP, n: Int, scale: Float32, i: Int):
    """dp <- P * (dP - sum(P * dP)) * scale, row i."""
    var s: Float32 = 0.0
    for j in range(n):
        s += p[unsafe_offset=i * n + j] * dp[unsafe_offset=i * n + j]
    for j in range(n):
        var o = i * n + j
        dp[unsafe_offset=o] = p[unsafe_offset=o] * (dp[unsafe_offset=o] - s) * scale


def b_embed(x: FP, w: FP, ids: IP, cls: IP, o_tok: Int, o_pos: Int, o_cls: Int,
            D: Int, V: Int, i: Int):
    var r = i // D
    var d = i - r * D
    var b = r // T
    var t = r - b * T
    var v = w[unsafe_offset=o_pos + t * D + d] + w[unsafe_offset=o_cls + Int(cls[unsafe_offset=b]) * D + d]
    for k in range(PP):
        var id = Int(ids[unsafe_offset=b * NPIX + pix_index(t, k)])
        v += w[unsafe_offset=o_tok + (k * (V + 1) + id) * D + d]
    x[unsafe_offset=i] = v


def b_embed_bwd_pos(g: FP, dx: FP, o_pos: Int, B: Int, D: Int, i: Int):
    var s: Float32 = 0.0
    for b in range(B):
        s += dx[unsafe_offset=b * T * D + i]
    g[unsafe_offset=o_pos + i] += s


def b_embed_bwd_cls(g: FP, dx: FP, cls: IP, o_cls: Int, B: Int, D: Int, i: Int):
    var c = i // D
    var d = i - c * D
    var s: Float32 = 0.0
    for b in range(B):
        if Int(cls[unsafe_offset=b]) == c:
            for t in range(T):
                s += dx[unsafe_offset=(b * T + t) * D + d]
    g[unsafe_offset=o_cls + i] += s


def b_embed_bwd_tok(g: FP, dx: FP, ids: IP, o_tok: Int, B: Int, D: Int, V: Int, i: Int):
    var d = i % D
    var kv = i // D
    var k = kv // (V + 1)
    var v = kv - k * (V + 1)
    var s: Float32 = 0.0
    for b in range(B):
        for t in range(T):
            if Int(ids[unsafe_offset=b * NPIX + pix_index(t, k)]) == v:
                s += dx[unsafe_offset=(b * T + t) * D + d]
    g[unsafe_offset=o_tok + i] += s


def k_onehot(oh: FP, ids: IP, r0: Int32, n: Int32, nvp: Int32, V: Int32):
    """oh[rr, s * (V + 1) + v] = 1 where row r0 + rr has id v in slot s, one thread per (row, slot),
    on a zeroed oh of n rows and nvp columns (the last nvp - PP * (V + 1) are padding): the
    token-embedding gradient is oh^T @ dx."""
    var i = Int(global_idx.x)
    if i >= Int(n) * PP:
        return
    var rr = i // PP
    var s = i - rr * PP
    var r = Int(r0) + rr
    var b = r // T
    var v = Int(ids[unsafe_offset=b * NPIX + pix_index(r - b * T, s)])
    oh[unsafe_offset=rr * Int(nvp) + s * (Int(V) + 1) + v] = 1.0


def b_loss(dl: FP, lossbuf: FP, logits: FP, tgt: IP, mask: IP, gscale: FP, V: Int, i: Int):
    """One (token, sub-pixel) per thread. lossbuf sums to the batch-mean loss."""
    var r = i // PP
    var k = i - r * PP
    var b = r // T
    var t = r - b * T
    var row = logits.unsafe_offset(i * V)
    var drow = dl.unsafe_offset(i * V)
    if mask[unsafe_offset=r] == 0:
        for j in range(V):
            drow[unsafe_offset=j] = 0.0
        lossbuf[unsafe_offset=i] = 0.0
        return
    var mx = row[unsafe_offset=0]
    for j in range(1, V):
        mx = max(mx, row[unsafe_offset=j])
    var s: Float32 = 0.0
    for j in range(V):
        var e = exp(row[unsafe_offset=j] - mx)
        drow[unsafe_offset=j] = e
        s += e
    var tg = Int(tgt[unsafe_offset=b * NPIX + pix_index(t, k)])
    var gs = gscale[unsafe_offset=b]
    lossbuf[unsafe_offset=i] = -(row[unsafe_offset=tg] - mx - log(s)) * gs
    for j in range(V):
        var p = drow[unsafe_offset=j] / s
        drow[unsafe_offset=j] = gs * (p - (Float32(1.0) if j == tg else Float32(0.0)))


def b_band_mask(p: FP, i: Int):
    """Entry i of the T x T scores: very negative outside the band (the fused kernels skip
    those tiles)."""
    if not in_band((i // T) % T, i % T):
        p[unsafe_offset=i] = -3.0e38


def k_band_mask(p: FP, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_band_mask(p, i)


def k_softmax(p: FP, cols: Int32, rows: Int32):
    var i = Int(global_idx.x)
    if i < Int(rows):
        b_softmax(p, Int(cols), i)


def block_reduce(red: Pointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
                 v: Float32, tid: Int, is_max: Bool) -> Float32:
    """Max or sum of v over the 256 threads of a block (every thread must call it)."""
    red[unsafe_offset=tid] = v
    barrier()
    var step = 128
    while step > 0:
        if tid < step:
            var o = red[unsafe_offset=tid + step]
            var mine = red[unsafe_offset=tid]
            red[unsafe_offset=tid] = max(mine, o) if is_max else mine + o
        barrier()
        step //= 2
    var r = red[unsafe_offset=0]
    barrier()
    return r


def k_softmax_rows(p: FP, cols: Int32):
    """Row softmax with one 256-thread block per row, so neighbouring threads read
    neighbouring columns (b_softmax's one-thread-per-row reads are scattered)."""
    var red = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var n = Int(cols)
    var row = p.unsafe_offset(Int(block_idx.x) * n)
    var mx: Float32 = -3.0e38
    for j in range(tid, n, 256):
        mx = max(mx, row[unsafe_offset=j])
    mx = block_reduce(red, mx, tid, True)
    var s: Float32 = 0.0
    for j in range(tid, n, 256):
        var e = exp(row[unsafe_offset=j] - mx)
        row[unsafe_offset=j] = e
        s += e
    var inv = 1.0 / block_reduce(red, s, tid, False)
    for j in range(tid, n, 256):
        row[unsafe_offset=j] *= inv


def k_softmax_bwd_rows(dp: FP, p: FP, cols: Int32, scale: Float32):
    var red = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var n = Int(cols)
    var base = Int(block_idx.x) * n
    var s: Float32 = 0.0
    for j in range(tid, n, 256):
        s += p[unsafe_offset=base + j] * dp[unsafe_offset=base + j]
    s = block_reduce(red, s, tid, False)
    for j in range(tid, n, 256):
        var o = base + j
        dp[unsafe_offset=o] = p[unsafe_offset=o] * (dp[unsafe_offset=o] - s) * scale


def k_softmax_bwd(dp: FP, p: FP, cols: Int32, scale: Float32, rows: Int32):
    var i = Int(global_idx.x)
    if i < Int(rows):
        b_softmax_bwd(dp, p, Int(cols), scale, i)


def k_attn_dvec(dvec: FP, att: FP, datt: FP, D: Int32, H: Int32, hd: Int32, n: Int32):
    """dvec[(b*H + h)*T + t] = sum_d dO * O over head h of row t: the fused backward's
    softmax correction."""
    var i = Int(global_idx.x)
    if i < Int(n):
        var bh = i // T
        var b = bh // Int(H)
        var off = (b * T + i - bh * T) * Int(D) + (bh - b * Int(H)) * Int(hd)
        var s: Float32 = 0.0
        for d in range(Int(hd)):
            s += att[unsafe_offset=off + d] * datt[unsafe_offset=off + d]
        dvec[unsafe_offset=i] = s


def k_embed(x: FP, w: FP, ids: IP, cls: IP, o_tok: Int32, o_pos: Int32, o_cls: Int32,
            D: Int32, V: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_embed(x, w, ids, cls, Int(o_tok), Int(o_pos), Int(o_cls), Int(D), Int(V), i)


def k_embed_bwd_pos(g: FP, dx: FP, o_pos: Int32, B: Int32, D: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_embed_bwd_pos(g, dx, Int(o_pos), Int(B), Int(D), i)


def k_embed_bwd_cls(g: FP, dx: FP, cls: IP, o_cls: Int32, B: Int32, D: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_embed_bwd_cls(g, dx, cls, Int(o_cls), Int(B), Int(D), i)


def k_embed_bwd_tok(g: FP, dx: FP, ids: IP, o_tok: Int32, B: Int32, D: Int32, V: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_embed_bwd_tok(g, dx, ids, Int(o_tok), Int(B), Int(D), Int(V), i)


def k_loss(dl: FP, lossbuf: FP, logits: FP, tgt: IP, mask: IP, gscale: FP, V: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_loss(dl, lossbuf, logits, tgt, mask, gscale, Int(V), i)


struct Engine(Movable):
    var cfg: Config
    var B: Int
    var R: Int  # B * T rows
    var dev: Device
    var fused: Bool  # Metal fused attention: no T x T matrices, o_probs holds lse, o_dp dvec
    # offsets into the float arena
    var o_x: Int
    var o_h1: Int
    var o_xhat1: Int
    var o_rstd1: Int
    var o_qkv: Int
    var o_probs: Int
    var o_att: Int
    var o_h2: Int
    var o_xhat2: Int
    var o_rstd2: Int
    var o_u: Int
    var o_act: Int
    var o_hf: Int
    var o_xhatf: Int
    var o_rstdf: Int
    var o_logits: Int
    var o_dx: Int
    var o_dh: Int
    var o_dqkv: Int
    var o_datt: Int
    var o_du: Int
    var o_dact: Int
    var o_dp: Int
    var o_dlogits: Int
    var o_lossbuf: Int
    var o_gscale: Int
    # offsets into the int arena
    var i_ids: Int
    var i_tgt: Int
    var i_mask: Int
    var i_cls: Int

    def __init__(out self, cfg: Config, B: Int, use_gpu: Bool) raises:
        self.cfg = cfg.copy()
        self.B = B
        self.R = B * T
        self.fused = cfg.hd() == 32 and cfg.D % 4 == 0 and T % 64 == 0 and metal_kernels_available(use_gpu)
        var att_rows = 1 if self.fused else T  # per (sprite, head, query): lse / dvec, or a row of P
        var R = self.R
        var D = cfg.D
        var L = cfg.L
        var F = cfg.F
        var V = cfg.V
        var o = 0

        def take(mut o: Int, size: Int) -> Int:
            var start = o
            o += size
            return start

        self.o_x = take(o, R * D)
        self.o_h1 = take(o, L * R * D)
        self.o_xhat1 = take(o, L * R * D)
        self.o_rstd1 = take(o, L * R)
        self.o_qkv = take(o, L * R * 3 * D)
        self.o_probs = take(o, L * B * cfg.H * T * att_rows)
        self.o_att = take(o, L * R * D)
        self.o_h2 = take(o, L * R * D)
        self.o_xhat2 = take(o, L * R * D)
        self.o_rstd2 = take(o, L * R)
        self.o_u = take(o, L * R * F)
        self.o_act = take(o, L * R * F)
        self.o_hf = take(o, R * D)
        self.o_xhatf = take(o, R * D)
        self.o_rstdf = take(o, R)
        self.o_logits = take(o, R * PP * V)
        self.o_dx = take(o, R * D)
        self.o_dh = take(o, R * D)
        self.o_dqkv = take(o, R * 3 * D)
        self.o_datt = take(o, R * D)
        self.o_du = take(o, R * F)
        self.o_dact = take(o, R * F)
        self.o_dp = take(o, B * cfg.H * T * att_rows)
        self.o_dlogits = take(o, R * PP * V)
        self.o_lossbuf = take(o, R * PP)
        self.o_gscale = take(o, B)
        # Scratch for split-K partials of the weight gradients (largest is W_qkv, W_1/W_2 or W_out).
        var splitk_cap = SPLITK * max(3 * D * D, F * D, PP * V * D, 4 * F)
        var oi = 0
        self.i_ids = take(oi, B * NPIX)
        self.i_tgt = take(oi, B * NPIX)
        self.i_mask = take(oi, R)
        self.i_cls = take(oi, B)

        self.dev = Device(R, D, Model(cfg).n, splitk_cap, o, oi, use_gpu)

    def band_mask(self, p: FP, n: Int) raises:
        comptime if has_accelerator():
            if self.dev.gpu:
                self.dev.ctx.value().enqueue_function[k_band_mask](p, Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
                return
        for i in range(n):
            b_band_mask(p, i)

    def softmax(self, p: FP, rows: Int, cols: Int) raises:
        comptime if has_accelerator():
            if self.dev.gpu:
                self.dev.ctx.value().enqueue_function[k_softmax_rows](p, Int32(cols), grid_dim=rows, block_dim=256)
                return
        for i in range(rows):
            b_softmax(p, cols, i)

    def softmax_bwd(self, dp: FP, p: FP, rows: Int, cols: Int, scale: Float32) raises:
        comptime if has_accelerator():
            if self.dev.gpu:
                self.dev.ctx.value().enqueue_function[k_softmax_bwd_rows](dp, p, Int32(cols), scale,
                                                                      grid_dim=rows, block_dim=256)
                return
        for i in range(rows):
            b_softmax_bwd(dp, p, cols, scale, i)

    def attn_fwd_fused(self, att: FP, lse: FP, qkv: FP, scale: Float32) raises:
        """att = softmax(scale Q K^T) V with the Metal flash kernel; lse gets each row's log-sum-exp."""
        comptime if has_accelerator():
            self.dev.ctx.value().enqueue_function(
                self.dev.a_fwd.value(), att, lse, qkv, Int32(self.cfg.D), Int32(self.cfg.H), scale, Int32(T),
                Int32(FRAME_T), Int32(BAND), grid_dim=self.B * self.cfg.H * (T // 64), block_dim=256)

    def attn_bwd_fused(self, dqkv: FP, qkv: FP, att: FP, datt: FP, lse: FP, dvec: FP, scale: Float32) raises:
        """Overwrites dQ, dK, dV in dqkv, recomputing the probabilities from lse."""
        comptime if has_accelerator():
            var ctx = self.dev.ctx.value()
            var D = self.cfg.D
            var H = self.cfg.H
            var rows = self.B * H * T
            ctx.enqueue_function[k_attn_dvec](dvec, att, datt, Int32(D), Int32(H), Int32(self.cfg.hd()), Int32(rows),
                                              grid_dim=blocks(rows), block_dim=BLOCK)
            var grid = self.B * H * (T // 64)
            ctx.enqueue_function(self.dev.a_bkv.value(), dqkv, qkv, datt, lse, dvec, Int32(D), Int32(H), scale, Int32(T),
                                 Int32(FRAME_T), Int32(BAND), grid_dim=grid, block_dim=256)
            ctx.enqueue_function(self.dev.a_bq.value(), dqkv, qkv, datt, lse, dvec, Int32(D), Int32(H), scale, Int32(T),
                                 Int32(FRAME_T), Int32(BAND), grid_dim=grid, block_dim=256)

    def upload_batch(self, ids: List[Int32], tgt: List[Int32], mask: List[Int32],
                     cls: List[Int32], gscale: List[Float32]) raises:
        self.dev.upload_i(self.i_ids, ids)
        self.dev.upload_i(self.i_tgt, tgt)
        self.dev.upload_i(self.i_mask, mask)
        self.dev.upload_i(self.i_cls, cls)
        self.dev.upload_f(self.o_gscale, gscale)

    def forward(self, m: Model) raises:
        var D = self.cfg.D
        var F = self.cfg.F
        var V = self.cfg.V
        var H = self.cfg.H
        var L = self.cfg.L
        var R = self.R
        var hd = self.cfg.hd()
        var scale = Float32(1.0 / sqrt(Float64(hd)))
        var W = self.dev.F(self.dev.o_w)
        var x = self.dev.F(self.o_x)

        var n = R * D
        comptime if has_accelerator():
            if self.dev.gpu:
                self.dev.ctx.value().enqueue_function[k_embed](
                    x, W, self.dev.I(self.i_ids), self.dev.I(self.i_cls), Int32(m.o_tok), Int32(m.o_pos),
                    Int32(m.o_cls), Int32(D), Int32(V), Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
        if not self.dev.gpu:
            for i in range(n):
                b_embed(x, W, self.dev.I(self.i_ids), self.dev.I(self.i_cls), m.o_tok, m.o_pos, m.o_cls, D, V, i)

        for l in range(L):
            var h1 = self.dev.F(self.o_h1 + l * R * D)
            var qkv = self.dev.F(self.o_qkv + l * R * 3 * D)
            var att_rows = 1 if self.fused else T
            var probs = self.dev.F(self.o_probs + l * self.B * H * T * att_rows)
            var att = self.dev.F(self.o_att + l * R * D)
            var h2 = self.dev.F(self.o_h2 + l * R * D)
            var u = self.dev.F(self.o_u + l * R * F)
            var act = self.dev.F(self.o_act + l * R * F)

            self.dev.ln_fwd(h1, self.dev.F(self.o_xhat1 + l * R * D), self.dev.F(self.o_rstd1 + l * R), x,
                        W.unsafe_offset(m.lo(l, m.r_ln1g)), W.unsafe_offset(m.lo(l, m.r_ln1b)))
            self.dev.linear_fwd(qkv, h1, W.unsafe_offset(m.lo(l, m.r_wqkv)), W.unsafe_offset(m.lo(l, m.r_bqkv)), D, 3 * D)
            if self.fused:
                self.attn_fwd_fused(att, probs, qkv, scale)
            else:
                # S = scale * Q K^T for every (sprite, head)
                self.dev.gemm(probs, qkv, qkv.unsafe_offset(D), T, T, hd, self.B * H, H,
                          3 * D, 1, 1, 3 * D, T, 1,
                          T * 3 * D, hd, T * 3 * D, hd, H * T * T, T * T, scale, False)
                if BAND < FRAME_T // 64:
                    self.band_mask(probs, self.B * H * T * T)
                self.softmax(probs, self.B * H * T, T)
                # O = P V, written straight into the head's columns of att
                self.dev.gemm(att, probs, qkv.unsafe_offset(2 * D), T, hd, T, self.B * H, H,
                          T, 1, 3 * D, 1, D, 1,
                          H * T * T, T * T, T * 3 * D, hd, T * D, hd, 1.0, False)
            self.dev.linear_fwd(x, att, W.unsafe_offset(m.lo(l, m.r_wo)), W.unsafe_offset(m.lo(l, m.r_bo)), D, D, True)

            self.dev.ln_fwd(h2, self.dev.F(self.o_xhat2 + l * R * D), self.dev.F(self.o_rstd2 + l * R), x,
                        W.unsafe_offset(m.lo(l, m.r_ln2g)), W.unsafe_offset(m.lo(l, m.r_ln2b)))
            self.dev.linear_fwd(u, h2, W.unsafe_offset(m.lo(l, m.r_w1)), W.unsafe_offset(m.lo(l, m.r_b1)), D, F)
            self.dev.gelu_fwd(act, u, R * F)
            self.dev.linear_fwd(x, act, W.unsafe_offset(m.lo(l, m.r_w2)), W.unsafe_offset(m.lo(l, m.r_b2)), F, D, True)

        self.dev.ln_fwd(self.dev.F(self.o_hf), self.dev.F(self.o_xhatf), self.dev.F(self.o_rstdf), x,
                    W.unsafe_offset(m.o_lnfg), W.unsafe_offset(m.o_lnfb))
        self.dev.linear_fwd(self.dev.F(self.o_logits), self.dev.F(self.o_hf), W.unsafe_offset(m.o_wout),
                        W.unsafe_offset(m.o_bout), D, PP * V)

    def loss(self) raises -> Float32:
        """Masked cross-entropy; fills dlogits. Returns the batch-mean loss."""
        var V = self.cfg.V
        var n = self.R * PP
        var dl = self.dev.F(self.o_dlogits)
        var lb = self.dev.F(self.o_lossbuf)
        var lg = self.dev.F(self.o_logits)
        var done = False
        comptime if has_accelerator():
            if self.dev.gpu:
                self.dev.ctx.value().enqueue_function[k_loss](
                    dl, lb, lg, self.dev.I(self.i_tgt), self.dev.I(self.i_mask), self.dev.F(self.o_gscale),
                    Int32(V), Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
                done = True
        if not done:
            for i in range(n):
                b_loss(dl, lb, lg, self.dev.I(self.i_tgt), self.dev.I(self.i_mask), self.dev.F(self.o_gscale), V, i)
        return self.dev.chunk_sum(lb, n, False)

    def backward(self, m: Model) raises:
        var D = self.cfg.D
        var F = self.cfg.F
        var V = self.cfg.V
        var H = self.cfg.H
        var L = self.cfg.L
        var R = self.R
        var B = self.B
        var hd = self.cfg.hd()
        var scale = Float32(1.0 / sqrt(Float64(hd)))
        var W = self.dev.F(self.dev.o_w)
        var G = self.dev.F(self.dev.o_g)
        var dx = self.dev.F(self.o_dx)
        var dh = self.dev.F(self.o_dh)
        var dp = self.dev.F(self.o_dp)
        var dqkv = self.dev.F(self.o_dqkv)
        var datt = self.dev.F(self.o_datt)

        self.dev.linear_bwd(dh, G.unsafe_offset(m.o_wout), G.unsafe_offset(m.o_bout), self.dev.F(self.o_dlogits),
                        self.dev.F(self.o_hf), W.unsafe_offset(m.o_wout), D, PP * V, False)
        self.dev.fill(dx, R * D, 0.0)
        self.dev.ln_bwd(dx, G.unsafe_offset(m.o_lnfg), G.unsafe_offset(m.o_lnfb), dh, self.dev.F(self.o_xhatf),
                    self.dev.F(self.o_rstdf), W.unsafe_offset(m.o_lnfg))

        for l in range(L - 1, -1, -1):
            var h1 = self.dev.F(self.o_h1 + l * R * D)
            var qkv = self.dev.F(self.o_qkv + l * R * 3 * D)
            var probs = self.dev.F(self.o_probs + l * B * H * T * (1 if self.fused else T))
            var att = self.dev.F(self.o_att + l * R * D)
            var h2 = self.dev.F(self.o_h2 + l * R * D)
            var u = self.dev.F(self.o_u + l * R * F)
            var act = self.dev.F(self.o_act + l * R * F)

            # MLP block
            self.dev.linear_bwd(self.dev.F(self.o_dact), G.unsafe_offset(m.lo(l, m.r_w2)), G.unsafe_offset(m.lo(l, m.r_b2)),
                            dx, act, W.unsafe_offset(m.lo(l, m.r_w2)), F, D, False)
            self.dev.gelu_bwd(self.dev.F(self.o_du), self.dev.F(self.o_dact), u, R * F)
            self.dev.linear_bwd(dh, G.unsafe_offset(m.lo(l, m.r_w1)), G.unsafe_offset(m.lo(l, m.r_b1)),
                            self.dev.F(self.o_du), h2, W.unsafe_offset(m.lo(l, m.r_w1)), D, F, False)
            self.dev.ln_bwd(dx, G.unsafe_offset(m.lo(l, m.r_ln2g)), G.unsafe_offset(m.lo(l, m.r_ln2b)), dh,
                        self.dev.F(self.o_xhat2 + l * R * D), self.dev.F(self.o_rstd2 + l * R),
                        W.unsafe_offset(m.lo(l, m.r_ln2g)))

            # attention block
            self.dev.linear_bwd(datt, G.unsafe_offset(m.lo(l, m.r_wo)), G.unsafe_offset(m.lo(l, m.r_bo)),
                            dx, att, W.unsafe_offset(m.lo(l, m.r_wo)), D, D, False)
            if self.fused:
                self.attn_bwd_fused(dqkv, qkv, att, datt, probs, dp, scale)
            else:
                # dP = dO V^T
                self.dev.gemm(dp, datt, qkv.unsafe_offset(2 * D), T, T, hd, B * H, H,
                          D, 1, 1, 3 * D, T, 1,
                          T * D, hd, T * 3 * D, hd, H * T * T, T * T, 1.0, False)
                # dV = P^T dO
                self.dev.gemm(dqkv.unsafe_offset(2 * D), probs, datt, T, hd, T, B * H, H,
                          1, T, D, 1, 3 * D, 1,
                          H * T * T, T * T, T * D, hd, T * 3 * D, hd, 1.0, False)
                self.softmax_bwd(dp, probs, B * H * T, T, scale)
                # dQ = dS K
                self.dev.gemm(dqkv, dp, qkv.unsafe_offset(D), T, hd, T, B * H, H,
                          T, 1, 3 * D, 1, 3 * D, 1,
                          H * T * T, T * T, T * 3 * D, hd, T * 3 * D, hd, 1.0, False)
                # dK = dS^T Q
                self.dev.gemm(dqkv.unsafe_offset(D), dp, qkv, T, hd, T, B * H, H,
                          1, T, 3 * D, 1, 3 * D, 1,
                          H * T * T, T * T, T * 3 * D, hd, T * 3 * D, hd, 1.0, False)
            self.dev.linear_bwd(dh, G.unsafe_offset(m.lo(l, m.r_wqkv)), G.unsafe_offset(m.lo(l, m.r_bqkv)),
                            dqkv, h1, W.unsafe_offset(m.lo(l, m.r_wqkv)), D, 3 * D, False)
            self.dev.ln_bwd(dx, G.unsafe_offset(m.lo(l, m.r_ln1g)), G.unsafe_offset(m.lo(l, m.r_ln1b)), dh,
                        self.dev.F(self.o_xhat1 + l * R * D), self.dev.F(self.o_rstd1 + l * R),
                        W.unsafe_offset(m.lo(l, m.r_ln1g)))

        # embeddings
        var ids = self.dev.I(self.i_ids)
        var cls = self.dev.I(self.i_cls)
        var npos = T * D
        var ncls = (self.cfg.C + 1) * D
        var ntok = PP * (V + 1) * D
        comptime if has_accelerator():
            if self.dev.gpu:
                var ctx = self.dev.ctx.value()
                ctx.enqueue_function[k_embed_bwd_pos](G, dx, Int32(m.o_pos), Int32(B), Int32(D), Int32(npos),
                                                      grid_dim=blocks(npos), block_dim=BLOCK)
                ctx.enqueue_function[k_embed_bwd_cls](G, dx, cls, Int32(m.o_cls), Int32(B), Int32(D), Int32(ncls),
                                                      grid_dim=blocks(ncls), block_dim=BLOCK)
                # oh^T @ dx over row chunks that fit the split-K scratch (one-hot chunk, then the
                # padded product); gemm_tiled, since gemm's own split-K would reuse that scratch.
                var nvp = (PP * (V + 1) + 63) // 64 * 64
                var o_oh = (self.dev.o_splitk + 63) // 64 * 64
                var rc = min(R, (self.dev.splitk_cap - 64 - nvp * D) // nvp // 64 * 64)
                if rc >= 64:
                    var oh = self.dev.F(o_oh)
                    var prod = self.dev.F(o_oh + rc * nvp)
                    for r0 in range(0, R, rc):
                        var n = min(rc, R - r0)
                        self.dev.fill(oh, n * nvp, 0.0)
                        ctx.enqueue_function[k_onehot](oh, ids, Int32(r0), Int32(n), Int32(nvp), Int32(V),
                                                       grid_dim=blocks(n * PP), block_dim=BLOCK)
                        self.dev.gemm_tiled(prod, oh, dx.unsafe_offset(r0 * D), nvp, D, n, 1, 1, 1, nvp, D, 1, D, 1,
                                            0, 0, 0, 0, 0, 0, 1.0, 0 if r0 == 0 else 1)
                    self.dev.axpy(G.unsafe_offset(m.o_tok), prod, 1.0, ntok)
                else:
                    ctx.enqueue_function[k_embed_bwd_tok](G, dx, ids, Int32(m.o_tok), Int32(B), Int32(D), Int32(V),
                                                          Int32(ntok), grid_dim=blocks(ntok), block_dim=BLOCK)
        if not self.dev.gpu:
            for i in range(npos):
                b_embed_bwd_pos(G, dx, m.o_pos, B, D, i)
            for i in range(ncls):
                b_embed_bwd_cls(G, dx, cls, m.o_cls, B, D, i)
            for i in range(ntok):
                b_embed_bwd_tok(G, dx, ids, m.o_tok, B, D, V, i)
