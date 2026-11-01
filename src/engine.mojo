# Batched training engine, shared by every architecture: embeddings, loss, buffer layout and
# the forward/backward schedule. Generic ops (matmul, layer norm, AdamW, ...) live in
# src/gpu.mojo, the fused attention in src/attention.mojo. Rows are batch-major: r = b * n + i,
# row i of canvas b being canvas token tok[r]. n = T and tok[r] = i unless set_tokens chose
# fewer tokens (CROP, see token_order in src/model.mojo).

from std.math import sqrt, exp, log
from std.sys import has_accelerator
from std.memory import stack_allocation
from max.gpu import global_idx, thread_idx, block_idx, barrier
from max.gpu.memory import AddressSpace
from src.gpu import FP, IP, BLOCK, NCHUNK, SPLITK, blocks, Device
from arch.variant import DYT, CROP
from src.model import Config, Model, T, PP, SLOTS, NPIX, tgt_pix, slot_pix, crop_check, run_len
from src.attention import (
    BQ, b_attn_fwd, b_attn_dvec, b_attn_bwd_kv, b_attn_bwd_q,
    k_attn_fwd, k_attn_dvec, k_attn_bwd_kv, k_attn_bwd_q,
)


def b_embed(x: FP, w: FP, ids: IP, cls: IP, tok: IP, o_tok: Int, o_pos: Int, o_cls: Int,
            D: Int, V: Int, nt: Int, i: Int):
    var r = i // D
    var d = i - r * D
    var b = r // nt
    var t = Int(tok[unsafe_offset=r])
    var v = w[unsafe_offset=o_pos + t * D + d] + w[unsafe_offset=o_cls + Int(cls[unsafe_offset=b]) * D + d]
    for s in range(SLOTS):
        var id = Int(ids[unsafe_offset=b * NPIX + slot_pix(t, s)])
        v += w[unsafe_offset=o_tok + (s * (V + 1) + id) * D + d]
    x[unsafe_offset=i] = v


def b_embed_bwd_pos(g: FP, dx: FP, inv: IP, o_pos: Int, B: Int, D: Int, nt: Int, i: Int):
    """i = t * D + d; inv[b * T + t] is canvas b's row of token t (-1: not run)."""
    var t = i // D
    var d = i - t * D
    var s: Float32 = 0.0
    for b in range(B):
        var r = Int(inv[unsafe_offset=b * T + t])
        if r >= 0:
            s += dx[unsafe_offset=(b * nt + r) * D + d]
    g[unsafe_offset=o_pos + i] += s


def b_embed_bwd_cls(g: FP, dx: FP, cls: IP, o_cls: Int, B: Int, D: Int, nt: Int, i: Int):
    var c = i // D
    var d = i - c * D
    var s: Float32 = 0.0
    for b in range(B):
        if Int(cls[unsafe_offset=b]) == c:
            for t in range(nt):
                s += dx[unsafe_offset=(b * nt + t) * D + d]
    g[unsafe_offset=o_cls + i] += s


def b_embed_bwd_tok(g: FP, dx: FP, ids: IP, tok: IP, o_tok: Int, B: Int, D: Int, V: Int, nt: Int, i: Int):
    var d = i % D
    var sv = i // D
    var slot = sv // (V + 1)
    var v = sv - slot * (V + 1)
    var s: Float32 = 0.0
    for b in range(B):
        for t in range(nt):
            var r = b * nt + t
            if Int(ids[unsafe_offset=b * NPIX + slot_pix(Int(tok[unsafe_offset=r]), slot)]) == v:
                s += dx[unsafe_offset=r * D + d]
    g[unsafe_offset=o_tok + i] += s


def k_onehot(oh: FP, ids: IP, tok: IP, r0: Int32, n: Int32, nvp: Int32, V: Int32, TN: Int32):
    """oh[rr, s * (V + 1) + v] = 1 where row r0 + rr has id v in slot s, one thread per (row, slot),
    on a zeroed oh of n rows and nvp columns (the last nvp - SLOTS * (V + 1) are padding): the
    token-embedding gradient is oh^T @ dx."""
    var i = Int(global_idx.x)
    if i >= Int(n) * SLOTS:
        return
    var rr = i // SLOTS
    var s = i - rr * SLOTS
    var r = Int(r0) + rr
    var b = r // run_len(TN)
    var v = Int(ids[unsafe_offset=b * NPIX + slot_pix(Int(tok[unsafe_offset=r]), s)])
    oh[unsafe_offset=rr * Int(nvp) + s * (Int(V) + 1) + v] = 1.0


def b_loss(dl: FP, lossbuf: FP, logits: FP, tgt: IP, mask: IP, gscale: FP, tok: IP, V: Int, nt: Int, i: Int):
    """One (row, sub-pixel) per thread. lossbuf sums to the batch-mean loss."""
    var r = i // PP
    var k = i - r * PP
    var b = r // nt
    var t = Int(tok[unsafe_offset=r])
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
    var tg = Int(tgt[unsafe_offset=b * NPIX + tgt_pix(t, k)])
    var gs = gscale[unsafe_offset=b]
    lossbuf[unsafe_offset=i] = -(row[unsafe_offset=tg] - mx - log(s)) * gs
    for j in range(V):
        var p = drow[unsafe_offset=j] / s
        drow[unsafe_offset=j] = gs * (p - (Float32(1.0) if j == tg else Float32(0.0)))


def k_embed(x: FP, w: FP, ids: IP, cls: IP, tok: IP, o_tok: Int32, o_pos: Int32, o_cls: Int32,
            D: Int32, V: Int32, TN: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_embed(x, w, ids, cls, tok, Int(o_tok), Int(o_pos), Int(o_cls), Int(D), Int(V), run_len(TN), i)


def k_embed_bwd_pos(g: FP, dx: FP, inv: IP, o_pos: Int32, B: Int32, D: Int32, TN: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_embed_bwd_pos(g, dx, inv, Int(o_pos), Int(B), Int(D), run_len(TN), i)


def k_embed_bwd_cls(g: FP, dx: FP, cls: IP, o_cls: Int32, B: Int32, D: Int32, TN: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_embed_bwd_cls(g, dx, cls, Int(o_cls), Int(B), Int(D), run_len(TN), i)


def k_embed_bwd_tok(g: FP, dx: FP, ids: IP, tok: IP, o_tok: Int32, B: Int32, D: Int32, V: Int32, TN: Int32,
                    n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_embed_bwd_tok(g, dx, ids, tok, Int(o_tok), Int(B), Int(D), Int(V), run_len(TN), i)


def k_loss(dl: FP, lossbuf: FP, logits: FP, tgt: IP, mask: IP, gscale: FP, tok: IP, V: Int32, TN: Int32,
           n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        b_loss(dl, lossbuf, logits, tgt, mask, gscale, tok, Int(V), run_len(TN), i)


struct Engine(Movable):
    var cfg: Config
    var B: Int
    var R: Int  # B * n rows this batch
    var Rcap: Int  # B * T: the rows the buffers hold (and the per-layer stride)
    var n: Int  # tokens per canvas this batch (set_tokens)
    var dev: Device
    # offsets into the float arena
    var o_x: Int
    var o_h1: Int
    var o_xhat1: Int
    var o_rstd1: Int
    var o_qkv: Int
    var o_lse: Int  # per layer: log-sum-exp of each attention row [B*H*T]
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
    var o_dvec: Int
    var o_dlogits: Int
    var o_lossbuf: Int
    var o_gscale: Int
    # offsets into the int arena
    var i_ids: Int
    var i_tgt: Int
    var i_mask: Int
    var i_cls: Int
    var i_tok: Int  # [B * n] the canvas token of each row
    var i_inv: Int  # [B * T] each canvas token's row, -1 if it doesn't run

    def __init__(out self, cfg: Config, B: Int, use_gpu: Bool) raises:
        self.cfg = cfg.copy()
        self.B = B
        self.R = B * T
        self.Rcap = B * T
        self.n = T
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
        self.o_lse = take(o, L * B * cfg.H * T)
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
        self.o_dvec = take(o, B * cfg.H * T)
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
        self.i_tok = take(oi, R)
        self.i_inv = take(oi, R)

        if cfg.hd() != 16 and cfg.hd() != 32:
            raise Error("fused attention needs a head size (dim / heads) of 16 or 32, got " + String(cfg.hd()))
        crop_check()
        self.dev = Device(R, D, Model(cfg).n, splitk_cap, o, oi, use_gpu)
        var ident = List[Int32](capacity=R)
        for _ in range(B):
            for t in range(T):
                ident.append(Int32(t))
        self.set_tokens(ident, T)

    def set_tokens(mut self, toks: List[Int32], n: Int) raises:
        """Runs only n tokens per canvas from now on: canvas b's row i is token toks[b * n + i]
        (ascending). n must be a multiple of the attention tile (BQ), and below T only in CROP
        architectures. Masks uploaded with
        upload_batch are then per row [B * n], and logits are [B * n, PP, V]."""
        if n <= 0 or n > T or n % BQ != 0 or len(toks) != self.B * n:
            raise Error("set_tokens: bad token count " + String(n))
        comptime if CROP < 0:  # the kernels use the constant T (run_len)
            if n != T:
                raise Error("set_tokens: only CROP architectures run fewer than T tokens")
        var inv = List[Int32](length=self.Rcap, fill=-1)
        for b in range(self.B):
            for i in range(n):
                inv[b * T + Int(toks[b * n + i])] = Int32(i)
        self.n = n
        self.R = self.B * n
        self.dev.R = self.R
        self.dev.upload_i(self.i_tok, toks)
        self.dev.upload_i(self.i_inv, inv)

    def attn_fwd(self, att: FP, lse: FP, qkv: FP) raises:
        """att = softmax(scale Q K^T) V per (sprite, head), plus each row's log-sum-exp."""
        var D = self.cfg.D
        var H = self.cfg.H
        var hd = self.cfg.hd()
        var scale = Float32(1.0 / sqrt(Float64(hd)))
        comptime if has_accelerator():
            if self.dev.gpu:
                var ctx = self.dev.ctx.value()
                var grid = self.B * H * (self.n // BQ)
                if hd == 16:
                    ctx.enqueue_function[k_attn_fwd[16]](att, lse, qkv, Int32(D), Int32(H), scale, Int32(self.n),
                                                         grid_dim=grid, block_dim=256)
                else:
                    ctx.enqueue_function[k_attn_fwd[32]](att, lse, qkv, Int32(D), Int32(H), scale, Int32(self.n),
                                                         grid_dim=grid, block_dim=256)
                return
        for i in range(self.B * H * self.n):
            b_attn_fwd(att, lse, qkv, D, H, hd, scale, self.n, i)

    def attn_bwd(self, dqkv: FP, qkv: FP, att: FP, datt: FP, lse: FP) raises:
        """Overwrites every head's dQ, dK, dV columns of dqkv, recomputing the probabilities."""
        var D = self.cfg.D
        var H = self.cfg.H
        var hd = self.cfg.hd()
        var scale = Float32(1.0 / sqrt(Float64(hd)))
        var dvec = self.dev.F(self.o_dvec)
        var nt = self.n
        var rows = self.B * H * nt
        comptime if has_accelerator():
            if self.dev.gpu:
                var ctx = self.dev.ctx.value()
                var grid = self.B * H * (nt // BQ)
                ctx.enqueue_function[k_attn_dvec](dvec, att, datt, Int32(D), Int32(H), Int32(hd), Int32(rows), Int32(nt),
                                                  grid_dim=blocks(rows * hd), block_dim=BLOCK)
                if hd == 16:
                    ctx.enqueue_function[k_attn_bwd_kv[16]](dqkv, qkv, datt, lse, dvec, Int32(D), Int32(H), scale,
                                                            Int32(nt), grid_dim=grid, block_dim=256)
                    ctx.enqueue_function[k_attn_bwd_q[16]](dqkv, qkv, datt, lse, dvec, Int32(D), Int32(H), scale,
                                                           Int32(nt), grid_dim=grid, block_dim=256)
                else:
                    ctx.enqueue_function[k_attn_bwd_kv[32]](dqkv, qkv, datt, lse, dvec, Int32(D), Int32(H), scale,
                                                            Int32(nt), grid_dim=grid, block_dim=256)
                    ctx.enqueue_function[k_attn_bwd_q[32]](dqkv, qkv, datt, lse, dvec, Int32(D), Int32(H), scale,
                                                           Int32(nt), grid_dim=grid, block_dim=256)
                return
        for i in range(rows):
            b_attn_dvec(dvec, att, datt, D, H, hd, nt, i)
        for i in range(rows):
            b_attn_bwd_kv(dqkv, qkv, datt, lse, dvec, D, H, hd, scale, nt, i)
            b_attn_bwd_q(dqkv, qkv, datt, lse, dvec, D, H, hd, scale, nt, i)

    def upload_batch(self, ids: List[Int32], tgt: List[Int32], mask: List[Int32],
                     cls: List[Int32], gscale: List[Float32]) raises:
        self.dev.upload_i(self.i_ids, ids)
        self.dev.upload_i(self.i_tgt, tgt)
        self.dev.upload_i(self.i_mask, mask)
        self.dev.upload_i(self.i_cls, cls)
        self.dev.upload_f(self.o_gscale, gscale)

    def norm_fwd(self, y: FP, xhat: FP, rstd: FP, x: FP, g: FP, b: FP, a: FP) raises:
        """Layer norm, or DyT (a: its scalar) in the variants that set DYT; xhat then holds x."""
        comptime if DYT:
            self.dev.dyt_fwd(y, xhat, x, g, b, a)
        else:
            self.dev.ln_fwd(y, xhat, rstd, x, g, b)

    def norm_bwd(self, dx: FP, dg: FP, db: FP, dy: FP, xhat: FP, rstd: FP, g: FP, da: FP, a: FP) raises:
        comptime if DYT:
            self.dev.dyt_bwd(dx, dg, db, da, dy, xhat, g, a)
        else:
            self.dev.ln_bwd(dx, dg, db, dy, xhat, rstd, g)

    def forward(self, m: Model) raises:
        var D = self.cfg.D
        var F = self.cfg.F
        var V = self.cfg.V
        var H = self.cfg.H
        var L = self.cfg.L
        var R = self.R
        var S = self.Rcap  # per-layer stride of the activation buffers
        var W = self.dev.F(self.dev.o_w)
        var x = self.dev.F(self.o_x)
        var tok = self.dev.I(self.i_tok)

        var n = R * D
        comptime if has_accelerator():
            if self.dev.gpu:
                self.dev.ctx.value().enqueue_function[k_embed](
                    x, W, self.dev.I(self.i_ids), self.dev.I(self.i_cls), tok, Int32(m.o_tok), Int32(m.o_pos),
                    Int32(m.o_cls), Int32(D), Int32(V), Int32(self.n), Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
        if not self.dev.gpu:
            for i in range(n):
                b_embed(x, W, self.dev.I(self.i_ids), self.dev.I(self.i_cls), tok, m.o_tok, m.o_pos, m.o_cls, D, V,
                        self.n, i)

        for l in range(L):
            var h1 = self.dev.F(self.o_h1 + l * S * D)
            var qkv = self.dev.F(self.o_qkv + l * S * 3 * D)
            var lse = self.dev.F(self.o_lse + l * self.B * H * T)
            var att = self.dev.F(self.o_att + l * S * D)
            var h2 = self.dev.F(self.o_h2 + l * S * D)
            var u = self.dev.F(self.o_u + l * S * F)
            var act = self.dev.F(self.o_act + l * S * F)

            self.norm_fwd(h1, self.dev.F(self.o_xhat1 + l * S * D), self.dev.F(self.o_rstd1 + l * S), x,
                        W.unsafe_offset(m.lo(l, m.r_ln1g)), W.unsafe_offset(m.lo(l, m.r_ln1b)), W.unsafe_offset(m.o_alpha + 2 * l))
            self.dev.linear_fwd(qkv, h1, W.unsafe_offset(m.lo(l, m.r_wqkv)), W.unsafe_offset(m.lo(l, m.r_bqkv)), D, 3 * D)
            self.attn_fwd(att, lse, qkv)
            self.dev.linear_fwd(x, att, W.unsafe_offset(m.lo(l, m.r_wo)), W.unsafe_offset(m.lo(l, m.r_bo)), D, D, True)

            self.norm_fwd(h2, self.dev.F(self.o_xhat2 + l * S * D), self.dev.F(self.o_rstd2 + l * S), x,
                        W.unsafe_offset(m.lo(l, m.r_ln2g)), W.unsafe_offset(m.lo(l, m.r_ln2b)), W.unsafe_offset(m.o_alpha + 2 * l + 1))
            self.dev.linear_fwd(u, h2, W.unsafe_offset(m.lo(l, m.r_w1)), W.unsafe_offset(m.lo(l, m.r_b1)), D, F)
            self.dev.gelu_fwd(act, u, R * F)
            self.dev.linear_fwd(x, act, W.unsafe_offset(m.lo(l, m.r_w2)), W.unsafe_offset(m.lo(l, m.r_b2)), F, D, True)

        self.norm_fwd(self.dev.F(self.o_hf), self.dev.F(self.o_xhatf), self.dev.F(self.o_rstdf), x,
                    W.unsafe_offset(m.o_lnfg), W.unsafe_offset(m.o_lnfb), W.unsafe_offset(m.o_alpha + 2 * L))
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
                    self.dev.I(self.i_tok), Int32(V), Int32(self.n), Int32(n), grid_dim=blocks(n), block_dim=BLOCK)
                done = True
        if not done:
            for i in range(n):
                b_loss(dl, lb, lg, self.dev.I(self.i_tgt), self.dev.I(self.i_mask), self.dev.F(self.o_gscale),
                       self.dev.I(self.i_tok), V, self.n, i)
        return self.dev.chunk_sum(lb, n, False)

    def backward(self, m: Model) raises:
        var D = self.cfg.D
        var F = self.cfg.F
        var V = self.cfg.V
        var H = self.cfg.H
        var L = self.cfg.L
        var R = self.R
        var S = self.Rcap
        var B = self.B
        var W = self.dev.F(self.dev.o_w)
        var G = self.dev.F(self.dev.o_g)
        var dx = self.dev.F(self.o_dx)
        var dh = self.dev.F(self.o_dh)
        var dqkv = self.dev.F(self.o_dqkv)
        var datt = self.dev.F(self.o_datt)

        self.dev.linear_bwd(dh, G.unsafe_offset(m.o_wout), G.unsafe_offset(m.o_bout), self.dev.F(self.o_dlogits),
                        self.dev.F(self.o_hf), W.unsafe_offset(m.o_wout), D, PP * V, False)
        self.dev.fill(dx, R * D, 0.0)
        self.norm_bwd(dx, G.unsafe_offset(m.o_lnfg), G.unsafe_offset(m.o_lnfb), dh, self.dev.F(self.o_xhatf),
                    self.dev.F(self.o_rstdf), W.unsafe_offset(m.o_lnfg), G.unsafe_offset(m.o_alpha + 2 * L), W.unsafe_offset(m.o_alpha + 2 * L))

        for l in range(L - 1, -1, -1):
            var h1 = self.dev.F(self.o_h1 + l * S * D)
            var qkv = self.dev.F(self.o_qkv + l * S * 3 * D)
            var lse = self.dev.F(self.o_lse + l * B * H * T)
            var att = self.dev.F(self.o_att + l * S * D)
            var h2 = self.dev.F(self.o_h2 + l * S * D)
            var u = self.dev.F(self.o_u + l * S * F)
            var act = self.dev.F(self.o_act + l * S * F)

            # MLP block
            self.dev.linear_bwd(self.dev.F(self.o_dact), G.unsafe_offset(m.lo(l, m.r_w2)), G.unsafe_offset(m.lo(l, m.r_b2)),
                            dx, act, W.unsafe_offset(m.lo(l, m.r_w2)), F, D, False)
            self.dev.gelu_bwd(self.dev.F(self.o_du), self.dev.F(self.o_dact), u, R * F)
            self.dev.linear_bwd(dh, G.unsafe_offset(m.lo(l, m.r_w1)), G.unsafe_offset(m.lo(l, m.r_b1)),
                            self.dev.F(self.o_du), h2, W.unsafe_offset(m.lo(l, m.r_w1)), D, F, False)
            self.norm_bwd(dx, G.unsafe_offset(m.lo(l, m.r_ln2g)), G.unsafe_offset(m.lo(l, m.r_ln2b)), dh,
                        self.dev.F(self.o_xhat2 + l * S * D), self.dev.F(self.o_rstd2 + l * S),
                        W.unsafe_offset(m.lo(l, m.r_ln2g)), G.unsafe_offset(m.o_alpha + 2 * l + 1), W.unsafe_offset(m.o_alpha + 2 * l + 1))

            # attention block
            self.dev.linear_bwd(datt, G.unsafe_offset(m.lo(l, m.r_wo)), G.unsafe_offset(m.lo(l, m.r_bo)),
                            dx, att, W.unsafe_offset(m.lo(l, m.r_wo)), D, D, False)
            self.attn_bwd(dqkv, qkv, att, datt, lse)
            self.dev.linear_bwd(dh, G.unsafe_offset(m.lo(l, m.r_wqkv)), G.unsafe_offset(m.lo(l, m.r_bqkv)),
                            dqkv, h1, W.unsafe_offset(m.lo(l, m.r_wqkv)), D, 3 * D, False)
            self.norm_bwd(dx, G.unsafe_offset(m.lo(l, m.r_ln1g)), G.unsafe_offset(m.lo(l, m.r_ln1b)), dh,
                        self.dev.F(self.o_xhat1 + l * S * D), self.dev.F(self.o_rstd1 + l * S),
                        W.unsafe_offset(m.lo(l, m.r_ln1g)), G.unsafe_offset(m.o_alpha + 2 * l), W.unsafe_offset(m.o_alpha + 2 * l))

        # embeddings
        var ids = self.dev.I(self.i_ids)
        var cls = self.dev.I(self.i_cls)
        var tok = self.dev.I(self.i_tok)
        var inv = self.dev.I(self.i_inv)
        var nt = self.n
        var npos = T * D
        var ncls = (self.cfg.C + 1) * D
        var ntok = SLOTS * (V + 1) * D
        comptime if has_accelerator():
            if self.dev.gpu:
                var ctx = self.dev.ctx.value()
                ctx.enqueue_function[k_embed_bwd_pos](G, dx, inv, Int32(m.o_pos), Int32(B), Int32(D), Int32(nt), Int32(npos),
                                                      grid_dim=blocks(npos), block_dim=BLOCK)
                ctx.enqueue_function[k_embed_bwd_cls](G, dx, cls, Int32(m.o_cls), Int32(B), Int32(D), Int32(nt), Int32(ncls),
                                                      grid_dim=blocks(ncls), block_dim=BLOCK)
                # oh^T @ dx over row chunks that fit the split-K scratch (one-hot chunk, then the
                # padded product); gemm_tiled, since gemm's own split-K would reuse that scratch.
                var nvp = (SLOTS * (V + 1) + 63) // 64 * 64
                var o_oh = (self.dev.o_splitk + 63) // 64 * 64
                var rc = min(R, (self.dev.splitk_cap - 64 - nvp * D) // nvp // 64 * 64)
                if rc >= 64:
                    var oh = self.dev.F(o_oh)
                    var prod = self.dev.F(o_oh + rc * nvp)
                    for r0 in range(0, R, rc):
                        var n = min(rc, R - r0)
                        self.dev.fill(oh, n * nvp, 0.0)
                        ctx.enqueue_function[k_onehot](oh, ids, tok, Int32(r0), Int32(n), Int32(nvp), Int32(V), Int32(nt),
                                                       grid_dim=blocks(n * SLOTS), block_dim=BLOCK)
                        self.dev.gemm_tiled(prod, oh, dx.unsafe_offset(r0 * D), nvp, D, n, 1, 1, 1, nvp, D, 1, D, 1,
                                            0, 0, 0, 0, 0, 0, 1.0, 0 if r0 == 0 else 1)
                    self.dev.axpy(G.unsafe_offset(m.o_tok), prod, 1.0, ntok)
                else:
                    ctx.enqueue_function[k_embed_bwd_tok](G, dx, ids, tok, Int32(m.o_tok), Int32(B), Int32(D), Int32(V),
                                                          Int32(nt), Int32(ntok), grid_dim=blocks(ntok), block_dim=BLOCK)
        if not self.dev.gpu:
            for i in range(npos):
                b_embed_bwd_pos(G, dx, inv, m.o_pos, B, D, nt, i)
            for i in range(ncls):
                b_embed_bwd_cls(G, dx, cls, m.o_cls, B, D, nt, i)
            for i in range(ntok):
                b_embed_bwd_tok(G, dx, ids, tok, m.o_tok, B, D, V, nt, i)
