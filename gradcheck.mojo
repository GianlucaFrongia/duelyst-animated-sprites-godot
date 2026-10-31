# Verifies the hand-written backward pass against finite differences.
# Run this whenever you change a forward or backward function:
#
#     mojo run gradcheck.mojo
#
# It builds a tiny model, picks random parameters from every tensor, and compares
# the analytic gradient with (loss(w + eps) - loss(w - eps)) / (2 eps).

from std.random import seed, random_ui64, random_float64
from arch.model import Config, Model, Work, forward, backward, masked_loss, T, PP, NPIX, tgt_pix


def loss_at(mut m: Model, mut wk: Work, ids: List[Int], tgt: List[Int], mask: List[Bool],
            cls: Int, gs: Float32) -> Float32:
    forward(m, wk, ids, cls)
    return masked_loss(wk, m.cfg.V, tgt, mask, gs)


def main() raises:
    seed(42)
    var cfg = Config(V=5, C=2, D=16, L=2, H=2, F=32)
    var m = Model(cfg)
    m.init_weights()
    # Larger weights than training init so every path carries a visible gradient.
    for i in range(m.n):
        if m.decay[i] == 1:
            m.w[i] *= 10.0
    var wk = Work(cfg)

    var tgt = List[Int](length=NPIX, fill=0)
    for i in range(NPIX):
        tgt[i] = Int(random_ui64(0, UInt64(cfg.V - 1)))
    var mask = List[Bool](length=T, fill=False)
    var n_masked = 0
    for t in range(T):
        mask[t] = random_float64() < 0.5
        if mask[t]:
            n_masked += 1
    var ids = tgt.copy()
    for t in range(T):
        if mask[t]:
            for k in range(PP):
                ids[tgt_pix(t, k)] = cfg.V
    var cls = 1
    var gs = Float32(1.0 / Float64(n_masked * PP))

    m.zero_grad()
    var base = loss_at(m, wk, ids, tgt, mask, cls, gs)
    backward(m, wk, ids, cls)
    print("loss", base)

    var names = List[String]()
    var starts = List[Int]()
    var sizes = List[Int]()

    def add(mut names: List[String], mut starts: List[Int], mut sizes: List[Int],
            name: String, s: Int, n: Int):
        names.append(name)
        starts.append(s)
        sizes.append(n)

    var D = cfg.D
    var F = cfg.F
    add(names, starts, sizes, "tok_emb", m.o_tok, m.o_pos - m.o_tok)
    add(names, starts, sizes, "pos_emb", m.o_pos, T * D)
    add(names, starts, sizes, "cls_emb", m.o_cls + cls * D, D)
    for l in range(cfg.L):
        var p = "L" + String(l) + "."
        add(names, starts, sizes, p + "ln1_g", m.lo(l, m.r_ln1g), D)
        add(names, starts, sizes, p + "ln1_b", m.lo(l, m.r_ln1b), D)
        add(names, starts, sizes, p + "W_qkv", m.lo(l, m.r_wqkv), 3 * D * D)
        add(names, starts, sizes, p + "b_qkv", m.lo(l, m.r_bqkv), 3 * D)
        add(names, starts, sizes, p + "W_o", m.lo(l, m.r_wo), D * D)
        add(names, starts, sizes, p + "ln2_g", m.lo(l, m.r_ln2g), D)
        add(names, starts, sizes, p + "W_1", m.lo(l, m.r_w1), F * D)
        add(names, starts, sizes, p + "b_1", m.lo(l, m.r_b1), F)
        add(names, starts, sizes, p + "W_2", m.lo(l, m.r_w2), D * F)
        add(names, starts, sizes, p + "b_2", m.lo(l, m.r_b2), D)
    add(names, starts, sizes, "lnf_g", m.o_lnfg, D)
    add(names, starts, sizes, "W_out", m.o_wout, PP * cfg.V * D)
    add(names, starts, sizes, "b_out", m.o_bout, PP * cfg.V)

    var eps: Float32 = 1e-2
    var worst: Float32 = 0.0
    var failed = 0
    for r in range(len(names)):
        var worst_r: Float32 = 0.0
        for _ in range(4):
            var i = starts[r] + Int(random_ui64(0, UInt64(sizes[r] - 1)))
            var w0 = m.w[i]
            m.w[i] = w0 + eps
            var lp = loss_at(m, wk, ids, tgt, mask, cls, gs)
            m.w[i] = w0 - eps
            var lm = loss_at(m, wk, ids, tgt, mask, cls, gs)
            m.w[i] = w0
            var num = (lp - lm) / (2.0 * eps)
            var ana = m.g[i]
            var err = abs(num - ana) / max(Float32(1e-3), abs(num) + abs(ana))
            worst_r = max(worst_r, err)
        worst = max(worst, worst_r)
        var ok = worst_r < 0.05
        if not ok:
            failed += 1
        print(("  ok  " if ok else "  FAIL"), names[r], "max rel err", worst_r)
    print("worst relative error:", worst)
    if failed > 0:
        raise Error(String(failed) + " tensors failed the gradient check")
    print("gradient check passed")
