# Curate the exported dataset: near-duplicate removal + cluster-based rebalancing.
#
# Adapted from the image-generation data recipe in "LongCat-Next" (arXiv
# 2603.27538, Sec. 4.2.2 and 4.3.2-4.3.3):
#   1. Embed every clip.        (paper: SigLIP2; here: a cheap pixel-art
#                                descriptor built from the frames, no model needed)
#   2. Deduplicate.             Within each action (attack, idle, ...) drop clips
#                                whose embedding is near-identical to one already
#                                kept. Duelyst has many reskins / alt-generals.
#   3. Cluster (k-means).       Group clips by look + motion.
#   4. Power-law rebalancing.   Repeats = (largest_cluster / cluster_size)^alpha,
#                                so rare looks are seen more often per epoch.
#   5. Cluster exemplars.       The clip nearest each centroid, for an optional
#                                short final "SFT-style" pass on a clean, diverse set.
#
# The descriptor of a clip is the concatenation of three L2-normalised blocks:
#   appearance: mean 16x16 RGB thumbnail of the frames, minus the background
#   motion:     mean |frame[t+1] - frame[t]| as a 16x16 grayscale map
#   palette:    4x4x4 RGB histogram of the non-background pixels
#
# Output (in <dataset>/curation/):
#   curation.jsonl                 one record per clip: cluster, dup_of, repeats, exemplar
#   <media>_r<N>.jsonl             musubi-tuner jsonl lists, one per repeat tier
#   <media>_exemplars.jsonl        cluster exemplars only
# where <media> is "video" (Wan clips) and/or "image" (FLUX.2 grids).
# train_lora.mojo picks these up with --curated / --exemplars.
#
# Usage:
#   mojo run training/curate_dataset.mojo [--dataset training/dataset]
#       [--clusters 0 (=sqrt N)] [--dup 0.985] [--alpha 0.5] [--max-repeats 4]
#       [--seed 42] [--bg 64,64,64]

from std.python import Python, PythonObject
from std.sys import argv
from std.collections import List


comptime THUMB = 16


def frames_of(
    pil: PythonObject, iio: PythonObject, np: PythonObject, dataset: String, rec: PythonObject
) raises -> PythonObject:
    """The frames of one clip as a list of RGB PIL images (from the grid PNG if
    there is one, otherwise from the video)."""
    var out = Python.list()
    if "grid" in rec:
        var img = pil.open(dataset + "/" + String(rec["grid"])).convert("RGB")
        var cols = Int(py=rec["grid_cols"])
        var rows = Int(py=rec["grid_rows"])
        var cw = Int(py=img.width) // cols
        var ch = Int(py=img.height) // rows
        var n = Int(py=rec["frames"])
        for k in range(n):
            var c = k % cols
            var r = k // cols
            out.append(img.crop(Python.tuple(c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)))
    elif "video" in rec:
        for arr in iio.mimread(dataset + "/" + String(rec["video"]), memtest=False):
            out.append(pil.fromarray(np.asarray(arr)[:, :, :3]))
    return out


def l2(np: PythonObject, v: PythonObject) raises -> PythonObject:
    return v / (np.linalg.norm(v) + 1e-8)


def describe(
    pil: PythonObject, np: PythonObject, frames: PythonObject, bg: PythonObject
) raises -> PythonObject:
    var thumbs = Python.list()
    var hist = np.zeros(64, dtype="float64")
    for f in frames:
        # BOX keeps thin details visible as partial values instead of dropping them.
        var t = np.asarray(f.resize(Python.tuple(THUMB, THUMB), pil.BOX), dtype="float32") / 255.0
        thumbs.append(t - bg / 255.0)
        var px = np.asarray(f.resize(Python.tuple(64, 64), pil.NEAREST), dtype="int16").reshape(-1, 3)
        var fg = px[np.abs(px - bg).max(axis=1) > 12]
        if Int(py=fg.shape[0]) > 0:
            var q = (fg // 64).T
            hist += np.bincount(q[0] * 16 + q[1] * 4 + q[2], minlength=64)
    var stack = np.stack(thumbs)
    var appearance = stack.mean(axis=0).ravel()
    var motion = np.zeros(THUMB * THUMB, dtype="float32")
    if len(thumbs) > 1:
        motion = np.abs(np.diff(stack.mean(axis=3), axis=0)).mean(axis=0).ravel()
    var palette = np.sqrt(hist)  # Hellinger-style, softens dominant colours
    var mot = l2(np, motion) * PythonObject(0.7)
    var v = np.concatenate(Python.list(l2(np, appearance), mot, l2(np, palette)))
    return l2(np, v).astype("float32")


def kmeans(np: PythonObject, x: PythonObject, k: Int, seed: Int, iters: Int) raises -> PythonObject:
    """Spherical k-means with k-means++ init. Returns (labels, centroids)."""
    var rng = np.random.default_rng(seed)
    var n = Int(py=x.shape[0])
    var idx = Python.list(Int(py=rng.integers(n)))
    var d2 = 1.0 - np.dot(x, x[idx[0]])
    for _ in range(1, k):
        var p = np.clip(d2, 0, None) ** 2
        var s = Float64(py=p.sum())
        var j: Int
        if s <= 0:
            j = Int(py=rng.integers(n))
        else:
            j = Int(py=rng.choice(n, p=p / s))
        idx.append(j)
        d2 = np.minimum(d2, 1.0 - np.dot(x, x[j]))
    var cent = x[np.array(idx)].copy()
    var labels = np.zeros(n, dtype="int64")
    for _ in range(iters):
        var new_labels = np.dot(x, cent.T).argmax(axis=1)
        var changed = Int(py=(new_labels != labels).sum())
        labels = new_labels
        for c in range(k):
            var members = x[labels == c]
            if Int(py=members.shape[0]) == 0:
                # Re-seed an empty cluster with the point worst served by its centroid.
                var worst = Int(py=(x * cent[labels]).sum(axis=1).argmin())
                cent[c] = x[worst]
            else:
                cent[c] = l2(np, members.mean(axis=0))
        if changed == 0:
            break
    return Python.tuple(labels, cent)


def main() raises:
    var dataset = String("training/dataset")
    var k = 0
    var dup = 0.985
    var alpha = 0.5
    var max_repeats = 4
    var seed = 42
    var bg_s = String("64,64,64")

    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if i + 1 >= len(args):
            raise Error("missing value for " + a)
        i += 1
        var v = String(args[i])
        if a == "--dataset": dataset = v
        elif a == "--clusters": k = Int(v)
        elif a == "--dup": dup = Float64(v)
        elif a == "--alpha": alpha = Float64(v)
        elif a == "--max-repeats": max_repeats = Int(v)
        elif a == "--seed": seed = Int(v)
        elif a == "--bg": bg_s = v
        else:
            raise Error("unknown argument: " + a)
        i += 1

    var os = Python.import_module("os")
    var json = Python.import_module("json")
    var np = Python.import_module("numpy")
    var pil = Python.import_module("PIL.Image")
    var iio: PythonObject = Python.none()
    try:
        iio = Python.import_module("imageio.v2")
    except:
        pass
    var read = Python.evaluate("lambda p: open(p).read()")
    var bgp = bg_s.split(",")
    var bg = np.array(Python.list(Int(bgp[0].strip()), Int(bgp[1].strip()), Int(bgp[2].strip())), dtype="float32")

    # ---- 1. embed ----
    var recs = Python.list()
    for line in String(read(dataset + "/metadata.jsonl")).split("\n"):
        if String(line.strip()) != "":
            recs.append(json.loads(String(line)))
    var n = len(recs)
    if n < 2:
        raise Error("need at least 2 clips in " + dataset + "/metadata.jsonl")
    print("Embedding", n, "clips...")
    var feats = Python.list()
    for j in range(n):
        feats.append(describe(pil, np, frames_of(pil, iio, np, dataset, recs[j]), bg))
        if (j + 1) % 500 == 0:
            print("  ", j + 1)
    var x = np.stack(feats)

    # ---- 2. dedup within each action, keeping longer / larger clips first ----
    var dup_of = Python.list()
    for _ in range(n):
        dup_of.append(Python.none())
    var actions = Python.evaluate("lambda rs: sorted({r['action'] for r in rs})")(recs)
    var n_dup = 0
    for act in actions:
        var members = Python.evaluate(
            "lambda rs, a: sorted([i for i, r in enumerate(rs) if r['action'] == a],"
            " key=lambda i: (-rs[i]['frames'], -rs[i]['frame_size'], rs[i]['id']))"
        )(recs, act)
        var kept = Python.list()
        for m in members:
            var mi = Int(py=m)
            if len(kept) > 0:
                var sims = np.dot(x[np.array(kept)], x[mi])
                # Only another unit can be a duplicate (reskins); two animations of
                # the same unit (e.g. caststart / castend) are different motions.
                for q in range(len(kept)):
                    if recs[Int(py=kept[q])]["asset"] == recs[mi]["asset"]:
                        sims[q] = -1.0
                var best = Int(py=sims.argmax())
                if Float64(py=sims[best]) >= dup:
                    dup_of[mi] = recs[Int(py=kept[best])]["id"]
                    n_dup += 1
                    continue
            kept.append(mi)
    var keep_idx = np.array(Python.evaluate("lambda d: [i for i, v in enumerate(d) if v is None]")(dup_of))
    var nk = Int(py=keep_idx.shape[0])
    print("Near-duplicates removed:", n_dup, "| kept:", nk)

    # ---- 3. cluster ----
    if k <= 0:
        k = max(2, Int(py=np.sqrt(nk).round()))
    k = min(k, nk)
    var res = kmeans(np, x[keep_idx], k, seed, 50)
    var labels = res[0]
    var cent = res[1]
    var sizes = np.bincount(labels, minlength=k)

    # ---- 4. power-law repeats: (largest_cluster / size)^alpha, so the biggest
    #      cluster is seen once per epoch and rarer ones proportionally more ----
    var w = (Float64(py=sizes.max()) / sizes[labels].astype("float64")) ** alpha
    var repeats = np.clip(np.rint(w), 1, max_repeats).astype("int64")

    # ---- 5. exemplar = member closest to each centroid ----
    var sim_to_c = (x[keep_idx] * cent[labels]).sum(axis=1)
    var is_ex = np.zeros(nk, dtype="bool")
    for c in range(k):
        var members = np.nonzero(labels == c)[0]
        if Int(py=members.shape[0]) > 0:
            is_ex[members[sim_to_c[members].argmax()]] = True

    # ---- write ----
    var out = dataset + "/curation"
    os.makedirs(out, exist_ok=True)
    var cur = Python.evaluate("open")(out + "/curation.jsonl", "w")
    var pos = Python.dict()
    for j in range(nk):
        pos[keep_idx[j]] = j
    var tiers = Python.dict()  # (media, repeats) -> list of jsonl lines
    for j in range(n):
        var rec = recs[j]
        var row = Python.dict()
        row["id"] = rec["id"]
        row["action"] = rec["action"]
        row["dup_of"] = dup_of[j]
        if dup_of[j] is None:
            var p = Int(py=pos[j])
            var r = Int(py=repeats[p])
            row["cluster"] = Int(py=labels[p])
            row["cluster_size"] = Int(py=sizes[labels[p]])
            row["repeats"] = r
            row["exemplar"] = Bool(py=is_ex[p])
            var medias = Python.list()
            if "video" in rec:
                medias.append(Python.tuple("video", rec["video"]))
            if "grid" in rec:
                medias.append(Python.tuple("image", rec["grid"]))
            for m in medias:
                var media = String(m[0])
                var path = String(os.path.abspath(dataset + "/" + String(m[1])))
                # Read the caption file so hand/VLM-refined captions are used.
                var cap_path = String(os.path.splitext(path)[0]) + ".txt"
                var entry = Python.dict()
                entry[media + "_path"] = path
                entry["caption"] = String(read(cap_path)).strip()
                var line = String(json.dumps(entry))
                var key = media + "_r" + String(r)
                if key not in tiers:
                    tiers[key] = Python.list()
                tiers[key].append(line)
                if Bool(py=is_ex[p]):
                    var ek = media + "_exemplars"
                    if ek not in tiers:
                        tiers[ek] = Python.list()
                    tiers[ek].append(line)
        else:
            row["repeats"] = 0
        cur.write(json.dumps(row) + "\n")
    cur.close()

    # Drop stale tier files from earlier runs before writing the new ones.
    for f in os.listdir(out):
        var fs = String(f)
        if (fs.startswith("video_") or fs.startswith("image_")) and fs.endswith(".jsonl"):
            os.remove(out + "/" + fs)
    for key in Python.evaluate("sorted")(tiers.keys()):
        var lines = tiers[key]
        Python.evaluate("lambda p, s: open(p, 'w').write(s)")(
            out + "/" + String(key) + ".jsonl", String(PythonObject("\n").join(lines)) + "\n"
        )
        print("  ", String(key) + ".jsonl:", len(lines), "clips")

    var hist = np.bincount(repeats, minlength=max_repeats + 1)
    print("Clusters:", k, "| sizes min/median/max:", Int(py=sizes.min()), Int(py=np.median(sizes)), Int(py=sizes.max()))
    var parts = List[String]()
    for r in range(1, max_repeats + 1):
        parts.append("x" + String(r) + "=" + String(Int(py=hist[r])))
    print("Repeats:", String(", ").join(parts))
    var eff = Int(py=repeats.sum())
    print("Effective clips per epoch:", eff, "(was", n, "before curation)")
    print("Wrote", out)
