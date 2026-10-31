"""Turn animated sprites into the frame-pair binary format the Mojo trainer reads.

Each training example is K consecutive frames of one animation stacked vertically: the
first K-1 are context, the last is the target. K = --context + 1, default 2 (a 64x128 pair:
frame t on top, frame t+1 below). Architectures with two context frames (ctx2) use K = 3.

--inbetween writes keyframe triples for the inbetween architecture instead: frames t and t+g
as context, the middle frame t+g/2 as the target, for gaps g = 2, 4, 8.

Two input layouts are supported, and both can be given in one run to merge them:

    --atlases DIR   TexturePacker atlases: <name>.plist + <name>.png per unit (e.g. Duelyst).
                    Frame keys look like <unit>_<anim>_<NNN>.png; the class is <anim>.
    --frames DIR    One folder per class, one sub-folder per animation, frames sorted by name:
                    sprites/slime/0001/000.png 001.png ...

Every frame of an animation is cropped with the same offset (union bounding box, placed
bottom-center) so motion is preserved. Animations larger than 64x64 are skipped: the art
is kept at its native pixel scale rather than resized.

Output (in --out):
    pixels.bin   N * K * 4096 bytes: palette indices of each 64x(64K) example, row-major
    labels.bin   N bytes, class id per example
    gaps.bin     N bytes, frame distance between the first and last frame of each example
    palette.txt  one "r g b" line per palette index (index 0 is transparent)
    classes.txt  one class name per line, in class-id order
    meta.txt     N / V (palette size incl. transparent) / C (classes) / S (frame size) / K /
                 O (frame order: next, or inbetween)
    preview.png  a few examples rendered back from the indices, to check the conversion
    val/         the same files for held-out units (--val-frac): every animation of a unit goes
                 to one side, so validation measures units the model never saw

    python tools/prepare_animations.py --atlases raw/units --out data
"""
import argparse
import plistlib
import random
import re
import shutil
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

S = 64
ALIASES = {"breathe": "breathing"}
DEFAULT_CLASSES = "attack,breathing,death,hit,idle,run"
DEFAULT_LOOPING = "breathing,idle,run"


def parse_rect(s):
    return list(map(int, re.findall(r"-?\d+", s)))


def load_atlases(root, classes):
    """Yields (class, unit, [RGBA frames at full source size]) for every animation in every atlas."""
    for plist in sorted(Path(root).glob("*.plist")):
        sheet_path = plist.with_suffix(".png")
        if not sheet_path.exists():
            continue
        frames = plistlib.load(open(plist, "rb"))["frames"]
        sheet = np.asarray(Image.open(sheet_path).convert("RGBA"))
        groups = defaultdict(list)
        for key, v in frames.items():
            m = re.match(r"(.*)_([a-z]+)_(\d+)\.png$", key)
            if not m:
                continue
            anim = ALIASES.get(m.group(2), m.group(2))
            if anim in classes:
                groups[(m.group(1), anim)].append((int(m.group(3)), v))
        for (unit, anim), fr in sorted(groups.items()):
            out = []
            for _, v in sorted(fr, key=lambda e: e[0]):
                x, y, w, h = parse_rect(v["frame"])
                ox, oy, _, _ = parse_rect(v["sourceColorRect"])
                sw, sh = parse_rect(v["sourceSize"])
                crop = sheet[y:y + h, x:x + w]
                if v.get("rotated"):  # TexturePacker stores rotated frames 90 degrees clockwise
                    crop = np.rot90(sheet[y:y + w, x:x + h], 1)
                full = np.zeros((sh, sw, 4), dtype=np.uint8)
                full[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
                out.append(full)
            yield anim, plist.stem + "/" + unit, out


def folder_unit(cls, name):
    """The unit an animation folder belongs to: "<unit>_<Anim>" -> "<unit>". Names like
    "attack_023f2c93" (class first) don't name the unit, so the folder stands alone."""
    head, _, _ = name.rpartition("_")
    return name if not head or head.lower() == cls.lower() else head


def load_frame_folders(root, classes):
    for cdir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if classes and cdir.name not in classes:
            continue
        for adir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            paths = sorted(adir.glob("*.png"))
            if paths:
                yield cdir.name, folder_unit(cdir.name, adir.name), [np.asarray(Image.open(p).convert("RGBA")) for p in paths]


def write_set(out, pairs, labels, gaps, palette, classes, K, order):
    out.mkdir(parents=True, exist_ok=True)
    pairs.tofile(out / "pixels.bin")
    np.array(labels, dtype=np.uint8).tofile(out / "labels.bin")
    np.array(gaps, dtype=np.uint8).tofile(out / "gaps.bin")
    with open(out / "palette.txt", "w") as f:
        f.write("0 0 0\n")  # index 0: transparent (color unused)
        for r, g, b in palette:
            f.write(f"{r} {g} {b}\n")
    (out / "classes.txt").write_text("\n".join(classes) + "\n")
    (out / "meta.txt").write_text(f"N {len(pairs)}\nV {len(palette) + 1}\nC {len(classes)}\nS {S}\nK {K}\nO {order}\n")


def fit_animation(frames, alpha_threshold):
    """Crops every frame with the animation's union bbox, placed bottom-center in SxS."""
    opaque = [f[..., 3] >= alpha_threshold for f in frames]
    ys = [np.nonzero(o.any(1))[0] for o in opaque]
    xs = [np.nonzero(o.any(0))[0] for o in opaque]
    if not any(len(y) for y in ys):
        return None
    y0 = min(y[0] for y in ys if len(y))
    y1 = max(y[-1] for y in ys if len(y)) + 1
    x0 = min(x[0] for x in xs if len(x))
    x1 = max(x[-1] for x in xs if len(x)) + 1
    h, w = y1 - y0, x1 - x0
    if h > S or w > S:
        return None
    dy, dx = S - h, (S - w) // 2
    out = np.zeros((len(frames), S, S, 4), dtype=np.uint8)
    for i, (f, o) in enumerate(zip(frames, opaque)):
        part = f[y0:y1, x0:x1].copy()
        part[..., 3] = np.where(o[y0:y1, x0:x1], 255, 0)
        out[i, dy:dy + h, dx:dx + w] = part
    return out


def pack(rgb):
    rgb = rgb.astype(np.int32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def unpack(c):
    return np.stack([(c >> 16) & 255, (c >> 8) & 255, c & 255], axis=-1).astype(np.int32)


def build_palette(rgb_opaque, n_slots, rng):
    colors, counts = np.unique(pack(rgb_opaque), return_counts=True)
    if len(colors) <= n_slots:
        return unpack(colors[np.argsort(-counts)])
    # Median cut on a subsample: fast, and the palette barely changes beyond a few million pixels.
    take = min(len(rgb_opaque), 4_000_000)
    sample = rgb_opaque[rng.choice(len(rgb_opaque), take, replace=False)]
    q = Image.fromarray(sample.reshape(1, -1, 3).astype(np.uint8), "RGB").quantize(
        colors=n_slots, method=Image.Quantize.MEDIANCUT)
    print(f"{len(colors)} unique colors -> quantized to {n_slots}")
    return np.array(q.getpalette()[: n_slots * 3], dtype=np.int32).reshape(-1, 3)


def to_indices(rgba, palette):
    flat = pack(rgba[..., :3]).reshape(-1)
    op = rgba[..., 3].reshape(-1) > 0
    idx = np.zeros(len(flat), dtype=np.uint8)
    # Map each distinct color once instead of every pixel.
    colors, inv = np.unique(flat[op], return_inverse=True)
    d = ((unpack(colors)[:, None, :] - palette[None, :, :]) ** 2).sum(-1)
    idx[op] = (d.argmin(1) + 1).astype(np.uint8)[inv.reshape(-1)]
    return idx.reshape(rgba.shape[:-1])


def render(idx, palette):
    pal = np.vstack([[0, 0, 0], palette]).astype(np.uint8)
    rgba = np.zeros(idx.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = pal[idx]
    rgba[..., 3] = np.where(idx == 0, 0, 255)
    return rgba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlases", help="folder of TexturePacker .plist + .png atlases")
    ap.add_argument("--frames", help="folder of <class>/<animation>/<frame>.png")
    ap.add_argument("--out", default="data")
    ap.add_argument("--classes", default=None,
                    help=f"comma-separated animation types to keep (atlases default: {DEFAULT_CLASSES})")
    ap.add_argument("--looping", default=DEFAULT_LOOPING,
                    help="classes whose animations loop (they also get a last -> first frame pair)")
    ap.add_argument("--max-colors", type=int, default=64,
                    help="palette size including the transparent index (at most 255)")
    ap.add_argument("--alpha-threshold", type=int, default=128)
    ap.add_argument("--context", type=int, default=1,
                    help="context frames per example (1 for ctx and stacked, 2 for ctx2)")
    ap.add_argument("--inbetween", action="store_true",
                    help="keyframe triples (t, t+g, middle) for the inbetween architecture")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="fraction of units held out in <out>/val for validation (0 for none)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.inbetween:
        args.context = 2
    K = args.context + 1
    if not 2 <= args.max_colors <= 255:
        raise SystemExit("--max-colors must be between 2 and 255")
    rng = np.random.default_rng(args.seed)

    if not (args.atlases or args.frames):
        raise SystemExit("give --atlases, --frames, or both")
    if args.classes:
        classes = sorted(args.classes.split(","))
    else:
        found = set(DEFAULT_CLASSES.split(",")) if args.atlases else set()
        if args.frames:
            found |= {p.name for p in Path(args.frames).iterdir() if p.is_dir()}
        classes = sorted(found)
    # Both sources can be combined into one dataset with a shared palette.
    sources = []
    if args.atlases:
        sources.append(load_atlases(args.atlases, set(classes)))
    if args.frames:
        sources.append(load_frame_folders(args.frames, set(classes)))
    source = (item for src in sources for item in src)

    anims, labels, units, skipped = [], [], [], 0
    for name, unit, frames in source:
        if len(frames) < 2:
            continue
        fitted = fit_animation(frames, args.alpha_threshold)
        if fitted is None:
            skipped += 1
            continue
        anims.append(fitted)
        labels.append(classes.index(name))
        units.append(unit)
    if not anims:
        raise SystemExit("no usable animations found")
    print(f"{len(anims)} animations kept, {skipped} skipped (larger than {S}x{S})")

    allf = np.concatenate(anims)
    palette = build_palette(allf[..., :3][allf[..., 3] > 0], args.max_colors - 1, rng)
    idx_anims = [to_indices(a, palette) for a in anims]

    # K consecutive frames (or keyframe triples); looping animations also wrap around
    # (last -> first) so samples can loop.
    looping = set(args.looping.split(","))
    # A hash of the unit name picks the side, so the split stays the same across re-runs.
    def held_out(unit):
        return zlib.crc32(unit.encode()) % 10000 < args.val_frac * 10000

    sets = {False: ([], [], []), True: ([], [], [])}  # held out -> (pairs, labels, gaps)
    for a, c, unit in zip(idx_anims, labels, units):
        pairs, pair_labels, gaps = sets[held_out(unit)]
        n = len(a)
        loops = classes[c] in looping
        if args.inbetween:
            for g in (2, 4, 8):
                if g >= n:  # with wrap-around, frame t + g would be t itself (or earlier) again
                    continue
                for i in range(n if loops else n - g):
                    pairs.append(np.concatenate([a[i], a[(i + g) % n], a[(i + g // 2) % n]], axis=0))
                    pair_labels.append(c)
                    gaps.append(g)
            continue
        for i in range(n if loops else n - K + 1):
            pairs.append(np.concatenate([a[(i + k) % n] for k in range(K)], axis=0))
            pair_labels.append(c)
            gaps.append(K - 1)
    pairs, pair_labels, gaps = sets[False]
    if not pairs:
        raise SystemExit(f"no animation has {K} frames")
    pairs = np.stack(pairs)

    out = Path(args.out)
    order = "inbetween" if args.inbetween else "next"
    V = len(palette) + 1
    write_set(out, pairs, pair_labels, gaps, palette, classes, K, order)
    shutil.rmtree(out / "val", ignore_errors=True)  # a stale split would not match this one
    vpairs, vlabels, vgaps = sets[True]
    if vpairs:
        write_set(out / "val", np.stack(vpairs), vlabels, vgaps, palette, classes, K, order)
    held = {u for u in units if held_out(u)}
    print(f"validation: {len(vpairs)} examples from {len(held)} of {len(set(units))} units -> {out}/val/")

    # Preview: 12 random examples, each shown as [context ... | target] side by side.
    pick = random.Random(args.seed).sample(range(len(pairs)), min(12, len(pairs)))
    sheet = Image.new("RGBA", (4 * (K * S + 6), 3 * (S + 4)), (40, 40, 48, 255))
    for j, i in enumerate(pick):
        rgba = render(pairs[i], palette)
        x, y = (j % 4) * (K * S + 6), (j // 4) * (S + 4)
        for k in range(K):
            sheet.alpha_composite(Image.fromarray(rgba[k * S:(k + 1) * S]), (x + 2 + k * (S + 2), y + 2))
    sheet.resize((sheet.width * 3, sheet.height * 3), Image.NEAREST).save(out / "preview.png")

    per_class = np.bincount(pair_labels, minlength=len(classes))
    print(f"{len(pairs)} examples of {K} frames, {V} palette entries, {len(classes)} classes -> {out}/")
    print("  " + ", ".join(f"{c}: {n}" for c, n in zip(classes, per_class)))


if __name__ == "__main__":
    main()
