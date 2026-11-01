"""Random LPC characters -> animated frame folders for prepare_animations.py --frames.

LPC (Liberated Pixel Cup) is the animated pixel-art source behind the "Sprites" dataset used on
arXiv (DSVAE 1803.02991, Visual Dynamics 1807.09245, MDI-GAN 2409.10721, ...). Those papers use a
few hundred recolors of one body; this composes body, head, hair, clothes, shoes and weapons from
the full Universal LPC generator, so characters vary in shape too. Licenses: CC-BY-SA 3.0 / GPL 3.0
/ OGA-BY (per layer, see raw/lpc/CREDITS.csv): credit the authors if you ship anything trained on it.

    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator.git raw/lpc
    git -C raw/lpc sparse-checkout set --no-cone /sheet_definitions/ /palette_definitions/ /CREDITS.csv \
        '/spritesheets/**/walk.png' '/spritesheets/**/walk/*.png'   # ... one pair per animation in ANIMS
    python tools/make_lpc_dataset.py --lpc raw/lpc --out raw/lpc_frames --n 2000
    python tools/prepare_animations.py --frames raw/lpc_frames --out data_lpc \
        --classes attack,breathing,death,idle,run,walk --looping breathing,idle,run,walk

Output: <out>/<class>/lpcNNNNN_<anim without "_">/NNN.png, 64x64 RGBA, right-facing (hurt only exists facing down).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

F = 64
# LPC animation file -> class (the trainer's class names; walk is new)
ANIMS = {
    "idle": "idle", "combat_idle": "breathing", "walk": "walk", "run": "run", "hurt": "death",
    "slash": "attack", "backslash": "attack", "halfslash": "attack", "thrust": "attack",
    "spellcast": "attack", "shoot": "attack",
}
RIGHT = 3  # sheet rows: up, left, down, right
BODY_TYPES = ["male", "female", "muscular"]
# slot -> (definition folders, probability the character wears one)
SLOTS = {
    "head": (["head/heads"], 1.0),
    "eyes": (["head/eyes"], 0.05),  # only cyclops eyes; normal eyes come with the head's recolor
    "hair": (["hair"], 0.8),
    "torso": (["torso"], 0.9),
    "legs": (["legs"], 0.9),
    "feet": (["feet"], 0.75),
    "headwear": (["headwear"], 0.25),
    "weapon": (["weapons"], 0.5),
}


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def load_palettes(root):
    """material -> (base colors, {name: colors}) from the ULPC palette files."""
    out = {}
    for meta in (root / "palette_definitions").glob("*/meta_*.json"):
        m = json.loads(meta.read_text())
        f = meta.parent / f"{meta.parent.name}_ulpc.json"
        if m.get("type") != "material" or not f.exists():
            continue
        pals = {k: [hexrgb(c) for c in v] for k, v in json.loads(f.read_text()).items()}
        if m["base"] in pals:
            out[meta.parent.name] = (pals[m["base"]], pals)
    return out


def load_items(root):
    """slot -> [(definition, layers)] usable for at least one body type."""
    items = {s: [] for s in SLOTS}
    for slot, (dirs, _) in SLOTS.items():
        for d in dirs:
            for f in sorted((root / "sheet_definitions" / d).rglob("*.json")):
                if f.name.startswith("meta_"):
                    continue
                dfn = json.loads(f.read_text())
                layers = [v for k, v in sorted(dfn.items()) if k.startswith("layer_")
                          and isinstance(v, dict) and "custom_animation" not in v]
                if layers:
                    items[slot].append((dfn, layers))
    return items


class Sheets:
    def __init__(self, root):
        self.root = root / "spritesheets"
        self.cache = {}

    def get(self, rel):
        if rel not in self.cache:
            p = self.root / rel
            self.cache[rel] = np.asarray(Image.open(p).convert("RGBA")) if p.exists() else None
        return self.cache[rel]


def recolor(img, colors):
    """colors: [(base palette, target palette)]; masks come from the original so maps don't chain."""
    out = img.copy()
    rgb = img[..., :3]
    for src, dst in colors:
        for s, d in zip(src, dst):
            out[(rgb == s).all(-1) & (img[..., 3] > 0), :3] = d
    return out


def pick(item, body, skin, rng, palettes):
    """One worn item: its layers' paths for this body type and a color choice, or None."""
    dfn, layers = item
    paths = [(l.get("zPos", 0), l[body]) for l in layers if isinstance(l.get(body), str) and "${" not in l[body]]
    if not paths:
        return None
    variant, colors = None, None
    if dfn.get("variants"):
        variant = rng.choice(dfn["variants"])
    elif "recolors" in dfn:
        r = dfn["recolors"]
        colors = []
        for spec in [r] if "material" in r else r.values():  # one material, or color_1, color_2, ...
            mat = spec.get("material")
            if mat in palettes:
                base, pals = palettes[mat]
                base = pals.get(spec.get("base"), base)  # an item may be drawn in another base color
                colors.append((base, skin if mat == "body" and dfn.get("match_body_color") else rng.choice(list(pals.values()))))
    return paths, variant, colors


def render(worn, anim, sheets):
    """Composites the worn layers for one animation; returns right-facing frames or None."""
    layers = []
    body_paths = worn[0][0]
    if sheets.get(f"{body_paths[0][1]}{anim}.png") is None:  # e.g. muscular has no backslash/shoot
        return None
    for paths, variant, colors in worn:
        for z, base in paths:
            img = sheets.get(f"{base}{anim}/{variant}.png" if variant else f"{base}{anim}.png")
            if img is not None:
                layers.append((z, img, colors))
    if not layers:
        return None
    shape = layers[0][1].shape  # body is first (zPos 10 sorts it low anyway)
    canvas = np.zeros(shape, dtype=np.float32)
    for _, img, colors in sorted(layers, key=lambda e: e[0]):
        if img.shape != shape:  # oversize or mismatched sheet
            continue
        if colors:
            img = recolor(img, colors)
        a = img[..., 3:4] / 255.0
        canvas[..., :3] = img[..., :3] * a + canvas[..., :3] * (1 - a)
        canvas[..., 3:4] = a * 255 + canvas[..., 3:4] * (1 - a)
    canvas = canvas.astype(np.uint8)
    rows = shape[0] // F
    row = RIGHT if rows >= 4 else 0
    strip = canvas[row * F:(row + 1) * F]
    frames = [strip[:, i * F:(i + 1) * F] for i in range(shape[1] // F)]
    while frames and frames[-1][..., 3].max() == 0:
        frames.pop()
    return frames if len(frames) >= 2 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lpc", default="raw/lpc")
    ap.add_argument("--out", default="raw/lpc_frames")
    ap.add_argument("--n", type=int, default=2000, help="characters (each gives up to 11 animations)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="resume: skip writing characters before this one")
    args = ap.parse_args()
    root, out = Path(args.lpc), Path(args.out)
    rng = random.Random(args.seed)
    palettes = load_palettes(root)
    items = load_items(root)
    body_def = json.loads((root / "sheet_definitions/body/body.json").read_text())
    body_item = (body_def, [body_def["layer_1"]])
    sheets = Sheets(root)
    assert sheets.get(f"{body_def['layer_1']['male']}walk.png") is not None, "walk sheets missing: check the sparse checkout"
    counts = {}
    for n in range(args.n):
        body = rng.choice(BODY_TYPES)
        skin = rng.choice(list(palettes["body"][1].values()))
        worn = [pick(body_item, body, skin, rng, palettes)]
        for slot, (_, p) in SLOTS.items():
            fits = [it for it in items[slot] if isinstance(it[1][0].get(body), str)]
            if fits and rng.random() < p:
                worn.append(pick(rng.choice(fits), body, skin, rng, palettes))
        worn = [w for w in worn if w]
        if n < args.start:  # still drew its choices, so later characters match an uninterrupted run
            continue
        for anim, cls in ANIMS.items():
            frames = render(worn, anim, sheets)
            if frames is None:
                continue
            d = out / cls / f"lpc{n:05d}_{anim.replace('_', '')}"  # one "_": prepare_animations reads the unit before it
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                Image.fromarray(f).save(d / f"{i:03d}.png")
            counts[cls] = counts.get(cls, 0) + 1
        if len(sheets.cache) > 300:  # an RGBA attack sheet is up to ~0.9 MB; there are ~70k of them
            sheets.cache.clear()
        if n % 100 == 0:
            print(f"character {n} / {args.n}", flush=True)
    print("animations per class:", counts)


if __name__ == "__main__":
    main()
