"""Group Unity sprites named <Unit>_Default_<n>_<Action>_<frame> into animations, aligned on the sprite pivot.

Writes <out>/<class>/<bundle>.<Unit>_<Action>/<NNN>.png, ready for prepare_animations.py --frames.
Collision, Emission and Colors layers and the static art (UI, portraits, buildings) have no frames
to pair, so they are left out.

    uv run --python 3.12 --with UnityPy python -I tools/export_unity_frames.py <aa/StandaloneOSX> raw/songsofconquest_frames
"""
import os, re, sys, glob, json, collections
import UnityPy
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
ACTION = {"Idle": "idle", "Move": "run", "MeleeAttack": "attack", "RangedAttack": "attack", "Attack": "attack",
          "Melee": "attack", "Hit": "hit", "Death": "death"}
NAME = re.compile(r"^(.+?)_Default_\d+_([A-Za-z]+)_(\d+)$")
stats = collections.Counter(); sizes = collections.Counter()
for f in sorted(glob.glob(src + "/*.bundle")):
    bundle = os.path.basename(f).replace("_assets_all.bundle", "").replace(".bundle", "")
    groups = collections.defaultdict(list)
    for o in UnityPy.load(f).objects:
        if o.type.name != "Sprite": continue
        s = o.read()
        m = NAME.match(s.m_Name)
        if not m or m.group(2) not in ACTION:
            stats["sprites_not_animation"] += 1; continue
        groups[(m.group(1), m.group(2))].append((int(m.group(3)), s))
    for (unit, action), items in groups.items():
        items.sort(key=lambda t: t[0])
        if len(items) < 2: stats["single_frame"] += 1; continue
        frames = []
        try:
            for _, s in items:
                img = s.image.convert("RGBA")
                w, h = img.size
                frames.append((img, round(s.m_Pivot.x * w), round(s.m_Pivot.y * h), w, h))
        except Exception:
            stats["read_error"] += 1; continue
        left = max(px for _, px, _, _, _ in frames); right = max(w - px for _, px, _, w, _ in frames)
        top = max(h - py for _, _, py, _, h in frames); bottom = max(py for _, _, py, _, _ in frames)
        W, H = left + right, top + bottom
        if W > 2048 or H > 2048: stats["bad_pivot"] += 1; continue
        cls = ACTION[action]
        d = os.path.join(out, cls, f"{bundle}.{unit}_{action}")
        os.makedirs(d, exist_ok=True)
        for i, (img, px, py, w, h) in enumerate(frames):
            c = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            c.paste(img, (left - px, top - (h - py)))
            c.save(os.path.join(d, f"{i:03d}.png"))
        stats[f"anim_{cls}"] += 1; stats["frames"] += len(frames)
        sizes["<=64" if W <= 64 and H <= 64 else ">64"] += 1
    print(bundle, flush=True)
print(json.dumps(stats, indent=1)); print(dict(sizes))
