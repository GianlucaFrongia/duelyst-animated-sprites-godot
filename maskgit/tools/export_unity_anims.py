"""Export a Unity game's sprite AnimationClips as <out>/<class>/<clip>_<id>/<NNN>.png, aligned on the sprite pivot.

Needs UnityPy, which is not in requirements.txt:
    uv run --python 3.12 --with UnityPy python -I tools/export_unity_anims.py <bundle> raw/hullrupture
"""
import os, re, sys, json, collections
import UnityPy
from PIL import Image
bundle, out = sys.argv[1], sys.argv[2]
CLASSES = {"idle": "idle", "run": "run", "attack": "attack", "death": "death", "hit": "hit", "hurt": "hit", "breathing": "breathing"}
env = UnityPy.load(bundle)
stats = collections.Counter(); seen = set(); report = []
for o in env.objects:
    if o.type.name != "AnimationClip": continue
    clip = o.read()
    m = re.match(r"^([a-z]+)(?:-\d+)?(?:_\d+)?$", clip.m_Name)
    cls = CLASSES.get(m.group(1)) if m else None
    if not cls: stats["clip_other_class"] += 1; continue
    b = clip.m_ClipBindingConstant
    pptr = [g for g in b.genericBindings if g.isPPtrCurve]
    if len(pptr) != 1: stats["clip_multi_or_no_sprite_curve"] += 1; continue
    keys = tuple(p.path_id for p in b.pptrCurveMapping)
    if len(keys) < 2: stats["clip_single_frame"] += 1; continue
    if keys in seen: stats["clip_duplicate"] += 1; continue
    seen.add(keys)
    frames = []
    try:
        for p in b.pptrCurveMapping:
            s = p.read(); img = s.image.convert("RGBA")
            w, h = img.size
            px, py = round(s.m_Pivot.x * s.m_Rect.width), round(s.m_Pivot.y * s.m_Rect.height)
            frames.append((img, px, py, w, h))
    except Exception as e:
        stats["clip_read_error"] += 1; continue
    left = max(px for _, px, _, _, _ in frames); right = max(w - px for _, px, _, w, _ in frames)
    top = max(h - py for _, _, py, _, h in frames); bottom = max(py for _, _, py, _, _ in frames)
    W, H = left + right, top + bottom
    if W > 2048 or H > 2048: stats["clip_bad_pivot"] += 1; continue
    d = os.path.join(out, cls, f"{clip.m_Name}_{o.path_id & 0xffffffff:08x}")
    os.makedirs(d, exist_ok=True)
    for i, (img, px, py, w, h) in enumerate(frames):
        c = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        c.paste(img, (left - px, top - (h - py)))
        c.save(os.path.join(d, f"{i:03d}.png"))
    stats[f"kept_{cls}"] += 1
    report.append((cls, clip.m_Name, len(frames), W, H))
print(json.dumps(stats, indent=1))
sz = collections.Counter((c, "<=64" if W <= 64 and H <= 64 else ">64") for c, _, _, W, H in report)
print(sorted(sz.items()))
