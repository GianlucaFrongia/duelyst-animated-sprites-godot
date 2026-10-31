"""Export every Sprite in a folder of Unity Addressables bundles as <out>/<bundle>/<name>.png.

Needs UnityPy, which is not in requirements.txt:
    uv run --python 3.12 --with UnityPy python -I tools/export_unity_sprites.py <aa/StandaloneOSX> raw/songsofconquest
"""
import os, sys, glob, collections
import UnityPy
src, out = sys.argv[1], sys.argv[2]
stats = collections.Counter()
for f in sorted(glob.glob(src + "/*.bundle")):
    name = os.path.basename(f).replace("_assets_all.bundle", "").replace(".bundle", "")
    d = os.path.join(out, name)
    seen = collections.Counter()
    for o in UnityPy.load(f).objects:
        if o.type.name != "Sprite": continue
        try:
            s = o.read(); img = s.image
        except Exception:
            stats["read_error"] += 1; continue
        base = s.m_Name.replace("/", "_")
        seen[base] += 1
        fn = base if seen[base] == 1 else f"{base}_{o.path_id & 0xffffffff:08x}"
        os.makedirs(d, exist_ok=True)
        img.save(os.path.join(d, fn + ".png"))
        stats["sprites"] += 1
    print(name, flush=True)
print(dict(stats))
