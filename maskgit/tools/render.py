"""Render samples.bin (written by sample.mojo) into sprite sheets and GIF previews.

    python tools/render.py --samples out/samples.bin --palette data/palette.txt --out out/

For each sampled animation writes:
    sheet_NNN.png   the frames side by side at true size, with transparency
    anim_NNN.gif    an upscaled looping preview
plus grid.png, all sheets stacked and upscaled.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

BG = (40, 40, 48, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="out/samples.bin")
    ap.add_argument("--palette", default="data/palette.txt")
    ap.add_argument("--out", default="out")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    meta = dict(line.split() for line in Path(args.samples + ".txt").read_text().splitlines() if line)
    F, S = int(meta["frames"]), int(meta["size"])
    pal = np.loadtxt(args.palette, dtype=np.uint8).reshape(-1, 3)
    idx = np.fromfile(args.samples, dtype=np.uint8).reshape(-1, F, S, S)
    rgba = np.zeros(idx.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = pal[np.minimum(idx, len(pal) - 1)]
    rgba[..., 3] = np.where(idx == 0, 0, 255)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = args.scale
    grid = Image.new("RGBA", (F * (S + 2), len(rgba) * (S + 2)), BG)
    for n, frames in enumerate(rgba):
        sheet = Image.new("RGBA", (F * S, S), (0, 0, 0, 0))
        gif = []
        for f, a in enumerate(frames):
            img = Image.fromarray(a, "RGBA")
            sheet.paste(img, (f * S, 0))
            grid.alpha_composite(img, (f * (S + 2) + 1, n * (S + 2) + 1))
            bg = Image.new("RGBA", (S, S), BG)
            bg.alpha_composite(img)
            gif.append(bg.convert("RGB").resize((S * s, S * s), Image.NEAREST))
        sheet.save(out / f"sheet_{n:03d}.png")
        gif[0].save(out / f"anim_{n:03d}.gif", save_all=True, append_images=gif[1:],
                    duration=int(1000 / args.fps), loop=0)
    grid.resize((grid.width * s, grid.height * s), Image.NEAREST).save(out / "grid.png")
    print(f"rendered {len(rgba)} sheets of {F} frames -> {out}/grid.png, sheet_*.png, anim_*.gif")


if __name__ == "__main__":
    main()
