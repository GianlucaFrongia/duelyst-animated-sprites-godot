"""Generate a small procedural dataset of looping animations so the pipeline runs end to end.

Each animation is 8 frames of 64x64 (a 32x32 sprite placed bottom-center): slimes squash,
coins spin, potions bubble and swords bob. Writes <out>/<class>/<NNNNN>/<frame>.png, the
layout prepare_animations.py reads with --frames.

    python tools/make_toy_dataset.py --out sprites --per-class 200
"""
import argparse
import math
import random
from pathlib import Path

from PIL import Image

# A 16-color palette (index 0 is unused here; transparency is handled via alpha).
PALETTE = [
    (0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
    (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
    (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
    (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170),
]
OUTLINE = PALETTE[1]


def new_canvas():
    return [[None] * 32 for _ in range(32)]


def outline(px):
    """Add a 1-pixel dark outline around every filled pixel."""
    out = [row[:] for row in px]
    for y in range(32):
        for x in range(32):
            if px[y][x] is None:
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 32 and 0 <= nx < 32 and px[ny][nx] is not None:
                        out[y][x] = OUTLINE
                        break
    return out


def slime(rng, ph):
    px = new_canvas()
    body = rng.choice([PALETTE[i] for i in (3, 8, 11, 12, 14)])
    hi = PALETTE[7]
    cx, base = 16 + rng.randint(-2, 2), 26 + rng.randint(-1, 2)
    w, h = rng.randint(8, 12), rng.randint(8, 13)
    squash = math.sin(2 * math.pi * ph)  # squash and stretch
    w, h = round(w * (1 - 0.12 * squash)), round(h * (1 + 0.2 * squash))
    for y in range(32):
        for x in range(32):
            dy = (base - y) / h
            dx = (x - cx) / w
            if 0 <= dy <= 1 and dx * dx + dy * dy <= 1:
                px[y][x] = body
    ey = base - h // 2
    for ex in (cx - 3, cx + 2):
        px[ey][ex] = PALETTE[0]
        px[ey - 1][ex] = PALETTE[0]
    px[base - h + 3][cx - w // 2 + 2] = hi
    return outline(px)


def potion(rng, ph):
    px = new_canvas()
    liquid = rng.choice([PALETTE[i] for i in (8, 9, 11, 12, 14, 2)])
    glass = PALETTE[6]
    cx, r = 16 + rng.randint(-1, 1), rng.randint(6, 9)
    cy = 22
    for y in range(32):
        for x in range(32):
            d = (x - cx) ** 2 + (y - cy) ** 2
            if d <= r * r:
                px[y][x] = liquid if y >= cy - r // 3 else glass
    neck_h = rng.randint(4, 6)
    for y in range(cy - r - neck_h, cy - r + 1):
        for x in range(cx - 2, cx + 2):
            px[y][x] = glass
    for x in range(cx - 3, cx + 3):
        px[cy - r - neck_h - 1][x] = PALETTE[4]
    px[cy - 2][cx - r + 2] = PALETTE[7]
    by = cy + r - 2 - int(ph * (r + r // 3))  # a bubble rising through the liquid
    bx = cx + rng.randint(-r // 2, r // 2)
    px[by][bx] = PALETTE[7]
    return outline(px)


def sword(rng, ph):
    px = new_canvas()
    blade = rng.choice([PALETTE[6], PALETTE[7], PALETTE[12]])
    hilt = rng.choice([PALETTE[4], PALETTE[9], PALETTE[2]])
    length = rng.randint(14, 19)
    # Diagonal from bottom-left to top-right.
    sx, sy = 7 + rng.randint(-1, 1), 24 + rng.randint(-1, 1)
    sy += round(2 * math.sin(2 * math.pi * ph))  # hovering bob
    for i in range(length):
        x, y = sx + i, sy - i
        if 0 <= x < 31 and 1 <= y < 32:
            px[y][x] = blade
            px[y - 1][x] = blade
    for i in range(-3, 4):
        x, y = sx + i, sy + i
        if 0 <= x < 32 and 0 <= y < 32:
            px[y][x] = hilt
    for i in range(1, 4):
        x, y = sx - i, sy + i
        if 0 <= x < 32 and 0 <= y < 32:
            px[y][x] = PALETTE[4]
    return outline(px)


def coin(rng, ph):
    px = new_canvas()
    face = rng.choice([PALETTE[10], PALETTE[9], PALETTE[6]])
    rim = PALETTE[4] if face != PALETTE[6] else PALETTE[5]
    cx, cy = 16 + rng.randint(-2, 2), 16 + rng.randint(-2, 2)
    rx, ry = rng.randint(4, 10), rng.randint(9, 11)
    rx = max(1, round(rx * abs(math.cos(math.pi * ph))))  # spinning
    for y in range(32):
        for x in range(32):
            v = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
            if v <= 1:
                px[y][x] = face if v <= 0.6 else rim
    for y in range(cy - ry // 2, cy + ry // 2):
        px[y][cx] = rim
    return outline(px)


CLASSES = {"slime": slime, "potion": potion, "sword": sword, "coin": coin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sprites")
    ap.add_argument("--per-class", type=int, default=200, help="animations per class")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for name, fn in CLASSES.items():
        for i in range(args.per_class):
            d = Path(args.out) / name / f"{i:05d}"
            d.mkdir(parents=True, exist_ok=True)
            look = rng.random()  # same random look for every frame of this animation
            for f in range(args.frames):
                px = fn(random.Random(look), f / args.frames)
                img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                for y in range(32):
                    for x in range(32):
                        if px[y][x] is not None:
                            img.putpixel((16 + x, 32 + y), px[y][x] + (255,))
                img.save(d / f"{f:03d}.png")
    print(f"wrote {args.per_class * len(CLASSES)} animations of {args.frames} frames to {args.out}/")


if __name__ == "__main__":
    main()
