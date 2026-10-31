"""Turn your own keyframe PNGs into a one-example dataset that the sampler can start from.

    python tools/keyframes.py --like data_duelyst_hr_ib --out keys/knight first.png last.png
    ./sample-inbetween --ckpt checkpoints/duelyst_hr_inbetween --data keys/knight --init 0 --class idle --frames 9

The frames are cropped together (union bounding box, placed bottom-center in 64x64, as in
training) and mapped to the --like dataset's palette, so a checkpoint trained on that dataset
can read them. Give one PNG per context frame of that dataset: 2 for inbetween (first and last
keyframe) and ctx2 (oldest first), 1 for ctx and stacked. keys_preview.png shows the frames
after palette mapping.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from prepare_animations import S, fit_animation, render, to_indices  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True, help="dataset whose palette, classes and layout to use")
    ap.add_argument("--out", required=True, help="folder for the one-example dataset")
    ap.add_argument("--alpha-threshold", type=int, default=128)
    ap.add_argument("frames", nargs="+", help="keyframe PNGs, in canvas order")
    args = ap.parse_args()

    like = Path(args.like)
    meta = dict(line.split() for line in (like / "meta.txt").read_text().splitlines() if line)
    K = int(meta.get("K", 2))
    if len(args.frames) != K - 1:
        raise SystemExit(f"{like} examples have {K - 1} context frame(s); got {len(args.frames)} PNG(s)")
    palette = np.loadtxt(like / "palette.txt", dtype=np.int32).reshape(-1, 3)[1:]  # row 0 is transparent

    frames = [np.asarray(Image.open(p).convert("RGBA")) for p in args.frames]
    fitted = fit_animation(frames, args.alpha_threshold)
    if fitted is None:
        raise SystemExit(f"keyframes are empty or larger than {S}x{S} together (no resizing: pixel art "
                         "must stay at its native scale)")
    idx = to_indices(fitted, palette)
    rgb = fitted[..., :3].astype(np.int32)
    op = fitted[..., 3] > 0
    err = np.sqrt(((rgb - palette[np.maximum(idx, 1) - 1]) ** 2).sum(-1))[op]
    example = np.concatenate([*idx, np.zeros((S, S), np.uint8)], axis=0)  # target frame left empty

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    example.astype(np.uint8).tofile(out / "pixels.bin")
    np.zeros(1, np.uint8).tofile(out / "labels.bin")
    np.full(1, K - 1, np.uint8).tofile(out / "gaps.bin")
    for f in ("palette.txt", "classes.txt"):
        shutil.copy(like / f, out / f)
    meta["N"] = "1"
    (out / "meta.txt").write_text("".join(f"{k} {v}\n" for k, v in meta.items()))

    sheet = Image.new("RGBA", ((K - 1) * (S + 2), S), (40, 40, 48, 255))
    for k, frame in enumerate(idx):
        sheet.alpha_composite(Image.fromarray(render(frame, palette)), (k * (S + 2), 0))
    sheet.resize((sheet.width * 4, sheet.height * 4), Image.NEAREST).save(out / "keys_preview.png")
    print(f"{len(args.frames)} keyframe(s) -> {out}/ (use --data {out} --init 0); "
          f"palette color error: mean {err.mean():.1f}, max {err.max():.1f} (0-441)")


if __name__ == "__main__":
    main()
