"""Summarizes the train_<name> stages of one or more psg runs: speed, examples seen and
held-out validation loss, plus the sample grids of each model stacked into one image.

    python tools/compare_runs.py out/compare runs/compare_archs [runs/more ...]

Reads runs/<run>/pipeline for the stage order and <i>.log for each stage's output; a stage
that ran in several runs is taken from the last run that finished it. Sample
grids are taken from <out>/<name>/grid.png (tools/render.py). Writes <out>/summary.md and
<out>/samples.png, and prints the summary.
"""

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

STEP = re.compile(r"^step (\d+) / (\d+) \| loss ([\d.]+) .*\| ([\d.]+) pairs/s")
VAL = re.compile(r"^val step (\d+) \| loss with context ([\d.]+) \| loss from scratch ([\d.]+)")
BATCH = re.compile(r"\| batch (\d+)")


def stage_names(run: Path) -> list[str]:
    return [line.split(None, 1)[1].strip() for line in (run / "pipeline").read_text().splitlines()
            if line.strip().startswith("stage ")]


def parse(log: str) -> dict:
    r = {"step": 0, "loss": None, "pairs": None, "batch": None, "val": []}
    for line in log.splitlines():
        if m := STEP.match(line):
            r["step"], r["loss"], r["pairs"] = int(m[1]), float(m[3]), float(m[4])
        elif m := VAL.match(line):
            r["val"].append((int(m[1]), float(m[2]), float(m[3])))
        elif r["batch"] is None and (m := BATCH.search(line)):
            r["batch"] = int(m[1])
    return r


def minutes(run: Path, i: int) -> float | None:
    try:
        return (int((run / f"{i}.end").read_text()) - int((run / f"{i}.start").read_text())) / 60
    except (OSError, ValueError):
        return None


def main() -> None:
    out, runs = Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]]
    found = {}
    for run in runs:
        for i, name in enumerate(stage_names(run)):
            if name.startswith("train_") and (run / f"{i}.end").exists():
                found[name[len("train_"):]] = (parse((run / f"{i}.log").read_text()), minutes(run, i))
    if not found:
        sys.exit(f"no finished train_* stages in {', '.join(map(str, runs))}")
    rows = [(name, r, mins) for name, (r, mins) in found.items()]

    lines = ["| model | steps | batch | examples seen | pairs/s | minutes | train loss | val loss (context) | best | val loss (from scratch) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r, mins in rows:
        last = r["val"][-1] if r["val"] else (None, None, None)
        best = min((v[1] for v in r["val"]), default=None)
        f = lambda x, p=4: "-" if x is None else f"{x:.{p}f}"
        seen = r["step"] * r["batch"] if r["batch"] else None
        lines.append(f"| {name} | {r['step']} | {r['batch'] or '-'} | {seen or '-'} | {f(r['pairs'], 1)} | "
                     f"{f(mins, 1)} | {f(r['loss'])} | {f(last[1])} | {f(best)} | {f(last[2])} |")
    lines += ["", "Validation loss (with context / from scratch) by step:", ""]
    for name, r, _ in rows:
        lines.append(f"- **{name}**: " + ", ".join(f"{s}: {c:.3f}/{z:.3f}" for s, c, z in r["val"]))
    summary = "\n".join(lines) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(summary)
    print(summary)

    grids = [(name, out / name / "grid.png") for name, _, _ in rows if (out / name / "grid.png").exists()]
    if grids:
        imgs = [(name, Image.open(p).convert("RGB")) for name, p in grids]
        label = 24
        w = max(im.width for _, im in imgs)
        sheet = Image.new("RGB", (w, sum(im.height + label for _, im in imgs)), "white")
        y = 0
        for name, im in imgs:
            ImageDraw.Draw(sheet).text((6, y + 6), name, fill="black")
            sheet.paste(im, (0, y + label))
            y += im.height + label
        sheet.save(out / "samples.png")
        print(f"samples of {len(imgs)} models -> {out / 'samples.png'}")


if __name__ == "__main__":
    main()
