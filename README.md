# pixel-sprite-gen

A small generator for **animated** 64×64 pixel-art sprite sheets, with a **hand-written training pipeline in Mojo**: no ML framework, no autograd. Every forward pass has a matching backward pass written by hand, checked against finite differences.

The model is a MaskGIT-style masked transformer that draws one frame at a time. It sees the previous frame (or two) and paints the next one. Several model architectures live side by side in `models/` (see [Model architectures](#model-architectures)). Frames are grids of palette indices (index 0 = transparent), so outputs are always crisp pixels that use only your palette.

## Setup

Targets AMD GPUs on Linux with ROCm (developed against an RX 6800 XT, `gfx1030`), with Mojo 1.2.0 and MAX 26.7.0 (MAX provides Mojo's GPU support). [docs/amd_gpu.md](docs/amd_gpu.md) covers the driver setup and a first speed check. When `librocblas.so` loads, rocBLAS does the matmuls; otherwise the Mojo kernels do.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mojo --version
```

`tools/build.sh` compiles for the GPU `rocminfo` reports; set `ACCEL=gfx…` to build for another one.

## psg: pipelines and progress

`psg` is a small Mojo CLI that runs stages in the background and shows their progress.

```bash
sh tools/build.sh            # builds train_gpu-ctx, sample-ctx, gpu_check-ctx and psg when sources changed
ARCH=ctx2 sh tools/build.sh  # the same for another architecture: train_gpu-ctx2, sample-ctx2, ...

./psg run pipelines/toy.pipe                  # start a pipeline and watch it (Ctrl-C detaches)
./psg train small -- --data data_duelyst --steps 4000 --dim 64   # one training → checkpoints/small
./psg train ib --arch inbetween -- --data data_duelyst_ib          # another architecture (default: ctx,
                                                                   # or the --resume checkpoint's)
./psg status                                  # all runs, once
./psg watch [name]                            # live: progress bar, loss, speed, ETA, loss sparkline
./psg logs <name> [stage]                     # tail of a stage's log
./psg stop <name>
./psg adopt <name> <pid> <log>                # track a train_gpu you started by hand
```

```
duelyst  ▶ running  started 2026-10-24 21:56:49
  train        ██░░░░░░░░░░░░░░░░░░░░░░  10%  step 2100/20000  loss 0.5556  12.4 pairs/s  eta 192 min  ▶ running  23m31s
               loss █▄▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁
```

A pipeline is a text file of stages. Each stage is a shell command run from the project root (see `pipelines/toy.pipe` and `pipelines/duelyst.pipe`, which trains two model sizes back to back):

```
set DATA data_duelyst                 # ${DATA} is substituted below
stage prepare
  cmd .venv/bin/python tools/prepare_animations.py --atlases raw/units --out ${DATA}
  skip_if ${DATA}/meta.txt            # skip when this path exists
stage train
  cmd ./train_gpu-ctx --data ${DATA} --out checkpoints/small --steps 4000
  gpu yes                             # automatic for ./train_gpu-*, ./sample-*, ./gpu_check-*, ./grpo-*
```

How runs behave:
- Stages run in order, and a failing stage stops its pipeline.
- Runs are detached, so closing the terminal or pressing Ctrl-C in `watch` doesn't stop them.
- GPU stages from different runs wait for each other, so you can start several pipelines and they queue up.
- Everything about a run lives in `runs/<name>/`: the pipeline it ran, per-stage logs, pids, exit codes and timings.
- Progress is parsed from lines like `step 900 / 20000 | loss 0.85 | ... | eta 204 min`. Any command that prints `<word> N / M` gets a progress bar.

## Quick start (toy dataset)

```bash
# 1. Make 800 procedural looping animations (slimes squash, coins spin, potions bubble, swords bob)
python tools/make_toy_dataset.py --out sprites --per-class 200

# 2. Convert frames into the binary frame-pair format
python tools/prepare_animations.py --frames sprites --out data --looping coin,potion,slime,sword
xdg-open data/preview.png   # pairs rendered back from the binary data: check them

# 3. Build, then check the hand-written gradients and the GPU kernels against the CPU reference
sh tools/build.sh
sh tools/build.sh gradcheck && ./gradcheck-ctx
./gpu_check-ctx

# 4. Train on the GPU
./train_gpu-ctx --data data --out checkpoints/toy --steps 2000

# 5. Generate sprite sheets and render them
./sample-ctx --ckpt checkpoints/toy --data data --class slime --n 4 --frames 8
python tools/render.py --samples out/samples.bin --palette data/palette.txt --out out
xdg-open out/anim_000.gif
```

## Training on Duelyst

The Duelyst unit animations (idle, breathing, run, attack, hit, death) are TexturePacker atlases: a `.png` sheet plus a `.plist` with each frame's rectangle. The plist is needed to cut the frames out.

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/tnantoka/duelyst-assets.git raw/duelyst
git -C raw/duelyst sparse-checkout set --no-cone \
  'public/duelyst-1.97.12/app/resources/units/*.png' 'public/duelyst-1.97.12/app/resources/units/*.plist'

python tools/prepare_animations.py --atlases raw/duelyst/public/duelyst-1.97.12/app/resources/units --out data_duelyst
./train_gpu-ctx --data data_duelyst --out checkpoints/duelyst --steps 20000 --dim 128 --layers 4 --lr 6e-4 --warmup 500
./sample-ctx --ckpt checkpoints/duelyst --data data_duelyst --class idle --n 4 --frames 10
python tools/render.py --samples out/samples.bin --palette data_duelyst/palette.txt --out out
```

This keeps 2,365 of the 4,122 animations: those whose motion fits in 64×64 at the art's native pixel size. That's 24,013 frame pairs and a shared 64-color palette quantized from 2,507 colors. The class is the animation type, so `--class run` asks for a run cycle.

## Training on LPC characters

The animated pixel-art source most arXiv sprite papers use is the Liberated Pixel Cup (LPC) spritesheet set: the "Sprites" dataset of DSVAE ([1803.02991](https://arxiv.org/abs/1803.02991)), Visual Dynamics ([1807.09245](https://arxiv.org/abs/1807.09245)), MDI-GAN ([2409.10721](https://arxiv.org/abs/2409.10721)) and the disentanglement work after them. `tools/make_lpc_dataset.py` composes random characters (body, head, hair, clothes, shoes, headwear, weapon, with palette recolors) from the Universal LPC generator. It writes their right-facing animations as 64×64 frames: idle, combat idle (`breathing`), walk, run, hurt (`death`, facing down) and six attacks (slash, backslash, halfslash, thrust, spellcast, shoot).

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator.git raw/lpc
pats=(); for a in idle combat_idle walk run slash backslash halfslash thrust spellcast shoot hurt; do
  pats+=("/spritesheets/**/$a.png" "/spritesheets/**/$a/*.png"); done
git -C raw/lpc sparse-checkout set --no-cone /sheet_definitions/ /palette_definitions/ /CREDITS.csv "${pats[@]}"   # ~300 MB

python tools/make_lpc_dataset.py --lpc raw/lpc --out raw/lpc_frames --n 2000   # ~35 min
python tools/prepare_animations.py --frames raw/lpc_frames --out data_lpc \
  --classes attack,breathing,death,idle,run,walk --looping breathing,idle,run,walk
```

`walk` is a new class; leave it out of `--classes` to keep the Duelyst class set. To train on it together with the other sources, add it to the merge: `sh tools/merge_frames.sh raw/frames_all raw/hullrupture raw/songsofconquest_frames raw/lpc_frames`. The art is CC-BY-SA 3.0 / GPL 3.0 / OGA-BY per layer (`raw/lpc/CREDITS.csv`): credit the artists for anything you release.

## Using your own sprites

Put frames in one folder per animation, grouped by class. Frames play in filename order.

```
sprites/
  idle/    knight/  000.png 001.png ...     (any size; each animation is cropped together)
           slime/   000.png 001.png ...
  attack/  knight/  ...
```

Then run `python tools/prepare_animations.py --frames sprites --out data`. Every frame of an animation is cropped with the same offset (the union bounding box, bottom-center), so motion is kept. Animations bigger than 64×64 are skipped rather than resized, because resizing pixel art ruins it. Classes in `--looping` (default `breathing,idle,run`) also get a last→first pair, so generated cycles can loop.

## Model architectures

Each folder in `models/` is one architecture, and all it holds is `arch/variant.mojo`: a handful of compile-time settings (how many context frames the canvas holds, whether they are extra tokens or embedding slots, the dataset's frame order, the attention band, layer norm or DyT, row or Hilbert token order). Everything else is shared and reads those settings: the model, its CPU reference forward/backward and checkpoints (`src/model.mojo`), the batched GPU engine (`src/engine.mojo`) and fused attention (`src/attention.mojo`), and the entry points `train_gpu`, `sample`, `grpo`, `gradcheck` and `gpu_check`, which are compiled against one architecture with `-I models/<name>` into `./<tool>-<name>`.

| Architecture | Context | Tokens | Dataset |
|---|---|---|---|
| `stacked` | previous frame, as 1,024 extra tokens | 2,048 | pairs |
| `ctx` | previous frame, added into each target token | 1,024 | pairs |
| `ctx2` | two previous frames, added into each target token | 1,024 | triples (`prepare_animations.py --context 2`) |
| `inbetween` | two keyframes; predicts the frame halfway between | 1,024 | keyframe triples (`prepare_animations.py --inbetween`) |
| `sparse` | `stacked` with band-sparse attention: each 4-pixel-row strip sees only the strips within 8 pixels, in both frames (about a third of the attention work) | 2,048 | pairs |
| `inbetween_sparse` | `inbetween` with the same band-sparse attention | 1,024 | keyframe triples |
| `inbetween_dyt` | `inbetween_sparse` with DyT (Dynamic Tanh) in place of every layer norm | 1,024 | keyframe triples |
| `inbetween_hilbert` | `inbetween_sparse` with tokens in Hilbert-curve order, so a band reaches as far up and down as sideways | 1,024 | keyframe triples |
| `inbetween_dyt_hilbert` | `inbetween_hilbert` with DyT in place of every layer norm | 1,024 | keyframe triples |
| `inbetween_crop` | `inbetween` running only the target tokens within 3 tokens of either keyframe's sprite (the rest of the frame is transparent), batches grouped by that count | ~570 on average (64–1,024) | keyframe triples |

`ctx` keeps detail much better than `stacked` but mostly copies the previous frame: from one frame it can't tell which way a unit is moving. `ctx2` sees two frames and moves more, but still far less than real animations. `inbetween` sidesteps the ambiguity: you give the two ends of the motion (keyframes) and it draws the frames between, the way an animator's in-betweener does. A 9-frame sheet is filled by halving gaps: frame 4 from frames 0 and 8, then 2 and 6, then the odd frames:

```bash
./sample-inbetween --ckpt checkpoints/duelyst_hr_inbetween --data data_duelyst_hr_ib --class idle --init 18250 --frames 9
```

`--init` picks a dataset example whose two keyframes become the sheet's first and last frame; `--frames` must be 3, 5, 9 or 17. To use **your own keyframes**, turn the PNGs into a one-example dataset with the training palette, then start from it:

```bash
python tools/keyframes.py --like data_duelyst_hr_ib --out keys/knight first.png last.png
./sample-inbetween --ckpt checkpoints/duelyst_hr_inbetween --data keys/knight --init 0 --class idle --frames 9 \
  --temperature 1.0 --choice-noise 10
```

Use `--temperature 1.0 --choice-noise 10` for in-betweening: with the defaults the sampler commits "transparent" first wherever it is unsure, and thin details (flames, chains) vanish from the middle frames.

The PNGs are cropped together at native scale (they must fit in 64×64 together) and mapped to the nearest palette colors; `keys/knight/keys_preview.png` shows the result. The same tool works for `ctx` (one PNG) and `ctx2` (two, oldest first) with their datasets as `--like`. Datasets record their frame order (`O next` or `O inbetween` in `meta.txt`), and each architecture checks it.

Checkpoints record their architecture (`A ctx` in the `.cfg`), and loading one with another architecture's binary stops with a message saying which build to use.

**Adding an architecture:** `cp -r models/ctx models/myidea`, set `comptime ARCH = "myidea"` and the other settings in its `arch/variant.mojo`, and `ARCH=myidea sh tools/build.sh` builds `train_gpu-myidea`, `sample-myidea`, ... Run `ARCH=myidea sh tools/build.sh gradcheck gpu_check`, then `./gradcheck-myidea` and `./gpu_check-myidea`, before training. An idea that needs more than a new combination of settings is a new setting in `variant.mojo` and a `comptime if` in the shared code.

## How it works (ctx)

Each training example is a **pair of consecutive frames**: the previous frame as context and the next frame as the target. The target frame is split into 1,024 tokens of 2×2 pixel patches. Each token's embedding is the sum of eight sub-pixel embeddings (its 4 target pixels, which may be MASK, and the 4 *context* pixels at the same spot), a position embedding and a class embedding. The context is not a second set of tokens: putting it inside each token makes "keep what was here last frame" the easy default, and attention still reaches context at other positions through the tokens there. It also keeps attention at 1,024 tokens instead of 2,048.

During training, a random fraction (cosine schedule) of the *target* frame's tokens is replaced by a MASK token. A bidirectional transformer predicts the 4 pixels of every masked token, with cross-entropy loss on masked tokens only. The context frame is never masked. Two kinds of dropout help at sampling time:
- **Context dropout** (`--context-drop 0.15`): the whole context frame is replaced by MASK, which teaches the model to draw a first frame from nothing.
- **Class dropout** (`--cfg-drop 0.1`): the class is dropped, which enables classifier-free guidance.

Sampling a sheet:
1. Start with an empty (all-MASK) context and draw frame 0. On each of ~12 steps the model proposes pixels for every masked token, keeps the most confident ones and re-masks the rest.
2. Move frame 0 into the context slot and draw frame 1.
3. Repeat for as many frames as you ask for.

| File | What it does |
|---|---|
| `src/kernels.mojo` | SIMD matmuls (3 layouts), layer norm, GELU, softmax: forward and backward (CPU reference) |
| `src/gpu.mojo` | The GPU device: memory arena, matmuls (tiled, split-K, or rocBLAS), linear layers, layer norm / DyT, GELU, reductions; the same kernels as CPU loops without a GPU |
| `src/gpu_kernels.mojo` | The kernels `src/gpu.mojo` launches, each a per-thread body shared with the CPU path |
| `src/optim.mojo` | Gradient clipping, AdamW, Muon (on `Model.matrices()`), the learning-rate schedule |
| `src/data.mojo` | Example loader, target-frame masking and context dropout, CLI helpers |
| `src/model.mojo` | Canvas geometry from `arch/variant.mojo`, flat parameter buffer, CPU reference forward, hand-written backward, checkpoints |
| `src/engine.mojo` | The batched GPU forward/backward: embeddings, loss, buffer layout, layer schedule |
| `src/attention.mojo` | Fused (flash-style) attention kernels for every architecture: online softmax forward, recomputing backward, attention band |
| `models/<arch>/arch/variant.mojo` | One architecture's settings |
| `train_gpu.mojo` | GPU training loop: warmup + cosine (or `--decay`) LR, gradient clipping, logging, checkpoints |
| `sample.mojo` | Frame-by-frame sheet generation with temperature and classifier-free guidance (GPU, or `--cpu`) |
| `grpo.mojo` | GRPO fine-tuning: group-normalized rewards of sampled frames, policy gradient through the masked loss |
| `gradcheck.mojo` | Finite-difference check of every parameter tensor |
| `gpu_check.mojo` | Compares GPU logits, loss, gradients and an AdamW step against the CPU model, for a small config and a wide one that exercises the tiled matmuls and chunked embedding gradient |
| `tools/prepare_animations.py` | Atlases or frame folders → palette-indexed frame pairs (or triples with `--context 2`), plus a preview |
| `tools/keyframes.py` | Your own keyframe PNGs → a one-example dataset (training palette and crop) for `sample --init 0` |
| `tools/render.py` | Samples → `sheet_NNN.png` (frames side by side), `anim_NNN.gif`, `grid.png` |
| `tools/make_toy_dataset.py` | Procedural looping toy animations |
| `psg.mojo`, `src/pipeline.mojo` | The pipeline CLI: detached workers, GPU queueing, progress parsing, live view |
| `tools/build.sh`, `pipelines/*.pipe` | Incremental builds (safe while training runs) and example pipelines |

## Useful flags

`train_gpu`: `--dim 64 --layers 4 --heads 4 --mlp 256 --batch 8 --lr 1e-3 --warmup 100 --steps 2000 --cfg-drop 0.1 --context-drop 0.15 --history-drop 0.1 --flip 0.0 --resume checkpoints/x`

`--decay 0.2` replaces the cosine schedule with warmup-stable-decay (DeepSeek LLM): `--lr` holds after warmup, then falls linearly to a tenth over the last 20% of the steps.

`--history-drop` (architectures with two context frames) hides the older one, so the model also learns the second frame of a sheet, when only one frame of history exists. The fused attention needs `dim / heads` of 16 or 32 in every architecture. `--cpu` runs the same kernels as CPU loops (slow; for debugging).

`--flip` mirrors pairs with that probability. Duelyst units all face right, so it's off by default.

**Validation:** `prepare_animations.py --val-frac 0.05` (the default) holds out 5% of units, with all their animations, in `<data>/val/`. `train_gpu` then logs `val step N | loss with context X | loss from scratch Y` every `--val-every` steps (default: `--save-every`), on `--val-n 504` held-out examples with fixed masks. The second number is the one that tracks drawing a first frame; the training loss mostly reflects continuing one.

`sample`: `--class <name|id|none> --n 4 --frames 8 --batch 4 --steps 12 --temperature 0.7 --guidance 2.0 --seed 0 --cpu --init <example>`

`grpo`: `--ckpt checkpoints/x --data <dir> --out checkpoints/x_grpo --iters 200 --prompts 8 --group 4 --lr 1e-5 --eval-every 25`. It fine-tunes a trained model with GRPO, reinforcement learning on the real sampler's output: it draws `--group` next frames per training example and rewards agreement with the real frame on non-transparent pixels. It logs that reward on held-out units every `--eval-every` iterations. Build it with `sh tools/build.sh grpo`.

`--init` opens every sheet with a dataset example's context frames (one for `ctx`, two for `ctx2`) instead of drawing them, and the model continues the animation. Drawing frame 0 from nothing tends to collapse to an empty frame, so this gives much better sheets.

## Speed and memory

Attention is fused in every architecture (it never stores the T×T matrices, including `stacked`'s 2,048×2,048), so memory is mostly the per-layer activations: `train_gpu` prints its buffer size at startup. Bigger batches keep the GPU busier; with a bigger batch, use fewer steps for the same number of examples and a higher learning rate (see `pipelines/duelyst_hr_ctx2.pipe`). The step counts in `pipelines/` were sized from speeds measured on earlier hardware: re-measure (docs/amd_gpu.md, step 4) before planning long runs.

## Limitations and next steps

- **Drift:** frames are chained, so small errors add up over long sheets, and a cycle won't close perfectly on its last frame. Conditioning on the sheet's first frame as well would anchor the unit's identity: one more embedding slot group per token, like `ctx2`'s second frame.
- **Big animations are skipped:** about 40% of Duelyst animations (mostly attacks with lunges) don't fit in 64×64. An 80×80 or 96×96 frame would keep them, at a steep attention cost.
- **Speed:** attention backward is the largest cost of a step. The kernels in `src/attention.mojo` and the tiled matmul in `src/gpu_kernels.mojo` are generic; tuning them for RDNA (tile sizes, wave32 shuffles) is the next step.
- **Token-embedding gradient:** the old kernel (`k_embed_bwd_tok`, one thread per entry looping over every row) silently left many entries at zero at large batches, different ones on every run. The engine now builds one-hot rows in chunks that fit the split-K scratch and multiplies them with the activations' gradient (`k_onehot` + the tiled matmul); the old kernel remains only as the fallback when that scratch is too small for one 64-row chunk.
- Checkpoints from the old 32×32 still-sprite model don't load in this version. The pre-animation source is in `docs/pixel-sprite-gen.zip`.
