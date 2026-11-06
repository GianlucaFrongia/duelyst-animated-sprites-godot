# Training a pixel-art animation LoRA on the Duelyst sprites

Mojo scripts that turn this repo's sprites into a captioned training set, train a
LoRA with [musubi-tuner](https://github.com/kohya-ss/musubi-tuner), and convert
the generated output back into Godot `SpriteFrames`. The goal is prompts like:

> `dlystsprite, pixel art sprite animation, attack animation, red dragon, fire element, flames, side view, facing right, plain dark gray background`

| Script | What it does |
|---|---|
| `export_dataset.mojo` | Reads every `.tres` + spritesheet and writes Wan video clips, FLUX.2 grid sheets, captions and `metadata.jsonl` |
| `train_lora.mojo` | Checks PyTorch/GPU from Mojo, writes the musubi-tuner dataset config and runs caching + training |
| `postprocess.mojo` | Generated video/sheet → downscaled, palette-quantized, background-removed frames + strip PNG + `.tres` |

Image work and PyTorch run through Mojo's Python interop (`from std.python import Python`).
The training math itself is musubi-tuner's PyTorch code, which Mojo launches.

The `training/` folder has a `.gdignore`, so Godot won't import the dataset. It is
also excluded from Asset Library downloads by `.gitattributes`.

## 1. Setup (Linux + AMD ROCm recommended)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt          # Mojo + Pillow/NumPy/imageio

# PyTorch: AMD (ROCm) build. Check pytorch.org for the current ROCm version.
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
# NVIDIA instead: --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/kohya-ss/musubi-tuner ~/musubi-tuner
pip install -e ~/musubi-tuner
```

AMD notes:
- RDNA2/RDNA3 cards that ROCm doesn't officially list usually work with
  `export HSA_OVERRIDE_GFX_VERSION=10.3.0` (RDNA2) or `11.0.0` (RDNA3).
- The trainer defaults to SDPA attention and plain AdamW (`--optimizer adamw`),
  because xformers and bitsandbytes (`adamw8bit`) are unreliable on ROCm.

## 2. Export the dataset

```bash
mojo run training/export_dataset.mojo            # → training/dataset/
```

Output (about 4,500 animations, ~520 MB, ~12 min):

```
training/dataset/wan/videos/<unit>__<anim>.mp4 + .txt    512x512 clips, 9 fps
training/dataset/flux/grids/<unit>__<anim>.png + .txt    frame grids, ≤1024 px
training/dataset/metadata.jsonl                          one record per animation
```

How frames are prepared:
- Each animation is cropped to the shared bounding box of all its frames, so the
  character stays aligned.
- Frames are upscaled by an **integer** factor with nearest-neighbour, so pixels
  stay square and crisp. The factor is stored as `video_scale` / `grid_scale` in
  `metadata.jsonl`.
- Frames are flattened onto a dark gray background (`--bg 64,64,64`). Holes in
  the art show as dark, like in-game.

Useful options: `--only fire` (only names containing "fire"), `--limit 50`,
`--no-fx`, `--no-video`, `--no-grid`, `--frames` (also dump raw frames),
`--trigger WORD`, `--video-size 512`, `--grid-max 1024`.

### Captions

Captions are built automatically from:
- the trigger word (`dlystsprite`)
- the action (attack / idle / run / hit / death / cast …)
- the unit name and faction
- **element tags guessed from keywords in the name** (`fire`, `ice`, `lightning`,
  `holy light`, `shadow`, `poison`, …)

Effect sprites (`fx_*`) are captioned as visual effects, e.g. "fire slash".

The element tags only look at names, so many units that do breathe fire aren't
tagged. Only ~100 clips get `fire`. For better prompt following, refine the
`.txt` files: by hand, or with a vision model (musubi-tuner includes
`caption_images_by_qwen_vl.py`).

## 3. Train

Start with `--dry-run` to see the config and exact commands without running anything.

### Wan 2.1 T2V 1.3B: animation, fits 16 GB

Download the files listed in musubi-tuner's [docs/wan.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md):
- DiT `wan2.1_t2v_1.3B_bf16.safetensors`
- VAE `wan_2.1_vae.safetensors`
- T5 `models_t5_umt5-xxl-enc-bf16.pth`

```bash
mojo run training/train_lora.mojo --model wan --musubi ~/musubi-tuner \
    --dit models/wan2.1_t2v_1.3B_bf16.safetensors \
    --vae models/wan_2.1_vae.safetensors \
    --t5 models/models_t5_umt5-xxl-enc-bf16.pth
```

A LoRA trained on the T2V 1.3B model can also be loaded into **Wan2.1-VACE-1.3B**
(same base weights) in ComfyUI. There you give it a unit's idle frame as the
reference image to animate.

### FLUX.2 [klein] base 4B: still spritesheet grids

Download the files listed in musubi-tuner's [docs/flux_2.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/flux_2.md):
- `flux2-klein-base-4b.safetensors`
- `ae.safetensors`
- the Qwen3 4B text encoder (pass the first split file)

```bash
mojo run training/train_lora.mojo --model flux2 --musubi ~/musubi-tuner \
    --dit models/flux2-klein-base-4b.safetensors \
    --vae models/ae.safetensors \
    --text-encoder models/qwen3-4b/model-00001-of-00002.safetensors
```

Common options: `--epochs 16`, `--rank 32`, `--lr`, `--resolution`,
`--fp8` (lower VRAM), `--blocks-to-swap N` (offload blocks to CPU RAM),
`--skip-cache` (reuse cached latents/text), `--name`, `--work DIR`.
LoRAs are saved to `training/lora_work/output/`, one per epoch.

## 4. Generate, then post-process

Generate with ComfyUI, or with musubi-tuner's own scripts, for example:

```bash
cd ~/musubi-tuner
python src/musubi_tuner/wan_generate_video.py --task t2v-1.3B \
    --dit ... --vae ... --t5 ... \
    --lora_weight /path/to/training/lora_work/output/duelyst_wan.safetensors \
    --video_size 512 512 --video_length 17 --fps 9 \
    --prompt "dlystsprite, pixel art sprite animation, attack animation, ice golem, fire element, flames, side view, facing right, plain dark gray background" \
    --save_path out/
```

Then convert the result to Godot assets:

```bash
# Video. --factor = the integer upscale used in training (often 4-8 at 512 px).
mojo run training/postprocess.mojo out/result.mp4 --name attack --factor 5

# FLUX grid sheet
mojo run training/postprocess.mojo out/sheet.png --grid 5x5 --factor 2 --name attack
```

This writes the following to `training/generated/`:
- `attack_frames/`
- a horizontal strip `attack.png`
- `attack.tres`, a SpriteFrames resource pointing at `res://generated/attack.png`
  (change the path with `--res-dir`)

Copy the PNG and `.tres` into your Godot project and use them with `AnimatedSprite2D`.

Other options: `--colors 64` (palette size), `--bg` / `--tol` (background key
colour and tolerance), `--fps 9`.

## 5. P-JiT: pixel-space text-to-image (Linum's Pyramid-JiT)

`training/pjit/` trains [P-JiT](https://www.linum.ai/field-notes/pjit-21k), Linum's 2.18B
pixel-space diffusion transformer (code: [Linum-AI/pjit-21k](https://github.com/Linum-AI/pjit-21k),
objective: [Linum-AI/pyramid-jit `loss.py`](https://github.com/Linum-AI/pyramid-jit)). It suits
pixel art well: there is **no VAE**. The model reads raw RGB as 32x32-pixel patches and predicts
the clean image directly (x-prediction), so no autoencoder smears the pixel grid.

| | |
|---|---|
| Trunk | 22 single-stream DiT blocks, width 2,944, 23 heads x 128, SwiGLU, 2 image + 2 text refiner blocks |
| Block | sandwich RMSNorm, low-rank AdaLN (tanh gates), q/k RMSNorm, sigmoid attention gate, 3-D RoPE |
| Text | frozen Qwen3.5-4B, hidden layers 7/15/27 concatenated (7,680 features), 256 tokens |
| Objective | x-prediction, velocity MSE weighted 1/t² (t clamped at 0.1), logit-normal t, noise x2 at 512px / x4 at 1024px, 5 % caption dropout |
| Training-only | pyramid readout heads after blocks 10 and 16 (¼ and ½ resolution), PixelREPA (DINOv3) after block 5, LPIPS, P-DINO |
| Sampling | 50 Euler steps, adaptive projected guidance (APG) at scale 15 |

`objective.py` is Linum's reference objective, adapted to work with smaller models. `train.py` adds
the data pipeline, LoRA, EMA, checkpointing and sampling during training. Linum did not release
an optimizer or training script, so those settings (AdamW, warmup, LR) are this repo's own defaults.

### Setup

```bash
pip install git+https://github.com/Linum-AI/pjit-21k     # model, sampler, Qwen3.5 text encoder
pip install timm lpips                                     # only for --pixel-repa / --pdino / --lpips
```

The first run downloads `Linum-AI/pjit-21k` (8.7 GB) and `Qwen/Qwen3.5-4B` (8 GB) from the Hugging Face Hub.
Captions are encoded once and cached in `<out>/text_cache/`. The encoder is unloaded before training, so it
never shares GPU memory with the DiT. On AMD, attention falls back to PyTorch SDPA automatically
(FlashAttention-3 is Hopper-only).

### On a 16 GB AMD card

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # RDNA3 cards ROCm doesn't list (10.3.0 for RDNA2)
python training/pjit/train.py --dry-run  # prints GPU, data, model and loss plan; downloads nothing
```

What fits in 16 GB:

| Job | GPU memory | 16 GB? |
|---|---|---|
| Caption caching (Qwen3.5-4B, bf16) | ~8 GB, runs alone before the DiT loads | yes |
| LoRA fine-tune (default): bf16 base 4.4 GB + LoRA + checkpointed activations | ~7–10 GB at 1024² | yes |
| Sampling (`sample.py`): prompts encoded first, encoder freed, then the fp32 DiT | ~10 GB | yes |
| Scratch `tiny` / `small` (full AdamW) | ~1 / ~6 GB + activations | yes |
| Scratch `base` (720M) | ~12 GB + activations | tight; `--force` |
| Full fine-tune (`--lora-rank 0`) or scratch `pjit` (2.18B) | 40 GB+ | no; the script stops early |

AMD notes:
- FlashAttention-3 is NVIDIA Hopper only. Attention uses PyTorch SDPA (a warning says so once).
- Sample noise uses the GPU's own `torch.randn`. `--h100-noise`, which reproduces Linum's
  exact seeds, needs NVIDIA plus Triton. The same seed gives a different image, but quality is unaffected.
- P-JiT runs under bf16 autocast. RDNA3 supports bf16. On RDNA2 the script warns if
  PyTorch reports no bf16 support.
- If you still run out of memory, lower `--max-side` (e.g. 768) to skip the largest
  sheets, or re-export with `--grid-max 768`.
- LoRA training at 1024² on a 16 GB card is slow. Expect several seconds per image. A first
  run of `--epochs 2` (about 9k images) is a good test of whether the LoRA is learning.

### Fine-tune the released P-JiT 21K (recommended)

Train on the spritesheet grids from step 2 (any folder of `<id>.png` + `<id>.txt` works):

```bash
python training/pjit/train.py --data training/dataset/flux/grids \
    --out training/pjit_work --name duelyst_pjit --epochs 10 \
    --sample-prompt "dlystsprite, pixel art spritesheet, 4x4 grid of 16 animation frames, read left to right, top to bottom, attack animation, fire dragon, fire element, flames, side view, facing right, plain dark gray background"
```

- The default is a rank-32 LoRA on every attention and FFN projection (≈37M trainable
  parameters), with the frozen base held in bf16 and gradient checkpointing on. That should fit
  in 16 GB at 1024px. `--lora-rank 0` trains all 2.18B weights instead, which needs roughly 40 GB
  or more with AdamW.
- Images train at their own size. Batches are bucketed by size, and nothing is resized, so the
  pixel grid stays crisp. The noise scale follows the size (2 at 512², 4 at 1024²), as in Linum's runs.
- The released model was trained on dense 50–80 word captions. The short template captions work
  for a LoRA, but richer ones (step 2, "Captions") help.
- LoRAs are saved to `training/pjit_work/output/*.safetensors`. `--resume training/pjit_work/state.pt` continues a run.

### Train the architecture from scratch

A fresh model learns whatever text features it is given, so it doesn't need Linum's 4B encoder.
The default is **Qwen3.5-0.8B** (about 1.6 GB). Any Qwen3.5 size works with `--text-encoder`.
The trainer reads the encoder's config and taps three full-attention layers at the depths Linum used
(22 %, 47 % and 84 %). It sets the model's text width to match and records the encoder in the saved
`config.json`, so `sample.py` picks the same one. Each encoder gets its own caption cache
(`<out>/text_cache/<encoder>_L<layers>/`).

For a 16 GB card, export smaller sheets at the sprites' native pixel size. Most units fit a 128 px
cell at 1x, so the model sees every art pixel once and a 512² sheet is only 1,024 tokens at patch 16:

```bash
mojo run training/export_dataset.mojo --out training/dataset_small --no-video \
    --grid-max 512 --cell-size 128
python training/pjit/train.py --init scratch --preset small \
    --data training/dataset_small/flux/grids --out training/pjit_scratch \
    --max-steps 30000 --batch-size 4 --grad-accum 2 \
    --sample-prompt "dlystsprite, pixel art spritesheet, 3x3 grid of 9 animation frames, read left to right, top to bottom, attack animation, firebreather, abyssian, fire element, flames, side view, facing right, plain dark gray background"
python training/pjit/sample.py --weights training/pjit_scratch/output/duelyst_pjit-step030000 \
    --height 512 --width 512 --guidance 5 --prompt "dlystsprite, ..."
```

This trains the full Pyramid-JiT objective, with the readout heads on by default
(`--pixel-repa`, `--lpips` and `--pdino` add the DINOv3 / LPIPS terms if `timm` / `lpips` are installed). EMA defaults
to a half-life of 3 % of the run (Linum used 3,907 of ~135k steps). The first 36 % of steps use
noise-heavy timesteps, as in Linum's first 50M of 138M samples.

| `--preset` | Size | Notes |
|---|---|---|
| `tiny` | ~41M, patch 16 | smoke tests, 128–256 px |
| `small` | ~280M, patch 16 | realistic for one GPU on this dataset |
| `base` | ~720M, patch 32 | |
| `pjit` | 2.18B, patch 32 | Linum's exact architecture; their run cost ~$21K (8,415 H100-hours, 660M samples) |

About 4,500 spritesheets is far too little data to train a general text-to-image model from scratch.
Expect a model that only knows Duelyst-style sprites. Fine-tuning the released weights gives much
better prompt following for the same compute. Smaller presets use patch 16. For pixel art, sprite
pixels line up best with the patch grid when the integer upscale (`grid_scale`) divides the patch size.

### Generate

```bash
python training/pjit/sample.py --lora training/pjit_work/output/duelyst_pjit-step002000.safetensors \
    --prompt "dlystsprite, pixel art spritesheet, ..." --height 1024 --width 1024 --seeds 0 1 2
# a full / scratch model: --weights training/pjit_scratch/output/duelyst_pjit-step050000
mojo run training/postprocess.mojo training/pjit_out/p0_seed0.png --grid 4x4 --factor 4 --name attack
```

`--dry-run` on `train.py` prints the data, model and objective plan without downloading
or training anything.
