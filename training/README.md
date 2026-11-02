# Training a pixel-art animation LoRA on the Duelyst sprites

Mojo scripts that turn this repo's sprites into a captioned training set, train a
LoRA with [musubi-tuner](https://github.com/kohya-ss/musubi-tuner), and convert
the generated output back into Godot `SpriteFrames`. The goal is prompts like:

> `dlystsprite, pixel art sprite animation, attack animation, red dragon, fire element, flames, side view, facing right, plain dark gray background`

| Script | What it does |
|---|---|
| `export_dataset.mojo` | Reads every `.tres` + spritesheet and writes Wan video clips, FLUX.2 grid sheets, captions and `metadata.jsonl` |
| `curate_dataset.mojo` | Optional: removes near-duplicate clips, clusters the rest and sets per-cluster repeats |
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

## 2b. Curate (optional, recommended)

```bash
mojo run training/curate_dataset.mojo            # → training/dataset/curation/
```

This follows the data recipe for image generation in
[LongCat-Next](https://arxiv.org/abs/2603.27538) (Sec. 4.3). That paper found
that training on raw data spends most of the model's capacity on the most common
looks:

1. **Embed** each clip with a cheap pixel-art descriptor: the mean thumbnail,
   the motion map (frame differences) and a colour histogram. The paper uses
   SigLIP2; here no model is needed (~40 s for the full set).
2. **Deduplicate** within each action: a clip that is near-identical
   (`--dup 0.985` cosine) to another *unit's* clip is dropped. On the full set
   this finds ~25 copies, e.g. `zodiac`/`zodiac02` and `shadow2`/`crawler`.
3. **Cluster** with k-means (`--clusters`, default √N ≈ 67).
4. **Rebalance**: each clip is repeated `(largest cluster / its cluster)^alpha`
   times (`--alpha 0.5`, `--max-repeats 4`). One-off looks are seen more often
   per epoch. Repeat counts on the full set: x1 2311, x2 1884, x3 275, x4 50.
5. **Exemplars**: the clip nearest each cluster centre, for a short final pass
   on a small, diverse set.

Captions are read from the `.txt` files when curation runs. If you refine
captions later, re-run curation.

Then train with `--curated`. An epoch is now ~7,100 samples instead of ~4,500,
so you can lower `--epochs`. Optionally, finish with a short pass on the
exemplars, continuing from the curated LoRA:

```bash
mojo run training/train_lora.mojo --model flux2 ... --curated --epochs 10
mojo run training/train_lora.mojo --model flux2 ... --exemplars --epochs 4 \
    --lr 5e-5 --name duelyst_flux2_ex \
    --network-weights training/lora_work/output/duelyst_flux2.safetensors
```

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
`--skip-cache` (reuse cached latents/text), `--name`, `--work DIR`,
`--curated` / `--exemplars` (see 2b), `--network-weights FILE` (continue a LoRA).
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
