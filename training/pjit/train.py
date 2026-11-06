"""Train P-JiT (Linum's pixel-space Pyramid-JiT) on a folder of captioned images.

Two modes:

  * --init pretrained (default): start from Linum's released P-JiT 21K weights
    (Linum-AI/pjit-21k, 2.18B) and train a LoRA (--lora-rank 32) or, on a large GPU, all
    weights (--lora-rank 0).
  * --init scratch: build a fresh model from a preset (tiny / small / base / pjit) and
    train the full Pyramid-JiT objective, readout heads included.

The objective is Linum's (see objective.py): x-prediction with a 1/t^2-weighted
velocity MSE, logit-normal timesteps, JiT-style scaled noise, 5 % caption dropout for
guidance, plus the optional training-only terms (readouts, PixelREPA, LPIPS, P-DINO).

Example (LoRA on the Duelyst spritesheet grids):

    python training/pjit/train.py --data training/dataset/flux/grids \
        --out training/pjit_work --name duelyst_pjit --epochs 10
"""

import argparse
import functools
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D  # noqa: E402
import lora as L  # noqa: E402
import objective as O  # noqa: E402

try:
    from pjit_21k import PyramidJiT, PyramidJiTConfig, SamplerConfig, generate, set_attention_backend
except ImportError:  # pragma: no cover
    sys.exit("pjit_21k is not installed: pip install git+https://github.com/Linum-AI/pjit-21k")

PRESETS = {
    # ~41M: a quick sanity run at 128-256 px
    "tiny": dict(dim=512, num_heads=4, num_layers=8, ffn_dim=2048, refiner_blocks=1,
                 refiner_ffn_dim=1536, adaln_rank=128, bottleneck_dim=256, patch=(1, 16, 16)),
    # ~280M
    "small": dict(dim=1024, num_heads=8, num_layers=16, ffn_dim=4096, refiner_blocks=2,
                  refiner_ffn_dim=2816, adaln_rank=256, bottleneck_dim=512, patch=(1, 16, 16)),
    # ~720M
    "base": dict(dim=1536, num_heads=12, num_layers=20, ffn_dim=6144, refiner_blocks=2,
                 refiner_ffn_dim=4096, adaln_rank=256, bottleneck_dim=768, patch=(1, 32, 32)),
    # Linum's exact architecture (2.18B)
    "pjit": dict(),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--data", default="training/dataset/flux/grids", help="folder of <id>.png + <id>.txt")
    g.add_argument("--text-cache", default=None,
                   help="caption embedding cache (default <out>/text_cache/<encoder>)")
    g.add_argument("--text-encoder", default=None,
                   help="Qwen3.5 model for captions. Pretrained: must be Qwen/Qwen3.5-4B (the "
                        "default). Scratch: any Qwen3.5 size, default Qwen/Qwen3.5-0.8B")
    g.add_argument("--max-side", type=int, default=1280, help="skip images with a longer side")
    g.add_argument("--workers", type=int, default=2)

    g = p.add_argument_group("model")
    g.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained")
    g.add_argument("--weights", default="Linum-AI/pjit-21k", help="HF id or local dir (pretrained)")
    g.add_argument("--preset", choices=list(PRESETS), default="small", help="architecture (scratch)")
    g.add_argument("--patch", type=int, default=None, help="override the preset's patch size (scratch)")
    g.add_argument("--lora-rank", type=int, default=32, help="0 = train all weights (pretrained)")
    g.add_argument("--lora-alpha", type=float, default=None, help="default = rank")
    g.add_argument("--base-dtype", choices=["bf16", "fp32"], default="bf16",
                   help="dtype of frozen base weights in LoRA mode")
    g.add_argument("--attention", choices=["auto", "sdpa", "flash3"], default="auto")
    g.add_argument("--no-grad-checkpointing", action="store_true")

    g = p.add_argument_group("objective")
    g.add_argument("--noise-scale", type=float, default=None,
                   help="fixed noise scale (default: 4*sqrt(H*W)/1024 -> 2 at 512px, 4 at 1024px)")
    g.add_argument("--t-mean", type=float, default=-0.2)
    g.add_argument("--t-std", type=float, default=1.0)
    g.add_argument("--noisy-phase", type=float, default=None,
                   help="fraction of steps sampled noise-heavy (logit-normal 0.8/0.8) first; "
                        "default 0.36 for scratch (Linum: 50M of 138M samples), 0 otherwise")
    g.add_argument("--text-dropout", type=float, default=0.05)
    g.add_argument("--readouts", choices=["auto", "on", "off"], default="auto",
                   help="pyramid readout heads (auto: on for scratch, off for pretrained)")
    g.add_argument("--pixel-repa", action="store_true", help="DINOv3 alignment (needs timm)")
    g.add_argument("--lpips", action="store_true", help="LPIPS-VGG term (needs lpips)")
    g.add_argument("--pdino", action="store_true", help="perceptual DINOv3 term (needs timm)")

    g = p.add_argument_group("optimization")
    g.add_argument("--lr", type=float, default=None, help="default 1e-4 (LoRA), 2e-5 (full ft), 1e-4 (scratch)")
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    g.add_argument("--warmup", type=int, default=None, help="warmup steps (default 2%% of steps)")
    g.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--batch-size", type=int, default=1)
    g.add_argument("--grad-accum", type=int, default=4)
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--max-steps", type=int, default=0, help="optimizer steps; overrides --epochs")
    g.add_argument("--ema-half-life", type=float, default=None,
                   help="EMA half-life in steps, 0 = off (default: 3%% of steps for scratch, off otherwise)")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("output")
    g.add_argument("--out", default="training/pjit_work")
    g.add_argument("--name", default="duelyst_pjit")
    g.add_argument("--save-every", type=int, default=500)
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--resume", default=None, help="path to a state.pt written by this script")
    g.add_argument("--sample-every", type=int, default=500)
    g.add_argument("--sample-prompt", action="append", default=[],
                   help="prompt to sample during training (repeatable)")
    g.add_argument("--sample-size", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="default: the most common training size")
    g.add_argument("--sample-steps", type=int, default=30)
    g.add_argument("--sample-guidance", type=float, default=7.5)
    g.add_argument("--dry-run", action="store_true", help="build everything, print the plan, exit")
    g.add_argument("--force", action="store_true", help="skip the GPU memory check")
    return p.parse_args()


# --------------------------------
# MODEL
# --------------------------------

def init_scratch_weights(model: PyramidJiT):
    """adaLN-Zero style init: xavier linears, zero AdaLN up-projections (tanh gates start
    at 0, so every block starts as the identity) and a zero output head."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            w = m.weight.view(m.weight.shape[0], -1)
            nn.init.xavier_uniform_(w)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    for blocks in (model.blocks, model.image_refiner, model.text_refiner):
        for b in blocks:
            nn.init.zeros_(b.adaln_up.weight)
            nn.init.zeros_(b.adaln_up.bias)
    nn.init.normal_(model.time_embed[0].weight, std=0.02)
    nn.init.normal_(model.time_embed[2].weight, std=0.02)
    nn.init.zeros_(model.head.proj.weight)
    nn.init.zeros_(model.head.proj.bias)


def build_model(args, device, text_layers, text_dim):
    if args.init == "pretrained":
        if device.type == "meta":   # dry run: the released architecture, no weights
            with torch.device("meta"):
                return PyramidJiT(PyramidJiTConfig())
        # Stays on the CPU: main() casts the frozen base to bf16 before moving it, so the
        # 8.7 GB fp32 copy never lands on a 16 GB card.
        return PyramidJiT.from_pretrained(args.weights, device="cpu")
    cfg = dict(PRESETS[args.preset])
    if args.patch:
        cfg["patch"] = (1, args.patch, args.patch)
    # Linum's 0.5 RoPE scale comes from moving a 512px model to 1024px (Position
    # Interpolation); a model trained from scratch uses plain integer positions.
    cfg["rope_spatial_scale"] = 1.0
    cfg["text_layers"] = tuple(text_layers)
    cfg["text_dim"] = text_dim
    config = PyramidJiTConfig(**cfg)
    if config.dim % config.num_heads or (config.dim // config.num_heads) % 2:
        raise ValueError("dim must split into an even head_dim")
    with torch.device("meta" if device.type == "meta" else "cpu"):
        model = PyramidJiT(config)
    if device.type != "meta":
        init_scratch_weights(model)
    return model


def check_memory(args, device, scratch, use_lora):
    """Report the GPU and stop early on setups that cannot fit, instead of failing with an
    out-of-memory error after the downloads."""
    if device.type != "cuda":
        return
    props = torch.cuda.get_device_properties(device)
    gb = props.total_memory / 1024 ** 3
    backend = f"ROCm {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
    print(f"GPU: {props.name}, {gb:.1f} GB ({backend})")
    if not torch.cuda.is_bf16_supported():
        print("WARNING: this GPU reports no bf16 support; P-JiT trains under bf16 autocast. "
              "On RDNA2 try a recent ROCm build; RDNA3 and newer support bf16.")
    if args.force:
        return
    need = None
    if not scratch and not use_lora:
        need = 40   # 2.18B fp32 weights + grads + AdamW moments
    elif scratch and args.preset == "pjit":
        need = 40
    elif scratch and args.preset == "base":
        need = 20   # 720M x 16 bytes for weights, grads and AdamW, plus activations
    if need and gb < need:
        sys.exit(f"This setup needs roughly {need} GB of GPU memory; this GPU has {gb:.0f} GB. "
                 "Use the default LoRA fine-tune (--lora-rank 32) or --init scratch --preset "
                 "small, or pass --force to try anyway.")


def enable_grad_checkpointing(model):
    from torch.utils.checkpoint import checkpoint

    for blocks in (model.blocks, model.image_refiner, model.text_refiner):
        for block in blocks:
            fwd = block.forward

            def wrapped(*a, _fwd=fwd, **kw):
                if not torch.is_grad_enabled():
                    return _fwd(*a, **kw)
                return checkpoint(_fwd, *a, use_reentrant=False, **kw)

            block.forward = wrapped


class EMA:
    def __init__(self, params, half_life):
        self.params = list(params)
        self.beta = math.exp(-math.log(2.0) / half_life)
        self.shadow = [p.detach().clone().float() for p in self.params]
        self.backup = None

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.lerp_(p.detach().float(), 1 - self.beta)

    @torch.no_grad()
    def swap_in(self):
        self.backup = [p.detach().clone() for p in self.params]
        for s, p in zip(self.shadow, self.params):
            p.copy_(s.to(p.dtype))

    @torch.no_grad()
    def swap_out(self):
        for b, p in zip(self.backup, self.params):
            p.copy_(b)
        self.backup = None


def param_count(params):
    return sum(p.numel() for p in params)


# --------------------------------
# SAVE / SAMPLE
# --------------------------------

def save_weights(args, model, ema, step, lora_cfg):
    tag = f"{args.name}-step{step:06d}"
    if ema is not None:
        ema.swap_in()
    try:
        if lora_cfg:
            path = os.path.join(args.out, "output", tag + ".safetensors")
            L.save_lora(model, path, extra={"step": step}, **lora_cfg)
        else:
            from safetensors.torch import save_file

            path = os.path.join(args.out, "output", tag)
            os.makedirs(path, exist_ok=True)
            state = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()}
            save_file(state, os.path.join(path, "model.safetensors"))
            with open(os.path.join(path, "config.json"), "w") as fh:
                json.dump({"architecture": model.config.to_dict(),
                           "text_encoder": args.text_encoder}, fh, indent=2)
    finally:
        if ema is not None:
            ema.swap_out()
    print(f"  saved {path}" + (" (EMA)" if ema is not None else ""))
    return path


def save_state(args, step, epoch, trainable, optimizer, ema, extra_modules):
    path = os.path.join(args.out, "state.pt")
    torch.save({
        "step": step, "epoch": epoch,
        "trainable": [p.detach().cpu() for p in trainable],
        "optimizer": optimizer.state_dict(),
        "ema": [s.cpu() for s in ema.shadow] if ema is not None else None,
        "extra": {k: m.state_dict() for k, m in extra_modules.items()},
    }, path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def write_samples(args, model, ema, step, sample_hw, device):
    if not args.sample_prompt:
        return
    from PIL import Image

    was_training = model.training
    model.eval()
    if ema is not None:
        ema.swap_in()
    try:
        h, w = sample_hw
        sampler = SamplerConfig(
            height=h, width=w, sampling_steps=args.sample_steps,
            guidance_scale=args.sample_guidance, negative_prompt="",
            noise_scale=O.noise_scale_for(h, w, O.LossConfig(noise_scale=args.noise_scale)),
            noise_sm_count=None)
        encoder = D.CachedTextEncoder(args.text_cache, device)
        out_dir = os.path.join(args.out, "samples")
        os.makedirs(out_dir, exist_ok=True)
        for k, prompt in enumerate(args.sample_prompt):
            img = generate(model=model, text_encoder=encoder, prompt=prompt, seeds=[k],
                           sampler=sampler, device=str(device), quiet=True)[0]
            Image.fromarray(img[:, 0].permute(1, 2, 0).cpu().numpy()).save(
                os.path.join(out_dir, f"step{step:06d}_{k}.png"))
        print(f"  wrote {len(args.sample_prompt)} samples to {out_dir}")
    finally:
        if ema is not None:
            ema.swap_out()
        model.train(was_training)


# --------------------------------
# MAIN
# --------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU visible to PyTorch; running on CPU (only useful for --preset tiny tests).")
    set_attention_backend(args.attention)
    os.makedirs(args.out, exist_ok=True)
    scratch = args.init == "scratch"
    if args.text_encoder is None:
        args.text_encoder = "Qwen/Qwen3.5-0.8B" if scratch else D.LINUM_ENCODER
    if not scratch and os.path.basename(args.text_encoder.rstrip("/")) != "Qwen3.5-4B":
        sys.exit("The released P-JiT weights only understand Qwen3.5-4B features; a smaller "
                 "encoder works only with --init scratch.")
    if scratch:
        try:
            text_layers, text_dim = D.encoder_layout(args.text_encoder)
        except Exception as exc:   # dry run offline: show the plan with Linum's layout
            if not args.dry_run:
                raise
            print(f"(could not read {args.text_encoder} config: {exc}; showing the 4B layout)")
            text_layers, text_dim = D.LINUM_LAYERS, D.TEXT_DIM
    else:
        text_layers, text_dim = D.LINUM_LAYERS, D.TEXT_DIM
    print(f"Text: {args.text_encoder}, layers {list(text_layers)} -> {text_dim} features per token")
    args.text_cache = args.text_cache or os.path.join(
        args.out, "text_cache", D.encoder_slug(args.text_encoder, text_layers))
    use_lora = not scratch and args.lora_rank > 0

    # ---- data ----
    items = D.scan_folder(args.data)
    big = [it for it in items if max(it["h"], it["w"]) > args.max_side]
    items = [it for it in items if max(it["h"], it["w"]) <= args.max_side]
    if big:
        print(f"Skipping {len(big)} images larger than --max-side {args.max_side}")
    if not items:
        sys.exit(f"No <id>.png + <id>.txt pairs in {args.data}")
    sizes = {}
    for it in items:
        sizes[(it["h"], it["w"])] = sizes.get((it["h"], it["w"]), 0) + 1
    sample_hw = tuple(args.sample_size) if args.sample_size else max(sizes, key=sizes.get)

    sampler = D.BucketBatchSampler(items, args.batch_size, seed=args.seed)
    steps_per_epoch = max(1, len(sampler) // args.grad_accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = args.warmup if args.warmup is not None else max(1, int(0.02 * total_steps))
    lr = args.lr or (1e-4 if (use_lora or scratch) else 2e-5)
    noisy = args.noisy_phase if args.noisy_phase is not None else (0.36 if scratch else 0.0)
    half_life = args.ema_half_life if args.ema_half_life is not None else (
        max(50.0, 0.03 * total_steps) if scratch else 0.0)

    print(f"Data: {len(items)} images in {len(sizes)} size buckets "
          f"(most common {sample_hw[0]}x{sample_hw[1]}), {len(sampler)} batches/epoch")
    print(f"Plan: {total_steps} optimizer steps (batch {args.batch_size} x accum {args.grad_accum}), "
          f"lr {lr}, warmup {warmup}, EMA half-life {half_life or 'off'}")

    check_memory(args, device, scratch, use_lora)

    # ---- captions ----
    # Encoded before the DiT is built, so the 8 GB Qwen3.5-4B encoder has the GPU to itself.
    captions = [it["caption"] for it in items] + args.sample_prompt + ([""] if args.sample_prompt else [])
    if not args.dry_run:
        D.ensure_text_cache(captions, args.text_cache, args.text_encoder, device,
                            layers=text_layers)

    # ---- model ----
    # A dry run builds the model on the meta device: no download, no memory.
    model = build_model(args, torch.device("meta") if args.dry_run else device,
                        text_layers, text_dim)
    model.train()
    for p in model.parameters():
        p.requires_grad_(not use_lora)
    lora_cfg = None
    if use_lora and args.base_dtype == "bf16" and device.type == "cuda":
        model.to(torch.bfloat16)          # on the CPU: 4.4 GB of frozen weights to move
    if not args.dry_run:
        model.to(device)
    if use_lora:
        alpha = args.lora_alpha or float(args.lora_rank)
        trainable = L.inject_lora(model, args.lora_rank, alpha)
        lora_cfg = dict(rank=args.lora_rank, alpha=alpha, scopes=L.DEFAULT_SCOPES, targets=L.DEFAULT_TARGETS)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
    if not args.no_grad_checkpointing:
        enable_grad_checkpointing(model)

    # ---- objective ----
    readouts_on = args.readouts == "on" or (args.readouts == "auto" and scratch)
    phases = [(None, args.t_mean, args.t_std)]
    if noisy > 0:
        phases = [(int(noisy * total_steps), 0.8, 0.8)] + phases
    loss_cfg = O.LossConfig(
        noise_scale=args.noise_scale, timestep_phases=phases, text_dropout=args.text_dropout,
        readouts=O.default_readouts(model.config.num_layers) if readouts_on else [],
        pixel_repa_block=O.default_repa_block(model.config.num_layers))
    extra = {}
    readout_heads = O.build_readout_heads(model, loss_cfg).to(device)
    if len(readout_heads):
        extra["readout_heads"] = readout_heads
    adapter = dino = lpips = None
    if (args.pixel_repa or args.pdino) and not args.dry_run:
        dino = O.load_dino(device)
    if args.pixel_repa and (dino is not None or args.dry_run):
        adapter = O.PixelRepaAdapter.for_model(model, loss_cfg.pixel_repa_mask_ratio).to(device)
        extra["adapter"] = adapter
    if args.lpips and not args.dry_run:
        lpips = O.load_lpips(device)
    extra_params = [p for m in extra.values() for p in m.parameters()]

    print(f"Model: {param_count(model.parameters()) / 1e6:.1f}M params "
          f"({'LoRA r%d, ' % args.lora_rank if use_lora else ''}"
          f"{param_count(trainable) / 1e6:.2f}M trainable + {param_count(extra_params) / 1e6:.2f}M "
          f"training-only heads), patch {model.patch[1]}, {model.config.num_layers} blocks x {model.config.dim}")
    print(f"Objective: readouts {loss_cfg.readouts or 'off'}, PixelREPA {'block %d' % loss_cfg.pixel_repa_block if adapter else 'off'}, "
          f"LPIPS {'on' if lpips else 'off'}, P-DINO {'on' if args.pdino else 'off'}, "
          f"timestep phases {phases}, noise scale "
          f"{args.noise_scale or 'auto (%.2f at %dx%d)' % (O.noise_scale_for(*sample_hw, loss_cfg), *sample_hw)}")

    if args.dry_run:
        print("Dry run: nothing trained.")
        return

    optimizer = torch.optim.AdamW(trainable + extra_params, lr=lr, betas=tuple(args.betas),
                                  weight_decay=args.weight_decay)

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        if args.schedule == "cosine":
            prog = (step - warmup) / max(1, total_steps - warmup)
            return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return lr

    ema = EMA(trainable, half_life) if half_life > 0 else None
    step, epoch = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        with torch.no_grad():
            for p, v in zip(trainable, state["trainable"]):
                p.copy_(v.to(p.device, p.dtype))
        optimizer.load_state_dict(state["optimizer"])
        for k, m in extra.items():
            m.load_state_dict(state["extra"][k])
        if ema is not None and state["ema"] is not None:
            ema.shadow = [s.to(device) for s in state["ema"]]
        step, epoch = state["step"], state["epoch"]
        print(f"Resumed from {args.resume} at step {step}")

    dataset = D.CaptionImageDataset(items, args.text_cache, model.text_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.workers,
        collate_fn=functools.partial(D.collate, text_len=model.text_len),
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    with open(os.path.join(args.out, "run.json"), "w") as fh:
        json.dump({"args": vars(args), "architecture": model.config.to_dict(),
                   "loss": {k: v for k, v in vars(loss_cfg).items()}}, fh, indent=2, default=str)

    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    running, t0, micro = {}, time.time(), 0
    while step < total_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            x_0 = (batch["image"].to(device, non_blocking=True).to(torch.bfloat16) / 127.5 - 1.0)
            text = batch["text"].to(device, non_blocking=True)
            text_lens = batch["text_lens"].to(device)
            with autocast:
                losses = O.pjit_loss(model, readout_heads, x_0, text, text_lens, step, loss_cfg,
                                     adapter=adapter, dino_features=dino, lpips_vgg=lpips,
                                     use_pdino=args.pdino)
            (losses["total"] / args.grad_accum).backward()
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            micro += 1
            if micro % args.grad_accum:
                continue

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            gnorm = torch.nn.utils.clip_grad_norm_(trainable + extra_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update()
            step += 1

            if step % args.log_every == 0 or step == 1:
                n = micro
                msg = " ".join(f"{k} {v / n:.4f}" for k, v in running.items())
                print(f"step {step}/{total_steps} ep {epoch} | {msg} | gnorm {float(gnorm):.3f} "
                      f"| lr {lr_at(step - 1):.2e} | {(time.time() - t0) / max(1, n) * args.grad_accum:.2f}s/step",
                      flush=True)
                running, micro, t0 = {}, 0, time.time()
            if step % args.save_every == 0 or step == total_steps:
                save_weights(args, model, ema, step, lora_cfg)
                save_state(args, step, epoch, trainable, optimizer, ema, extra)
            if args.sample_every and (step % args.sample_every == 0 or step == total_steps):
                write_samples(args, model, ema, step, sample_hw, device)
            if step >= total_steps:
                break
        epoch += 1
    print("Done.")


if __name__ == "__main__":
    main()
