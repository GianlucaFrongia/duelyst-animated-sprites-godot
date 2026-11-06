# P-JiT training objective, adapted from Linum's reference implementation
# (github.com/Linum-AI/pyramid-jit, loss.py; Copyright 2026 Linum Inc., Apache-2.0).
#
# Changes from the reference:
#   * imports the model pieces from `pjit_21k` (the 21K release) instead of `pyramid_jit`
#   * the noise scale follows the image size (2 at 512 px, 4 at 1024 px, as Linum used)
#     unless it is fixed
#   * readout blocks, the PixelREPA tap and the adapter width are set from the model's
#     depth and width, so smaller from-scratch configs work
#   * perceptual terms and the adapter are optional
#
# Notation (as in the reference): x_0 clean image in [-1, 1], eps ~ N(0, I), t in (0, 1)
# with t=0 clean and t=1 noise, shapes (B, 3, 1, H, W).
#
#   L = L_final + sum_k w_k * L_readout_k
#       + w_repa * L_pixel_repa + w_lpips * L_lpips + w_pdino * L_pdino

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pjit_21k.attention import VarlenMeta, build_varlen_metadata
from pjit_21k.model import (
    OutputHead,
    PyramidJiT,
    SelfAttention,
    SwiGLUFFN,
    build_rope_table,
    build_text_mask,
    compute_rotary_frequencies,
    sinusoidal_embedding_1d,
    unpatchify,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class LossConfig:
    # None = size-dependent: 4 * sqrt(H * W) / 1024 (2.0 at 512 px, 4.0 at 1024 px).
    noise_scale: Optional[float] = None
    t_eps: float = 0.1                # clamp on t in the 1/t^2 weight (max weight 100x)
    t_min: float = 1e-3
    num_timesteps: int = 1000
    # Logit-normal timesteps by optimizer step: (until_step or None, mean, std).
    timestep_phases: List[Tuple[Optional[int], float, float]] = field(
        default_factory=lambda: [(None, -0.2, 1.0)])
    # x-prediction readouts: (1-indexed trunk block, spatial downsample, weight).
    readouts: List[Tuple[int, int, float]] = field(default_factory=list)
    pixel_repa_weight: float = 0.1
    pixel_repa_mask_ratio: float = 0.2
    pixel_repa_block: int = 5
    lpips_weight: float = 0.1
    pdino_weight: float = 0.01
    perceptual_gate_t: float = 0.7
    text_dropout: float = 0.05


def default_readouts(num_layers: int) -> List[Tuple[int, int, float]]:
    """Linum's readouts sit after blocks 10 and 16 of 22; scale that to other depths."""
    a = max(1, round(num_layers * 10 / 22))
    b = max(a + 1, round(num_layers * 16 / 22))
    b = min(b, num_layers - 1)
    if b <= a:
        return [(a, 4, 1.0)] if a < num_layers else []
    return [(a, 4, 1.0), (b, 2, 1.0)]


def default_repa_block(num_layers: int) -> int:
    return max(1, round(num_layers * 5 / 22))


def noise_scale_for(h: int, w: int, cfg: LossConfig) -> float:
    if cfg.noise_scale is not None:
        return cfg.noise_scale
    return 4.0 * math.sqrt(h * w) / 1024.0


# --------------------------------
# FORWARD PROCESS
# --------------------------------

def sample_timesteps(batch_size, global_step, cfg, device, dtype=torch.bfloat16):
    mean, std = cfg.timestep_phases[-1][1:]
    for until_step, m, s in cfg.timestep_phases:
        if until_step is None or global_step < until_step:
            mean, std = m, s
            break
    z = torch.empty(batch_size, dtype=dtype, device=device).normal_()
    return torch.sigmoid(z * std + mean).clamp(min=cfg.t_min, max=1.0 - cfg.t_min)


def add_noise(x_0, t, cfg):
    scale = noise_scale_for(x_0.shape[-2], x_0.shape[-1], cfg)
    x_1 = torch.empty_like(x_0).normal_() * scale
    t5 = t.view(-1, 1, 1, 1, 1)
    return (1 - t5) * x_0 + t5 * x_1, (t * cfg.num_timesteps).long()


def drop_text(text, cfg):
    """CFG dropout: whole captions replaced by zero rows (lengths unchanged)."""
    if cfg.text_dropout <= 0:
        return text
    keep = torch.bernoulli(torch.full((text.shape[0],), 1 - cfg.text_dropout, device=text.device))
    return text * keep.view(-1, 1, 1).to(text.dtype)


# --------------------------------
# DIFFUSION LOSSES (x-prediction, velocity-space MSE)
# --------------------------------

def velocity_mse(x_pred, x_0, t, cfg):
    """||v_pred - v||^2 = ||x_pred - x_0||^2 / t^2, with t clamped at t_eps."""
    t5 = t.view(-1, 1, 1, 1, 1).clamp_min(cfg.t_eps)
    return (((x_pred.float() - x_0.float()) / t5.float()) ** 2).mean()


def readout_target(x_0, size):
    small = F.interpolate(
        x_0[:, :, 0].float(), size=size, mode="bilinear", align_corners=False, antialias=True)
    return small.to(x_0.dtype).unsqueeze(2)


# --------------------------------
# READOUTS (deep supervision of the trunk)
# --------------------------------

def build_readout_heads(model: PyramidJiT, cfg: LossConfig) -> nn.ModuleDict:
    pt, ph, pw = model.patch
    heads = {}
    for block, down, _ in cfg.readouts:
        if ph % down or pw % down:
            raise ValueError(f"patch {ph}x{pw} is not divisible by readout downsample {down}")
        heads[f"block{block}"] = OutputHead(
            dim=model.config.dim, out_channels=model.config.out_channels,
            patch=(pt, ph // down, pw // down))
    return nn.ModuleDict(heads)


def apply_readout(head, tokens, e, grid, image_fhw, model_patch, downsample):
    pt, ph, pw = model_patch
    out_patch = (pt, ph // downsample, pw // downsample)
    x = unpatchify(x_head=head(x=tokens, e=e), f_p=grid[0], h_p=grid[1], w_p=grid[2],
                   patch=out_patch, out_channels=3)
    f, h, w = image_fhw
    return x[:, :, :f, :h * out_patch[1] // ph, :w * out_patch[2] // pw]


# --------------------------------
# PIXEL-REPA
# --------------------------------

class AdapterBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim=dim, num_heads=num_heads, eps=eps)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SwiGLUFFN(dim=dim, hidden_dim=ffn_dim)
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))

    def forward(self, x, e, mask, rope_table, varlen_meta):
        e = (self.modulation.to(e.dtype) + e).chunk(6, dim=1)
        y = self.attn(x=self.norm1(x) * (1 + e[1]) + e[0], mask=mask, rope_table=rope_table,
                      varlen_meta=varlen_meta)
        x = x + y * e[2]
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        y = y * mask.unsqueeze(-1)
        return x + y * e[5]


class PixelRepaAdapter(nn.Module):
    """Masked transformer adapter that maps a trunk block's tokens to DINOv3 features
    (training only). Linum's: 2944 -> 2048, 2 blocks of 16 x 128 heads, FFN 8192."""

    def __init__(self, in_dim, dim=2048, num_heads=16, ffn_dim=8192, num_blocks=2,
                 freq_dim=256, eps=1e-6, out_dim=1024, mask_ratio=0.2,
                 rope_max_positions=1024):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.mask_ratio = mask_ratio
        self.input_proj = nn.Linear(in_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.time_embed = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            AdapterBlock(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, out_dim))
        head_dim = dim // num_heads
        with torch.device("cpu"):
            self.freqs = torch.cat([
                compute_rotary_frequencies(
                    max_seq_len=rope_max_positions, dim=head_dim - 4 * (head_dim // 6)),
                compute_rotary_frequencies(max_seq_len=rope_max_positions, dim=2 * (head_dim // 6)),
                compute_rotary_frequencies(max_seq_len=rope_max_positions, dim=2 * (head_dim // 6)),
            ], dim=1)

    @classmethod
    def for_model(cls, model: PyramidJiT, mask_ratio: float, out_dim: int = 1024):
        dim = model.config.dim
        if dim >= 2048:
            return cls(in_dim=dim, mask_ratio=mask_ratio, out_dim=out_dim)
        width = max(128, (dim // 128) * 128)
        return cls(in_dim=dim, dim=width, num_heads=width // 128, ffn_dim=4 * width,
                   mask_ratio=mask_ratio, out_dim=out_dim)

    def forward(self, tokens, mask, t, grid):
        if self.freqs.device != tokens.device:
            self.freqs = self.freqs.to(tokens.device)
        f_p, h_p, w_p = grid
        l_img = f_p * h_p * w_p
        x = self.input_proj(tokens)
        keep = torch.rand(x.shape[0], l_img, device=x.device) >= self.mask_ratio
        x_img = torch.where(keep.unsqueeze(-1), x[:, :l_img], self.mask_token.to(x.dtype))
        x = torch.cat([x_img, x[:, l_img:]], dim=1)
        e = self.time_embed(sinusoidal_embedding_1d(dim=self.freq_dim, position=t, dtype=x.dtype))
        e0 = self.time_proj(e).unflatten(1, (6, self.dim))
        meta = build_varlen_metadata(mask=mask)
        rope = build_rope_table(
            freqs=self.freqs, f_s=f_p, h_s=h_p, w_s=w_p, text_len=tokens.shape[1] - l_img)
        for block in self.blocks:
            x = block(x=x, e=e0, mask=mask, rope_table=rope, varlen_meta=meta)
        return self.head(x[:, :l_img])


def _imagenet_normalize(x):
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def dino_alignment_target(x_0, dino_features, model_patch, dino_patch=16):
    """DINOv3 patch features of the clean image on the trunk's token grid."""
    h_raw, w_raw = x_0.shape[-2:]
    grid_h, grid_w = math.ceil(h_raw / model_patch), math.ceil(w_raw / model_patch)
    canvas_h, canvas_w = grid_h * dino_patch, grid_w * dino_patch
    scale = dino_patch / model_patch
    content_h = min(max(1, round(h_raw * scale)), canvas_h)
    content_w = min(max(1, round(w_raw * scale)), canvas_w)
    with torch.no_grad():
        x = F.interpolate(x_0[:, :, 0].float(), size=(content_h, content_w), mode="bicubic",
                          align_corners=False, antialias=True).clamp_(-1.0, 1.0)
        x = F.pad(x, (0, canvas_w - content_w, 0, canvas_h - content_h), value=0.0)
        return dino_features(_imagenet_normalize((x + 1.0) * 0.5))


def pixel_repa_loss(z_adapter, z_dino):
    return -(F.normalize(z_dino.float(), dim=-1)
             * F.normalize(z_adapter.float(), dim=-1)).sum(-1).mean()


# --------------------------------
# PERCEPTUAL LOSSES
# --------------------------------

def gated_mean(per_sample, gate):
    return (per_sample.float() * gate).sum() / gate.sum().clamp_min(1.0)


def lpips_loss(x_pred, x_0, lpips_vgg, gate, crop=224):
    pred, target = x_pred[:, :, 0], x_0[:, :, 0]
    h, w = pred.shape[-2:]
    if h < crop or w < crop:
        size = (max(h, crop), max(w, crop))
        pred = F.interpolate(pred, size=size, mode="bicubic", align_corners=False)
        target = F.interpolate(target, size=size, mode="bicubic", align_corners=False)
    top = int(torch.randint(0, pred.shape[-2] - crop + 1, (1,)))
    left = int(torch.randint(0, pred.shape[-1] - crop + 1, (1,)))
    pred = pred[..., top:top + crop, left:left + crop]
    target = target[..., top:top + crop, left:left + crop]
    return gated_mean(lpips_vgg(pred, target).view(-1), gate)


def pdino_loss(x_pred, x_0, dino_features, gate, max_side=224, dino_patch=16):
    h, w = x_pred.shape[-2:]
    scale = min(1.0, max_side / max(h, w))
    size = (max(dino_patch, round(h * scale / dino_patch) * dino_patch),
            max(dino_patch, round(w * scale / dino_patch) * dino_patch))

    def prepare(x):
        x = _imagenet_normalize((x[:, :, 0].float() + 1.0) * 0.5)
        return F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)

    with torch.no_grad():
        feat_target = dino_features(prepare(x_0))
    feat_pred = dino_features(prepare(x_pred))
    per_sample = (1 - F.cosine_similarity(feat_pred, feat_target, dim=-1)).mean(dim=-1)
    return gated_mean(per_sample, gate)


def load_dino(device):
    """Frozen DINOv3 ViT-L/16 through timm, returning final-norm patch tokens."""
    import timm

    net = timm.create_model(
        "vit_large_patch16_dinov3.lvd1689m", pretrained=True, dynamic_img_size=True)
    net = net.eval().requires_grad_(False).to(device)
    prefix = getattr(net, "num_prefix_tokens", 1)

    def features(x):
        return net.forward_features(x)[:, prefix:]

    return features


def load_lpips(device):
    import lpips

    return lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False).to(device)


# --------------------------------
# TAPPED FORWARD + TOTAL
# --------------------------------

def forward_with_taps(model, readout_heads, adapter, x_t, text, text_lens, t_int, cfg):
    taps: Dict[object, torch.Tensor] = {}
    wanted = {block for block, _, _ in cfg.readouts}
    if adapter is not None:
        wanted.add(cfg.pixel_repa_block)
    hooks = [
        model.blocks[block - 1].register_forward_hook(
            lambda m, i, o, k=block: taps.__setitem__(k, o))
        for block in sorted(wanted)]
    if cfg.readouts:
        hooks.append(model.head.register_forward_hook(
            lambda m, args, kwargs, o: taps.__setitem__("e", kwargs["e"]), with_kwargs=True))
    try:
        x_pred = model(x_t=x_t, text=text, text_lens=text_lens, t=t_int)
    finally:
        for hook in hooks:
            hook.remove()

    f_in, h_in, w_in = x_t.shape[2:]
    pt, ph, pw = model.patch
    grid = (-(-f_in // pt), -(-h_in // ph), -(-w_in // pw))
    l_img = math.prod(grid)
    out = {"x_pred": x_pred}
    for block, down, _ in cfg.readouts:
        out[f"block{block}"] = apply_readout(
            head=readout_heads[f"block{block}"], tokens=taps[block][:, :l_img], e=taps["e"],
            grid=grid, image_fhw=(f_in, h_in, w_in), model_patch=model.patch, downsample=down)
    if adapter is not None:
        img_mask = torch.ones(x_t.shape[0], l_img, dtype=torch.bool, device=x_t.device)
        mask = torch.cat(
            [img_mask, build_text_mask(seq_len=model.text_len, lengths=text_lens)], dim=1)
        out["z_adapter"] = adapter(tokens=taps[cfg.pixel_repa_block], mask=mask, t=t_int, grid=grid)
    return out


def pjit_loss(model, readout_heads, x_0, text, text_lens, global_step, cfg,
              adapter=None, dino_features=None, lpips_vgg=None, use_pdino=False):
    """One step's losses. Call under bf16 autocast."""
    t = sample_timesteps(x_0.shape[0], global_step, cfg, x_0.device, x_0.dtype)
    x_t, t_int = add_noise(x_0, t, cfg)
    use_repa = adapter is not None and dino_features is not None
    z_dino = (dino_alignment_target(x_0, dino_features, model_patch=model.patch[1])
              if use_repa else None)
    out = forward_with_taps(
        model, readout_heads, adapter if use_repa else None, x_t,
        drop_text(text, cfg), text_lens, t_int, cfg)

    losses = {"final": velocity_mse(out["x_pred"], x_0, t, cfg)}
    for block, _, weight in cfg.readouts:
        pred = out[f"block{block}"]
        losses[f"block{block}"] = weight * velocity_mse(
            pred, readout_target(x_0, tuple(pred.shape[-2:])), t, cfg)
    if use_repa:
        losses["pixel_repa"] = cfg.pixel_repa_weight * pixel_repa_loss(out["z_adapter"], z_dino)
    gate = (t <= cfg.perceptual_gate_t).float()
    if lpips_vgg is not None:
        losses["lpips"] = cfg.lpips_weight * lpips_loss(out["x_pred"], x_0, lpips_vgg, gate)
    if use_pdino and dino_features is not None:
        losses["pdino"] = cfg.pdino_weight * pdino_loss(out["x_pred"], x_0, dino_features, gate)
    losses["total"] = sum(losses.values())
    return losses
