"""Minimal LoRA for P-JiT: wraps nn.Linear layers of the DiT blocks."""

import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("attn.q", "attn.k", "attn.v", "attn.o", "ffn.w13", "ffn.w2")
DEFAULT_SCOPES = ("blocks", "image_refiner", "text_refiner")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        dev = base.weight.device
        self.lora_down = nn.Linear(base.in_features, rank, bias=False, device=dev)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False, device=dev)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale


def _targets(model: nn.Module, scopes: Iterable[str], targets: Iterable[str]) -> List[Tuple[str, nn.Module, str]]:
    found = []
    targets = tuple(targets)
    for scope in scopes:
        for idx, block in enumerate(getattr(model, scope)):
            for target in targets:
                parent_path, _, attr = target.rpartition(".")
                parent = block.get_submodule(parent_path) if parent_path else block
                if isinstance(getattr(parent, attr, None), (nn.Linear, LoRALinear)):
                    found.append((f"{scope}.{idx}.{target}", parent, attr))
    return found


def inject_lora(model, rank, alpha, scopes=DEFAULT_SCOPES, targets=DEFAULT_TARGETS) -> List[nn.Parameter]:
    params = []
    for _, parent, attr in _targets(model, scopes, targets):
        layer = LoRALinear(getattr(parent, attr), rank=rank, alpha=alpha)
        setattr(parent, attr, layer)
        params += [layer.lora_down.weight, layer.lora_up.weight]
    return params


def lora_state_dict(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().float().cpu().contiguous()
            for k, v in model.state_dict().items() if ".lora_" in k}


def save_lora(model, path: str, rank: int, alpha: float, scopes, targets, extra=None):
    from safetensors.torch import save_file

    meta = {"format": "pjit-lora", "rank": str(rank), "alpha": str(alpha),
            "scopes": json.dumps(list(scopes)), "targets": json.dumps(list(targets))}
    if extra:
        meta.update({k: str(v) for k, v in extra.items()})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(lora_state_dict(model), path, metadata=meta)


def merge_lora_file(model, path: str, strength: float = 1.0) -> int:
    """Add a saved LoRA into the model's base weights (no runtime overhead). Returns the
    number of merged layers."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as fh:
        meta = fh.metadata() or {}
        state = {k: fh.get_tensor(k) for k in fh.keys()}
    rank = int(meta["rank"])
    scale = float(meta["alpha"]) / rank * strength
    merged = 0
    for key in state:
        if not key.endswith(".lora_down.weight"):
            continue
        path_ = key[: -len(".lora_down.weight")]
        down = state[key].float()
        up = state[path_ + ".lora_up.weight"].float()
        layer = model.get_submodule(path_)
        weight = layer.base.weight if isinstance(layer, LoRALinear) else layer.weight
        with torch.no_grad():
            delta = (up @ down) * scale
            weight.add_(delta.to(weight.device, weight.dtype))
        merged += 1
    return merged
