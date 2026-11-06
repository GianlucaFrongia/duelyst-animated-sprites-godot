"""Image + caption data for P-JiT training.

A dataset is a folder of `<id>.png` images with `<id>.txt` captions, e.g. the FLUX
spritesheet grids written by export_dataset.mojo (training/dataset/flux/grids). Captions
are encoded once with the frozen Qwen3.5-4B text encoder and cached as safetensors, so the
8 GB encoder is never in memory while the DiT trains.
"""

import gc
import hashlib
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

TEXT_DIM = 7680


def caption_key(caption: str) -> str:
    return hashlib.sha1(caption.encode("utf-8")).hexdigest()


def scan_folder(folder: str) -> List[Dict]:
    items = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".png"):
            continue
        stem = name[:-4]
        txt = os.path.join(folder, stem + ".txt")
        if not os.path.exists(txt):
            continue
        with open(txt, encoding="utf-8") as fh:
            caption = fh.read().strip()
        with Image.open(os.path.join(folder, name)) as im:
            w, h = im.size
        items.append({"id": stem, "image": os.path.join(folder, name), "caption": caption,
                      "h": h, "w": w})
    return items


def text_cache_path(cache_dir: str, caption: str) -> str:
    return os.path.join(cache_dir, caption_key(caption) + ".safetensors")


def ensure_text_cache(captions: Sequence[str], cache_dir: str, encoder_path: str,
                      device, batch_size: int = 1) -> None:
    """Encode every caption not yet cached, then free the encoder. One caption per call by
    default, as Linum's sampler encodes them: QwenTextEncoder trims padding assuming it is on
    the right, so a padded batch is only safe with a right-padding tokenizer."""
    from safetensors.torch import save_file

    os.makedirs(cache_dir, exist_ok=True)
    todo = sorted({c for c in captions if not os.path.exists(text_cache_path(cache_dir, c))})
    if not todo:
        return
    print(f"Encoding {len(todo)} captions with {encoder_path} -> {cache_dir}")
    from pjit_21k import QwenTextEncoder

    encoder = QwenTextEncoder(encoder_path, device=device)
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        for caption, emb in zip(chunk, encoder(chunk)):
            save_file({"emb": emb.to(torch.bfloat16).cpu().contiguous()},
                      text_cache_path(cache_dir, caption))
        done = min(i + batch_size, len(todo))
        if done % 200 < batch_size or done == len(todo):
            print(f"  {done}/{len(todo)}")
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_text(cache_dir: str, caption: str) -> torch.Tensor:
    from safetensors.torch import load_file

    return load_file(text_cache_path(cache_dir, caption))["emb"]


def pad_text(embs: List[torch.Tensor], text_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros(len(embs), text_len, embs[0].shape[-1], dtype=torch.bfloat16)
    lens = torch.zeros(len(embs), dtype=torch.int32)
    for i, e in enumerate(embs):
        n = min(e.shape[0], text_len)
        out[i, :n] = e[:n]
        lens[i] = n
    return out, lens


class CaptionImageDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Dict], cache_dir: str, text_len: int):
        self.items = items
        self.cache_dir = cache_dir
        self.text_len = text_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        with Image.open(item["image"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        img = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(1)   # (3, 1, H, W)
        return {"image": img, "text": load_text(self.cache_dir, item["caption"])}


def collate(batch, text_len: int):
    images = torch.stack([b["image"] for b in batch])
    text, lens = pad_text([b["text"] for b in batch], text_len)
    return {"image": images, "text": text, "text_lens": lens}


class BucketBatchSampler(torch.utils.data.Sampler):
    """Batches of same-sized images (the spritesheet grids come in many sizes; P-JiT
    takes any size, but one batch needs one size). Shuffled every epoch."""

    def __init__(self, items: List[Dict], batch_size: int, seed: int = 0):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i, it in enumerate(items):
            self.buckets[(it["h"], it["w"])].append(i)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _batches(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for idxs in self.buckets.values():
            idxs = idxs[:]
            rng.shuffle(idxs)
            batches += [idxs[i:i + self.batch_size] for i in range(0, len(idxs), self.batch_size)]
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return sum(-(-len(v) // self.batch_size) for v in self.buckets.values())


class CachedTextEncoder:
    """Stands in for QwenTextEncoder in pjit_21k.generate, serving cached embeddings."""

    def __init__(self, cache_dir: str, device):
        self.cache_dir = cache_dir
        self.device = device

    def __call__(self, texts: List[str]) -> List[torch.Tensor]:
        return [load_text(self.cache_dir, t).to(self.device) for t in texts]
