"""Generate images with a P-JiT model trained by train.py.

    # LoRA on top of the released P-JiT 21K
    python training/pjit/sample.py --lora training/pjit_work/output/duelyst_pjit-step002000.safetensors \
        --prompt "dlystsprite, pixel art spritesheet, ..." --height 1024 --width 1024 --seeds 0 1 2

    # A full model (fine-tuned or from scratch): the saved directory
    python training/pjit/sample.py --weights training/pjit_work/output/duelyst_pjit-step010000 ...

The output PNGs are the same kind of sheet the model was trained on, so postprocess.mojo
turns them into Godot frames (--grid CxR --factor <grid_scale>).
"""

import argparse
import gc
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lora as L  # noqa: E402
import objective as O  # noqa: E402

from pjit_21k import PyramidJiT, QwenTextEncoder, SamplerConfig, generate, set_attention_backend  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="Linum-AI/pjit-21k", help="HF id or a directory saved by train.py")
    p.add_argument("--lora", action="append", default=[], help="LoRA .safetensors to merge (repeatable)")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--text-encoder", default="Qwen/Qwen3.5-4B")
    p.add_argument("--prompt", action="append", required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=15.0, help="APG scale (Linum's default 15)")
    p.add_argument("--noise-scale", type=float, default=None, help="default: same rule as training")
    p.add_argument("--h100-noise", action="store_true",
                   help="draw noise as an H100 does (Linum's seeds; NVIDIA + Triton only)")
    p.add_argument("--attention", choices=["auto", "sdpa", "flash3"], default="auto")
    p.add_argument("--bf16-weights", action="store_true",
                   help="hold the DiT in bf16 (4.4 GB instead of 8.7 GB); tiny rounding changes")
    p.add_argument("--out-dir", default="training/pjit_out")
    args = p.parse_args()

    device = "cuda"
    set_attention_backend(args.attention)

    # Encode every prompt first and free the 8 GB encoder before the DiT (8.7 GB in fp32)
    # is loaded: both at once do not fit on a 16 GB card.
    encoder = QwenTextEncoder(args.text_encoder, device=torch.device(device))
    texts = list(dict.fromkeys(args.prompt + [args.negative_prompt]))
    # One caption per call, as Linum's sampler does (no padding inside a batch).
    cached = {t: encoder([t])[0] for t in texts}
    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    model = PyramidJiT.from_pretrained(args.weights, device="cpu")
    for path in args.lora:
        n = L.merge_lora_file(model, path, args.lora_strength)
        print(f"merged {n} LoRA layers from {path}")
    if args.bf16_weights:
        model = model.to(torch.bfloat16)
    model = model.to(device).eval()

    def encoder(batch):
        return [cached[t] for t in batch]

    noise = args.noise_scale or O.noise_scale_for(args.height, args.width, O.LossConfig())
    sampler = SamplerConfig(
        height=args.height, width=args.width, sampling_steps=args.steps,
        guidance_scale=args.guidance, noise_scale=noise, negative_prompt=args.negative_prompt,
        noise_sm_count=132 if args.h100_noise else None)

    from PIL import Image

    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    for i, prompt in enumerate(args.prompt):
        images = generate(model=model, text_encoder=encoder, prompt=prompt, seeds=args.seeds,
                          sampler=sampler, device=device)
        for seed, img in zip(args.seeds, images):
            path = os.path.join(args.out_dir, f"p{i}_seed{seed}.png")
            Image.fromarray(img[:, 0].permute(1, 2, 0).cpu().numpy()).save(path)
            records.append({"file": path, "prompt": prompt, "seed": seed})
            print("wrote", path)
    with open(os.path.join(args.out_dir, "prompts.json"), "w") as fh:
        json.dump({"sampler": sampler.to_dict(), "lora": args.lora, "images": records}, fh, indent=2)


if __name__ == "__main__":
    main()
