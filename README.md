# pixel-art-training

Training animated pixel-art sprite generators. Each folder is its own project with its own README and setup.

| Folder | What it is | Start here |
|---|---|---|
| [`data/duelyst/`](data/duelyst/) | The source sprites: 600+ Duelyst units with attack, idle, run and death animations, as a Godot 4 project (fork of [Jordyfel/duelyst-animated-sprites-godot](https://github.com/Jordyfel/duelyst-animated-sprites-godot), CC0) | [README](data/duelyst/README.md) |
| [`maskgit/`](maskgit/) | pixel-sprite-gen: a MaskGIT-style transformer, hand-written in Mojo, that paints 64×64 palette-indexed frames one at a time and fills in frames between keyframes. Trains on AMD GPUs (ROCm). | [README](maskgit/README.md), [GPU setup](maskgit/docs/amd_gpu.md) |
| [`lora/`](lora/) | A LoRA on Wan 2.1 1.3B (video) or FLUX.2 klein (sprite sheets), trained with musubi-tuner, plus conversion of the output back into Godot `SpriteFrames` | [README](lora/README.md) |

Run `lora/` scripts from the repository root. Run `maskgit/` commands from inside `maskgit/`.

Datasets, checkpoints and run outputs are not in git (see each folder's `.gitignore`).
