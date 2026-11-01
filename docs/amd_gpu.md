# Training on an AMD Radeon (RX 6800 XT) under Linux

The code targets AMD GPUs through ROCm, developed against the RX 6800/6900 XT (`gfx1030`, RDNA2,
listed in Modular's GPU compatibility table). Every architecture uses the fused attention in
`src/attention.mojo`, and rocBLAS does the matmuls when `librocblas.so` is installed (the
tiled Mojo matmul otherwise).

## 1. Linux and the AMD driver (once)

Ubuntu 24.04 installed natively (dual boot) is the simplest; WSL also works but is slower to set up.

1. Install the AMD GPU driver with ROCm by following AMD's *ROCm quick start installation
   guide* for Ubuntu (rocm.docs.amd.com): download the `amdgpu-install` package it links, then
   `sudo amdgpu-install --usecase=rocm`. Mojo needs AMD GPU driver 6.3.3 or later.
2. `sudo usermod -aG render,video $USER`, then reboot.
3. Check: `rocminfo | grep gfx` prints `gfx1030`, and `rocm-smi` lists the card.

## 2. Get the project

```bash
git clone git@github.com:GianlucaFrongia/pixel-sprite-gen.git
```

Datasets, checkpoints and `raw/` are not in git: copy them over (for example
`rsync -av data_duelyst_hr_soc_ib_val pc:pixel-sprite-gen/`, 0.8 GB, the inbetween dataset) or
rebuild them with `tools/prepare_animations.py`.

## 3. Build and check (on the PC)

```bash
cd pixel-sprite-gen
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sh tools/build.sh psg            # the GPU target comes from rocminfo (ACCEL=gfx1030 to force it)
for a in ctx ctx2 stacked sparse inbetween inbetween_sparse inbetween_dyt inbetween_hilbert inbetween_crop; do
  ARCH=$a sh tools/build.sh train_gpu sample gpu_check
  ./gpu_check-$a                 # must print "engine matches the CPU reference" twice (3x for inbetween_crop)
done
```

## 4. Measure speed

```bash
for a in inbetween inbetween_sparse; do
  ./train_gpu-$a --data data_duelyst_hr_soc_ib_val --out checkpoints/amd_test --steps 300 \
    --batch 42 --warmup 100 --dim 128 --layers 4 --heads 4 --muon --lr 1e-2 \
    --context-drop 0 --history-drop 0 --log-every 50 --val-every 0 | grep -E "device|step 300"
done
```

The `step 300` lines give pairs/s per architecture: the step counts in `pipelines/` were sized
from earlier hardware, so re-size long runs from these numbers.
