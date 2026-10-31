# Training on an AMD Radeon (RX 6800 XT) under Linux

The code cross-compiles for the RX 6800/6900 XT (`gfx1030`, RDNA2, listed in Modular's
GPU compatibility table). The fast Metal kernels in `src/metal.mojo` are Apple-only; on
AMD the generic Mojo kernels run. `stacked` and `sparse` build full 2048x2048 attention
matrices there and won't fit at batch 42: use `ctx`, `ctx2`, `inbetween` or `inbetween_sparse`.

## 1. Linux and the AMD driver (once)

Ubuntu 24.04 installed natively (dual boot) is the simplest; WSL also works but is slower to set up.

1. Install the AMD GPU driver with ROCm by following AMD's *ROCm quick start installation
   guide* for Ubuntu (rocm.docs.amd.com): download the `amdgpu-install` package it links, then
   `sudo amdgpu-install --usecase=rocm`. Mojo needs AMD GPU driver 6.3.3 or later.
2. `sudo usermod -aG render,video $USER`, then reboot.
3. Check: `rocminfo | grep gfx` prints `gfx1030`, and `rocm-smi` lists the card.

## 2. Copy the project from the Mac

From the Mac, in the project folder (replace `pc` with the PC's user@host):

```bash
rsync -av --exclude /.venv --exclude /raw --exclude /runs --exclude /out --exclude '/data*' \
  --exclude /checkpoints --exclude /ui --exclude '/docs/*.tgz' --exclude '/docs/*.zip' \
  --exclude /train_gpu --exclude '/train_gpu-*' --exclude /sample --exclude '/sample-*' \
  --exclude /gpu_check --exclude '/gpu_check-*' --exclude '/grpo-*' --exclude /psg --exclude /train \
  ./ pc:pixel-sprite-gen/
rsync -av data_duelyst_hr_soc_ib_val pc:pixel-sprite-gen/      # 0.8 GB, the inbetween dataset
```

The `*.mojo` sources are copied; the excluded binaries were built for macOS.

## 3. Build and check (on the PC)

```bash
cd pixel-sprite-gen
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ACCEL=gfx1030 sh tools/build.sh psg
ACCEL=gfx1030 ARCH=inbetween sh tools/build.sh train_gpu sample gpu_check
ACCEL=gfx1030 ARCH=inbetween_sparse sh tools/build.sh train_gpu gpu_check
./gpu_check-inbetween            # must end with "engine matches the CPU reference" (twice)
./gpu_check-inbetween_sparse
```

## 4. Measure speed

```bash
for a in inbetween inbetween_sparse; do
  ./train_gpu-$a --data data_duelyst_hr_soc_ib_val --out checkpoints/amd_test --steps 300 \
    --batch 42 --warmup 100 --dim 128 --layers 4 --heads 4 --muon --lr 1e-2 \
    --context-drop 0 --history-drop 0 --log-every 50 --val-every 0 | grep -E "device|step 300"
done
```

For comparison, the M2 Pro does about 130 pairs/s (`inbetween`) and 178 (`inbetween_sparse`).
Send back the `rocminfo | grep gfx` line, both `gpu_check` outputs and the two `step 300`
lines. AMD-specific kernels are tuned from there, ideally over SSH
(`sudo apt install openssh-server`) so they can be built and timed on the card directly.
