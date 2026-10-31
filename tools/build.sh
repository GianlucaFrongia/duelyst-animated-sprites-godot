#!/bin/sh
# Builds the Mojo binaries that are older than their sources. Each binary is built to a
# temporary file and renamed into place, so a training that is already running keeps
# its executable intact.
#
# ARCH picks the model architecture (a folder in models/, default ctx). The ctx binaries
# are ./train_gpu, ./sample, ...; any other architecture gets a suffix: ./train_gpu-stacked.
#
#     sh tools/build.sh                     # train_gpu sample gpu_check psg
#     sh tools/build.sh train_gpu           # just one
#     ARCH=stacked sh tools/build.sh        # another architecture
#     ACCEL=gfx1030 sh tools/build.sh       # a non-Apple GPU (here an RX 6800 XT)
set -e
cd "$(dirname "$0")/.."
MOJO=${MOJO:-.venv/bin/mojo}
ARCH=${ARCH:-ctx}
[ -d "models/$ARCH/arch" ] || { echo "no architecture models/$ARCH/arch" >&2; exit 1; }
# GPU to compile for: ACCEL=gfx1030 (RX 6800/6900 XT); Apple chips are detected.
CHIP=${ACCEL:-$(sysctl -n machdep.cpu.brand_string 2>/dev/null | sed -n 's/^Apple \(M[0-9]*\).*/apple-\1/p' | tr 'M' 'm')}
TARGETS=${*:-"train_gpu sample gpu_check psg"}

for t in $TARGETS; do
    out=$t
    [ "$ARCH" = ctx ] || [ "$t" = psg ] || out="$t-$ARCH"
    if [ -f "$out" ] && [ -z "$(find -L "$t.mojo" src "models/$ARCH" -name '*.mojo' -newer "$out" 2>/dev/null)" ]; then
        echo "$out is up to date"
        continue
    fi
    flags="-O3 -I models/$ARCH"
    if [ -n "$CHIP" ] && [ "$t" != "psg" ]; then
        flags="$flags --target-accelerator $CHIP"
    fi
    echo "building $out ($flags)"
    "$MOJO" build $flags "$t.mojo" -o "$out.tmp" 2>&1 | grep -v -e warning -e Crashpad -e "^ " -e "^<unknown>" -e "^Imported" -e "^Included" || true
    [ -f "$out.tmp" ] || { echo "build of $out failed" >&2; exit 1; }
    mv "$out.tmp" "$out"
done
