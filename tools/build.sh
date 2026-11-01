#!/bin/sh
# Builds the Mojo binaries that are older than their sources. Each binary is built to a
# temporary file and renamed into place, so a training that is already running keeps
# its executable intact.
#
# ARCH picks the model architecture (a folder in models/, default ctx); every tool except psg
# is built per architecture as ./<tool>-<arch>, e.g. ./train_gpu-ctx, ./sample-inbetween.
#
#     sh tools/build.sh                     # train_gpu sample gpu_check psg
#     sh tools/build.sh train_gpu           # just one
#     ARCH=inbetween sh tools/build.sh      # another architecture
#     ACCEL=gfx1100 sh tools/build.sh       # another AMD GPU than the one rocminfo reports
set -e
cd "$(dirname "$0")/.."
MOJO=${MOJO:-.venv/bin/mojo}
ARCH=${ARCH:-ctx}
[ -f "models/$ARCH/arch/variant.mojo" ] || { echo "no architecture models/$ARCH/arch/variant.mojo" >&2; exit 1; }
# GPU to compile for: the AMD GPU rocminfo reports (gfx1030 is an RX 6800/6900 XT), or ACCEL.
CHIP=${ACCEL:-$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' || true)}
TARGETS=${*:-"train_gpu sample gpu_check psg"}

for t in $TARGETS; do
    out=$t
    [ "$t" = psg ] || out="$t-$ARCH"
    if [ -f "$out" ] && [ -z "$(find *.mojo src "models/$ARCH" -name '*.mojo' -newer "$out" 2>/dev/null)" ]; then
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
