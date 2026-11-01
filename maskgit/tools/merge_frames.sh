#!/bin/sh
# Merges frame folders (<class>/<animation>/<frame>.png) into one folder of symlinks, so
# prepare_animations.py --frames can read several sources at once.
#     sh tools/merge_frames.sh raw/frames_all raw/hullrupture raw/songsofconquest_frames
set -e
out=$1; shift
rm -rf "$out"
for src in "$@"; do
    for c in "$src"/*/; do
        mkdir -p "$out/$(basename "$c")"
        for a in "$c"*/; do
            ln -s "$PWD/${a%/}" "$out/$(basename "$c")/$(basename "$a")"
        done
    done
done
