#!/bin/sh
# usage: sh bench.sh [suffix]  -> pairs/s at the inbetween_sparse_night config
./train_gpu-inbetween_sparse${1} --data data_duelyst_hr_soc_ib_val --out /tmp/bench_ckpt --steps 200 --batch 42 --warmup 1000 --dim 256 --layers 6 --heads 8 --muon --lr 1e-2 --context-drop 0 --history-drop 0 --log-every 50 --save-every 100000 --val-every 0 2>&1 | grep -E "step (150|200) "
