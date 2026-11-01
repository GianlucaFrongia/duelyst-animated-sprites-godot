#!/bin/sh
# per-kernel attention timings + registers for the inbetween_sparse_night config
rm -rf /tmp/profa && rocprofv3 --kernel-trace -d /tmp/profa -o run -- ./train_gpu-inbetween_sparse${1} --data data_duelyst_hr_soc_ib_val --out /tmp/bench_ckpt --steps 12 --batch 42 --warmup 1000 --dim 256 --layers 6 --heads 8 --muon --lr 1e-2 --context-drop 0 --history-drop 0 --log-every 10 --save-every 100000 --val-every 0 >/dev/null 2>&1
python3 - <<'P'
import sqlite3
c=sqlite3.connect("/tmp/profa/run_results.db")
for n,a,v,sc,l in c.execute("select name, avg(duration), max(vgpr_count), max(scratch_size), max(lds_size) from kernels where name like '%attn%' group by name order by 2 desc"):
    print(f"{a/1e3:8.1f}us  vgpr {v:3d} scratch {sc:4d} lds {l:6d}  {n[:40]}")
P
