#!/bin/bash
set -u
base=/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110
work="$base/src/paper_bench_20261111/ray_uts_curve"
cd "$base/src" || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages
export VT_DIR=$PWD/claude_hurdle_20261110 LAW_UJI_CSV=/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv
for method in refi wknn_HS retained_gp; do
    start=$(date +%s)
    nice -n 10 /home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python -u "$work/run_curve.py" "$method" > "$work/logs/${method}.log" 2>&1
    rc=$?
    end=$(date +%s)
    printf '{"method":"%s","start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s,"exit_code":%s}\n' \
        "$method" "$start" "$end" "$((end-start))" "$rc" > "$work/logs/${method}.exit.json"
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
