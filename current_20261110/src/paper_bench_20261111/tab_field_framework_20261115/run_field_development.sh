#!/bin/bash
set -u
BASE=/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110
WORK="$BASE/results/tab_field_framework_formal_20261115"
cd "$WORK/source" || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=$PWD:$BASE/src:/home/panyushuo/projects/panyushuo/ReFi-UJI-Core-20261031:/home/panyushuo/projects/panyushuo/RNP-UJI/.deps_tabicl:/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages
export LAW_UJI_CSV=/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv
start=$(date +%s)
timeout --signal=TERM 5400 /home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python -u formal_source.py --phase development --seed 1 --epochs 3 --output "$WORK/field_development" > "$WORK/field_development.log" 2>&1
status=$?
end=$(date +%s)
printf '{"exit_code":%s,"start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s}\n' "$status" "$start" "$end" "$((end-start))" > "$WORK/field_development.exit.json"
exit "$status"
