#!/bin/bash
set -u
base=/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110
work="$base/results/dense_calibration_reference_execution_20261118"
output="$base/results/dense_calibration_reference_20261118"
cd "$work/source" || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=$PWD:$base/src:/home/panyushuo/projects/panyushuo/ReFi-UJI-Core-20261031:/home/panyushuo/projects/panyushuo/RNP-UJI/.deps_tabicl:/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages
export LAW_UJI_CSV=/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv
start=$(date +%s)
/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python -u dense_reference.py --phase run --output "$output"
status=$?
end=$(date +%s)
printf '{"exit_code":%s,"start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s}\n' "$status" "$start" "$end" "$((end-start))" > "$work/run.exit.json"
exit "$status"
