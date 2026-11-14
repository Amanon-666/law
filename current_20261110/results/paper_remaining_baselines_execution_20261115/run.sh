#!/bin/bash
set -u
base=/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110
work="$base/results/paper_remaining_baselines_execution_20261115"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/panyushuo/projects/panyushuo/claude_fp_wt/current_20261110/src:/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages
export LAW_UJI_CSV=/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv
export LAW_ROOT=/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/Law-Probabilistic-Transfer-20261023
cd /home/panyushuo/projects/panyushuo/claude_fp_wt/current_20261110/src || exit 1
start=$(date +%s)
/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python -u "$work/batch.py" > "$work/batch.log" 2>&1
status=$?
end=$(date +%s)
printf '{"start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s,"exit_code":%s}\n' "$start" "$end" "$((end-start))" "$status" > "$work/run.exit.json"
exit "$status"
