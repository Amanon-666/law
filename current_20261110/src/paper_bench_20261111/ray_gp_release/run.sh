#!/bin/bash
set -u
base=/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110
work="$base/results/paper_tampere_TIE1_ext_execution_20261113/release_20261114"
cd "$base/src" || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages
export VT_DIR=$PWD/claude_hurdle_20261110 LAW_UJI_CSV=/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv
python=/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python
if [ "${1-}" = --threads-only ]; then
    exec "$python" -u "$work/resume_release.py" --threads-only
fi
start=$(date +%s)
nice -n 10 "$python" -u "$work/resume_release.py" --dataset tampere --building TIE1 --source-dataset uji --source-buildings B0 B1 B2 --protocol static-resample-v2 --seed 20261111 --repeats 0 --budgets 1 3 5 --R 3 --query-positions 60 --methods refi_GP --method-options "$base/results/paper_tampere_TIE1_ext_execution_20261113/options.json" --output "$base/results/paper_tampere_TIE1_ext_20261113" --resume > "$work/run.log" 2>&1
status=$?
end=$(date +%s)
printf '{"start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s,"exit_code":%s}\n' "$start" "$end" "$((end-start))" "$status" > "$work/run.exit.json"
exit "$status"
