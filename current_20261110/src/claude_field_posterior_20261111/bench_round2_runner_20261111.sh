#!/bin/bash
cd /home/panyushuo/projects/panyushuo/claude_fp_wt/current_20261110/src || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages
export VT_DIR=$PWD/claude_hurdle_20261110 LAW_UJI_CSV=/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv
start=$(date +%s)
echo "start=$(date -u +%FT%TZ) runner_pid=$$"
/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python -u -m law_bench.run_claude "$@" &
child=$!
echo "child_pid=$child"
wait "$child"
rc=$?
echo "end=$(date -u +%FT%TZ) elapsed_seconds=$(($(date +%s)-start)) exit_code=$rc"
exit "$rc"
