#!/bin/bash
set -u
work=/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110/results/paper_uji_B1_F0_F1_copied_gp_execution_20261114
export PYTHONPATH=/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110/src:/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
started=$(date +%s)
/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python -u "$work/run_copy_gp.py" > "$work/run.log" 2>&1
result=$?
ended=$(date +%s)
printf '{"exit_code":%d,"start_epoch":%d,"end_epoch":%d,"elapsed_seconds":%d}\n' "$result" "$started" "$ended" "$((ended-started))" > "$work/run.exit.json"
exit "$result"
