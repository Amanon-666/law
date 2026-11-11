#!/bin/bash
cd /home/panyushuo/projects/panyushuo/claude_xacq_wt/current_20261110/src/xacq_selfcal_claude_20261111 || exit 1
export PYTHONPATH=/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo $$ > results/runner.pid
date -u +%FT%TZ > results/runner.started
if [ "$#" -eq 0 ]; then set -- uji ipin; fi
for phase in "$@"; do
    /usr/bin/time -v -o results/$phase.resources.txt nice -n 10 /home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python -u run.py "$phase" > results/$phase.log 2>&1 &
    task_pid=$!
    echo "$task_pid" > results/$phase.pid
    wait "$task_pid"
    task_rc=$?
    echo "$task_rc" > results/$phase.exit
    if [ "$task_rc" != 0 ]; then exit "$task_rc"; fi
done
date -u +%FT%TZ > results/runner.completed
