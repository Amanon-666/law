#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies
law_python=/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python
law_output=results/uji_history_bank_v1_20261026
trap 'law_status=$?; printf "%s\n" "$law_status" > results/uji_history_bank_v1_20261026_exit.txt' EXIT
"$law_python" uji_history_bank_study.py check
if [ ! -d "$law_output" ]; then
    "$law_python" uji_history_bank_study.py prepare
fi
"$law_python" uji_history_bank_study.py history
for law_floor in B1F1 B1F2; do
    for law_seed in 0 1 2; do
        "$law_python" uji_history_bank_study.py event --event "${law_floor}_s${law_seed}"
    done
done
"$law_python" uji_history_bank_study.py summarize
