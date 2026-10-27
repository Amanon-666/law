#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies
law_python=/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python
trap 'law_status=$?; printf "%s\n" "$law_status" > results/tcms_history_bank_v1_20261026_exit.txt' EXIT
"$law_python" run_tcms_history_bank.py check
if [ ! -d results/tcms_history_bank_v1_20261026 ]; then
    "$law_python" run_tcms_history_bank.py prepare
fi
run_variant() {
    local law_variant="$1" law_gpu="$2"
    for law_floor in B1F1 B1F2; do
        for law_seed in 0 1 2; do
            CUDA_VISIBLE_DEVICES="$law_gpu" "$law_python" run_tcms_history_bank.py event --event "${law_floor}_s${law_seed}" --variant "$law_variant"
        done
    done
}
run_variant tcms_h2_group_loo 0 > results/tcms_history_bank_h2.log 2>&1 &
law_h2=$!
run_variant tcms_all5_group_loo 1 > results/tcms_history_bank_all5.log 2>&1 &
law_all5=$!
law_status=0
wait "$law_h2" || law_status=$?
wait "$law_all5" || law_status=$?
if [ "$law_status" -ne 0 ]; then exit "$law_status"; fi
"$law_python" run_tcms_history_bank.py summarize
