#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies
law_python=/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python
law_output=results/uji_bank_calibration_transfer_v1_20261026
trap 'law_status=$?; printf "%s\n" "$law_status" > results/uji_bank_calibration_transfer_v1_20261026_exit.txt' EXIT
"$law_python" run_bank_calibration_transfer.py check
if [ ! -d "$law_output" ]; then
    "$law_python" run_bank_calibration_transfer.py prepare
fi
run_floor() {
    local law_floor="$1" law_gpu="$2"
    for law_seed in 0 1 2; do
        CUDA_VISIBLE_DEVICES="$law_gpu" "$law_python" run_bank_calibration_transfer.py event --event "${law_floor}_s${law_seed}"
    done
}
run_floor B1F1 0 > results/uji_bank_calibration_transfer_B1F1.log 2>&1 &
law_f1=$!
run_floor B1F2 1 > results/uji_bank_calibration_transfer_B1F2.log 2>&1 &
law_f2=$!
law_status=0
wait "$law_f1" || law_status=$?
wait "$law_f2" || law_status=$?
if [ "$law_status" -ne 0 ]; then exit "$law_status"; fi
"$law_python" run_bank_calibration_transfer.py summarize
