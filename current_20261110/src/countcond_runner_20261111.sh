#!/bin/bash
set -euo pipefail
cd /home/panyushuo/projects/panyushuo/claude_fp_wt/current_20261110/src
export PYTHONPATH=/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages
export VT_DIR=$PWD/claude_hurdle_20261110
export LAW_UJI_CSV=/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python
for b in B0 B1 B2; do
 export COUNTCOND_ORIGINAL=$PWD/results/bench_uji_${b}_A_20261111
 resume=; test ! -f results/bench_uji_${b}_countcond_20261111/freeze.json || resume=--resume
 echo START static $b $(date -u +%FT%TZ) >&2
 "$PY" countcond_gate_20261111.py --dataset uji --building "$b" --methods ours_loaded_same ours_loaded_countcond --method-options countcond_bench_uji_${b}_options_20261111.json --output results/bench_uji_${b}_countcond_20261111 ${resume:-}
 echo DONE static $b $(date -u +%FT%TZ) >&2
done
for b in B1 B2; do
 export COUNTCOND_ORIGINAL=$PWD/results/legacy54_${b}_claude_20261111
 resume=; test ! -f results/legacy54_${b}_countcond_20261111/freeze.json || resume=--resume
 echo START legacy54 $b $(date -u +%FT%TZ) >&2
 "$PY" countcond_gate_20261111.py --dataset uji --building "$b" --protocol legacy54 --manifest /home/panyushuo/projects/panyushuo/ReFi-UJI-Core-20261031/results/building_information_assets_20261109/manifest.json --methods ours_loaded_same ours_loaded_countcond --method-options countcond_legacy54_${b}_options_20261111.json --output results/legacy54_${b}_countcond_20261111 ${resume:-}
 echo DONE legacy54 $b $(date -u +%FT%TZ) >&2
done
