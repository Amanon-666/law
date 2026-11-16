#!/bin/bash
set -eu
base=/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110
work="$base/results/relation_encoding_ablation_execution_20261115/source"
output="$base/results/relation_encoding_ablation_20261115"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages
export LAW_UJI_CSV=/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv
python=/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python
cd "$work"
"$python" -u source_candidate.py --phase train --output "$output"
"$python" -u evaluate.py --output "$output" --gp-cache "$base/results/tab_field_framework_formal_20261115/gp_surfaces"
