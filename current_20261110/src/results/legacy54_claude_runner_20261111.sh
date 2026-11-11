#!/bin/bash
set -u
cd /home/panyushuo/projects/panyushuo/claude_fp_wt/current_20261110/src
export PYTHONPATH=/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages
export LAW_UJI_CSV=/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv
export VT_DIR=$PWD/claude_hurdle_20261110
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
for building in B1 B2; do
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  tick=$SECONDS
  echo "START $building $start runner=$$" >> results/legacy54_claude_runner_20261111.status
  /home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python -u -m law_bench.run_claude --dataset uji --protocol legacy54 --manifest /home/panyushuo/projects/panyushuo/ReFi-UJI-Core-20261031/results/building_information_assets_20261109/manifest.json --source-buildings B0 --building "$building" --budgets 1 3 5 --repeats 0 1 --methods wknn_S wknn_HS gp_histmean ours_naive ours_v1 ours_A --method-options results/legacy54_claude_method_options_20261111.json --output "results/legacy54_${building}_claude_20261111" > "results/legacy54_${building}_claude_20261111.log" 2>&1
  rc=$?
  echo "END $building $(date -u +%Y-%m-%dT%H:%M:%SZ) elapsed_s=$((SECONDS-tick)) rc=$rc" >> results/legacy54_claude_runner_20261111.status
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
