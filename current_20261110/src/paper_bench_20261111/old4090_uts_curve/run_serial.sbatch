#!/bin/bash
#SBATCH --job-name=UTS-Kcurve
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/data/users/yanrongen/projects/panyushuo/PaperGP-20261113/uts_kcurve/logs/%x-%j.log
set -u
base=/data/users/yanrongen/projects/panyushuo/PaperGP-20261113
work="$base/uts_kcurve"
cd "$work" || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=/data/users/yanrongen/projects/panyushuo/TabAssets-B1-20261111/payload/RNP-UJI/.deps_tabicl
export LAW_UJI_CSV=/data/users/yanrongen/projects/panyushuo/TabAssets-B1-20261111/payload/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv
for method in refi wknn_HS retained_gp; do
    if [ "$method" = refi ]; then
        python="$base/refi_rebuild/env/bin/python"
    else
        python="$base/env/bin/python"
    fi
    start=$(date +%s)
    "$python" -u "$work/run_curve.py" "$method" > "$work/logs/${method}-${SLURM_JOB_ID}.log" 2>&1
    rc=$?
    end=$(date +%s)
    printf '{"method":"%s","job_id":"%s","start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s,"exit_code":%s}\n' \
        "$method" "$SLURM_JOB_ID" "$start" "$end" "$((end-start))" "$rc" > "$work/logs/${method}-${SLURM_JOB_ID}.exit.json"
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
