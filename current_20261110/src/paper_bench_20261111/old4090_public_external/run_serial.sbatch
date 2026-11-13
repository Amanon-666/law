#!/bin/bash
#SBATCH --job-name=PaperTabExternal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/data/users/yanrongen/projects/panyushuo/PaperGP-20261113/public_tab/logs/%x-%j.log
set -u
work=/data/users/yanrongen/projects/panyushuo/PaperGP-20261113/public_tab
assets=/data/users/yanrongen/projects/panyushuo/TabAssets-B1-20261111
cd "$assets/source_snapshot" || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$work/cache/huggingface" XDG_CACHE_HOME="$work/cache"
export PYTHONPATH="$assets/source_snapshot:$assets/payload/RNP-UJI/.deps_tabicl"
export LAW_UJI_CSV="$assets/payload/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv"
for group in TIE1 SAH1 SOD UTS; do
    start=$(date +%s)
    timeout --signal=TERM 1200 /data/users/yanrongen/conda-envs/forestry_llm/bin/python \
        -u "$work/run_group.py" "$group" > "$work/logs/${group}-${SLURM_JOB_ID}.log" 2>&1
    rc=$?
    end=$(date +%s)
    printf '{"group":"%s","job_id":"%s","start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s,"exit_code":%s}\n' \
        "$group" "$SLURM_JOB_ID" "$start" "$end" "$((end-start))" "$rc" > "$work/logs/${group}-${SLURM_JOB_ID}.exit.json"
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
