#!/bin/bash
#SBATCH --job-name=PaperTab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/data/users/yanrongen/projects/panyushuo/PaperTab-20261112/logs/%x-%j.log
set -u
assets=/data/users/yanrongen/projects/panyushuo/TabAssets-B1-20261111
work=/data/users/yanrongen/projects/panyushuo/PaperTab-20261112
building=$1
shift
output="$work/results/paper_uji_${building}_tabicl_20261112"
cd "$assets/source_snapshot" || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$assets/cache/huggingface" XDG_CACHE_HOME="$assets/cache"
export PYTHONPATH="$assets/source_snapshot:$assets/payload/RNP-UJI/.deps_tabicl"
export LAW_UJI_CSV="$assets/payload/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv"
start=$(date +%s)
echo "building=$building job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
sha256sum "$LAW_UJI_CSV" tabicl_history.py law_bench/method_tabicl.py law_bench/protocol.py
/data/users/yanrongen/conda-envs/forestry_llm/bin/python -u -m law_bench.run \
  --dataset uji --root "$LAW_UJI_CSV" --building "$building" \
  --protocol static-resample-v2 --seed 20261111 --budgets 1 3 5 --repeats "$@" \
  --R 3 --query-positions 60 --methods tabicl \
  --method-options "$work/options.json" --output "$output"
rc=$?
end=$(date +%s)
printf '{"building":"%s","job_id":"%s","start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s,"exit_code":%s}\n' \
  "$building" "$SLURM_JOB_ID" "$start" "$end" "$((end-start))" "$rc" > "$output.exit.json"
exit "$rc"
