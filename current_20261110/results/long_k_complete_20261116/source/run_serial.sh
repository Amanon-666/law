#!/bin/bash
set -u
BASE=/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110
WORK="$BASE/results/long_k_complete_20261116"
OUT="$WORK/data"
PYTHON=/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python
cd "$WORK/source" || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=$PWD:$BASE/src:/home/panyushuo/projects/panyushuo/ReFi-UJI-Core-20261031:/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages
export LAW_UJI_CSV=/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv
start=$(date +%s)
finish() {
    status=$?
    end=$(date +%s)
    printf '{"exit_code":%s,"start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s}\n' "$status" "$start" "$end" "$((end-start))" > "$WORK/run.exit.json"
}
trap finish EXIT
printf '{"status":"running","stage":"fixed_K20_timing"}\n' > "$WORK/run_status.json"
tick=$(date +%s)
timeout --signal=TERM 900 "$PYTHON" -u run_long_k.py --output "$OUT" --timing-task > "$WORK/timing.log" 2>&1
status=$?
end=$(date +%s)
printf '{"exit_code":%s,"elapsed_seconds":%s}\n' "$status" "$((end-tick))" > "$WORK/timing.exit.json"
if [ "$status" -ne 0 ]; then
    printf '{"status":"failed","stage":"fixed_K20_timing"}\n' > "$WORK/run_status.json"
    exit "$status"
fi
"$PYTHON" - "$OUT/timing_estimate.json" <<'PY'
import json,sys
estimate=json.load(open(sys.argv[1]))
print(json.dumps(estimate),flush=True)
if estimate['forecast_hours'] > 4:
    raise SystemExit('Forecast exceeds the predeclared routine four-hour planning limit')
PY
status=$?
if [ "$status" -ne 0 ]; then
    printf '{"status":"cost_exceeds_routine_limit","stage":"after_K20_timing"}\n' > "$WORK/run_status.json"
    exit "$status"
fi
printf '{"status":"running","stage":"complete195"}\n' > "$WORK/run_status.json"
tick=$(date +%s)
timeout --signal=TERM 18000 "$PYTHON" -u run_long_k.py --output "$OUT" > "$WORK/full.log" 2>&1
status=$?
end=$(date +%s)
printf '{"exit_code":%s,"elapsed_seconds":%s}\n' "$status" "$((end-tick))" > "$WORK/full.exit.json"
if [ "$status" -ne 0 ]; then
    printf '{"status":"failed","stage":"complete195"}\n' > "$WORK/run_status.json"
    exit "$status"
fi
printf '{"status":"completed","stage":"complete195"}\n' > "$WORK/run_status.json"
