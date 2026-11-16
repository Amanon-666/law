#!/bin/bash
set -u
work=/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110/results/relation_encoding_ablation_execution_20261115
start=$(date +%s)
printf '{"status":"running","start_epoch":%s,"launcher_pid":%s}\n' "$start" "$$" > "$work/run_status.json"
bash "$work/source/run_after_tab.sh"
status=$?
end=$(date +%s)
printf '{"exit_code":%s,"start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s}\n' "$status" "$start" "$end" "$((end-start))" > "$work/run.exit.json"
if [ "$status" -eq 0 ]; then
    state=completed
else
    state=failed
fi
printf '{"status":"%s","exit_code":%s,"start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s}\n' "$state" "$status" "$start" "$end" "$((end-start))" > "$work/run_status.json"
exit "$status"
