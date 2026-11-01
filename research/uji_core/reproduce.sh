#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Pass the directory containing the two UCI CSVs.
data_dir=${1:?UJI CSV directory required}
out_dir=${2:?New output directory required}
mkdir -p "$out_dir"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
module=research.uji_core
python -m "$module.check_core"
python -m "$module.study" "$data_dir" "$out_dir"
python -m "$module.wknn_reference" "$data_dir" "$out_dir"
python -m "$module.forward" "$data_dir" --output "$out_dir"
python -m "$module.forward_reliability" "$data_dir" --output "$out_dir"
python -m "$module.residual_covariance" "$data_dir" "$out_dir"
python -m "$module.forward_match" "$data_dir" --output "$out_dir"
python -m "$module.device_stress" "$data_dir" "$out_dir" "$out_dir"
python -m "$module.field_shape" "$data_dir" "$out_dir"
python -m "$module.smooth_scale" "$data_dir" "$out_dir"
python -m "$module.propagation_control" "$data_dir" "$out_dir/propagation.csv"
python -m "$module.map_readout" "$data_dir" "$out_dir"
python -m "$module.verify_deployment" "$data_dir" "$out_dir"
python -m "$module.summarize" "$out_dir"
python -m "$module.plot" "$out_dir"
