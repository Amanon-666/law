#!/bin/bash
# Usage: bash reproduce_multi_history.sh /path/to/python /new/output/directory
# Uses existing local data and frozen weights; never replaces an old run.
set -euo pipefail
cd "$(dirname "$0")"
law_python="${1:?provide an existing Python with torch/numpy/scipy/pypmc}"
law_output="${2:?provide a new output directory}"
"$law_python" check_multi_history.py
"$law_python" run_multi_history.py prepare --out "$law_output"
"$law_python" run_multi_history.py audit --out "$law_output"
"$law_python" run_multi_history.py train --out "$law_output"
"$law_python" run_multi_history.py evaluate --out "$law_output"
"$law_python" diagnose_multi_history.py --out "$law_output"
