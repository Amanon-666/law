"""Claude's runner wrapper: registers Claude-owned method entries IN-PROCESS and delegates to law_bench.run (root's
file is not modified). Usage (from current_20261110/src):
  python -m law_bench.run_claude --dataset uji --building B0 --methods ours_A --output results/...
"""
from . import run as R

R.METHODS.setdefault('ours_A', ('methods_claude', 'OursMethod', {'variant': 'A'}))

if __name__ == '__main__':
    R.main()
