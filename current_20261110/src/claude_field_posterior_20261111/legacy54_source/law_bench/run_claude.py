"""Claude's runner wrapper: registers Claude-owned method entries IN-PROCESS and delegates to law_bench.run (root's
file is not modified). Usage (from current_20261110/src):
  python -m law_bench.run_claude --dataset uji --building B0 --methods ours_A --output results/...
"""
from . import run as R

R.METHODS.setdefault('ours_A', ('methods_claude', 'OursMethod', {'variant': 'A'}))
# reader ablations of a SAVED model (model_path supplied per fold through --method-options)
for _tag, _reader in (('same', {}), ('no_offset', {'offsets': False}), ('no_robust', {'eps': 1e-12}),
                      ('no_temper', {'temper': False}), ('knn3', {'knn3': True})):
    R.METHODS.setdefault(f'ours_loaded_{_tag}', ('methods_claude', 'OursLoaded', {'reader': _reader, 'tag': _tag}))

if __name__ == '__main__':
    R.main()
