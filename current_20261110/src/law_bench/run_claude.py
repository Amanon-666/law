"""Claude's runner wrapper: registers Claude-owned method entries IN-PROCESS and delegates to law_bench.run (root's
file is not modified). Usage (from current_20261110/src):
  python -m law_bench.run_claude --dataset uji --building B0 --methods ours_A --output results/...
"""
from . import run as R

R.METHODS.setdefault('ours_A', ('methods_claude', 'OursMethod', {'variant': 'A'}))
# matched single-factor 'w/o learning': learned H-only field read WITHOUT S conditioning (vs ours_naive_noM2)
R.METHODS.setdefault('ours_loaded_fieldonly', ('methods_claude', 'OursLoaded', {'tag': 'fieldonly', 'reader': {'field_only': True}}))
# A0: number of history floors (saved model, no training)
for _n in (1, 2):
    R.METHODS.setdefault(f'ours_loaded_h{_n}', ('methods_claude', 'OursLoaded', {'tag': f'h{_n}', 'hfloors': _n}))
# standard DL transfer baselines (AP-vector DNN localiser; same building's history only)
R.METHODS.setdefault('finetune_dnn', ('methods_dl', 'FineTuneDNN', {}))
R.METHODS.setdefault('pretrain_dnn', ('methods_dl', 'PretrainDNN', {}))
R.METHODS.setdefault('maml_dnn', ('methods_dl', 'MAMLDNN', {}))
# reader ablations of a SAVED model (model_path supplied per fold through --method-options)
for _tag, _reader in (('same', {}), ('no_offset', {'offsets': False}), ('no_robust', {'eps': 1e-12}),
                      ('no_temper', {'temper': False}), ('knn3', {'knn3': True})):
    R.METHODS.setdefault(f'ours_loaded_{_tag}', ('methods_claude', 'OursLoaded', {'reader': _reader, 'tag': _tag}))

if __name__ == '__main__':
    R.main()
