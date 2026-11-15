"""Cache the original B1 GP score surfaces once for every matched field arm."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from law_bench import run
from law_bench.data import load
from law_bench.method_retained_gp import RetainedGPMethod
from law_bench.protocol import make_tasks
import predict as gp_prediction


BASE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110')
REFERENCE = BASE/'results/paper_uji_B1_ext_20261111'


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    original = json.loads((REFERENCE/'freeze.json').read_text())
    config = original['configuration']
    options = config['method_options']['refi_GP']['gp_options']
    assert options['fit_missing_history'] is False
    buildings, axes = load('uji')
    sources = [dict(dataset='uji',building=b,floors=buildings[b],ap_ids=axes[b]) for b in ['B0','B2']]
    assert run._data_identity({b:buildings[b] for b in ['B0','B2']},{b:axes[b] for b in ['B0','B2']}) == original['source_data']
    assert run._data_identity({'B1':buildings['B1']},{'B1':axes['B1']}) == original['target_data']
    freeze = dict(reference_freeze_sha256=run._sha(REFERENCE/'freeze.json'),
        target_data=original['target_data'],source_data=original['source_data'],gp_options=options,
        code_sha256=run._sha(Path(__file__)),
        purpose='Reuse original exact GP adaptation state and score surface across all new field arms.',
        target_GT_scored=False,training_performed=False,new_history_fitting_allowed=False)
    path = args.output/'freeze.json'
    if path.exists():
        assert json.loads(path.read_text()) == freeze
    else:
        run._write(path,freeze)
    model = RetainedGPMethod(**options).fit_source(sources)
    records = []
    for floor in sorted(buildings['B1']):
        for repeat in (0,1,2):
            tasks = make_tasks('uji','B1',buildings['B1'],axes['B1'],floor,
                budgets=(1,3,5),repeat=repeat,seed=config['seed'],R=config['R'],query_positions=config['query_positions'])
            for task, truth in tasks:
                source = REFERENCE/'events/refi_GP'/(task.task_id+'.json')
                old = json.loads(source.read_text())
                assert old['status'] == 'completed'
                assert old['query_row_indices'] == truth['row_indices'].tolist()
                assert old['query_groups'] == truth['groups'].tolist() and old['meta'] == task.meta
                destination = args.output/(task.task_id+'.npz')
                metadata = destination.with_suffix('.json')
                binding = dict(task_id=task.task_id,reference_event_sha256=run._sha(source),
                    query_rows=old['query_row_indices'],query_groups=old['query_groups'],meta=old['meta'])
                if metadata.exists():
                    record = json.loads(metadata.read_text())
                    assert record['binding'] == binding and run._sha(destination) == record['sha256']
                else:
                    tick = time.monotonic()
                    with patch.object(gp_prediction,'adapt',return_value=old['diagnostics']['GP']['state']):
                        gp = model.predict_scores(task)
                    delta = float(np.max(np.abs(gp['projected']-np.asarray(old['diagnostics']['GP_projected']))))
                    assert delta <= 1e-6
                    assert all(f['cache'] != 'fitted' for f in model.diagnostics['history_fields'])
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(destination,**{key:gp[key] for key in ['grid','scores','raw','projected','aps','bounds']})
                    record = dict(binding=binding,sha256=run._sha(destination),seconds=time.monotonic()-tick,
                        original_projection_max_abs_difference=delta,new_history_fields=0,target_GT_scored=False)
                    run._write(metadata,record)
                records.append(dict(task_id=task.task_id,path=str(destination),metadata=str(metadata),
                    sha256=record['sha256'],original_projection_max_abs_difference=record['original_projection_max_abs_difference']))
                print(json.dumps(dict(stage='original_GP_score_cache',completed=len(records),task_id=task.task_id)),flush=True)
    assert len(records) == 36
    run._write(args.output/'complete.json',dict(complete=True,tasks=records,
        freeze_sha256=run._sha(path),training_performed=False,new_history_fields=0,target_GT_scored=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    with threadpool_limits(limits=2):
        main(args)
