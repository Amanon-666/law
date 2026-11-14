"""Compare the CUDA/streaming executor with completed UJI task predictions."""
import json
import os
from pathlib import Path
import resource
import sys
import time

BASE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110')
sys.path.insert(0, str(BASE / 'src'))

from law_bench import run
from law_bench.data import load
from law_bench.protocol import make_tasks
from paper_bench_20261111.ext_gp_cuda import ReFiGPCUDA
from paper_bench_20261111.ext_gp_streaming import _NoFieldCache
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.set_num_threads(4)
    root = BASE / 'results/paper_uji_B0_ext_20261111'
    freeze = json.loads((root / 'freeze.json').read_text())
    config = freeze['configuration']
    buildings, aps = load('uji')
    sources = [dict(dataset='uji', building=b, floors=buildings[b], ap_ids=aps[b])
               for b in sorted(buildings) if b != 'B0']
    options = dict(config['method_options']['refi_GP'])
    options['gp_options'] = dict(options['gp_options'], fit_missing_history=True)
    model = ReFiGPCUDA(**options).fit_source(sources)
    assert isinstance(model.gp._fields, _NoFieldCache)
    tasks = make_tasks('uji', 'B0', buildings['B0'], aps['B0'], 0,
        budgets=tuple(config['budgets']), repeat=0, seed=config['seed'],
        R=config['R'], query_positions=config['query_positions'])
    results = []
    for task, truth in tasks:
        tick = time.monotonic()
        expected = json.loads((root / 'events/refi_GP' / (task.task_id + '.json')).read_text())
        assert expected['status'] == 'completed'
        assert expected['query_row_indices'] == truth['row_indices'].tolist()
        actual = model.predict(task)
        delta = float(np.abs(actual - np.asarray(expected['prediction'])).max())
        gp_delta = float(np.abs(np.asarray(model.diagnostics['GP_projected']) -
                               np.asarray(expected['diagnostics']['GP_projected'])).max())
        state = model.gp.diagnostics['state']
        expected_state = expected['diagnostics']['GP']['state']
        weight_delta = float(np.abs(np.asarray(state['weights']) -
                                   np.asarray(expected_state['weights'])).max())
        assert weight_delta <= 1e-7, (task.task_id, weight_delta)
        assert state['sigma']['intercept_std'] == expected_state['sigma']['intercept_std']
        assert delta <= 1e-7, (task.task_id, delta)
        assert gp_delta <= 1e-7, (task.task_id, gp_delta)
        assert not model.gp._fields
        assert all(field['cache'] != 'fitted' for field in model.gp.diagnostics['history_fields'])
        record = dict(task_id=task.task_id, max_prediction_difference=delta,
            max_gp_prediction_difference=gp_delta, seconds=time.monotonic()-tick,
            max_adaptation_weight_difference=weight_delta,
            intercept_std=state['sigma']['intercept_std'],
            retained_history_objects=len(model.gp._fields))
        results.append(record)
        print(json.dumps(record), flush=True)
    complete = dict(passed=True, original_protocol=config, results=results,
        model_sha256=model.provenance['model_sha256'],
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (Path(__file__).parent / 'uji_complete_check.json').write_text(json.dumps(complete, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tasks=len(results))), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
