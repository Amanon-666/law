"""Separate execution-lifetime equivalence from original thread-rounding differences."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

BASE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110')
sys.path.insert(0, str(BASE/'src'))
from law_bench import run
from law_bench.data import load
from law_bench.protocol import make_tasks
from paper_bench_20261111.ext_gp_cuda import ReFiGPCUDA
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.set_num_threads(4)
    root = BASE/'results/paper_uji_B0_ext_20261111'
    config = json.loads((root/'freeze.json').read_text())['configuration']
    buildings, aps = load('uji')
    sources = [dict(dataset='uji', building=b, floors=buildings[b], ap_ids=aps[b])
               for b in sorted(buildings) if b != 'B0']
    options = dict(config['method_options']['refi_GP'])
    options['gp_options'] = dict(options['gp_options'], fit_missing_history=True)
    candidate = ReFiGPCUDA(**options).fit_source(sources)
    original = joblib.load(root/'source_models/refi_GP.joblib')
    tasks = make_tasks('uji', 'B0', buildings['B0'], aps['B0'], 0,
        budgets=tuple(config['budgets']), repeat=0, seed=config['seed'],
        R=config['R'], query_positions=config['query_positions'])
    task, truth = next((task, truth) for task, truth in tasks if task.K == 5)
    expected = json.loads((root/'events/refi_GP'/(task.task_id+'.json')).read_text())
    assert expected['query_row_indices'] == truth['row_indices'].tolist()
    tick = time.monotonic()
    old_prediction = original.predict(task)
    new_prediction = candidate.predict(task)
    old_state = original.gp.diagnostics['state']
    new_state = candidate.gp.diagnostics['state']
    differences = dict(
        same_thread_complete=float(np.abs(old_prediction-new_prediction).max()),
        same_thread_gp=float(np.abs(np.asarray(original.diagnostics['GP_projected'])-
                                    np.asarray(candidate.diagnostics['GP_projected'])).max()),
        same_thread_weights=float(np.abs(np.asarray(old_state['weights'])-
                                         np.asarray(new_state['weights'])).max()),
        stored_complete=float(np.abs(new_prediction-np.asarray(expected['prediction'])).max()),
        stored_gp=float(np.abs(np.asarray(candidate.diagnostics['GP_projected'])-
                              np.asarray(expected['diagnostics']['GP_projected'])).max()),
        stored_weights=float(np.abs(np.asarray(new_state['weights'])-
                                   np.asarray(expected['diagnostics']['GP']['state']['weights'])).max()))
    assert differences['same_thread_complete'] <= 1e-7, differences
    assert differences['same_thread_gp'] <= 1e-7, differences
    assert differences['same_thread_weights'] <= 1e-7, differences
    assert old_state['sigma']['intercept_std'] == new_state['sigma']['intercept_std']
    assert not candidate.gp._fields
    assert all(item['cache'] != 'fitted' for item in candidate.gp.diagnostics['history_fields'])
    wrapper_path = Path(__file__).parent.parent/'cuda_streaming_longrun_20261114/resume_cuda.py'
    spec = importlib.util.spec_from_file_location('checked_execution_wrapper', wrapper_path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    try:
        wrapper.verify_model(original, 'negative_old_execution_model')
    except ValueError:
        old_rejected = True
    else:
        raise AssertionError('Old eager execution model was accepted')
    current = joblib.load(BASE/'results/paper_tampere_TIE1_ext_cuda_20261114/source_models/refi_GP.joblib')
    wrapper.verify_model(current, 'positive_current_execution_model')
    result = dict(passed=True, task_id=task.task_id, threads=4,
        seconds=time.monotonic()-tick, differences=differences,
        intercept_std=new_state['sigma']['intercept_std'],
        retained_history_objects=len(candidate.gp._fields), old_model_rejected=old_rejected)
    (Path(__file__).parent/'uji_k5_same_threads_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
