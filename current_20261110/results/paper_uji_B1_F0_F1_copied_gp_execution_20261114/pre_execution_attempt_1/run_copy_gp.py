"""Fixed copied-field replacement inside the existing complete ReFi system."""
import hashlib
import json
from pathlib import Path
import sys
import time

BASE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110')
sys.path.insert(0, str(BASE / 'src'))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from law_bench import run, methods_claude
from law_bench.data import load
from law_bench.method_retained_gp import RetainedGPMethod
from law_bench.protocol import make_tasks
from law_bench.score import score, aggregate
from hurdle_gp_pool import fuse_hurdle_gp

REFERENCE = BASE / 'results/paper_uji_B1_ext_20261111'
COPIED = Path('/home/panyushuo/projects/panyushuo/claude_fp_wt/current_20261110/src/results/bench_uji_B1_extra_20261111')
OUTPUT = BASE / 'results/paper_uji_B1_F0_F1_copied_gp_20261114'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    original = json.loads((REFERENCE / 'freeze.json').read_text())
    config = original['configuration']
    copy_freeze = json.loads((COPIED / 'freeze.json').read_text())
    copy_fit = json.loads((COPIED / 'source_models/ours_naive_noM2.json').read_text())
    model_path = COPIED / 'source_models/ours_naive_noM2.joblib'
    assert run._sha(model_path) == copy_fit['sha256']
    assert original['target_data'] == copy_freeze['target_data']
    assert original['source_data'] == copy_freeze['source_data']
    buildings, aps = load('uji')
    sources = [dict(dataset='uji', building=b, floors=buildings[b], ap_ids=aps[b])
               for b in ['B0', 'B2']]
    assert run._data_identity({s['building']: s['floors'] for s in sources},
                              {s['building']: s['ap_ids'] for s in sources}) == original['source_data']
    assert run._data_identity({'B1': buildings['B1']}, {'B1': aps['B1']}) == original['target_data']
    copied = joblib.load(model_path)
    assert copied.variant == copied.model.variant == 'naive_noM2'
    assert copied.model.ops == [] and not copied.inf
    gp_options = config['method_options']['refi_GP']['gp_options']
    assert gp_options['fit_missing_history'] is False
    gp_model = RetainedGPMethod(**gp_options).fit_source(sources)
    freeze = dict(created_epoch=time.time(), configuration=dict(dataset='uji', building='B1',
        floors=[0, 1], budgets=[1, 3, 5], repeats=[0], seed=config['seed'], R=config['R'],
        query_positions=config['query_positions']), reference_freeze_sha256=run._sha(REFERENCE / 'freeze.json'),
        copied_freeze_sha256=run._sha(COPIED / 'freeze.json'), copied_model_sha256=copy_fit['sha256'],
        copied_variance=list(copied.model.naive_var), target_data=original['target_data'],
        source_data=original['source_data'], gp_options=gp_options,
        code={str(Path(p).resolve()): run._sha(p) for p in
              [__file__, methods_claude.__file__, methods_claude.O.__file__]},
        training_performed=False, new_history_fitting_allowed=False,
        review=dict(question='Does learned history reconstruction improve the complete system relative to copying?',
            changed='Replace learned observation field with the archived source-calibrated copied field.',
            fixed='H/S/Q, candidate ordering and empirical S atoms, AP axis, reader, GP adaptation and half-log pooling.',
            deployment='Copied fields use H, S and Query RSS only. Query XY enters score after prediction.',
            counterexamples='Identical field distributions yield identical pooling. A uniform field tempers GP. Incorrect copied priors can degrade it. Both sides retain the same H/S candidate and coordinate bounds.',
            support_and_missing_AP='Existing copied reader and GP retain their original support/new-AP handling; only K1/3/5 are evaluated.',
            invariance='Historical positions, physical floor IDs and AP axis remain unchanged between arms.',
            decision='Retain all six tasks regardless of outcome. This is an exposed-development mechanism comparison.'))
    run._write(OUTPUT / 'freeze.json', freeze)
    prepared = []
    for floor in [0, 1]:
        tasks = make_tasks('uji', 'B1', buildings['B1'], aps['B1'], floor,
            budgets=(1, 3, 5), repeat=0, seed=config['seed'], R=config['R'],
            query_positions=config['query_positions'])
        for task, truth in tasks:
            expected = json.loads((REFERENCE / 'events/refi_GP' / (task.task_id + '.json')).read_text())
            old_copy = json.loads((COPIED / 'events/ours_naive_noM2' / (task.task_id + '.json')).read_text())
            assert expected['status'] == old_copy['status'] == 'completed'
            assert expected['query_row_indices'] == old_copy['query_row_indices'] == truth['row_indices'].tolist()
            assert expected['query_groups'] == old_copy['query_groups'] and expected['meta'] == old_copy['meta']
            prediction, candidates, log_mass, bounds = methods_claude.predict_mass(copied, task)
            delta = float(np.max(np.abs(prediction - np.asarray(old_copy['prediction']))))
            assert delta <= 1e-8, (task.task_id, delta)
            expected_candidates = np.vstack([
                np.unique(np.concatenate([d['xy'] for d in task.H.values()]), axis=0),
                np.unique(task.S['xy'], axis=0)])
            assert np.array_equal(candidates, expected_candidates)
            prepared.append((task, truth, expected, candidates, log_mass, bounds, delta))
    run._write(OUTPUT / 'pre_execution_check.json', dict(completed=True, tasks=len(prepared),
        max_copied_prediction_difference=max(p[-1] for p in prepared), code_sha256=run._sha(__file__)))
    records = []
    for task, truth, expected, candidates, log_mass, bounds, copy_delta in prepared:
        started = time.monotonic()
        gp = gp_model.predict_scores(task)
        gp_delta = float(np.max(np.abs(gp['projected'] - np.asarray(expected['diagnostics']['GP_projected']))))
        assert gp_delta <= 1e-6 and np.array_equal(bounds, gp['bounds']), (task.task_id, gp_delta)
        assert np.array_equal(gp['aps'], methods_claude._hs_aps(task))
        assert all(field['cache'] != 'fitted' for field in gp_model.diagnostics['history_fields'])
        fused = fuse_hurdle_gp(candidates, log_mass, gp['grid'], gp['scores'], gp['raw'], bounds)
        prediction = fused['predictions']['probability_half']
        record = dict(task_id=task.task_id, floor=task.floor, K=task.K, repeat=0,
            meta=expected['meta'], status='completed', prediction=prediction.tolist(),
            scores=score(prediction, truth), query_row_indices=expected['query_row_indices'],
            query_groups=expected['query_groups'], seconds=time.monotonic() - started,
            diagnostics=dict(copied_prediction_max_difference=copy_delta,
                gp_prediction_max_difference=gp_delta, original_event_sha256=run._sha(
                    REFERENCE / 'events/refi_GP' / (task.task_id + '.json')),
                GP=gp_model.diagnostics, pool='probability_half'))
        run._write(OUTPUT / 'events/copied_gp' / (task.task_id + '.json'), record)
        records.append(record)
        print(json.dumps(dict(task_id=task.task_id, seconds=record['seconds'], scores={
            k: record['scores'][k] for k in ['mean', 'p90']})), flush=True)
    run._write(OUTPUT / 'summary.json', dict(status='completed', tasks=6,
        by_K=aggregate(records), training_performed=False, new_history_fields=0))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
