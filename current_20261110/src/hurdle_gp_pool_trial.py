"""One fixed original-Hurdle/retained-GP complementarity comparison.

Load the unchanged five-model original recipe and all 54 cached GP position
score maps. Make one Hurdle forward per context, retaining its own candidate
rows and scores. No training, GP fit/integration, lambda sweep or new data.
The original core preloads the CSV; only explicit H/S and Query RSS enter
prediction. Query coordinates enter scoring after predictions are saved.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import json
import socket
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from building_information_tab_trial import digest, now, write
from frozen_hurdle_translator import FrozenHurdleTranslator
from hurdle_gp_pool import fuse_hurdle_gp

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT/'results/building_information_assets_20261109'
GP_RECORDS = ROOT/'results/position_fusion_20261109_continued/target/complete.json'
PACKAGE = ROOT/'claude_hurdle_20261110'
METHODS = ('hurdle', 'GP', 'GP_candidates', 'coordinate_half', 'probability_half')
CODE = ('hurdle_gp_pool_trial.py', 'hurdle_gp_pool.py', 'frozen_hurdle_translator.py',
        'vertical_translator/translator.py', 'building_information_tab_trial.py',
        'claude_hurdle_20261110/core.py', 'claude_hurdle_20261110/protocol9.py')


def read(path):
    return json.loads(Path(path).read_text())


def summarize(manifest, records):
    """Original equal-building/floor/direction/K/position metric and pooled p90."""
    event_map = {event['id']: event for event in manifest['events']}
    samples = []
    for record in records:
        event = event_map[record['event']]
        with np.load(record['scored_path']) as saved:
            _, group = np.unique(saved['truth'], axis=0, return_inverse=True)
            group = group.ravel()
            count = int(group.max())+1
            errors = {method: np.bincount(group, weights=saved[method])/np.bincount(group)
                      for method in METHODS}
        nf = sum(f.startswith(event['building']) for f in manifest['floors'])
        samples.append(dict(event=event, errors=errors, weight=.5/nf/2/3/count))
    def aggregate(selected, method):
        values = np.concatenate([sample['errors'][method] for sample in selected])
        weights = np.concatenate([np.full(len(sample['errors'][method]), sample['weight'])
                                  for sample in selected])
        weights /= weights.sum()
        order = np.argsort(values)
        return dict(mean=float(weights @ values), position_mean_p90=float(
            values[order[np.searchsorted(np.cumsum(weights[order]), .9)]]))
    return {method: dict(aggregate(samples, method), **{
        'by_'+key: {str(value): aggregate(
            [sample for sample in samples if sample['event'][key] == value], method)
            for value in choices}
        for key, choices in (('K', (1, 3, 5)), ('floor', manifest['floors']),
                             ('building', manifest['buildings']))}) for method in METHODS}


def run(args):
    os.chdir(ROOT)
    start = time.monotonic()
    manifest = read(ASSETS/'manifest.json')
    csv = Path(manifest['csv_path'])
    assert digest(csv) == manifest['csv_sha256']
    assert digest(ASSETS/'manifest.json') == read(ASSETS/'complete.json')['manifest_sha256']
    assert len(manifest['events']) == 54 and manifest['K'] == [1, 3, 5]
    source = read(args.models/'complete.json')
    check = read(args.models/'deployment_check.json')
    assert source['complete'] and check['passed']
    assert source['freeze_sha256'] == digest(args.models/'freeze.json')
    gp = read(GP_RECORDS)
    assert gp['complete']
    gp_records = {record['event']: record for record in gp['records']}
    assert set(gp_records) == {event['id'] for event in manifest['events']}
    inputs = [csv, ASSETS/'manifest.json', ASSETS/'complete.json', GP_RECORDS,
              args.models/'complete.json', args.models/'freeze.json', args.models/'deployment_check.json']
    for record in source['models']:
        assert digest(record['path']) == record['sha256']
        inputs.append(Path(record['path']))
    for record in gp_records.values():
        path = ROOT/record['gp_path']
        assert digest(path) == record['gp_sha256']
        inputs.append(path)
    args.output.mkdir(exist_ok=False)
    write(args.output/'freeze.json', dict(created=now(),
        code={name: digest(ROOT/name) for name in CODE},
        inputs={str(path): digest(path) for path in inputs}, contract=dict(
            events=manifest['events'], methods=METHODS, original_hurdle=read(args.models/'freeze.json'),
            gp='Original cached retained GP complete-scan position scores and native grid mean; no changes or new fits',
            gp_recorded_freeze_sha256=gp['freeze_sha256'],
            interpolation='Probability-surface bilinear interpolation from the original rectangular 2m grid, computed by logsumexp; no extrapolation',
            pool='Fixed0.5*original Hurdle log mass+0.5*interpolated GP log density, normalized over original candidate entries',
            candidates='Unchanged Hurdle H-unique coordinates plus empirical S entries, retaining coordinate duplicates',
            controls='Original Hurdle, native GP, candidate-grid GP endpoint, native-coordinate half; no new temperature or selected weight',
            question='Does the fixed complete probability pool improve the best adopted Hurdle and simple coordinate combination?',
            adoption='Lower mean than Hurdle and coordinate_half; no overall or K1 p90 regression versus Hurdle; candidate-GP endpoint reported to expose discretization effects',
            counterexamples='Flat GP tempers Hurdle rather than disappearing; both confident and wrong can compound error; mismatched or out-of-grid candidates stop this cache path',
            scope='Predictive combination with overlapping H/S and Query, not independent Bayesian evidence; existing exposed development events',
            ordering='Original core eagerly parses CSV; explicit H/S and RSS only predict, save all method predictions per event before truth-based scoring',
            seconds=args.seconds, resources=dict(CPU_threads=2, GPU=0, downloads=0, external_timeout=900)),
        runtime=dict(python=sys.executable, pythonpath=os.environ.get('PYTHONPATH', ''),
                     versions={n:version(n) for n in ('numpy','scipy','scikit-learn','pandas','joblib')})))
    os.environ['LAW_UJI_CSV'] = str(csv)
    sys.path.insert(0, str(PACKAGE))
    import core as C
    assert C.MAN['events'] == manifest['events'] and C.MAN['history_rows'] == manifest['history_rows']
    model = FrozenHurdleTranslator(args.models)
    records = []
    for event in manifest['events']:
        if time.monotonic()-start >= args.seconds:
            raise TimeoutError('Explicit comparison budget exhausted; retain artifacts, no automatic continuation')
        def arrays(ids):
            rows = np.asarray(ids, dtype=int)-1
            return dict(rss=C.RSS[rows], xy=C.XY[rows])
        histories = {name:arrays(manifest['history_rows'][name]) for name in event['histories']}
        support = arrays(event['support_rows'])
        prepared = model.prepare(histories, support, event['floor'])
        query_indices = np.asarray(event['query_rows'], dtype=int)-1
        hurdle = prepared.predict(C.RSS[query_indices])
        with np.load(ROOT/gp_records[event['id']]['gp_path']) as cached:
            assert np.array_equal(cached['query_rows'], event['query_rows'])
            assert np.array_equal(cached['aps'], prepared.aps)
            result = fuse_hurdle_gp(hurdle['cand_xy'], hurdle['log_position_mass'],
                cached['grid'], cached['scores'], cached['raw'], prepared.bounds)
        predictions = result['predictions']
        # The unchanged single-path prediction is retained verbatim.
        assert np.max(np.abs(predictions['hurdle']-hurdle['projected'])) <= 1e-8
        predictions['hurdle'] = hurdle['projected']
        path = args.output/f"{event['id']}_unscored.npz"
        np.savez_compressed(path, **predictions, query_rows=event['query_rows'], aps=prepared.aps,
            cand_xy=hurdle['cand_xy'], log_hurdle=hurdle['log_position_mass'],
            log_probability_half=result['log_probability_half'], log_gp_candidates=result['log_gp_candidates'])
        truth = C.XY[query_indices]
        _, group, counts = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
        weights = 1/(len(counts)*counts[group.ravel()])
        errors = {name:np.linalg.norm(value-truth, axis=1) for name,value in predictions.items()}
        scored = args.output/f"{event['id']}_scored.npz"
        np.savez_compressed(scored, truth=truth, **errors)
        records.append(dict(event=event['id'], floor=event['floor'], K=event['K'],
            mde={name:float(weights @ value) for name,value in errors.items()},
            prediction_path=str(path), prediction_sha256=digest(path),
            scored_path=str(scored), scored_sha256=digest(scored)))
        write(args.output/'progress.json', dict(records=records))
        print(f'Completed {len(records)}/54 fixed contexts', flush=True)
    summary = summarize(manifest, records)
    pooled, original, coordinate = (summary[name] for name in ('probability_half','hurdle','coordinate_half'))
    adoption = dict(mean=pooled['mean'] < min(original['mean'],coordinate['mean']),
        overall_tail=pooled['position_mean_p90'] <= original['position_mean_p90'],
        K1_tail=pooled['by_K']['1']['position_mean_p90'] <= original['by_K']['1']['position_mean_p90'],
        better_than_candidate_GP=pooled['mean'] < summary['GP_candidates']['mean'])
    write(args.output/'complete.json', dict(complete=True, created=now(), seconds=time.monotonic()-start,
        freeze_sha256=digest(args.output/'freeze.json'), records=records, summary=summary,
        adoption=adoption, training_updates=0, GP_fits=0, weight_selection=False,
        target_performance_reproduction_is_not_the_purpose=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=ROOT/'results/claude_hurdle_b0_ens5_20261110')
    parser.add_argument('--output', type=Path, default=ROOT/'results/hurdle_gp_pool_20261110')
    parser.add_argument('--seconds', type=float, default=840.)
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 840
    with threadpool_limits(limits=2):
        run(args)
