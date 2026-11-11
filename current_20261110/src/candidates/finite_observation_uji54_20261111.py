"""Strict54 retention check: finite-observation fields, existing exact count reader.

Compare with completed original-five-model+count and retain the adopted raw
full-pool reference. Reuse the original manifest, GP caches, five readouts,
save-before-score order and hierarchical metrics. No fitting or selection.
This entry is deliberately outside the frozen finite_observation package.
"""
import hurdle_gp_pool_trial as original  # CPU2, offline and no GPU before numpy.

import argparse
from importlib.metadata import version
import os
from pathlib import Path
import signal
import socket
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from candidates.count_conditioned_20261111.reader import CountConditionedHurdle
from candidates.finite_observation_20261111.deployment import FiniteObservationTranslator


ROOT = original.ROOT
MODELS = ROOT/'results/claude_hurdle_b0_finite_obs_ens5_20261111'
COUNT = ROOT/'results/uji54_count_conditioned_20261111/complete.json'
COUNT_SHA = '0f154db21c5b596e1e205ef1da4b1564f3cf7f36a5e1031e74b846f4bb8f2327'
ADOPTED = ROOT/'results/hurdle_gp_pool_20261110/complete.json'
OUTPUT = ROOT/'results/uji54_finite_observation_20261111'
COUNT_READER = 'candidates/count_conditioned_20261111/reader.py'
read, digest, write = original.read, original.digest, original.write
summarize, compare_summaries = original.summarize, original.compare_summaries
fuse_hurdle_gp = original.fuse_hurdle_gp


def run(args):
    os.chdir(ROOT)
    started = time.monotonic()
    for protected in (MODELS, COUNT.parent, ADOPTED.parent, original.ASSETS, original.GP_RECORDS.parent):
        assert args.output != protected and protected not in args.output.parents
    inputs = {}

    def bind(path, expected=None):
        path = Path(path)
        sha = digest(path)
        if expected is not None:
            assert sha == expected, str(path)
        inputs[str(path)] = sha
        return sha

    bind(original.ASSETS/'manifest.json', original.MANIFEST_SHA)
    bind(original.ASSETS/'complete.json')
    manifest = read(original.ASSETS/'manifest.json')
    assert read(original.ASSETS/'complete.json')['manifest_sha256'] == original.MANIFEST_SHA
    assert len(manifest['events']) == 54 and manifest['K'] == [1, 3, 5]
    events = manifest['events']
    expected = {event['id'] for event in events}
    assert len(expected) == 54
    csv = Path(manifest['csv_path'])
    bind(csv, manifest['csv_sha256'])
    bind(original.GP_RECORDS)
    gp = read(original.GP_RECORDS)
    assert gp['complete']
    gp_records = {r['event']: r for r in gp['records']}
    assert len(gp['records']) == len(gp_records) == 54 and set(gp_records) == expected
    for record in gp_records.values():
        bind(ROOT/record['gp_path'], record['gp_sha256'])

    references = {}
    for label, path, sha in (('original_count', COUNT, COUNT_SHA),
                             ('adopted_original_raw', ADOPTED, original.BASELINE_SHA)):
        bind(path, sha)
        value = read(path)
        assert value['complete']
        bind(path.parent/'freeze.json', value['freeze_sha256'])
        frozen = read(path.parent/'freeze.json')
        assert frozen['contract']['events'] == events
        assert frozen['contract']['methods'] == list(original.METHODS)
        assert frozen['contract']['gp_recorded_freeze_sha256'] == gp['freeze_sha256']
        for same in (csv, original.ASSETS/'manifest.json', original.GP_RECORDS,
                     *[ROOT/r['gp_path'] for r in gp_records.values()]):
            assert frozen['inputs'][str(same)] == inputs[str(same)]
        records = {r['event']: r for r in value['records']}
        assert len(value['records']) == len(records) == 54 and set(records) == expected
        references[label] = dict(path=path, complete=value, freeze=frozen, records=records)
    old = references['original_count']
    assert old['complete']['source_parity']['passed']
    for name in (COUNT_READER, 'hurdle_gp_pool.py', 'frozen_hurdle_translator.py',
                 'vertical_translator/translator.py'):
        assert old['freeze']['code'][name] == digest(ROOT/name), name
    old_fields = {r['event']: r for r in old['complete']['fields']}
    assert len(old['complete']['fields']) == len(old_fields) == 54 and set(old_fields) == expected
    for event in events:
        record = old['records'][event['id']]
        assert record['floor'] == event['floor'] and record['K'] == event['K']
        for kind in ('prediction', 'scored'):
            bind(ROOT/record[kind+'_path'], record[kind+'_sha256'])
        field = old_fields[event['id']]
        assert field['query_used'] is False
        bind(ROOT/field['path'], field['sha256'])

    bind(MODELS/'freeze.json', args.model_freeze_sha256)
    bind(MODELS/'complete.json')
    source, source_freeze = read(MODELS/'complete.json'), read(MODELS/'freeze.json')
    assert source['complete'] and source['freeze_sha256'] == args.model_freeze_sha256
    assert sorted(r['seed'] for r in source['models']) == [1, 2, 3, 4, 5]
    assert source_freeze['reader']['beta'] == original.BETA == 128.
    assert source_freeze['reader']['eps'] == original.EPS == .02
    assert source_freeze['reader']['offsets'] == original.OFFSETS.tolist()
    for record in source['models']:
        bind(MODELS/record['path'], record['sha256'])
        bind(MODELS/record['source_tasks']['path'], record['source_tasks']['sha256'])
    translator = FiniteObservationTranslator(MODELS)
    adopted_mean = references['adopted_original_raw']['complete']['summary']['probability_half']['mean']
    code = (*original.CODE, str(Path(__file__).relative_to(ROOT)), COUNT_READER,
            'candidates/finite_observation_20261111/features.py',
            'candidates/finite_observation_20261111/deployment.py')
    args.output.mkdir(exist_ok=False)
    write(args.output/'freeze.json', dict(created=original.now(), inputs=inputs,
        code={name: digest(ROOT/name) for name in code},
        model_generation=dict(path=str(MODELS), freeze_sha256=args.model_freeze_sha256),
        contract=dict(events=events, methods=list(original.METHODS),
            primary='New finite-observation five-model field plus exact count versus original five-model field plus identical count',
            secondary='Also retain adopted original raw-reader full combination; do not discard its existing gains',
            fixed='H/S/Q rows, AP axis, candidates/bounds, empirical S p/m/s2, EPS/OFFSETS/BETA, GP caches and half pool',
            changed='Learned H p/m/s2 from the frozen finite-observation source recipe',
            GP='Original retained scores/native output; no fitting, adaptation, integration or resampling',
            gp_recorded_freeze_sha256=gp['freeze_sha256'],
            comparisons='Same hierarchical mean and position-mean p90, overall and by K/floor/building, against both references',
            minimum_retention=dict(reference='adopted_original_raw probability_half',
                baseline_mean_m=adopted_mean, allowed_mean_increase_m=.3,
                ceiling_m=adopted_mean+.3, sufficient_for_adoption=False),
            parity='Candidates/AP/bounds and empirical S exact; both GP endpoints within1e-8m; learned A coordinates may change',
            ordering='Original core parses CSV; forward uses H/S and Query RSS only; save per-event predictions before truth scoring',
            field_cache='Save each new prepared H/S field before Query prediction',
            scope='Already exposed strict54 retention check; no independent-test or acquisition-shift guarantee',
            adoption='Main-thread judgment after all comparisons, never automatic',
            resources=dict(CPU_threads=2, GPU=0, training=0, GP_fits=0, seconds=args.seconds,
                           external_timeout=900, automatic_resume=False)),
        runtime=dict(python=sys.executable, versions={name: version(name) for name in
                     ('numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib')})))
    freeze_sha = digest(args.output/'freeze.json')
    os.environ['LAW_UJI_CSV'] = str(csv)
    sys.path.insert(0, str(original.PACKAGE))
    import core as C
    assert C.MAN['events'] == events and C.MAN['history_rows'] == manifest['history_rows']
    records, fields, parity = [], [], {}
    for event in events:
        if time.monotonic()-started >= args.seconds:
            raise TimeoutError('Comparison budget exhausted; artifacts retained, no automatic continuation')
        eid = event['id']

        def arrays(ids):
            rows = np.asarray(ids, dtype=int)-1
            return dict(rss=C.RSS[rows], xy=C.XY[rows])

        histories = {name: arrays(manifest['history_rows'][name]) for name in event['histories']}
        support = arrays(event['support_rows'])
        prepared = translator.prepare(histories, support, event['floor'])
        nS = len(np.unique(support['xy'], axis=0))
        assert nS == event['K'] > 0
        with np.load(ROOT/old_fields[eid]['path'], allow_pickle=False) as cached:
            for name in ('cand_xy', 'aps', 'bounds', 'ap_count'):
                assert np.array_equal(getattr(prepared, name), cached[name]), (eid, name)
            for name in ('p', 'm', 's2'):
                assert getattr(prepared, name).shape == cached[name].shape
                assert np.array_equal(getattr(prepared, name)[-nS:], cached[name][-nS:]), (eid, 'empirical S', name)
        field_path = args.output/f'{eid}_field.npz'
        np.savez_compressed(field_path, **{name: getattr(prepared, name) for name in
            ('cand_xy', 'aps', 'p', 'm', 's2', 'bounds', 'ap_count')})
        fields.append(dict(event=eid, path=str(field_path), sha256=digest(field_path), query_used=False))
        query_indices = np.asarray(event['query_rows'], dtype=int)-1
        counted = CountConditionedHurdle(prepared)
        hurdle = counted.predict(C.RSS[query_indices])
        with np.load(ROOT/gp_records[eid]['gp_path'], allow_pickle=False) as cached:
            assert np.array_equal(cached['query_rows'], event['query_rows'])
            assert np.array_equal(cached['aps'], prepared.aps)
            result = fuse_hurdle_gp(hurdle['cand_xy'], hurdle['log_position_mass'],
                cached['grid'], cached['scores'], cached['raw'], prepared.bounds)
        predictions = result['predictions']
        assert np.max(np.abs(predictions['hurdle']-hurdle['projected'])) <= 1e-8
        predictions['hurdle'] = hurdle['projected']
        checks = dict(candidates_AP_bounds_exact=True, empirical_S_p_m_s2_exact=True)
        with np.load(ROOT/old['records'][eid]['prediction_path'], allow_pickle=False) as prior:
            assert np.array_equal(prior['query_rows'], event['query_rows'])
            assert np.array_equal(prior['aps'], prepared.aps)
            assert np.array_equal(prior['cand_xy'], prepared.cand_xy)
            for method in ('GP', 'GP_candidates'):
                delta = float(np.max(np.abs(predictions[method]-prior[method])))
                assert delta <= 1e-8, (eid, method, delta)
                checks[method+'_coordinate_max_difference_m'] = delta
        parity[eid] = checks
        path = args.output/f'{eid}_unscored.npz'
        np.savez_compressed(path, **predictions, query_rows=event['query_rows'], aps=prepared.aps,
            cand_xy=hurdle['cand_xy'], log_hurdle=hurdle['log_position_mass'],
            log_probability_half=result['log_probability_half'], log_gp_candidates=result['log_gp_candidates'])
        # Preserve original event-level save-before-score ordering and weights.
        truth = C.XY[query_indices]
        with np.load(ROOT/old['records'][eid]['scored_path'], allow_pickle=False) as prior:
            assert np.array_equal(prior['truth'], truth)
        _, group, counts = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
        weights = 1/(len(counts)*counts[group.ravel()])
        errors = {name: np.linalg.norm(value-truth, axis=1) for name, value in predictions.items()}
        scored = args.output/f'{eid}_scored.npz'
        np.savez_compressed(scored, truth=truth, **errors)
        records.append(dict(event=eid, floor=event['floor'], K=event['K'],
            mde={name: float(weights @ value) for name, value in errors.items()},
            prediction_path=str(path), prediction_sha256=digest(path),
            scored_path=str(scored), scored_sha256=digest(scored)))
        write(args.output/'progress.json', dict(complete=False, freeze_sha256=freeze_sha,
            records=records, fields=fields, parity=parity))
        print(f'Completed {len(records)}/54 finite-observation contexts', flush=True)
    assert len(records) == len(fields) == len(parity) == 54
    summary = summarize(manifest, records)
    comparisons = {label: dict(path=str(ref['path']), sha256=inputs[str(ref['path'])],
        baseline_summary=ref['complete']['summary'],
        deltas=compare_summaries(ref['complete']['summary'], summary)) for label, ref in references.items()}
    write(args.output/'complete.json', dict(complete=True, created=original.now(),
        seconds=time.monotonic()-started, freeze_sha256=freeze_sha,
        model_freeze_sha256=args.model_freeze_sha256, records=records, fields=fields,
        summary=summary, comparisons=comparisons, parity=dict(passed=True, events=parity),
        minimum_retention=dict(reference='adopted_original_raw probability_half',
            baseline_mean_m=adopted_mean, ceiling_m=adopted_mean+.3,
            current_mean_m=summary['probability_half']['mean'],
            passed=summary['probability_half']['mean'] <= adopted_mean+.3,
            sufficient_for_adoption=False),
        adoption=dict(status='requires_main_thread_judgment', automatic_rule=False),
        delta_convention='Current minus reference; negative is improvement; original hierarchical weighting',
        training_updates=0, GP_fits=0, weight_selection=False, field_preparations=54))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-freeze-sha256', required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seconds', type=float, default=840.)
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 840

    def timeout(signum, frame):
        raise TimeoutError('840-second-or-smaller run budget exhausted; no automatic continuation')

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    try:
        with threadpool_limits(limits=2):
            run(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
