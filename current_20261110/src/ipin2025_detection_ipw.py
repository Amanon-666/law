"""Post-hoc det-only confirmation on the already exposed IPIN2025 streams.

prepare: new A distributions on the same 15 H/S contexts; inherit old GP files.
predict: original 30-stream runner with its verified fast cached-field readout.
score: original separate scorer, only after all streams and GP parity are sealed.
No source training, GP fitting/adaptation/sampling, old-model prediction or GPU.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import socket
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import ipin2025_confirmation as original
from cached_retained_readout import score_cached_fields
from frozen_hurdle_translator import FrozenHurdleTranslator, BETA, EPS, OFFSETS
from building_information_tab_trial import digest, now, write

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT/'results/ipin2025_confirmation_20261110'
MODELS = ROOT/'results/claude_hurdle_b0_detipw20_ens5_20261111'
BASE_SHA = '371f5815a44f7c4ddac67cbb1ad7932ba90e3189958d5723f4984f86cf1d08a7'
NEW_SHA = '419a16553719343694f31d9d231a0af04ba2be1374eee3c3cbee8024eb186deb'
OLD_SHA = {
    'freeze.json': '6fc0475cb9e23014645d1a31853bb8e8b606a68c027d2f6ddc937c6f821d648c',
    'contexts_complete.json': 'de03679c06d5b3f22b7b5d67c057e78971beb5ce2391cf2289af6fb21370a1f8',
    'predictions.json': '10fad9325ec3b0ee08837624ee7e0f4f67ce32feb64fb9499b30e7c17a297026',
    'scores.json': '50716b22d18af368b1ffbbe9c48f345d24a1dac5335b5792f0d8b216a387ef04',
}
FAST_SHA = '13384ea458e522b01fba46609114bb752827246b6bfae9d0b2b74afdcc08616c'
read, binding = original.read, original.binding


def checked(record):
    assert digest(record['path']) == record['sha256'], record['path']
    return read(record['path'])


def freeze(args):
    assert args.output != BASELINE and BASELINE not in args.output.parents
    assert args.output != MODELS and MODELS not in args.output.parents
    assert args.output != original.ASSETS and original.ASSETS not in args.output.parents
    for name, sha in OLD_SHA.items():
        assert digest(BASELINE/name) == sha
    old = read(BASELINE/'freeze.json')
    contexts = read(BASELINE/'contexts_complete.json')
    predictions = read(BASELINE/'predictions.json')
    assert contexts['complete'] and contexts['freeze_sha256'] == OLD_SHA['freeze.json']
    assert predictions['complete'] and predictions['freeze_sha256'] == OLD_SHA['freeze.json']
    assert predictions['contexts_complete_sha256'] == OLD_SHA['contexts_complete.json']
    expected = {(f, k) for f in range(2, 7) for k in (1, 3, 5)}
    assert len(contexts['contexts']) == 15
    assert {(r['floor'], r['K']) for r in contexts['contexts']} == expected
    assert len(predictions['predictions']) == 30
    assert {(r['log_id'], r['floor'], r['K']) for r in predictions['predictions']} == {
        (log, f, k) for log in ('03', '04') for f, k in expected}
    for key, value in dict(eligible_floors=[2, 3, 4, 5, 6], K=[1, 3, 5],
                           methods=original.METHODS).items():
        assert old[key] == predictions[key] == value
    assert old['logs'] == predictions['logs'] and old['R'] == 1
    # Verify historical data bindings, not old code equality: the reader evolved.
    for item in old['inputs']:
        assert digest(item['path']) == item['sha256']
    assets = read(original.ASSETS/'complete.json')
    for name, key in [('freeze.json', 'freeze_sha256'), ('manifest.json', 'manifest_sha256'),
                      ('histories.json', 'histories_sha256')]:
        assert digest(original.ASSETS/name) == assets[key]
    execution = checked(predictions['readout_execution'])
    assert digest(ROOT/'cached_retained_readout.py') == execution['binding']['source']['sha256'] == FAST_SHA
    assert read(BASELINE/'cached_readout_check.json')['passed']
    source = read(MODELS/'complete.json')
    source_freeze = read(MODELS/'freeze.json')
    inheritance = read(MODELS/'inheritance_check.json')
    assert digest(MODELS/'freeze.json') == source['freeze_sha256'] == NEW_SHA
    assert source['complete'] and source_freeze['recipe'] == 'detection_ipw20'
    assert source_freeze['base_generation']['freeze_sha256'] == BASE_SHA
    assert inheritance['passed'] and inheritance['reader_loaded']
    assert inheritance['base_freeze_sha256'] == BASE_SHA and inheritance['new_freeze_sha256'] == NEW_SHA
    assert inheritance['checked_seeds'] == [1, 2, 3, 4, 5]
    assert sorted(r['seed'] for r in inheritance['seeds']) == [1, 2, 3, 4, 5]
    assert all(r[key] is True for r in inheritance['seeds'] for key in (
        'mean_state_equal', 'scale_state_equal', 'model_cols_equal', 'det_cols_equal', 'det_parameters_equal'))
    assert source_freeze['refit_heads'] == ['det'] and source_freeze['inherited_heads'] == ['mean', 'scale']
    assert BETA == source_freeze['reader']['beta'] == 128.
    assert EPS == source_freeze['reader']['eps'] == .02
    assert OFFSETS.tolist() == source_freeze['reader']['offsets'] == [-8., -4., 0., 4., 8.]
    inputs = [BASELINE/name for name in OLD_SHA]
    inputs += [BASELINE/name for name in ('cached_readout_check.json', 'cached_readout_execution.json',
                                         'training_readout_cost.json', 'query_stream_inventory.json')]
    inputs += [Path(r['path']) for r in old['inputs']]
    inputs += [MODELS/name for name in ('complete.json', 'freeze.json', 'inheritance_check.json')]
    for record in source['models']:
        path = MODELS/record['path']
        assert digest(path) == record['sha256']
        inputs.append(path)
    for record in contexts['contexts']:
        complete = checked(record)
        assert complete['complete'] and complete['freeze_sha256'] == OLD_SHA['freeze.json']
        assert (complete['floor'], complete['K']) == (record['floor'], record['K'])
        inputs.append(Path(record['path']))
    for record in predictions['predictions']:
        assert digest(record['path']) == record['sha256']
        inputs.append(Path(record['path']))
    args.output.mkdir(exist_ok=False)
    code = (*original.CODE, 'cached_retained_readout.py', Path(__file__).name)
    write(args.output/'freeze.json', dict(created=now(), eligible_floors=old['eligible_floors'],
        K=old['K'], R=1, methods=old['methods'], logs=old['logs'],
        inputs=[binding(p) for p in dict.fromkeys(inputs)],
        code={name: digest(ROOT/name) for name in code},
        contract=dict(question='Does the already-defined det-only IPW20 correction improve the existing IPIN failure?',
            status='Exposed data, post-hoc mechanism confirmation; not a new blind test',
            changed='Only new A detection probabilities; require exact inherited m/s2, candidates/APs/bounds',
            history='Original FPS128 other-floor H and original nested Training S, R1; no new data selection',
            GP='Old adaptation state/grid/fields, transitively bound by each context complete.json; load_prepared verifies every field',
            readout='Original verified score_cached_fields; no saved GP score maps; cached-field readout only, not GP fitting or MC',
            timing=old['contract']['timing'], coverage=old['contract']['coverage'],
            pooling='Same fixed probability_half and all five original controls; no weight/reader selection',
            GP_parity_tolerance_m=1e-7, source_training=0, GP_fits=0, GP_adaptations=0, GP_sampling=0,
            scoring='Separate original scorer only after all30 streams and GP/clock parity checks; same coverage/log/K comparisons',
            input_binding='Frozen top-level indices recursively bind original H/S, states, grids and per-AP fields; original files are read-only',
            resources=dict(CPU_threads=2, GPU=0, downloads=0, segment_seconds=args.seconds))))


def prepare(args):
    if not args.resume:
        freeze(args)
    original.verify_freeze(args.output)
    assert not (args.output/'contexts_complete.json').exists()
    manifest = read(original.ASSETS/'manifest.json')
    translator = FrozenHurdleTranslator(MODELS)
    records, completed = [], 0
    start = time.monotonic()
    for record in read(BASELINE/'contexts_complete.json')['contexts']:
        directory = args.output/record['event']
        path = directory/'complete.json'
        if path.exists():
            current = read(path)
            assert current['freeze_sha256'] == digest(args.output/'freeze.json')
            assert digest(current['hurdle']['path']) == current['hurdle']['sha256']
        else:
            if completed >= args.max_contexts or time.monotonic()-start >= args.seconds:
                return 2
            tick = time.monotonic()
            old = checked(record)
            assert digest(old['hurdle']['path']) == old['hurdle']['sha256']
            histories, support, ap_ids = original.context(manifest, record['floor'], record['K'])
            assert old['ap_ids'] == ap_ids and old['sources'] == list(histories)
            h = translator.prepare(histories, support, f"IPIN2025_B2025F{record['floor']}")
            with np.load(old['hurdle']['path'], allow_pickle=False) as cached:
                for key in ('cand_xy', 'aps', 'm', 's2', 'bounds', 'ap_count'):
                    assert np.array_equal(getattr(h, key), cached[key]), f'{record["event"]}: changed {key}'
                p_change = float(np.max(np.abs(h.p-cached['p'])))
            directory.mkdir(exist_ok=True)
            hpath = directory/'hurdle.npz'
            np.savez_compressed(hpath, **{key: getattr(h, key) for key in (
                'cand_xy', 'aps', 'p', 'm', 's2', 'bounds', 'ap_count')})
            # Keep references to the ORIGINAL state/grid/fields; never copy or recompute them.
            current = dict(old, created=now(), hurdle=binding(hpath), freeze_sha256=digest(args.output/'freeze.json'),
                parent_context=record, seconds_this_segment=time.monotonic()-tick,
                inheritance=dict(exact_m_s2_candidates_AP_bounds=True, max_p_change=p_change),
                GP_fits=0, GP_adaptations=0, GP_sampling=0)
            write(path, current)
            completed += 1
        records.append(dict(event=record['event'], floor=record['floor'], K=record['K'], **binding(path)))
        print(f'Prepared {len(records)}/15 det-only contexts', flush=True)
    write(args.output/'contexts_complete.json', dict(complete=True, created=now(), contexts=records,
        freeze_sha256=digest(args.output/'freeze.json'), official_query_opened=False, GT_opened=False))
    return 0


def verify_streams(output):
    """No GT: same complete receipt expansion/clocks and unchanged GP controls."""
    baseline = {(r['log_id'], r['floor'], r['K']): r
                for r in read(BASELINE/'predictions.json')['predictions']}
    progress = output/'predictions_progress.json'
    saved = read(progress) if progress.exists() else []
    seen, comparisons = set(), []
    for record in saved:
        key = record['log_id'], record['floor'], record['K']
        assert key in baseline and key not in seen
        seen.add(key)
        old = baseline[key]
        for name in ('receipt_predictions', 'ticks', 'duplicate_bssid_rows', 'conflicting_rss_rows'):
            assert record[name] == old[name]
        assert digest(record['path']) == record['sha256'] and digest(old['path']) == old['sha256']
        with np.load(record['path'], allow_pickle=False) as new, np.load(old['path'], allow_pickle=False) as base:
            for name in ('times', 'snapshot_times'):
                assert np.array_equal(new[name], base[name])
            differences = {}
            for name in ('GP', 'GP_candidates'):
                differences[name] = float(np.max(np.abs(new[name]-base[name])))
                assert differences[name] <= 1e-7, (key, name, differences[name])
        comparisons.append(dict(log_id=key[0], floor=key[1], K=key[2], max_difference_m=differences))
    result = dict(passed=True, complete=len(seen) == 30, checked_streams=len(seen),
        baseline_predictions_sha256=OLD_SHA['predictions.json'], freeze_sha256=digest(output/'freeze.json'),
        progress_sha256=digest(progress) if progress.exists() else None,
        comparisons=comparisons, GT_opened=False)
    if (output/'predictions.json').exists():
        sealed = read(output/'predictions.json')
        assert sealed['predictions'] == saved and result['complete']
        result['predictions_sha256'] = digest(output/'predictions.json')
    write(output/'cached_GP_parity.json', result)
    return result


def predict(args):
    original.verify_freeze(args.output)
    if args.resume:
        verify_streams(args.output)
    # Reuse the original loader, full receipt reader, rss_matrix, batching,
    # 2Hz expansion, fixed pool, and 30-stream seal. Only the GP readout hook changes.
    with patch.object(original, 'score_retained_fields', score_cached_fields):
        result = original.predict_streams(args)
    verify_streams(args.output)
    return result


def compare_group(old, new):
    if 'methods' not in new:
        return {key: compare_group(old[key], value) for key, value in new.items()}
    coverage = ('total_points', 'covered_points', 'coverage', 'unsupported_floor', 'no_prior_prediction')
    assert all(old[key] == new[key] for key in coverage)
    methods = {}
    for method in original.METHODS:
        a, b = old['methods'][method], new['methods'][method]
        assert a['count'] == b['count']
        methods[method] = {metric: dict(baseline=a[metric], current=b[metric],
            delta=None if a[metric] is None else b[metric]-a[metric]) for metric in ('mean', 'p90')}
    return dict(coverage={key: new[key] for key in coverage}, methods=methods)


def score(args):
    original.verify_freeze(args.output)
    parity = verify_streams(args.output)
    assert parity['complete'] and parity['predictions_sha256'] == digest(args.output/'predictions.json')
    from ipin2025_score import score as original_score
    result = original_score(args.output/'predictions.json')
    write(args.output/'scores.json', result)
    baseline = read(BASELINE/'scores.json')  # Scoring phase only, never prepare/predict.
    for key in ('schema', 'gt_sha256', 'eligible_floors', 'methods', 'gt_rows', 'metric', 'coverage', 'reference'):
        assert baseline[key] == result[key]
    assert len(baseline['points']) == len(result['points'])
    for old, new in zip(baseline['points'], result['points']):
        assert {k: v for k, v in old.items() if k != 'errors'} == {k: v for k, v in new.items() if k != 'errors'}
    write(args.output/'comparison.json', dict(complete=True, created=now(),
        status='Exposed-data post-hoc confirmation of the already-defined detection_ipw20 mechanism',
        baseline=binding(BASELINE/'scores.json'), current=binding(args.output/'scores.json'),
        GP_parity=binding(args.output/'cached_GP_parity.json'),
        delta_convention='current minus original; negative is improvement; no K pooling or automatic adoption',
        by_K=compare_group(baseline['by_K'], result['by_K'])))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'predict', 'score'))
    parser.add_argument('--output', type=Path, default=ROOT/'results/ipin2025_detection_ipw20_20261111')
    parser.add_argument('--seconds', type=float, default=780.)
    parser.add_argument('--max-contexts', type=int, default=15)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.device = 'cpu'
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 780 and args.max_contexts > 0
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    raise SystemExit({'prepare': prepare, 'predict': predict, 'score': score}[args.phase](args))
