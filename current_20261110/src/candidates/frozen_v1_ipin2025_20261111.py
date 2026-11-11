"""Frozen v1_newcode: original C, no C, and the same C with exact S priors.

check: one real F2/K1 H/S-only augmentation-equivalence check; cache its fields.
prepare: one H-only ensemble call on [H candidates; Ps] per remaining context.
predict: original 30-stream runner per arm, original raw reader and cached GP.
score: original scorer only after all THREE arms/90 streams seal.
All source assets and original code remain read-only. No source or GP fit.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
from pathlib import Path
import socket
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import ipin2025_confirmation as original
import ipin2025_detection_ipw as shared
from building_information_tab_trial import digest, now, write
from frozen_hurdle_translator import PreparedFrozenHurdle, _support_fields, BETA, EPS, OFFSETS
from law_bench.method_frozen_field import FrozenFieldMethod, MODEL_DIR


ROOT, BASELINE = original.ROOT, shared.BASELINE
SOURCE = ROOT/'results/law_bench_v2_jitter_20261111/source_snapshot/claude_field_posterior_20261111'
OUTPUT = ROOT/'results/frozen_v1_ipin2025_20261111'
ARMS = ('full_v1', 'without_field_conditioning', 'exact_support_prior')
FIELD_KEYS = ('cand_xy', 'aps', 'p', 'm', 's2', 'bounds', 'ap_count')
REFERENCES = dict(original_A=BASELINE,
    count=ROOT/'results/ipin2025_count_conditioned_20261111',
    finite_observation=ROOT/'results/ipin2025_finite_observation_20261111')
SCORE_SHA = dict(original_A=shared.OLD_SHA['scores.json'],
    count='1b75d463d24c64335a09abab3e9fcd4ff598be43762ec6c4f83777d118078019',
    finite_observation='e7505b956a26585a9962740733faa7f27eed9b70379f6d2fcf742865ec90f190')
read, binding = original.read, original.binding


def source_reader():
    method = FrozenFieldMethod(source_dir=SOURCE, model_dir=MODEL_DIR, variant='v1')
    # This adapter's fit_source only verifies fixed source permissions/hashes.
    method.fit_source([dict(dataset='uji', building='B0')])
    return method


def remaining(args):
    return args.seconds-(time.monotonic()-args.started)


def freeze(args):
    for protected in (*REFERENCES.values(), original.ASSETS, original.SOURCE, MODEL_DIR, SOURCE):
        assert args.output != protected and protected not in args.output.parents
    for name, sha in shared.OLD_SHA.items():
        assert digest(BASELINE/name) == sha
    old = read(BASELINE/'freeze.json')
    contexts = read(BASELINE/'contexts_complete.json')
    expected = {(f, k) for f in range(2, 7) for k in (1, 3, 5)}
    assert contexts['complete'] and contexts['freeze_sha256'] == shared.OLD_SHA['freeze.json']
    assert len(contexts['contexts']) == 15 and {(r['floor'], r['K']) for r in contexts['contexts']} == expected
    assert old['eligible_floors'] == [2, 3, 4, 5, 6] and old['K'] == [1, 3, 5] and old['R'] == 1
    inputs = [BASELINE/name for name in shared.OLD_SHA]
    for item in old['inputs']:
        assert digest(item['path']) == item['sha256']
        inputs.append(Path(item['path']))
    manifest = read(original.ASSETS/'manifest.json')
    for item in [*manifest['histories'].values(), *manifest['support'].values()]:
        assert digest(item['path']) == item['sha256']
        inputs.append(Path(item['path']))
    for record in contexts['contexts']:
        parent = shared.checked(record)
        assert parent['complete'] and parent['freeze_sha256'] == shared.OLD_SHA['freeze.json']
        assert (parent['floor'], parent['K']) == (record['floor'], record['K'])
        assert digest(parent['hurdle']['path']) == parent['hurdle']['sha256']
        inputs += [Path(record['path']), Path(parent['hurdle']['path'])]
    for label, directory in REFERENCES.items():
        assert digest(directory/'scores.json') == SCORE_SHA[label]
        predictions = read(directory/'predictions.json')
        assert predictions['complete'] and predictions['phase'] == 'predictions'
        assert predictions['freeze_sha256'] == digest(directory/'freeze.json')
        assert predictions['contexts_complete_sha256'] == digest(directory/'contexts_complete.json')
        for key in ('eligible_floors', 'K', 'methods', 'logs'):
            assert predictions[key] == old[key]
        assert len(predictions['predictions']) == 30
        assert {(r['log_id'], r['floor'], r['K']) for r in predictions['predictions']} == {
            (log, f, k) for log in ('03', '04') for f, k in expected}
        inputs += [directory/name for name in ('freeze.json', 'contexts_complete.json', 'predictions.json', 'scores.json')]
        if label == 'original_A':
            for record in predictions['predictions']:
                assert digest(record['path']) == record['sha256']
                inputs.append(Path(record['path']))
    assert digest(ROOT/'cached_retained_readout.py') == shared.FAST_SHA
    assert BETA == 128. and EPS == .02 and OFFSETS.tolist() == [-8., -4., 0., 4., 8.]
    method = source_reader()
    inputs += [Path(r['path']) for r in method.provenance['models']]
    inputs += [MODEL_DIR/'theta_selection.json', MODEL_DIR.parent/'summary.json']
    source_csv = Path(read(MODEL_DIR/'theta_selection.json')['csv'])
    assert digest(source_csv) == method.provenance['source_csv_sha256']
    inputs.append(source_csv)
    assert digest(MODEL_DIR.parent/'summary.json') == 'e0b49966692b97f4c349b95991afe20adad96ce5d99bc97ff7dc56239767260f'
    sibling = SOURCE.parent
    inputs += [SOURCE/name for name in method.provenance['code_sha256']]
    inputs += [sibling/'history_reliability.py', sibling/'claude_hurdle_20261110/translator.py',
               sibling/'vertical_translator/translator.py']
    code = (*original.CODE, 'cached_retained_readout.py', 'ipin2025_detection_ipw.py',
            'law_bench/method_frozen_field.py', 'law_bench/method_adopted.py',
            str(Path(__file__).relative_to(ROOT)))
    args.output.mkdir(exist_ok=False)
    frozen = dict(created=now(), **{key: old[key] for key in ('eligible_floors', 'K', 'R', 'methods', 'logs')},
        inputs=[binding(p) for p in dict.fromkeys(inputs)], code={name: digest(ROOT/name) for name in code},
        arms=list(ARMS), original_model_provenance=method.provenance,
        source_history=dict(original_run_had_standalone_freeze=False,
            original_result=binding(MODEL_DIR.parent/'summary.json'),
            original_code_binding=binding(MODEL_DIR/'theta_selection.json'),
            binding_scope='This new prospective IPIN binding does not retroactively freeze the old UJI run'),
        contract=dict(status='Already exposed IPIN data; prospective three-arm complete-recipe comparison',
            question='Does the original five-model H-only prior plus its complete S-conditioning retain useful IPIN calibration gains?',
            full_v1='Original ensemble_field, condition(16,.3,1,0), original empirical S, raw Hurdle and GP probability_half',
            without_field_conditioning='Same five-model prior p/m/s2 without C; same empirical S and H/S AP axis, not pure no-S',
            exact_support_prior='Evaluate the same H-only prior at Ps, call unchanged condition on [H;Ps], keep only its H prefix before appending original empirical S',
            preparation='One ensemble call per context on [original unique H candidates;det_stats-order Ps]; save raw H/S priors, both condition results and all three prepared fields',
            forward='H maps plus permitted S coordinates -> same pointwise H-only field; S RSS enters C and empirical S, never prior values except the unchanged H/S-visible AP catalogue',
            reverse='Scored Query comes only from sealed raw-reader predictions; reader candidates stay original H plus empirical S; appended prior-evaluation Ps rows are discarded after C',
            change_boundary='Only replace nearest-H prior lookup at S by actual same-operator evaluation at S; exact self-matches remove the old >4m exclusion, not S/Q acquisition differences',
            prior_evidence='v1 is a strong original strict54 asset, not universally best: completed B1 static-resample-v2 pool mean/p90=9.8580/17.0945 versus adopted9.6488/16.9341',
            limits='Exact S evaluation does not remove original R1 pseudo-observation smoothing or establish bS=bQ; superiority to original A alone is insufficient',
            equivalence_check='F2/K1 H/S-only: standalone original H versus augmented H prefix p/m/s2, absolute tolerance1e-10; no Q/GP/score, saved context reused by prepare',
            counterexamples='Empty S returns the same prior; all S on H recovers old C; zero RSS residual alone does not freeze detection or variance; continuous geometry is translation-invariant but original 2m bounds remain',
            fixed='Seeds1..5,43D H-only,frozen source snapshot,raw EPS/OFFSETS/BETA,candidates/AP/bounds,empirical S,original GP fields',
            attribution='Exact versus full isolates prior evaluation at S; full versus without-C isolates old C; comparisons with A/count/finite-observation are whole recipes',
            prohibited='No count,selfcal,S-prior,v2 theta,new fit,weight search or original A rerun',
            timing=old['contract']['timing'], coverage=old['contract']['coverage'],
            comparison='All three arms versus original A/count/finite-observation, with both GP controls retained; per K/log/floor/log-floor mean and p90 and original point errors; no K pooling or automatic success',
            GP_parity='Both original GP endpoints must have zero coordinate difference',
            scoring='All three arms must seal all30 streams (90 total) before any original scorer opens GT',
            resources=dict(CPU_threads=2, GPU=0, fits=0, GP_adaptations=0, GP_sampling=0,
                           segment_seconds=args.seconds, external_timeout=900,
                           resume='After check exits0, prepare --resume reuses its frozen directory/context; otherwise explicit native exit2 segment continuation only')))
    write(args.output/'freeze.json', frozen)
    for arm in ARMS:
        directory = args.output/arm
        directory.mkdir()
        write(directory/'freeze.json', dict(frozen, arm=arm, parent_freeze_sha256=digest(args.output/'freeze.json')))
    return method


def prepare(args):
    check_only = args.phase == 'check'
    method = None if args.resume else freeze(args)
    original.verify_freeze(args.output)
    freeze_sha = digest(args.output/'freeze.json')
    assert not (args.output/'contexts_complete.json').exists()
    manifest = read(original.ASSETS/'manifest.json')
    arm_records = {arm: [] for arm in ARMS}
    pairs = []
    records = read(BASELINE/'contexts_complete.json')['contexts']
    assert (records[0]['floor'], records[0]['K']) == (2, 1)
    for record in records[:1] if check_only else records:
        directory = args.output/record['event']
        pair_path = directory/'complete.json'
        if pair_path.exists():
            pair = read(pair_path)
            assert pair['freeze_sha256'] == freeze_sha
            for value in pair['fields'].values():
                assert digest(value['path']) == value['sha256']
            assert digest(pair['conditioning_detail']['path']) == pair['conditioning_detail']['sha256']
        else:
            if remaining(args) <= 0:
                return 2
            if method is None:
                method = source_reader()
            if method.ops is None:
                method._load()
            parent = shared.checked(record)
            histories, support, ap_ids = original.context(manifest, record['floor'], record['K'])
            assert parent['ap_ids'] == ap_ids and parent['sources'] == list(histories)
            visible = (support['rss'] != 100).any(0)
            for h in histories.values():
                visible |= (h['rss'] != 100).any(0)
            aps = np.flatnonzero(visible)
            O = method.module
            maps = O.make_maps({int(name.rsplit('F', 1)[1]): h for name, h in histories.items()})
            sr, sx = support['rss'][:, aps], support['xy']
            Ps = O.det_stats(sr, sx)[0]
            X = np.unique(np.concatenate([m[0] for m in maps.values()]), axis=0)
            nH = len(X)
            X_aug = np.vstack([X, Ps])
            # Frozen rel=False features and HGB predictions are pointwise in X.
            # Only H maps enter the prior; no S RSS is passed to this call.
            returned, augmented = O.ensemble_field(method.ops, maps, record['floor'], aps, X=X_aug)
            assert np.array_equal(returned, X_aug)
            field = {key: augmented[key][:nH] for key in ('p', 'm', 's2')}
            equivalence = None
            if (record['floor'], record['K']) == (2, 1):
                old_X, old_field = O.ensemble_field(method.ops, maps, record['floor'], aps)
                assert np.array_equal(old_X, X)
                differences = {key: float(np.max(np.abs(old_field[key]-field[key]))) for key in field}
                assert all(value <= 1e-10 for value in differences.values()), differences
                equivalence = dict(passed=True, event=record['event'], max_abs_H_prior_difference=differences,
                    tolerance=1e-10, extra_ensemble_calls=1, query_used=False, scores_computed=False)
            conditioned = O.condition(X, field['p'], field['m'], field['s2'], sr, sx, (16., .3, 1., 0.))
            exact_aug = O.condition(X_aug, augmented['p'], augmented['m'], augmented['s2'], sr, sx,
                                    (16., .3, 1., 0.))
            exact = tuple(value[:nH] for value in exact_aug)
            distances = np.linalg.norm(Ps[:, None]-X[None], axis=2)
            nearest = distances.argmin(1)
            nearest_distance = distances.min(1)
            # Coincident H/Ps can choose the H copy; pointwise priors agree there.
            # The augmented Ps rows serve C only, never become extra reader atoms.
            assert (np.linalg.norm(Ps[:, None]-X_aug[None], axis=2).min(1) == 0).all()
            support_positions = Ps
            Ps, sp, sm, sv = _support_fields(sr, sx)
            assert np.array_equal(Ps, support_positions)
            nS = len(Ps)
            assert nS == record['K'] > 0
            candidates = np.vstack([X, Ps])
            bounds = np.stack([np.floor(candidates.min(0)/2)*2, np.ceil(candidates.max(0)/2)*2])
            values = dict(full_v1=conditioned, without_field_conditioning=(field['p'], field['m'], field['s2']),
                          exact_support_prior=exact)
            prepared = {arm: PreparedFrozenHurdle(candidates, aps,
                np.clip(np.vstack([p, sp]), 1e-4, 1-1e-4), np.vstack([m, sm]),
                np.vstack([v, sv]), bounds, support['rss'].shape[1]) for arm, (p, m, v) in values.items()}
            assert digest(parent['hurdle']['path']) == parent['hurdle']['sha256']
            with np.load(parent['hurdle']['path'], allow_pickle=False) as old:
                for h in prepared.values():
                    for name in ('cand_xy', 'aps', 'bounds', 'ap_count'):
                        assert np.array_equal(getattr(h, name), old[name]), (record['event'], name)
                    for name in ('p', 'm', 's2'):
                        assert np.array_equal(getattr(h, name)[-nS:], old[name][-nS:]), (record['event'], name, 'S')
            directory.mkdir(exist_ok=True)
            detail_path = directory/'conditioning_detail.npz'
            # Raw before-clipping priors/results, with explicit H/S row boundary.
            np.savez_compressed(detail_path, H_xy=X, S_xy=Ps, aps=aps,
                S_to_H_nearest=nearest, S_to_H_distance=nearest_distance, old_ignored=nearest_distance > 4.,
                **{f'H_prior_{key}': field[key] for key in field},
                **{f'S_prior_{key}': augmented[key][nH:] for key in field},
                **{f'legacy_H_{key}': value for key, value in zip(('p', 'm', 's2'), conditioned)},
                **{f'exact_H_{key}': value[:nH] for key, value in zip(('p', 'm', 's2'), exact_aug)},
                **{f'exact_S_{key}': value[nH:] for key, value in zip(('p', 'm', 's2'), exact_aug)})
            fields = {}
            for arm, h in prepared.items():
                path = directory/f'{arm}_field.npz'
                np.savez_compressed(path, **{key: getattr(h, key) for key in FIELD_KEYS})
                fields[arm] = binding(path)
            delta = {f'{a}_minus_{b}': {key: float(np.max(np.abs(
                getattr(prepared[a], key)[:nH]-getattr(prepared[b], key)[:nH]))) for key in ('p', 'm', 's2')}
                for a, b in (('full_v1', 'without_field_conditioning'), ('exact_support_prior', 'full_v1'))}
            pair = dict(complete=True, created=now(), freeze_sha256=freeze_sha, parent_context=record,
                fields=fields, max_abs_H_change=delta, same_empirical_S=True, same_candidates_AP_bounds=True,
                conditioning_detail=binding(detail_path), S_xy=Ps.tolist(), S_to_H_distance=nearest_distance.tolist(),
                old_ignored_count=int((nearest_distance > 4.).sum()), exact_ignored_count=0,
                equivalence=equivalence, ensemble_calls=1+int(equivalence is not None), query_used=False, fits=0)
            write(pair_path, pair)
        if (record['floor'], record['K']) == (2, 1):
            assert pair['equivalence']['passed']
            proof = dict(complete=True, freeze_sha256=freeze_sha, preparation=binding(pair_path), **pair['equivalence'])
            proof_path = args.output/'equivalence_check.json'
            if proof_path.exists():
                assert read(proof_path) == proof
            else:
                write(proof_path, proof)
        parent = shared.checked(record)
        for arm in ARMS:
            arm_dir = args.output/arm/record['event']
            arm_dir.mkdir(exist_ok=True)
            path = arm_dir/'complete.json'
            current = dict(parent, created=pair['created'], hurdle=pair['fields'][arm],
                freeze_sha256=digest(args.output/arm/'freeze.json'), arm=arm,
                paired_preparation=binding(pair_path), parent_context=record)
            if path.exists():
                assert read(path) == current
            else:
                write(path, current)
            arm_records[arm].append(dict(event=record['event'], floor=record['floor'], K=record['K'], **binding(path)))
        pairs.append(dict(event=record['event'], **binding(pair_path)))
        print(f'Prepared {len(pairs)}/15 three-arm frozen-v1 contexts', flush=True)
    if check_only:
        return 0
    for arm in ARMS:
        write(args.output/arm/'contexts_complete.json', dict(complete=True, created=now(), contexts=arm_records[arm],
            freeze_sha256=digest(args.output/arm/'freeze.json'), official_query_opened=False, GT_opened=False))
    write(args.output/'contexts_complete.json', dict(complete=True, created=now(), contexts=pairs,
        freeze_sha256=freeze_sha, arms={arm: binding(args.output/arm/'contexts_complete.json') for arm in ARMS},
        equivalence_check=binding(args.output/'equivalence_check.json'),
        official_query_opened=False, GT_opened=False))
    return 0


def verify_arm(output):
    parity = shared.verify_streams(output)
    assert all(delta == 0. for record in parity['comparisons'] for delta in record['max_difference_m'].values())
    return parity


def predict(args):
    original.verify_freeze(args.output)
    contexts = read(args.output/'contexts_complete.json')
    assert contexts['complete'] and contexts['freeze_sha256'] == digest(args.output/'freeze.json')
    sealed = {}
    for arm in ARMS:
        directory = args.output/arm
        assert digest(directory/'contexts_complete.json') == contexts['arms'][arm]['sha256']
        original.verify_freeze(directory)
        assert read(directory/'freeze.json')['parent_freeze_sha256'] == digest(args.output/'freeze.json')
        if not (directory/'predictions.json').exists():
            if remaining(args) <= 0:
                return 2
            arm_args = argparse.Namespace(output=directory, device='cpu', seconds=remaining(args), resume=args.resume)
            code = shared.predict(arm_args)
            if code == 2:
                verify_arm(directory)
                return 2
            assert code == 0
        parity = verify_arm(directory)
        assert parity['complete'] and parity['predictions_sha256'] == digest(directory/'predictions.json')
        sealed[arm] = binding(directory/'predictions.json')
    write(args.output/'predictions_complete.json', dict(complete=True, created=now(),
        freeze_sha256=digest(args.output/'freeze.json'), contexts_complete=binding(args.output/'contexts_complete.json'),
        arms=sealed, streams_per_arm=30, total_streams=90, GP_endpoints_exact=True, GT_opened=False))
    return 0


def compatible(a, b):
    for key in ('schema', 'gt_sha256', 'eligible_floors', 'methods', 'gt_rows', 'metric', 'coverage', 'reference'):
        assert a[key] == b[key]
    assert len(a['points']) == len(b['points'])
    for x, y in zip(a['points'], b['points']):
        assert {k: v for k, v in x.items() if k != 'errors'} == {k: v for k, v in y.items() if k != 'errors'}


def score(args):
    original.verify_freeze(args.output)
    seal = read(args.output/'predictions_complete.json')
    assert seal['complete'] and seal['freeze_sha256'] == digest(args.output/'freeze.json')
    assert digest(seal['contexts_complete']['path']) == seal['contexts_complete']['sha256']
    # Finish ALL THREE permission/parity checks before any scorer opens GT.
    for arm in ARMS:
        directory = args.output/arm
        original.verify_freeze(directory)
        assert digest(directory/'predictions.json') == seal['arms'][arm]['sha256']
        parity = verify_arm(directory)
        assert parity['complete'] and parity['predictions_sha256'] == seal['arms'][arm]['sha256']
    progress_path = args.output/'scoring_progress.json'
    if progress_path.exists():
        assert args.resume
        progress = read(progress_path)
        assert progress['predictions_complete_sha256'] == digest(args.output/'predictions_complete.json')
    else:
        progress = dict(predictions_complete_sha256=digest(args.output/'predictions_complete.json'), scores={})
    from ipin2025_score import score as original_score
    scores = {}
    for arm in ARMS:
        path = args.output/arm/'scores.json'
        if arm not in progress['scores']:
            if remaining(args) <= 0:
                return 2
            write(path, original_score(args.output/arm/'predictions.json'))
            progress['scores'][arm] = binding(path)
            write(progress_path, progress)
        assert digest(path) == progress['scores'][arm]['sha256']
        scores[arm] = read(path)
        assert scores[arm]['predictions_manifest_sha256'] == seal['arms'][arm]['sha256']
    for arm in ARMS[1:]:
        compatible(scores[ARMS[0]], scores[arm])
    comparisons = {}
    for label, directory in REFERENCES.items():
        baseline = read(directory/'scores.json')
        assert baseline['predictions_manifest_sha256'] == digest(directory/'predictions.json')
        for current in scores.values():
            compatible(baseline, current)
        comparisons[label] = dict(scores=binding(directory/'scores.json'), arms={arm:
            shared.compare_group(baseline['by_K'], scores[arm]['by_K']) for arm in ARMS})
    write(args.output/'comparison.json', dict(complete=True, created=now(), freeze_sha256=digest(args.output/'freeze.json'),
        predictions_complete=binding(args.output/'predictions_complete.json'), scores=progress['scores'],
        primary={f'{current}_vs_{reference}': dict(reference=reference, current=current,
            by_K=shared.compare_group(scores[reference]['by_K'], scores[current]['by_K']))
            for current, reference in (('exact_support_prior', 'full_v1'), ('full_v1', 'without_field_conditioning'))},
        references=comparisons, GP_endpoints_exact=True,
        delta_convention='Current minus reference; negative is improvement; per-K/log/floor/log-floor mean and p90',
        attribution='Exact vs full changes only prior evaluation at S; full vs without changes old C; external recipes are not S-only ablations',
        adoption=dict(automatic=False, status='requires_main_thread_judgment')))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('check', 'prepare', 'predict', 'score'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seconds', type=float, default=780.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output, args.started = args.output.resolve(), time.monotonic()
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 780
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    with threadpool_limits(limits=2):
        raise SystemExit({'check': prepare, 'prepare': prepare, 'predict': predict, 'score': score}[args.phase](args))
