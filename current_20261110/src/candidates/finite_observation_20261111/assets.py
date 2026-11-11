"""Source-only check and explicitly invoked five-seed asset generation.

check: four real B0 tasks and one duplicate-value representation check; no fit,
prediction or localization score. train: original HurdleField fits, replacing
only cell_table with the frozen finite-observation recipe; save each seed.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import signal
import socket
import sys
import time
from unittest.mock import patch

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import claude_hurdle_assets as original
from .deployment import KIND
from .features import SCHEMA, SCHEMA_ID, WIDTH, T, build_context
from .source import (CSV_SHA, EXPOSURE_RNG_TAG, MODES, UPSTREAM_FILES, array_sha, load_source,
                     make_cell_table, provenance, task)


ROOT = original.ROOT
DIRECTORY = Path(__file__).resolve().parent
CSV = Path('/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv')
GENERATION = ROOT/'results/claude_hurdle_b0_finite_obs_ens5_20261111'
BASELINE = ROOT/'results/claude_hurdle_b0_ens5_20261110'
SEEDS = [1, 2, 3, 4, 5]
digest, write = original.digest, original.write
read = lambda path: json.loads(Path(path).read_text())
now = lambda: datetime.now(timezone.utc).isoformat()


def code_bindings():
    files = [DIRECTORY/name for name in ('features.py', 'source.py', 'deployment.py', 'assets.py')]
    files += [ROOT/'claude_hurdle_assets.py', ROOT/'frozen_hurdle_translator.py', Path(T.__file__)]
    files += [original.PACKAGE/name for name in UPSTREAM_FILES]
    return {str(path.relative_to(ROOT)): digest(path) for path in files}


def versions():
    return {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')}


def source_index(C):
    exposure = {}
    for floor in range(4):
        _, counts = np.unique(C.XY[C.FLR == floor], axis=0, return_counts=True)
        exposure[str(floor)] = dict(positions=len(counts), minimum=int(counts.min()),
            median=float(np.median(counts)), maximum=int(counts.max()), positions_at_most3=int((counts <= 3).sum()))
    return dict(csv_sha256=CSV_SHA, source_buildings=[0], source_floors={'0': [0, 1, 2, 3]},
        AP_names=C.WAPS, row_id_convention='Original one-based CSV data-row IDs',
        source_rows_by_floor={f'B0F{f}': int((C.FLR == f).sum()) for f in range(4)},
        exposure_by_floor=exposure,
        rows_by_floor={str(f): C.ROW_IDS[C.FLR == f].tolist() for f in range(4)})


def check(args):
    C, V, H = load_source(args.csv)
    index = source_index(C)
    assert len(C.RSS) == 5249
    for floor, positions in enumerate((54, 67, 70, 68)):
        profile = index['exposure_by_floor'][str(floor)]
        assert profile['positions'] == positions and profile['median'] == 20
        assert profile['maximum'] == (42 if floor == 0 else 30) and profile['positions_at_most3'] == 0
    rng = np.random.default_rng([1, 0])
    records = []
    for draw in range(4):
        erng = np.random.default_rng([EXPOSURE_RNG_TAG, 1, 0, 0, 0, 1, draw])
        item = task(C, V, H, 0, 1, draw, rng, erng)
        ctx, srows, base, qrows = item['context'], item['S_rows'], item['base_S_rows'], item['Q_rows']
        assert not np.intersect1d(base, qrows).size
        assert np.isin(srows, base).all()
        assert (C.FLR[base] == 0).all() and (C.FLR[qrows] == 0).all()
        assert np.array_equal(qrows, np.setdiff1d(np.flatnonzero(C.FLR == 0), base))
        expected_positions = np.unique(C.XY[C.FLR != 0], axis=0)
        assert np.array_equal(ctx['cand_xy'], expected_positions)
        for floor, rows in item['H_rows'].items():
            assert (C.BLD[rows] == 0).all() and (C.FLR[rows] == floor).all() and floor != 0
            assert len(np.unique(rows)) == len(rows)
            assert np.array_equal(np.unique(C.XY[rows], axis=0), np.unique(C.XY[C.FLR == floor], axis=0))
            if draw in (1, 3):
                assert (ctx['history_stats'][floor]['n'] == 1).all()
            else:
                assert np.array_equal(rows, np.flatnonzero(C.FLR == floor))
        positions, counts = np.unique(C.XY[base], axis=0, return_counts=True)
        assert len(positions) == 1
        available = np.asarray([np.all(C.XY[C.FLR == 0] == xy, axis=1).sum() for xy in positions])
        assert np.array_equal(counts, np.minimum(3, available))
        if draw in (2, 3):
            assert (ctx['support_stats']['n'] == 1).all()
        else:
            assert np.array_equal(srows, base)
        visible_rows = np.concatenate([*item['H_rows'].values(), srows])
        assert np.array_equal(ctx['aps'], np.flatnonzero(C.DET[visible_rows].any(0)))
        assert ctx['features'].shape == (len(expected_positions), len(ctx['aps']), WIDTH)
        Pt = H.STATS[(0, 0)][0]
        D = np.linalg.norm(ctx['cand_xy'][:, None]-Pt[None], axis=2)
        assert np.array_equal(item['original_label'], D.argmin(1))
        good = item['eligible']
        assert np.array_equal(Pt[item['original_label'][good]], item['Q_positions'][item['labels'][good]])
        assert (item['table'][1] > 0).all()
        # Independent old-constructor path: original 43 H + 12 S, same raw scans.
        maps = {f: T.position_map(h['rss'], h['xy']) for f, h in item['histories'].items()}
        F, ref = T.history_features(ctx['cand_xy'], 0, maps)
        F = F.astype(np.float32)
        sr = T.naive_ref_at(item['support']['xy'], 0, maps)
        S = T.support_features(ctx['cand_xy'], ref, item['support']['xy'], item['support']['rss'], sr, F)
        old = np.concatenate([F[:, ctx['aps']], S.astype(np.float32)[:, ctx['aps']]], axis=2)
        assert np.array_equal(old, ctx['features'][:, :, :55], equal_nan=True)
        records.append(dict(provenance(C, item), permissions_passed=True, original_55_exact=True))
    # Duplicate values are solely an algebraic representation counterexample.
    # They are never additional independent observations or training tasks.
    repeated = {f: {key: np.repeat(value, 2, axis=0) for key, value in h.items()}
                for f, h in item['histories'].items()}
    doubled = build_context(repeated, item['support'], 0)
    assert np.array_equal(ctx['aps'], doubled['aps'])
    assert np.array_equal(ctx['features'][:, :, :55], doubled['features'][:, :, :55], equal_nan=True)
    assert not np.array_equal(ctx['features'][:, :, 55:71], doubled['features'][:, :, 55:71], equal_nan=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists()
    write(args.output, dict(passed=True, phase='source_only_check', created=now(),
        kind=KIND, schema_id=SCHEMA_ID, feature_schema=SCHEMA, feature_width=WIDTH,
        csv=dict(path=str(args.csv), sha256=CSV_SHA), code=code_bindings(), versions=versions(),
        source=index, tasks=records,
        prior_audit='Sol read-only B0 audit relayed by main on2026-11-11:5249 rows,259 positions,n>3 everywhere; rechecked here',
        duplicate_value_check=dict(old55_exact=True, exposure_features_changed=True,
                                   interpretation='Repeated values only; not extra independent evidence'),
        fits=0, predictions=0, localization_scores=0, target_RSS_parsed=False))
    print(json.dumps(dict(passed=True, source_tasks=4, fits=0, predictions=0,
                         path=str(args.output), sha256=digest(args.output))), flush=True)
    return 0


def train(args):
    started = time.monotonic()
    if args.resume:
        frozen = read(args.output/'freeze.json')
        assert frozen['kind'] == KIND and frozen['code'] == code_bindings()
        assert frozen['versions'] == versions() and frozen['schema_id'] == SCHEMA_ID
        assert frozen['inputs']['csv'] == dict(path=str(args.csv), sha256=digest(args.csv))
        assert not (args.output/'complete.json').exists()
        checked = read(args.output/'source_check.json')
        assert digest(args.output/'source_check.json') == frozen['source_check_sha256']
    else:
        assert args.source_check is not None, '--source-check is required for initial generation'
        checked = read(args.source_check)
    assert checked['passed'] and checked['phase'] == 'source_only_check'
    assert checked['kind'] == KIND and checked['code'] == code_bindings() and checked['versions'] == versions()
    assert checked['fits'] == checked['predictions'] == 0 and checked['csv']['sha256'] == CSV_SHA
    C, V, H = load_source(args.csv)
    index = source_index(C)
    assert checked['source'] == index
    if not args.resume:
        base = read(BASELINE/'complete.json')
        base_freeze = read(BASELINE/'freeze.json')
        assert base['complete'] and base['freeze_sha256'] == digest(BASELINE/'freeze.json')
        assert base_freeze['source_buildings'] == [0] and base_freeze['seeds'] == SEEDS
        assert base_freeze['source_rows_by_floor'] == index['source_rows_by_floor'] and base_freeze['draws'] == 4
        args.output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(args.source_check, args.output/'source_check.json')
        write(args.output/'source_index.json', index)
        deployment_names = [str((DIRECTORY/name).relative_to(ROOT)) for name in ('features.py', 'deployment.py')]
        deployment_names += ['vertical_translator/translator.py', 'frozen_hurdle_translator.py']
        write(args.output/'freeze.json', dict(created=now(), kind=KIND, schema_id=SCHEMA_ID,
            feature_width=WIDTH, feature_schema=SCHEMA, seeds=SEEDS, draws=4, K=[1, 3, 5],
            source_buildings=[0], source_floors=index['source_floors'], source_rows_by_floor=index['source_rows_by_floor'],
            original_B0_exposure=index['exposure_by_floor'],
            original_B0_support='Every sampled source position has >3 rows, so original base S is exactly3 rows/position',
            source_check_sha256=digest(args.output/'source_check.json'), source_index_sha256=digest(args.output/'source_index.json'),
            inputs=dict(csv=dict(path=str(args.csv), sha256=CSV_SHA),
                baseline_complete=dict(path=str(BASELINE/'complete.json'), sha256=digest(BASELINE/'complete.json')),
                baseline_freeze=dict(path=str(BASELINE/'freeze.json'), sha256=digest(BASELINE/'freeze.json'))),
            code=code_bindings(), deployment_code={name: digest(ROOT/name) for name in deployment_names},
            versions=versions(), upstream_commit=original.UPSTREAM,
            upstream_git_blobs={name: original.BLOBS[name] for name in UPSTREAM_FILES},
            fit='Verbatim HurdleField([0],seed,draws=4), with only its cell_table replaced',
            recipe=dict(draw_modes=list(MODES),
                support='Original sample_support(...,False), K positions and min(3,available) real scans; thin only within base S',
                history='All original B0 history positions; full rows or one real row per position',
                exposure_rng='default_rng([20261111,seed,parity_group,building,floor,K,draw]); independent of original selection RNG',
                query='Target source rows excluding ALL base S rows, including reserved rows not used by low-exposure S',
                labels='Original nearest target position within4m; skip if no remaining Q, never remap to another position',
                AP_axis='Visible H/S only; no full-H or Q AP supplementation',
                selection='Original info plus5% negative sampling, max6000 cells/task; original n/nd weights',
                attribution='Joint finite-observation task construction and information representation; not an n-only ablation',
                effective_n='Equal-position exposure summary L^2/sum(1/n), not independent sample size',
                limits='Does not identify unknown query detection shift b or assume bS=bQ'),
            reader=dict(beta=128., eps=.02, offsets=[-8., -4., 0., 4., 8.],
                        empirical_S='Original _support_fields, appended once after ensemble',
                        return_type='Original PreparedFrozenHurdle; no count/IPW/Gamma/S-prior inside'),
            resources=dict(CPU_threads=2, GPU=0, segment_seconds=args.seconds, max_seeds_per_segment=args.max_seeds,
                           natural_save='One completed seed', external_timeout_seconds=900, downloads=0),
            runtime=dict(python=sys.executable), predictions=0, localization_scores=0))
    frozen = read(args.output/'freeze.json')
    assert digest(args.output/'source_index.json') == frozen['source_index_sha256']
    assert read(args.output/'source_index.json') == index
    freeze_sha = digest(args.output/'freeze.json')
    progress_path = args.output/'progress.json'
    progress = read(progress_path) if args.resume and progress_path.exists() else dict(models=[])
    if 'freeze_sha256' in progress:
        assert progress['freeze_sha256'] == freeze_sha
    models = progress['models']
    assert [r['seed'] for r in models] == SEEDS[:len(models)]
    for record in models:
        assert digest(args.output/record['path']) == record['sha256']
        assert digest(args.output/record['source_tasks']['path']) == record['source_tasks']['sha256']
    completed_here = 0

    def timeout(signum, frame):
        raise TimeoutError('Generation time budget exhausted; completed seeds retained, explicit resume required')

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    remaining = args.seconds-(time.monotonic()-started)
    if remaining <= 0:
        signal.signal(signal.SIGALRM, previous_handler)
        return 2
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        with threadpool_limits(limits=2):
            for seed in SEEDS[len(models):]:
                if completed_here >= args.max_seeds:
                    return 2
                write(progress_path, dict(complete=False, phase='fit_seed', active_seed=seed,
                    models=models, freeze_sha256=freeze_sha))
                print(json.dumps(dict(phase='fit_seed', seed=seed)), flush=True)
                tick, records = time.monotonic(), []
                with patch.object(H, 'cell_table', make_cell_table(C, V, H, seed, records)):
                    field = H.HurdleField([0], seed=seed, draws=4)
                assert len(records) == 4*3*4
                task_path = args.output/f'seed_{seed}_tasks.json'
                write(task_path, dict(seed=seed, freeze_sha256=freeze_sha,
                    source_index_sha256=frozen['source_index_sha256'], tasks=records))
                task_record = dict(path=task_path.name, sha256=digest(task_path))
                payload = dict(kind=KIND, schema_id=SCHEMA_ID, feature_width=WIDTH, feature_schema=SCHEMA,
                    seed=seed, source_buildings=[0], source_floors=index['source_floors'],
                    source_rows_by_floor=index['source_rows_by_floor'], draws=4, cols=field.cols,
                    freeze_sha256=freeze_sha, source_tasks=task_record, versions=versions(),
                    **{name: dict(estimator=getattr(field, name).est, cols=getattr(field, name).cols)
                       for name in ('det', 'mean', 'scale')})
                path = args.output/f'seed_{seed}.joblib'
                temporary = args.output/f'.seed_{seed}.tmp.joblib'
                joblib.dump(payload, temporary)
                temporary.replace(path)
                models.append(dict(seed=seed, path=path.name, sha256=digest(path), source_tasks=task_record,
                    bytes=path.stat().st_size, seconds=time.monotonic()-tick))
                write(progress_path, dict(complete=False, phase='seed_saved', models=models, freeze_sha256=freeze_sha))
                print(json.dumps(models[-1]), flush=True)
                completed_here += 1
                del field, payload, records
        write(args.output/'complete.json', dict(complete=True, created=now(), kind=KIND,
            schema_id=SCHEMA_ID, feature_width=WIDTH, feature_schema=SCHEMA,
            freeze_sha256=freeze_sha, models=models, source_buildings=[0],
            source_floors=index['source_floors'], source_rows_by_floor=index['source_rows_by_floor'],
            seconds_this_segment=time.monotonic()-started, source_check_sha256=frozen['source_check_sha256'],
            source_index_sha256=frozen['source_index_sha256'], predictions=0, localization_scores=0,
            original_fit_code_reused=True, original_task_recipe_changed=True))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('check', 'train'))
    parser.add_argument('--csv', type=Path, default=CSV)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-check', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-seeds', type=int, default=5)
    parser.add_argument('--seconds', type=float, default=840.)
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 840 and 1 <= args.max_seeds <= 5
    args.csv = args.csv.resolve()
    args.output = (args.output or (DIRECTORY/'source_check.json' if args.phase == 'check' else GENERATION)).resolve()
    with threadpool_limits(limits=2):
        raise SystemExit({'check': check, 'train': train}[args.phase](args))
