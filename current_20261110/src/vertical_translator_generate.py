"""Generate the original B0-only five-seed translator assets, not Tab training.

Fixed seeds1..5, draws8, original four-floor B0 corpus and HGB settings. Each
seed's H and H+S regressors are fitted once, shared by both target buildings:
the package's original b0only loop repeated identical fits after resetting RNG.
Only that redundant work is removed. Per-seed float64 maps are retained before
the original mean-of-maps readout; final exported maps retain originalfloat32.

The unmodified package preloads the full CSV, coordinates and all floor maps.
Do not call this zero Query-file access: target Query labels are not used by
the B0 fitting or event prediction functions, and scoring is invoked only
after the event's predictions are saved. B0 four-floor/all-row permission is
broader than the current clean three-floor Tab training permission.

CPU-only, two threads, no retry/resume/seed extension. Launch only when
authorized, under an external timeout of at most900s; a boundary check also
stops work once900s have elapsed. This file is prepared without execution.
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT/'vertical_translator'
MANIFEST = ROOT/'results/building_information_assets_20261109/manifest.json'
SEEDS = (1, 2, 3, 4, 5)
ARCHIVE_SHA256 = '8486c2174a041cf649c5c093eab92631cb323c95e65248af92f0f90b0cc88561'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def run(args):
    started = time.monotonic()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['LAW_UJI_CSV'] = str(args.csv.resolve())
    sys.path.insert(0, str(PACKAGE))
    import numpy as np
    import joblib
    from threadpoolctl import threadpool_limits
    import core as C
    import vt_lib as V

    original = json.loads(MANIFEST.read_text())
    assert digest(args.csv) == original['csv_sha256'] == C.MAN['csv_sha256']
    for key in ('seed', 'events', 'history_rows', 'floors', 'buildings', 'K', 'R'):
        assert C.MAN[key] == original[key]
    assert len(C.MAN['events']) == 54 and V.FLOORS[0] == [0, 1, 2, 3]
    assert digest(PACKAGE/'protocol9.py') == digest(ROOT/'building_information_protocol.py')
    versions = {n: version(n) for n in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl')}
    code = {str(p): digest(p) for p in [Path(__file__), *sorted(PACKAGE.glob('*.py'))]}
    freeze = dict(created=datetime.now(timezone.utc).isoformat(), code=code,
        input_sha256={str(args.csv.resolve()): digest(args.csv), str(MANIFEST): digest(MANIFEST)},
        original_archive=dict(sha256=ARCHIVE_SHA256, bytes=47692), versions=versions,
        train='b0only', seeds=list(SEEDS), draws=8, single_acquisition=False,
        source_rows_by_floor={f'B0F{f}': int(((C.BLD == 0) & (C.FLR == f)).sum()) for f in V.FLOORS[0]},
        source_permission='All original B0 F0/F1/F2/F3 rows, no protected exclusion; broader than existing Tab clean three-floor assets',
        training='Original V.training_set([0],rng,draws8,single_acquisition=False), one H and one HS V.Model per seed; no hyperparameter change',
        model_settings=dict(max_iter=400, learning_rate=.06, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=1., random_state=0),
        map_semantics='Mean of encoded RSS with missing=-105, not physical latent RSS or full observation density',
        reader='Original V.read_knn k3, missing=-105; distinct from the project raw nearest3 missing=-110',
        naive_reference='Original V.hist_feats reference, same candidate/AP/bounds and V.read_knn plus real S; no fitting',
        label_permission='Package eagerly loads full CSV/maps; target Q labels do not enter fitting/prediction; score only after persisted predictions',
        output='Original final NPZ schema plus naive_ref/pred_3nn_naive; no thresholding, rounding or Tab/GP field replacement',
        resources=dict(cpu_threads=2, GPUs=0, max_seconds=900, hard_timeout='external launcher', downloads=0),
        limits=['Previously exposed target development events; no blind claim or same-training-information attribution',
                'No Tab/GP training, new seeds, target selection or automatic retry',
                'Five fixed seed maps are averaged, not selected by target errors'])
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'freeze.json', freeze)
    freeze_sha = digest(args.output/'freeze.json')

    def budget():
        if time.monotonic()-started >= 900:
            raise TimeoutError('900s boundary reached; no automatic continuation')

    seed_records = []
    with threadpool_limits(limits=2):
        for seed in SEEDS:
            budget(); tick = time.monotonic()
            directory = args.output/f'seed_{seed}'; directory.mkdir()
            Xh, Xs, y = V.training_set([0], np.random.default_rng(seed), draws=8, single_acquisition=False)
            models, model_records = [], []
            for label, X in (('H', Xh), ('HS', np.hstack([Xh, Xs]))):
                budget(); fit_started = time.monotonic()
                model = V.Model(X, y)
                path = directory/f'model_{label}.joblib'
                # Plain sklearn state + column map is reloadable without
                # unpickling V.Model and triggering core's eager CSV imports.
                joblib.dump(dict(estimator=model.m, cols=model.cols, feature_count=X.shape[1],
                    seed=seed, kind=label, source_buildings=[0], training_cells=len(y),
                    freeze_sha256=freeze_sha, versions=versions), path)
                models.append(model)
                model_records.append(dict(kind=label, path=str(path), sha256=digest(path),
                    training_cells=len(y), seconds=time.monotonic()-fit_started))
            del Xh, Xs, y, X
            map_records = []
            for event in C.MAN['events']:
                budget()
                b, t = int(event['building'][1:]), int(event['floor'].split('F')[1])
                H, S, _ = C.event_arrays(event)
                aps = np.flatnonzero(C.DET[np.concatenate([H, S])].any(0))
                xy, mH = V.translate(models[0], b, t, S, aps, False)
                _, mHS = V.translate(models[1], b, t, S, aps, True)
                path = directory/f'{event["id"]}.npz'
                np.savez_compressed(path, cand_xy=xy, map_H=mH, map_HS=mHS, aps=aps)
                map_records.append(dict(event=event['id'], path=str(path), sha256=digest(path)))
            record = dict(seed=seed, models=model_records, maps=map_records, seconds=time.monotonic()-tick)
            write(directory/'complete.json', dict(record, complete=True, freeze_sha256=freeze_sha))
            seed_records.append(record)
            print(json.dumps(dict(stage='seed_complete', seed=seed, seconds=record['seconds'],
                                  model_seconds=[r['seconds'] for r in model_records])), flush=True)
            del models, model
        errors = {name: {} for name in ('translated_H', 'translated_HS', 'naive_ref')}
        predictions = []
        for event in C.MAN['events']:
            budget()
            b, t = int(event['building'][1:]), int(event['floor'].split('F')[1])
            H, S, Q = C.event_arrays(event); ctx = np.concatenate([H, S])
            aps = np.flatnonzero(C.DET[ctx].any(0)); bounds = C.bounds_of(C.XY[ctx])
            xy, _, reference = V.hist_feats(b, t)
            maps = []
            for seed in SEEDS:
                with np.load(args.output/f'seed_{seed}'/f'{event["id"]}.npz') as saved:
                    assert np.array_equal(saved['aps'], aps) and np.array_equal(saved['cand_xy'], xy)
                    maps.append((saved['map_H'].copy(), saved['map_HS'].copy()))
            mH = np.mean([m[0] for m in maps], 0); mHS = np.mean([m[1] for m in maps], 0)
            naive = reference[:, aps]
            pH = V.read_knn(xy, mH, S, Q, aps, bounds)
            pHS = V.read_knn(xy, mHS, S, Q, aps, bounds)
            pN = V.read_knn(xy, naive, S, Q, aps, bounds)
            path = args.output/f'{event["id"]}.npz'
            np.savez_compressed(path, cand_xy=xy, map_H=mH.astype(np.float32), map_HS=mHS.astype(np.float32),
                aps=aps, bounds=bounds, query_rows=np.asarray(event['query_rows']),
                pred_3nn_H=pH, pred_3nn_HS=pHS, naive_ref=naive.astype(np.float32), pred_3nn_naive=pN)
            predictions.append(dict(event=event['id'], path=str(path), sha256=digest(path)))
            for name, pred in zip(errors, (pH, pHS, pN)):
                errors[name][event['id']] = C.event_error(event, pred)
    summary = {name: C.summarize(value) for name, value in errors.items()}
    write(args.output/'summary.json', dict(train='b0only', seeds=list(SEEDS), draws=8,
        summary=summary, events=errors, freeze_sha256=freeze_sha))
    write(args.output/'complete.json', dict(complete=True, freeze_sha256=freeze_sha,
        summary_sha256=digest(args.output/'summary.json'), records=predictions,
        seeds=seed_records, models=10, events=54, seconds=time.monotonic()-started,
        Tab_training=False, GP_fits=0, target_selection=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web', 'Generation is server-only; no local experiment'
    run(args)
