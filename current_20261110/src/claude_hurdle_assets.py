"""Persist Claude's original five-seed hurdle models for deployment.

Calls the verbatim HurdleField(sources, seed, draws=4) without changing corpus,
sampling, scan-count weights, estimator settings or early stopping. No target
localization evaluation is performed. The upstream modules eagerly load the
full existing CSV; only specified source pairs enter their fit. Export plain sklearn states
so deployment does not import those dataset-loading modules.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import csv
import hashlib
import json
import socket
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT/'claude_hurdle_20261110'
SEEDS = [1, 2, 3, 4, 5]
UPSTREAM = 'a9e353712e6112a5b33998a47cb409292dfcf069'
BLOBS = {
    'core.py': '0989ad8ad670ca622bbe4f8bd6f477fe6ade1755',
    'translator.py': '6f02006dcc539447222ca91ea75d4104ee218c79',
    'vt_lib.py': '672a54c8bb1a134c798eab6ce6eac9aea7ddb755',
    'protocol9.py': '4eb097c6d45fb12b1c7759b79f778bfe5a9c7865',
    'hurdle.py': '72d427d08a0e834dacd78a6b13c59dcce1adf526',
    'run_nine_floor_hurdle.py': '6ee9e07a8f5bcec916512a955f3fb40b0e4db6e9',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run(args):
    start = time.monotonic()
    args.output = args.output.resolve()
    args.csv = args.csv.resolve()
    sources = list(args.sources)
    if not sources or len(set(sources)) != len(sources) or not set(sources) <= {0, 1, 2}:
        raise ValueError('Sources must be an ordered, nonempty list of distinct UJI building IDs')
    source_rows = {}
    with args.csv.open(newline='') as stream:
        for row in csv.DictReader(stream):
            b, f = int(row['BUILDINGID']), int(row['FLOOR'])
            if b in sources:
                key = f'B{b}F{f}'
                source_rows[key] = source_rows.get(key, 0) + 1
    source_floors = {str(b): sorted(int(k.rsplit('F', 1)[1]) for k in source_rows
                                  if k.startswith(f'B{b}F')) for b in sources}
    for name, expected in BLOBS.items():
        data = (PACKAGE/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == expected
    assert digest(args.csv) == '45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168'
    versions = {name: version(name) for name in
                ('numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib', 'threadpoolctl')}
    args.output.mkdir(exist_ok=False)
    write(args.output/'freeze.json', dict(
        created=datetime.now(timezone.utc).isoformat(), upstream_commit=UPSTREAM,
        upstream_git_blobs=BLOBS, train='b0only' if sources == [0] else 'explicit_sources', seeds=SEEDS, draws=4,
        source_buildings=sources, source_floors=source_floors, source_rows_by_floor=source_rows,
        source_permission='Original full corpus of the ordered source buildings; original sampling, label-dependent cell selection and scan-count weights preserved',
        fit=f'Verbatim hurdle.HurdleField({sources}, seed=seed, draws=4), including original estimator defaults',
        reader=dict(beta=128., eps=.02, offsets=[-8., -4., 0., 4., 8.],
                    selection_source='UJI B0 development; not independent of B0', evaluation_status='exposed_development'),
        code={str(p): digest(p) for p in [Path(__file__), *[PACKAGE/n for n in BLOBS]]},
        inputs={str(args.csv): digest(args.csv)}, versions=versions,
        runtime=dict(python=sys.executable, pythonpath=os.environ.get('PYTHONPATH', '')),
        resources=dict(CPU_threads=2, GPU=0, seconds=args.seconds, external_timeout=900, downloads=0),
        purpose='Persist deployable five-model assets, not repeat the nine-floor evaluation',
        reference='One source-context scan, no error scoring; used only to bind the portable deployment reader',
        access='Upstream eagerly reads full CSV and precomputes floor statistics; only allowed sources enter fit and reference prediction',
        performance='Asset export only; no new localization score'))
    frozen_sha = digest(args.output/'freeze.json')
    os.environ['LAW_UJI_CSV'] = str(args.csv)
    sys.path.insert(0, str(PACKAGE))
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits
    import core as C
    import hurdle as H
    import run_nine_floor_hurdle as R

    assert all(H.V.FLOORS[b] == source_floors[str(b)] for b in sources)
    reference_manifest = C.DEV if 0 in sources else C.MAN
    event = next(e for e in reference_manifest['events'] if int(e['building'][1:]) in sources)
    reference_building = int(event['building'][1:])
    history_rows, support_rows, query_rows = C.event_arrays_m(reference_manifest, event)
    aps = np.flatnonzero(C.DET[np.concatenate((history_rows, support_rows))].any(0))
    query_rows = query_rows[:1]
    models, reference_fields = [], []
    def budget():
        if time.monotonic()-start >= args.seconds:
            raise TimeoutError('Explicit asset-generation budget exhausted; retain existing models, no automatic retry')
    with threadpool_limits(limits=2):
        for seed in SEEDS:
            budget()
            tick = time.monotonic()
            field = H.HurdleField(sources, seed=seed, draws=4)
            payload = dict(kind='claude_hurdle_v1', seed=seed, source_buildings=sources, draws=4,
                source_floors=source_floors, source_rows_by_floor=source_rows,
                cols=field.cols, freeze_sha256=frozen_sha, versions=versions,
                **{name: dict(estimator=getattr(field, name).est, cols=getattr(field, name).cols)
                   for name in ('det', 'mean', 'scale')})
            path = args.output/f'seed_{seed}.joblib'
            joblib.dump(payload, path)
            model = dict(seed=seed, path=str(path), sha256=digest(path),
                         bytes=path.stat().st_size, seconds=time.monotonic()-tick)
            models.append(model)
            xy, p, mean, variance = field.predict(reference_building, int(event['floor'].split('F')[1]),
                                                 support_rows, aps)
            reference_fields.append((p, mean, variance))
            write(args.output/'progress.json', dict(models=models, freeze_sha256=frozen_sha))
            print(json.dumps(dict(seed=seed, seconds=model['seconds'], model_bytes=model['bytes'])), flush=True)
            del field, payload
        budget()
        ps, means, variances = (np.stack([value[i] for value in reference_fields]) for i in range(3))
        mean_p, mean_m = ps.mean(0), means.mean(0)
        mean_v = variances.mean(0) + means.var(0)
        candidates, log_mass = R.posterior(xy, mean_p, mean_m, mean_v,
                                           support_rows, query_rows, aps)
        bounds = C.bounds_of(C.XY[np.concatenate((history_rows, support_rows))])
        reference = args.output/'deployment_reference.npz'
        np.savez_compressed(reference, cand_xy=candidates, log_position_mass=log_mass,
                            projected=np.clip(np.exp(log_mass) @ candidates, bounds[0], bounds[1]),
                            query_rows=query_rows+1, aps=aps)
        write(args.output/'deployment_reference.json', dict(
            event=event, history_rows={h:reference_manifest['history_rows'][h] for h in event['histories']},
            query_rows=(query_rows+1).tolist(), reference_path=str(reference),
            reference_sha256=digest(reference), score_computed=False))
    write(args.output/'complete.json', dict(complete=True,
        created=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-start,
        freeze_sha256=frozen_sha, models=models, source_buildings=sources,
        source_floors=source_floors, source_rows_by_floor=source_rows,
        reference_metadata_sha256=digest(args.output/'deployment_reference.json'),
        target_predictions=0, localization_scores=0, original_recipe_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv'))
    parser.add_argument('--output', type=Path, default=ROOT/'results/claude_hurdle_b0_ens5_20261110')
    parser.add_argument('--seconds', type=float, default=840.)
    parser.add_argument('--sources', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 840
    run(args)
