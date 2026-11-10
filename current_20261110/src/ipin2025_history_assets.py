"""Training-only, shared-budget history assets for fixed IPIN2025 confirmation.

The existing public-corpus coordinate-only FPS budget (128 positions/floor,
one real receipt/position) is applied prospectively to BOTH Hurdle and GP.
No official Query/GT, global hyperparameter fitting or Hurdle training occurs.
History posteriors use the unchanged full-Gaussian solver and current UJI
base/observation parameters. A segment saves complete AP fields and stops at
its time boundary; --resume explicitly continues the same frozen preparation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import history_fit
from uji import group_scans
from building_information_tab_trial import digest, now, write
from ipin2025_observation_data import (
    load_training, metric_xy, farthest_history_rows, rss_matrix, REF_LON, REF_LAT)

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/Law-Probabilistic-Transfer-20261023')
DATA = Path('/home/panyushuo/datasets/ipin_t3_2025/2025_IPIN_Competition_Track03')
PARAMETERS = ROOT/'results/paired_assets_20261106/parameters.json'
HURDLE = ROOT/'results/claude_hurdle_b0_ens5_20261110'
CODE = ('ipin2025_history_assets.py', 'ipin2025_observation_data.py', 'history_fit.py',
        'observations.py', 'adaptation.py', 'fields.py', 'uji.py')


def read(path):
    return json.loads(Path(path).read_text())


def freeze(output):
    """Use only Training logs and known model metadata; seal the shared rows."""
    sys.path.append(str(OLD))
    from public_corpus.adapter import parse_log
    training = load_training(DATA, parse_log=parse_log)
    parameters = read(PARAMETERS)
    source = read(HURDLE/'complete.json')
    assert source['complete'] and read(HURDLE/'deployment_check.json')['passed']
    assert source['freeze_sha256'] == digest(HURDLE/'freeze.json')
    output.mkdir(exist_ok=False)
    inputs = {str(p):digest(p) for p in map(Path, training['training_files'])}
    inputs.update({str(p):digest(p) for p in (PARAMETERS, HURDLE/'complete.json',
                                            HURDLE/'freeze.json')})
    rows_by_floor, histories, support = {}, {}, {}
    for floor, records in sorted(training['floors'].items()):
        rows = farthest_history_rows(metric_xy(records), floor)
        selected = [records[i] for i in rows]
        ap_ids = sorted({ap for r in selected for ap in r['rss']})
        path = output/f'history_F{floor}.npz'
        np.savez_compressed(path, xy=metric_xy(selected), rss=rss_matrix(selected, ap_ids),
                            ap_ids=np.asarray(ap_ids), scan_ids=np.asarray([r['scan_id'] for r in selected]))
        rows_by_floor[str(floor)] = rows.tolist()
        histories[str(floor)] = dict(path=str(path), sha256=digest(path),
            positions=len(rows), available_scans=len(records),
            available_positions=len(np.unique(metric_xy(records), axis=0)),
            ap_ids=ap_ids, selection_seed=261026+2025*100+floor,
            scan_ids=[r['scan_id'] for r in selected])
    for floor in training['eligible_floors']:
        selected = training['support_order'][floor]
        ap_ids = sorted({ap for r in selected for ap in r['rss']})
        path = output/f'support_F{floor}.npz'
        np.savez_compressed(path, xy=metric_xy(selected), rss=rss_matrix(selected, ap_ids),
                            ap_ids=np.asarray(ap_ids), scan_ids=np.asarray([r['scan_id'] for r in selected]))
        support[str(floor)] = dict(path=str(path), sha256=digest(path),
            scan_ids=[r['scan_id'] for r in selected], ap_ids=ap_ids,
            provenance=[dict(file=r['relative_file'], app_timestamp=r['app_timestamp'],
                landmark_lines=r['landmark_lines'], landmark_times=r['landmark_times'],
                fraction=r['fraction']) for r in selected])
    manifest = dict(eligible_floors=training['eligible_floors'],
        floor_metadata=training['floor_metadata'], histories=histories, support=support,
        history_rows=rows_by_floor, K=[1,3,5], R=1, reference=training['reference'])
    write(output/'manifest.json', manifest)
    write(output/'freeze.json', dict(created=now(), inputs=inputs,
        code={name:digest(ROOT/name) for name in CODE}, parser=dict(
            path=str(Path(parse_log.__code__.co_filename)), sha256=digest(parse_log.__code__.co_filename)),
        manifest_sha256=digest(output/'manifest.json'), parameters=parameters,
        contract=dict(scope='Training-only shared histories and support; official Query/GT unopened',
            method='Unchanged original Hurdle+retained GP fixed probability half',
            history='Existing public-corpus coordinate-only FPS; at most128 unique XY, one real receipt, shared by both paths',
            selection='Prospectively frozen for IPIN2025; inherited algorithm/budget, not a pre-existing IPIN2025 subset',
            support='Original hash-ranked nested5 positions, prefixes K1/3/5, one actual receipt each',
            coverage='All qualifying floors retained; missing Level1 reference remains in full-site coverage',
            coordinates=dict(longitude=REF_LON, latitude=REF_LAT, metres_per_degree=111320),
            solver='Original full-Gaussian history VI; max640 iterations, projected-gradient1e-3 as current history assets',
            resources=dict(CPU_threads=2, GPU=0, downloads=0),
            note='This prepares per-environment GP assets; source learned weights/global GP parameters are fixed')))


def run(args):
    output = args.output.resolve()
    if not args.resume:
        freeze(output)
    frozen, manifest = read(output/'freeze.json'), read(output/'manifest.json')
    assert digest(output/'manifest.json') == frozen['manifest_sha256']
    assert all(digest(ROOT/name) == sha for name,sha in frozen['code'].items())
    assert all(digest(path) == sha for path,sha in frozen['inputs'].items())
    assert digest(frozen['parser']['path']) == frozen['parser']['sha256']
    assert not (output/'complete.json').exists()
    progress_path = output/'fields.jsonl'
    records = [json.loads(line) for line in progress_path.read_text().splitlines()] if progress_path.exists() else []
    done = {(record['floor'],record['ap_index']) for record in records}
    assert len(done) == len(records)
    base, obs = frozen['parameters']['base'], frozen['parameters']['obs']
    start = time.monotonic()
    for floor, history in manifest['histories'].items():
        assert digest(history['path']) == history['sha256']
        with np.load(history['path']) as saved:
            scans = group_scans(saved['rss'], saved['xy'])
            ap_ids = saved['ap_ids'].astype(str).tolist()
        assert ap_ids == history['ap_ids']
        directory = output/f'fields_F{floor}'
        directory.mkdir(exist_ok=True)
        for ap, ap_id in enumerate(ap_ids):
            if (floor,ap) in done:
                continue
            if time.monotonic()-start >= args.seconds:
                write(output/'progress.json', dict(complete=False, created=now(),
                    fields=len(records), next_field=dict(floor=floor,ap_index=ap),
                    seconds=time.monotonic()-start, explicit_resume_required=True))
                return 2
            diagnostics = {}
            original = history_fit.minimize
            def monitored(*a, **kw):
                result = original(*a, **kw)
                gradient = np.asarray(result.jac).copy()
                for j,(lo,hi) in enumerate(kw['bounds']):
                    if lo is not None and result.x[j] <= lo+1e-10 and gradient[j] > 0: gradient[j]=0
                    if hi is not None and result.x[j] >= hi-1e-10 and gradient[j] < 0: gradient[j]=0
                diagnostics.update(success=bool(result.success), iterations=int(result.nit),
                    projected_gradient=float(np.abs(gradient).max()), message=str(result.message))
                if not (np.isfinite(result.fun) and np.isfinite(result.x).all()
                        and np.isfinite(gradient).all() and diagnostics['projected_gradient'] <= 1e-3):
                    raise RuntimeError(f'Unfinished history fit F{floor}/{ap_id}: {diagnostics}')
                return result
            tick = time.monotonic()
            with patch.object(history_fit, 'minimize', monitored):
                field = history_fit.single_history(dict(scans=scans), ap, base, obs, steps=640)
            assert field is not None
            path = directory/f'ap{ap}.npz'
            np.savez_compressed(path, z=field.z.numpy(), mean=field.m.numpy(), cov=field.c.numpy())
            record = dict(floor=floor,ap_index=ap,ap_id=ap_id,path=str(path),
                sha256=digest(path),seconds=time.monotonic()-tick,solver=diagnostics)
            with progress_path.open('a') as stream: stream.write(json.dumps(record)+'\n')
            records.append(record)
            if len(records)%50 == 0: print(f'Saved {len(records)} history fields', flush=True)
        print(f'Completed F{floor}: {len(ap_ids)} AP fields', flush=True)
    expected = sum(len(h['ap_ids']) for h in manifest['histories'].values())
    assert len(records) == expected
    write(output/'histories.json', dict(manifest_sha256=frozen['manifest_sha256'],
        parameters_path=str(PARAMETERS), parameters_sha256=digest(PARAMETERS), fields=records))
    write(output/'complete.json', dict(complete=True,created=now(),
        freeze_sha256=digest(output/'freeze.json'),manifest_sha256=frozen['manifest_sha256'],
        histories_sha256=digest(output/'histories.json'),history_fields=expected,
        fields_by_floor={f:len(h['ap_ids']) for f,h in manifest['histories'].items()},
        history_positions_by_floor={f:h['positions'] for f,h in manifest['histories'].items()},
        eligible_floors=manifest['eligible_floors'],floor_metadata=manifest['floor_metadata'],
        segment_seconds=time.monotonic()-start,total_fit_seconds=sum(r['seconds'] for r in records),
        max_projected_gradient=max(r['solver']['projected_gradient'] for r in records),
        official_query_opened=False,GT_opened=False,source_model_updates=0))
    return 0


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/ipin2025_history_assets_20261110')
    parser.add_argument('--seconds',type=float,default=780.)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 780
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    raise SystemExit(run(args))
