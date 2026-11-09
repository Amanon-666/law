"""Full same-building historical GP assets for the frozen B1/B2 protocol.

No model training, target adaptation or query scoring. Each floor is fitted
once, and can only be loaded as history for a different target floor.
"""
import argparse
import json
import socket
import time
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch

import history_fit
from building_information_protocol import build_manifest, read_metadata, verify_manifest
from paired_assets import read_allowed
from recomposition_trial import ROOT, CSV, digest, write, now, subset

PRIOR = ROOT/'results/paired_assets_20261106'
OUT = ROOT/'results/building_information_assets_20261109'
CODE = ('building_information_assets.py', 'building_information_protocol.py',
        'history_fit.py', 'observations.py', 'adaptation.py', 'fields.py',
        'uji.py', 'paired_assets.py', 'recomposition_trial.py')


def run(resume=False, seconds=600):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    parameters = json.loads((PRIOR/'parameters.json').read_text())
    old = json.loads((PRIOR/'freeze.json').read_text())
    done = json.loads((PRIOR/'complete.json').read_text())
    checked = json.loads((PRIOR/'verification.json').read_text())
    assert done['complete'] and checked['passed']
    assert digest(PRIOR/'freeze.json') == done['freeze_sha256']
    assert digest(PRIOR/'parameters.json') == done['parameters_sha256']
    assert old['source_floors'] == ['B0F0', 'B0F1', 'B0F2']
    assert not set(old['allowed_rows']) & set(old['protected_rows'])
    inputs = {str(p): digest(p) for p in (CSV, PRIOR/'parameters.json',
              PRIOR/'freeze.json', PRIOR/'complete.json', PRIOR/'verification.json')}
    assert inputs[str(CSV)] == old['inputs'][str(CSV)]
    code = {name: digest(ROOT/name) for name in CODE}
    if resume:
        frozen = json.loads((OUT/'freeze.json').read_text())
        assert frozen['inputs'] == inputs and frozen['code'] == code
        assert not (OUT/'complete.json').exists()
        manifest = json.loads((OUT/'manifest.json').read_text())
        assert digest(OUT/'manifest.json') == frozen['manifest_sha256']
        verify_manifest(read_metadata(CSV), manifest)
    else:
        OUT.mkdir(exist_ok=False)
        manifest = build_manifest(CSV)
        write(OUT/'manifest.json', manifest)
        frozen = dict(created=now(), inputs=inputs, code=code,
            manifest_sha256=digest(OUT/'manifest.json'),
            source_calibration=dict(floors=old['source_floors'],
                allowed_rows=old['allowed_rows'], protected_rows=old['protected_rows'],
                updates=parameters['updates'], global_optimum_claim=False),
            forward=dict(object='Fit the existing full-Gaussian history posterior for every legal H map',
                benefit='All methods can use the complete same-building H asset rather than two selected floors',
                condition='Observed AP identities and common horizontal coordinates across floors',
                falsifier='Subsequent matched localization may show no history or adaptation gain'),
            reverse_audit=dict(
                source='Reuse fixed B0-only base/observation calibration; no B1/B2 source refit',
                H='Each map uses all trainingData rows of that floor, never its target support/query labels',
                loading='A target event may load only event.histories, excluding its own floor',
                permissions='Rows used as Q for one target may be H for another target; all floors are parsed globally for this reason',
                S_Q='No target adaptation, query prediction, scoring or score-based selection in this preparation',
                invariance='Per-map mean XY origin changes stationary-kernel coordinates, not predictions',
                missing='No-detection AP has no fitted field; a missing expected asset is an error'),
            inner_maxiter=640, inner_max_projected_gradient=1e-3,
            resources=dict(cpu=2, gpu=0, seconds_per_invocation=seconds,
                           stop='After each complete AP; explicit resume required', downloads=0),
            limits=['Preparation is not evidence of localization performance or innovation',
                    'B1/B2 development exposure and unknown public pretraining remain',
                    'History fits reuse the existing solver and finite B0 calibration'])
        write(OUT/'freeze.json', frozen)
        (OUT/'code').mkdir()
        for name in CODE:
            (OUT/'code'/name).write_bytes((ROOT/name).read_bytes())
    needed = {i for ids in manifest['history_rows'].values() for i in ids}
    raw = read_allowed(CSV, needed)
    assert set(raw) == set(manifest['floors'])
    records = []
    index = OUT/'fields.jsonl'
    if index.exists():
        records = [json.loads(line) for line in index.read_text().splitlines()]
        for record in records:
            assert digest(ROOT/record['path']) == record['sha256']
    existing = {(r['floor'], r['ap']): r for r in records}
    assert len(existing) == len(records)
    base, obs = parameters['base'], parameters['obs']
    original_minimize = history_fit.minimize
    current = {}
    latest = {}

    def monitored(*args, **kwargs):
        result = original_minimize(*args, **kwargs)
        gradient = np.asarray(result.jac).copy()
        for j, (low, high) in enumerate(kwargs['bounds']):
            if low is not None and result.x[j] <= low+1e-10 and gradient[j] > 0:
                gradient[j] = 0
            if high is not None and result.x[j] >= high-1e-10 and gradient[j] < 0:
                gradient[j] = 0
        latest.clear()
        latest.update(success=bool(result.success), iterations=int(result.nit),
                      projected_gradient=float(np.abs(gradient).max()), message=str(result.message))
        if not (np.isfinite(result.fun) and np.isfinite(result.x).all() and np.isfinite(gradient).all()
                and latest['projected_gradient'] <= frozen['inner_max_projected_gradient']):
            raise RuntimeError(f'Unfinished historical fit: {current}, {latest}')
        return result

    expected = set()
    for floor in manifest['floors']:
        origin = raw[floor]['xy'].mean(0)
        scans = subset(raw[floor], manifest['history_rows'][floor], origin)
        aps = [int(ap) for ap in np.flatnonzero((scans['stats']['k'] > 0).any(0))]
        for ap in aps:
            key = floor, ap
            expected.add(key)
            if key in existing:
                continue
            if time.monotonic()-started >= seconds:
                write(OUT/'progress.json', dict(complete=False, created=now(),
                    invocation_seconds=time.monotonic()-started, saved_fields=len(records),
                    next_field=dict(floor=floor, ap=ap), reason='Explicit time budget reached'))
                print((OUT/'progress.json').read_text(), flush=True)
                return 2
            current.update(floor=floor, ap=ap)
            with patch.object(history_fit, 'minimize', monitored):
                field = history_fit.single_history(dict(scans=scans), ap, base, obs, steps=640)
            path = OUT/'histories'/floor/f'ap{ap}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, z=field.z.numpy(), mean=field.m.numpy(),
                                cov=field.c.numpy(), origin=origin)
            record = dict(floor=floor, ap=ap, path=str(path.relative_to(ROOT)),
                          sha256=digest(path), solver=dict(latest))
            with index.open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            records.append(record)
            existing[key] = record
        print(json.dumps(dict(stage='history', floor=floor, fields=len(aps),
                              saved_total=len(records), seconds=time.monotonic()-started)), flush=True)
    assert set(existing) == expected
    assert all(digest(ROOT/name) == sha for name, sha in code.items())
    write(OUT/'histories.json', dict(parameters_path=str(PRIOR/'parameters.json'),
        parameters_sha256=digest(PRIOR/'parameters.json'), fields=records,
        history_rows=manifest['history_rows'], manifest_sha256=digest(OUT/'manifest.json')))
    write(OUT/'complete.json', dict(complete=True, created=now(),
        invocation_seconds=time.monotonic()-started, history_fields=len(records),
        history_fields_by_floor=dict(Counter(r['floor'] for r in records)),
        history_rows=len(needed), events=len(manifest['events']),
        freeze_sha256=digest(OUT/'freeze.json'), manifest_sha256=digest(OUT/'manifest.json'),
        histories_sha256=digest(OUT/'histories.json'), parameters_sha256=digest(PRIOR/'parameters.json'),
        max_VI_projected_gradient=max(r['solver']['projected_gradient'] for r in records),
        field_files_bytes=sum((ROOT/r['path']).stat().st_size for r in records),
        target_scores=0, training_updates=0, calibration_updates=0, downloads=0))
    print((OUT/'complete.json').read_text(), flush=True)
    return 0


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seconds', type=int, default=600)
    args = parser.parse_args()
    raise SystemExit(run(args.resume, args.seconds))
