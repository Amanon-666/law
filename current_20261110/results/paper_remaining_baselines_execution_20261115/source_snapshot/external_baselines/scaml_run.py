"""Frozen development K curve: official ScaML core, RSS-to-coordinate adapter."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch
import gpytorch
import scamlgp
from residual_probe import prepare
from public_corpus import k_curve_episodes as episodes
from external_baselines.scaml_adapter import FitTrace, ScaMLCoordinates, fit_sources, parameter_hash
from full_ap_study import metrics
import run_k_curve as curve

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=Path(scamlgp.__file__).resolve().parents[1]
OUT=ROOT/'results/scaml_k_curve_v1_20261027'
RESTARTS=5
MAXITER=500


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open('x') as f:
        json.dump(data,f,indent=2,allow_nan=False)


def source_identity(meta, histories):
    identity={k:meta[k] for k in ('building','histories','AP_indices','AP_ids','origin')}
    identity['rows']={h:meta['history_metadata'][h]['rows'] for h in meta['history_metadata']}
    arrays=hashlib.sha256()
    for h in histories:
        for field in ('rss','xy'):
            a=np.ascontiguousarray(h[field])
            arrays.update(str((a.dtype.str,a.shape)).encode());arrays.update(a.tobytes())
    identity['array_sha256']=arrays.hexdigest()
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return key,identity


def exact_context():
    stack=ExitStack()
    stack.enter_context(gpytorch.settings.max_cholesky_size(100000))
    stack.enter_context(gpytorch.settings.fast_computations(covar_root_decomposition=False,log_prob=False,solves=False))
    return stack


def freeze():
    curve.verify();episodes.old.verify_freeze(episodes.old.PREPARATION)
    files={Path(__file__),ROOT/'external_baselines/scaml_adapter.py',Path(episodes.__file__),
           Path(episodes.old.__file__),Path(episodes.old.source_task.__code__.co_filename),
           ROOT/'data.py',ROOT/'full_ap_study.py'}
    files.update(REFERENCE.rglob('*.py'))
    OUT.mkdir(exist_ok=False)
    manifest=episodes.load_manifest();write(OUT/'manifest.json',manifest)
    write(OUT/'freeze.json',dict(UTC=datetime.now(timezone.utc).isoformat(),
        code_sha256={str(p):digest(p) for p in sorted(files)},
        inputs_sha256={str(p):digest(p) for p in [episodes.OUT/'manifest.json',episodes.OUT/'freeze.json',episodes.old.PREPARATION/'freeze.json',ROOT/'results/scaml_compatibility_20261027/data_interface.json']},
        manifest_sha256=digest(OUT/'manifest.json'),
        versions=dict(torch=torch.__version__,gpytorch=gpytorch.__version__,python=sys.version),
        restarts=RESTARTS,maxiter=MAXITER,
        protocol='Same 60 H/S/Q/AP settings; fixed RSS encoding and H-only coordinate frame; source and target scalar x/y GPs; no Q transforms, no output clipping, no extra target labels.',
        fitting='Unmodified official model and restart selection. Exact dense solves throughout. Record all attempts and selected parameter hash/status/objective. Finite return is not proof of global optimality.',
        failure='No retries beyond fixed restarts; one terminal result/event. Failed source cache reused as failure, never silently refitted.',
        scope='Repeatedly examined development buildings; no independent validation or learning novelty claim.'))


def verify():
    f=read(OUT/'freeze.json')
    for field in ['code_sha256','inputs_sha256']:
        for path,sha in f[field].items():assert digest(path)==sha,path
    assert digest(OUT/'manifest.json')==f['manifest_sha256']
    assert f['restarts']==RESTARTS and f['maxiter']==MAXITER
    assert f['versions']==dict(torch=torch.__version__,gpytorch=gpytorch.__version__,python=sys.version)
    curve.verify();episodes.old.verify_freeze(episodes.old.PREPARATION)


def source_models(meta, histories):
    key,identity=source_identity(meta,histories)
    folder=OUT/'sources'/key
    if (folder/'result.json').exists():
        record=read(folder/'result.json')
        assert record['identity']==identity
        if not record['success']:raise RuntimeError(f'Previously failed source fit: {key}')
        assert digest(folder/'models.pt')==record['checkpoint_sha256']
        # This is our own just-verified local checkpoint, never an external pickle.
        models=torch.load(folder/'models.pt',map_location='cpu',weights_only=False)
        assert [[parameter_hash(m) for m in axis.values()] for axis in models]==record['parameter_hashes']
        return models,key
    folder.mkdir(parents=True,exist_ok=False)
    trace=FitTrace(maxiter=MAXITER);record=dict(identity=identity,success=False)
    start=time.perf_counter()
    try:
        with trace.observe(),warnings.catch_warnings(record=True) as caught:
            models=fit_sources(histories,seed=int(key[:8],16),restarts=RESTARTS)
        selected=[[trace.selected(m) for m in axis.values()] for axis in models]
        assert all(s['optimizer_success'] for axis in selected for s in axis)
        torch.save(models,folder/'models.pt')
        record.update(success=True,selected=selected,checkpoint_sha256=digest(folder/'models.pt'),
                      parameter_hashes=[[parameter_hash(m) for m in axis.values()] for axis in models])
    except Exception as error:
        record['error']=repr(error)
        raise
    finally:
        record.update(attempts=trace.attempts,seconds=time.perf_counter()-start,
                      warnings=[str(w.message) for w in caught])
        write(folder/'result.json',record)
    return models,key


def event(meta):
    folder=OUT/'events'/meta['id']
    if (folder/'result.json').exists():return
    folder.mkdir(parents=True,exist_ok=False)
    trace=FitTrace(maxiter=MAXITER);start=time.perf_counter()
    record=dict(id=meta['id'],building=meta['building'],K=meta['K'],seed=meta['seed'],
                unit=meta['coordinate_unit'],success=False)
    caught=[]
    try:
        task=episodes.build_task(meta)
        assert task['Q']['scan_ids'].tolist()==meta['query_scan_ids']
        with exact_context():
            sources,key=source_models(meta,task['H'])
            record['source_key']=key
            with trace.observe(),warnings.catch_warnings(record=True) as caught:
                model=ScaMLCoordinates(sources,task['S'],seed=meta['seed'],restarts=RESTARTS)
                selected=[trace.selected(m) for m in model.models]
                assert all(s['optimizer_success'] for s in selected)
                pred,var=model.predict(task['Q']['rss'])
            assert np.isfinite(pred).all() and np.isfinite(var).all() and (var>=0).all()
        # Only the scorer below consumes Query coordinates.
        scores=metrics(pred,task['Q'])
        np.savez_compressed(folder/'predictions.npz',predictions=pred,variance=var,
                            query_scan_ids=task['Q']['scan_ids'],query_xy=task['Q']['xy'])
        record.update(success=True,metrics=scores,selected=selected,query_scan_ids=meta['query_scan_ids'])
    except Exception as error:
        record['error']=repr(error)
    record.update(seconds=time.perf_counter()-start,attempts=trace.attempts,warnings=[str(w.message) for w in caught])
    write(folder/'result.json',record)
    print(json.dumps(dict(event=meta['id'],success=record['success'],seconds=record['seconds'])),flush=True)


def run(building):
    verify()
    events=[e for e in read(OUT/'manifest.json')['events'] if e['building']==building]
    assert len(events)==15
    for e in events:event(e)
    write(OUT/f'{building}_finished.json',dict(building=building,terminal=15,
        success=sum(read(OUT/'events'/e['id']/'result.json')['success'] for e in events)))


if __name__=='__main__':
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);p.add_argument('--building')
    a=p.parse_args();freeze() if a.stage=='freeze' else run(a.building)
