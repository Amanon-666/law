"""Released TabICLv2 on the frozen LAW history bank, without target training.

This is a mature direct-regression comparator, not a new LAW component.
The context retains source-floor identity; no target query fits preprocessing.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import socket
import sys
import time

SERVER = Path('/home/panyushuo/projects/panyushuo')
PACKAGE = SERVER / 'RNP-UJI/.deps_tabicl'
sys.path.insert(0, str(PACKAGE))
import numpy as np
import torch
from tabicl import TabICLRegressor

from full_ap_study import metrics
from run_multi_history import digest, write
from uji_probability_study import DATA, load_raw
from uji_history_bank_study import OUT as BANK, SOURCES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/uji_tabicl_history_bank_v1_20261026'
CHECKPOINT = SERVER / 'RNP-UJI/tabicl_weights/tabicl-regressor-v2-20260212.ckpt'
METHODS = ['local_tabicl', 'bank_tabicl']


def log(**row):
    print(json.dumps(row, allow_nan=False), flush=True)


def model_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def loader():
    loaded = TabICLRegressor(model_path=CHECKPOINT, allow_auto_download=False, device='cpu')
    loaded._load_model()
    loaded.model_.to('cuda:0').eval()
    return loaded


def estimator(loaded, seed):
    # Same model-reuse pattern as the existing official fine-tuning wrapper.
    model = TabICLRegressor(model_path=CHECKPOINT, allow_auto_download=False,
        device='cuda:0', n_jobs=2, n_estimators=8, batch_size=1, random_state=seed)
    model.model_ = loaded.model_; model.model_config_ = loaded.model_config_
    model.model_path_ = CHECKPOINT; model._load_model = lambda: None
    return model


def features(rss, floor, aps):
    x = np.where(rss[:, aps] == 100, 0., (rss[:, aps].astype(float) + 105.) / 105.)
    return np.column_stack((x, np.full(len(x), float(floor))))


def prepare():
    OUT.mkdir(exist_ok=False)
    code = [Path(__file__), ROOT/'full_ap_study.py', ROOT/'uji_probability_study.py',
            ROOT/'data.py', ROOT/'run_multi_history.py']
    package_files = [PACKAGE/'tabicl'/p for p in ['_sklearn/regressor.py', '_sklearn/preprocessing.py',
        '_model/tabicl.py', '_model/embedding.py', '_model/learning.py', '_model/layers.py']]
    freeze = dict(created_unix=time.time(), hypothesis='A released whole-scan conditional regressor may use the same H+S more effectively than the independent-AP LAW field; this does not test new source learning.',
        methods=METHODS, checkpoint=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT),
        external_pretraining='Official released synthetic-task prior; not equal total pretraining compute to LAW',
        excluded_weights='Existing UJI-finetuned checkpoint is excluded because its training/selection includes current evaluation floors',
        history_bank=SOURCES, target_budget='Exact existing K5 R3 row IDs, six inspected development events',
        input='Same frozen AP columns plus known FLOOR number, available for H/S/query; no query position features',
        floor_semantics='Physical floor index inside the same registered building; not arbitrary source-ID embedding or cross-building alignment',
        local_control='S only; same floor feature is constant and removed by the official context-only filter',
        bank_control='All original 2316 historical scans plus 15 target scans; no duplicated S, source subsampling, target-context reweighting or invented paired scans',
        preprocessing='Existing RSS normalization with 100 mapped to 0; official preprocessing and y scaling fitted on context only',
        estimator='Official default mean prediction, two scalar axes, eight ensemble views, batch_size=1 for memory only',
        target_gradient_steps=0, new_UJI_source_updates=0, extra_unlabelled_target_data=False,
        area='Project predicted coordinates to the exact same H+S rectangle; also save unprojected predictions',
        acquisition='Original June support/internal queries and Sept-Oct official queries retained; different acquisition conditions are not pooled',
        reverse_audit=['H rows and S rows -> context RSS, floor and coordinate labels',
            'Context-only feature filtering/scaling -> frozen model',
            'Each query RSS + known floor -> conditional coordinate predictions',
            'H+S bounds -> projection; query XY -> scoring only'],
        counterexamples=dict(empty_history='S-only comparator', absent_ap='Official context-only constant-column filter',
            wrong_history='No guarantee against negative transfer; source floor is observable but the generic model has no explicit discrepancy GP',
            sparse_target='2316 source records can dominate 15 target records; this natural imbalance is retained, not tuned against query scores',
            translation='Center labels in float64 before float32 model preprocessing',
            unseen_building='Not claimed; context uses existing shared AP identity and registered coordinate frame',
            query_batch='Check isolated vs batched source predictions before target evaluation'),
        versions=dict(tabicl=importlib.metadata.version('tabicl'), torch=torch.__version__, numpy=np.__version__),
        code_sha256={str(p):digest(p) for p in code+package_files},
        inputs={str(p):digest(p) for p in [BANK/'manifest.json', BANK/'freeze.json', DATA/'trainingData.csv', DATA/'validationData.csv']})
    write(OUT/'prefreeze.json', freeze)
    # One B0-only API check: does batching import other query scans into prediction?
    f = load_raw('trainingData.csv', {'B0F0'})['B0F0']
    groups = np.unique(f['group']); s = np.array([np.flatnonzero(f['group']==g)[0] for g in groups[:12]])
    q = np.array([np.flatnonzero(f['group']==g)[0] for g in groups[12:15]])
    aps = np.flatnonzero((f['rss'][s]!=100).any(0))
    x, xq = features(f['rss'][s], 0, aps), features(f['rss'][q], 0, aps)
    loaded = loader(); before = model_hash(loaded.model_)
    model = estimator(loaded, 261026); y = f['xy'][s,0]; origin = y.mean()
    model.fit(x, y-origin)
    batched = model.predict(xq); single = model.predict(xq[:1])
    np.testing.assert_allclose(batched[:1], single, rtol=1e-4, atol=1e-3)
    assert before == model_hash(loaded.model_)
    freeze['source_check'] = dict(floor='B0F0', context_positions=len(s), query_positions=len(q),
        batch_prediction_max_abs_difference=float(np.abs(batched[:1]-single).max()), frozen_weights_unchanged=True,
        no_target_performance_evaluation=True)
    (OUT/'code').mkdir()
    for p in code: shutil.copy2(p, OUT/'code'/p.name)
    shutil.copy2(BANK/'manifest.json', OUT/'manifest.json')
    freeze['frozen_unix'] = time.time(); write(OUT/'freeze.json', freeze)
    log(stage='prepared', checks=freeze['source_check'])


def event(event_id):
    freeze = json.loads((OUT/'freeze.json').read_text())
    for group in ['code_sha256', 'inputs']:
        for path, expected in freeze[group].items():
            assert digest(Path(path)) == expected, path
    assert digest(CHECKPOINT) == freeze['checkpoint_sha256']
    dest = OUT/event_id; dest.mkdir(exist_ok=True)
    if (dest/'completed.json').exists(): return
    manifest = json.loads((OUT/'manifest.json').read_text())
    meta = next(e for e in manifest['events'] if e['id']==event_id)
    floors = load_raw('trainingData.csv', {*SOURCES, meta['floor']})
    target = floors[meta['floor']]; official = load_raw('validationData.csv', {meta['floor']})[meta['floor']]
    ids = {int(v):i for i,v in enumerate(target['row_ids'])}; qids = {int(v):i for i,v in enumerate(official['row_ids'])}
    sr = np.array([ids[i] for i in meta['support_rows']]); qi = np.array([ids[i] for i in meta['internal_rows']])
    qo = np.array([qids[i] for i in meta['official_rows']]); aps = meta['ap_columns']; number = int(meta['floor'].split('F')[1])
    sx = features(target['rss'][sr], number, aps); sy = target['xy'][sr]
    qx = features(np.concatenate((target['rss'][qi], official['rss'][qo])), number, aps)
    truth = np.concatenate((target['xy'][qi], official['xy'][qo]))
    # Exact raw source rows from the shared manifest, not a freshly sampled bank.
    hx, hy = [], []
    for name in SOURCES:
        f = floors[name]; assert f['row_ids'].tolist() == manifest['history'][name]['row_ids']
        hx.append(features(f['rss'], int(name.split('F')[1]), aps)); hy.append(f['xy'])
    loaded = loader(); before = model_hash(loaded.model_); rows = []; saved = {}
    bounds = np.asarray(meta['grid_bounds'])
    for method in METHODS:
        x,y = (sx,sy) if method=='local_tabicl' else (np.concatenate(hx+[sx]),np.concatenate(hy+[sy]))
        origin = y.mean(0); started = time.perf_counter(); predictions = []
        for axis in range(2):
            model = estimator(loaded, 261026+meta['seed'])
            model.fit(x, y[:,axis]-origin[axis]); predictions.append(model.predict(qx))
        raw = np.column_stack(predictions)+origin
        assert np.isfinite(raw).all()
        pred = np.clip(raw, bounds[0], bounds[1]); saved[method] = pred; saved[method+'_unprojected'] = raw
        for split, sl in [('internal',slice(0,len(qi))), ('official',slice(len(qi),None))]:
            xy = truth[sl]; _,g,n = np.unique(xy,axis=0,return_inverse=True,return_counts=True)
            weights = 1/(len(n)*n[g.ravel()])
            rows.append(dict(method=method, split=split, **metrics(pred[sl],dict(xy=xy,scan_weights=weights)),
                context_scans=len(x), seconds=time.perf_counter()-started,
                projected_scans=int(np.any(pred[sl]!=raw[sl],axis=1).sum())))
        log(stage='method',event=event_id,method=method,results=rows[-2:])
    assert before == model_hash(loaded.model_)
    np.savez_compressed(dest/'predictions.npz',**saved,query_xy=truth,internal_count=len(qi),support_xy=sy,
        support_rows=meta['support_rows'],internal_rows=meta['internal_rows'],official_rows=meta['official_rows'])
    write(dest/'completed.json',dict(completed=True,event=event_id,results=rows,weights_unchanged=True,
        predictions_sha256=digest(dest/'predictions.npz')))


def summarize():
    rows = []
    for meta in json.loads((OUT/'manifest.json').read_text())['events']:
        result = json.loads((OUT/meta['id']/'completed.json').read_text())
        rows.extend(dict(event=meta['id'],floor=meta['floor'],**r) for r in result['results'])
    summary = [dict(method=m,split=s,events=6,mde=float(np.mean([r['mde'] for r in rows if r['method']==m and r['split']==s])))
        for m in METHODS for s in ['internal','official']]
    write(OUT/'summary.json',dict(completed=True,summary=summary,rows=rows)); log(stage='summary',summary=summary)


if __name__=='__main__':
    if socket.gethostname()!='ray-web': raise RuntimeError('Numerical work is server-only')
    torch.set_num_threads(2)
    p = argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','event','summarize']); p.add_argument('--event')
    args = p.parse_args()
    if args.stage=='prepare': prepare()
    elif args.stage=='event': event(args.event)
    else: summarize()
