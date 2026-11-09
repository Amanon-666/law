"""One B0 source engineering check, not a localization performance trial.

Verify full-H preprocessing without a costly full-H backward, then use a small
declared source context for train/inference parity and one shared update. No
development/final floor RSS is parsed and no localization score is computed.
"""
import json
import resource
import socket
import time

import numpy as np
import torch

from acquisition_tasks import load_acquisition_rows, episode_scans
from tabicl_history import CHECKPOINT, PACKAGE, features, state_hash
from tabicl_history_learning import EpisodicHistoryRegressor
from recomposition_trial import ROOT, PLAN, CSV, digest, write, now

OUT = ROOT/'results/tabicl_history_learning_check_20261105'
SOURCE = ROOT/'results/acquisition_retention_trial_20261105/freeze.json'
SEED = 1150540


def first_positions(rows, ids, count):
    positions = sorted({rows[i]['xy'] for i in ids})[:count]
    return [i for xy in positions for i in [j for j in ids if rows[j]['xy']==xy][:3]]


def bounds_and_aps(histories, support):
    scans = [*histories.values(), support]
    xy = np.concatenate([s['xy'] for s in scans])
    aps = np.flatnonzero(np.logical_or.reduce([(s['rss']!=100).any(0) for s in scans]))
    return np.array([xy.min(0), xy.max(0)]), aps


def run():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    torch.manual_seed(SEED)
    plan = json.loads(PLAN.read_text())
    source = json.loads(SOURCE.read_text())
    expected = next(v for k,v in source['inputs'].items() if k.endswith('/trainingData.csv'))
    assert digest(CSV) == expected
    event = dict(source['events'][0], query_rows=source['events'][0]['fit_rows'])
    assert event['floor']=='B0F1' and event['role']=='train'
    rows = load_acquisition_rows(CSV, {'B0F0','B0F1','B0F2'})
    assert not set(event['support_rows']).intersection(event['query_rows'])
    assert not {rows[i]['xy'] for i in event['support_rows']}.intersection(
        rows[i]['xy'] for i in event['query_rows'])
    origin = np.asarray(plan['origin'])
    small = dict(event, history_rows={f:first_positions(rows, ids, 4)
                                    for f,ids in event['history_rows'].items()},
                 query_rows=event['query_rows'][:9])
    code = ['tabicl_history.py','tabicl_history_learning.py','check_tabicl_history_learning.py',
            'acquisition_tasks.py','uji.py','recomposition_trial.py']
    write(OUT/'freeze.json', dict(created=now(), seed=SEED,
        code={n:digest(ROOT/n) for n in code},
        inputs={str(p):digest(p) for p in [CSV,PLAN,SOURCE,CHECKPOINT]},
        package={str(p.relative_to(PACKAGE)):digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))},
        full_event=event, engineering_event=small,
        objective='Mean source coordinate pinball across axes/views and equal-weight query positions',
        limitation='Small H context is solely an engineering check, not full-H performance or source generalization',
        resources=dict(cpu=2,gpu=0,wall_seconds=600,updates=1,downloads=0)))
    for n in code: (OUT/n).write_bytes((ROOT/n).read_bytes())
    model = EpisodicHistoryRegressor()
    original_hash = state_hash(model.loaded.model_)
    packed = episode_scans(rows,event,origin)
    h,s = packed['histories'],packed['support']
    bounds,aps = bounds_and_aps(h,s)
    model.fit(h,s,event['floor'],aps,bounds,SEED)
    full_counts = [m.n_samples_in_ for m in model.estimators]
    assert full_counts == [sum(len(v['rss']) for v in h.values())+len(s['rss'])]*2
    full_views = model.views(packed['query']['rss'][:1],0)
    assert full_views[0].shape[1] == full_counts[0]+1
    assert full_views[1].shape == (8,full_counts[0])
    full_info = dict(context_rows=full_counts[0], APs=len(aps), views=8,
                     effective_columns=full_views[0].shape[-1])
    del full_views, packed
    packed = episode_scans(rows,small,origin)
    h,s,q = packed['histories'],packed['support'],packed['query']
    bounds,aps = bounds_and_aps(h,s)
    model.fit(h,s,event['floor'],aps,bounds,SEED)
    assert state_hash(model.loaded.model_) == original_hash
    x,y = model.views(q['rss'],0)
    assert x.shape[:2] == (8, len(y[0])+len(q['rss']))
    assert len(y[0]) == sum(len(v['rss']) for v in h.values())+len(s['rss'])
    assert all(m.model_ is model.loaded.model_ for m in model.estimators)
    # Changing a query must not refit the labeled context or its scaling.
    changed_q = q['rss'].copy(); changed_q[0,:] = 100
    changed_x, changed_y = model.views(changed_q,0)
    assert np.array_equal(y,changed_y)
    assert np.array_equal(x[:,:y.shape[1]], changed_x[:,:y.shape[1]])
    # Check the different official train/eval forward implementations actually agree.
    raw_train = []
    model.loaded.model_.train()
    with torch.no_grad():
        for v in range(8):
            raw_train.append(model.loaded.model_(torch.tensor(x[v:v+1],dtype=torch.float32),
                torch.tensor(y[v:v+1],dtype=torch.float32), embed_with_test=False).numpy())
    estimator = model.estimators[0]
    raw_train = np.concatenate(raw_train)
    normalized_mean = raw_train.mean(axis=(0,2))
    manual = estimator.y_scaler_.inverse_transform(normalized_mean[:,None]).ravel()+model.origin[0]
    model.loaded.model_.eval()
    official = estimator.predict(features(q['rss'],model.floor,aps))+model.origin[0]
    parity = float(np.max(np.abs(manual-official)))
    assert np.allclose(manual,official,rtol=0,atol=2e-3),parity
    # A fixed translation must preserve the centered training tables.
    shift = np.array([10000.,-20000.])
    shifted_h = {f:dict(v,xy=v['xy']+shift) for f,v in h.items()}
    model.fit(shifted_h,dict(s,xy=s['xy']+shift),event['floor'],aps,bounds+shift,SEED)
    shifted_x,shifted_y = model.views(q['rss'],0)
    assert np.array_equal(x,shifted_x) and np.allclose(y,shifted_y,rtol=0,atol=1e-6)
    model.fit(h,s,event['floor'],aps,bounds,SEED)
    parameters = [p for p in model.loaded.model_.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=1e-5,weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    loss = model.backward(q['rss'],q['xy'])
    gradients = {}
    for name in ('col_embedder','row_interactor','icl_predictor'):
        grads = [p.grad for p in getattr(model.loaded.model_,name).parameters() if p.requires_grad]
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
        gradients[name] = float(torch.sqrt(sum(g.square().sum() for g in grads)))
        assert gradients[name]>0
    norm = float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
    optimizer.step()
    updated_hash = state_hash(model.loaded.model_)
    assert updated_hash != original_hash
    assert int(optimizer.state[parameters[0]]['step']) == 1
    after = model.predict(q['rss'])['raw']
    # Re-fit another context on the SAME network without resetting its weights.
    model.fit(h,dict(s,rss=s['rss'][:3],xy=s['xy'][:3]),event['floor'],aps,bounds,SEED)
    assert state_hash(model.loaded.model_) == updated_hash
    checkpoint = OUT/'engineering_only.ckpt'
    model.save(checkpoint,source_updates=1,provenance=dict(engineering_only=True,freeze_sha256=digest(OUT/'freeze.json')))
    checkpoint_hash = digest(checkpoint)
    restored = EpisodicHistoryRegressor(checkpoint)
    assert state_hash(restored.loaded.model_) == updated_hash
    restored.fit(h,s,event['floor'],aps,bounds,SEED)
    reload_gap = float(np.abs(after-restored.predict(q['rss'])['raw']).max())
    assert reload_gap<1e-6,reload_gap
    checkpoint.unlink()  # This micro-update is not a model to deploy or warm-start.
    result = dict(complete=True,created=now(),seconds=time.monotonic()-started,
        full_history_preprocessing=full_info,
        small_context_rows=y.shape[1],query_rows=len(q['rss']),APs=len(aps),
        train_inference_mean_max_gap=parity,source_pinball_before_step=loss,
        group_gradient_norms=gradients,gradient_norm=norm,updates=1,
        original_state_sha256=original_hash,updated_state_sha256=updated_hash,
        checkpoint_sha256=checkpoint_hash,reload_prediction_max_gap=reload_gap,
        temporary_checkpoint_removed=not checkpoint.exists(),
        peak_resident_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        localization_scores=0,development_or_final_floors_parsed=0)
    write(OUT/'complete.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    torch.set_num_threads(2)
    run()
