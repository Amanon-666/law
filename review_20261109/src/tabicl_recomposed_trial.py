"""Full Tab training on recomposed historical tasks; source-only, no auto-run.

The original full24 remains the strong control. Same 24 updates and 72 H/S
contexts, but 144 acquisition-query events: not equal query exposure or FLOPs.
No spatial operator, new architecture, or target parameter updates are claimed.
"""
import json
import signal
import socket
import time

import numpy as np
import torch

import tabicl_multitask_full_trial as full
from tabicl_recomposed_tasks import training_contexts
from tabicl_history import CHECKPOINT, PACKAGE, state_hash
from tabicl_history_learning import EpisodicHistoryRegressor
from source_episode_catalog import read_metadata
from paired_acquisition_tasks import verify_quartet, make_quartet, build_pairs
from acquisition_tasks import load_acquisition_rows, episode_scans
from check_tabicl_history_learning import bounds_and_aps
from uji import group_scans
from recomposition_trial import ROOT, CSV, digest, write, now

OUT = ROOT/'results/tabicl_recomposed_trial_20261108'
CONTROL = full.OUT
WALL_SECONDS = 1800


def pack_context(rows, record):
    first, second = record['events']
    a = episode_scans(rows, first, np.zeros(2))
    b = episode_scans(rows, second, np.zeros(2))
    assert first['support_rows']==second['support_rows']
    assert np.array_equal(a['query']['xy'], b['query']['xy'])
    # Each query position has 3 rows from each acquisition. The existing
    # backward's position weights give both acquisition conditions half mass.
    q = group_scans(np.concatenate((a['query']['rss'],b['query']['rss'])),
                    np.concatenate((a['query']['xy'],b['query']['xy'])))
    assert len(q['positions'])==12 and np.all(q['stats']['n']==6)
    bounds, aps = bounds_and_aps(a['histories'],a['support'])
    return dict(a,query=q,event=first,bounds=bounds,aps=aps)


def run():
    started = time.monotonic()
    torch.manual_seed(full.SEED); torch.cuda.manual_seed_all(full.SEED)
    catalog_path = full.CATALOG/'catalog.json'
    catalog = json.loads(catalog_path.read_text())
    control = json.loads((CONTROL/'complete.json').read_text())
    previous = json.loads((CONTROL/'freeze.json').read_text())
    reference = json.loads((CONTROL/'step_24.json').read_text())
    assert control['complete'] and control['source_updates']==24
    for path in (catalog_path,CSV,CHECKPOINT):
        assert digest(path)==previous['inputs'][str(path)]
    for name in ('tabicl_history.py','tabicl_history_learning.py','acquisition_tasks.py',
                 'source_episode_catalog.py','uji.py','check_tabicl_history_learning.py'):
        assert digest(ROOT/name)==previous['code'][name]
    package = {str(p.relative_to(PACKAGE)):digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package==previous['package']
    schedule = training_contexts(catalog)
    _, validation = full.events_for(catalog)
    assert validation==previous['validation']
    metadata = read_metadata(CSV,set(catalog['source_floors']))
    pairs = {p['id']:p for p in build_pairs(catalog)}
    for batch in schedule:
        for record in batch:
            quartet = make_quartet(catalog,pairs[record['pair_id']],record['draw_index'],
                                   k=record['K'],histories=record['histories'],phase='fit')
            verify_quartet(metadata,catalog,quartet,'fit')
    OUT.mkdir(exist_ok=False)
    full.OUT = OUT  # Reuse original fit/score without modifying its saved run.
    freeze = dict(created=now(),schedule=schedule,validation=validation,
        code={p.name:digest(p) for p in ROOT.glob('*.py')},package=package,
        inputs={str(p):digest(p) for p in (catalog_path,CSV,CHECKPOINT,CONTROL/'complete.json',
                                         CONTROL/'freeze.json',CONTROL/'step_24.json')},
        optimizer=dict(name='AdamW',lr=1e-5,weight_decay=.01,steps=24,clip=1.,seed=full.SEED),
        training=dict(contexts=72,acquisition_events=144,views=8,source_floors=catalog['source_floors'],
                      loss='Original two-axis quantile pinball, equal floor/position/acquisition'),
        comparison='Original full24; same authorized corpus and updates, changed history/task distribution and query exposure',
        criterion='Original 27 full-H tasks: >=3% mean MDE gain vs full24; no worse by floor or K',
        limitations=['Source development only; 24 steps not convergence',
                     'Joint recipe attribution, not a single-factor or equal-FLOP comparison',
                     'No proof of spatial operator value, framework novelty, or strong-GP superiority'],
        resources=dict(cpu=2,gpu=1,wall_seconds=WALL_SECONDS,downloads=0),target_opened=False)
    write(OUT/'freeze.json',freeze)
    rows = load_acquisition_rows(CSV,set(catalog['source_floors']))
    model = EpisodicHistoryRegressor(device='cuda:0')
    model.enable_recompute()
    assert state_hash(model.loaded.model_)==control['original_state_sha256']
    parameters = [p for p in model.loaded.model_.parameters() if p.requires_grad]
    assert sum(p.numel() for p in parameters)==28544983
    optimizer = torch.optim.AdamW(parameters,lr=1e-5,weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',init_scale=1024.)
    training = []
    for step,batch in enumerate(schedule,1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for record in batch:
            packed = pack_context(rows,record)
            full.fit(model,packed)
            losses.append(model.backward(packed['query']['rss'],packed['query']['xy'],scaler))
        scaler.unscale_(optimizer)
        for p in parameters:
            p.grad.div_(3)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
        old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
        assert scaler.get_scale()>=old_scale, 'AMP overflow; no automatic retry'
        training.append(dict(step=step,mean_pinball=float(np.mean(losses)),gradient_norm=norm,
                             contexts=3*step,acquisition_events=6*step,seconds=time.monotonic()-started))
        write(OUT/'training.json',training)
        full.progress('training',**training[-1])
    checkpoint = OUT/'final24.ckpt'
    model.save(checkpoint,source_updates=24,provenance=dict(freeze_sha256=digest(OUT/'freeze.json'),
        source_floors=catalog['source_floors'],source_only=True))
    trained_hash = state_hash(model.loaded.model_)
    del model,optimizer,parameters
    torch.cuda.empty_cache()
    restored = EpisodicHistoryRegressor(checkpoint,device='cuda:0')
    assert state_hash(restored.loaded.model_)==trained_hash
    records = []
    for spec,old in zip(validation,reference):
        p = full.pack(rows,catalog,spec)
        assert p['event']['id']==old['event']
        records.append(full.score(restored,p,'final24'))
        write(OUT/'validation.json',records)
    candidate = full.aggregate(records,lambda r:r['mde']['projected'])
    baseline = full.aggregate(reference,lambda r:r['mde']['projected'])
    gains = dict(mean=1-candidate['mean']/baseline['mean'],
                 by_floor={f:1-v/baseline['by_floor'][f] for f,v in candidate['by_floor'].items()},
                 by_K={k:1-v/baseline['by_K'][k] for k,v in candidate['by_K'].items()})
    gate = gains['mean']>=.03 and all(v>=0 for v in gains['by_floor'].values()) and all(v>=0 for v in gains['by_K'].values())
    assert state_hash(restored.loaded.model_)==trained_hash
    assert all(digest(p)==h for p,h in freeze['inputs'].items())
    assert all(digest(ROOT/n)==h for n,h in freeze['code'].items())
    write(OUT/'complete.json',dict(complete=True,created=now(),seconds=time.monotonic()-started,
        candidate=candidate,full24=baseline,relative_gains=gains,gate=bool(gate),
        source_contexts=72,source_acquisition_events=144,target_opened=False,
        checkpoint_sha256=digest(checkpoint),trained_state_sha256=trained_hash))


def deadline(signum,frame):
    raise TimeoutError('Frozen resource budget exhausted')


if __name__=='__main__':
    assert socket.gethostname()=='ray-web' and torch.cuda.is_available()
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.8,0)
    signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,WALL_SECONDS)
    try:
        run()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),error=repr(error),complete=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
