"""Identifiability check on the same historical smoothing operator.

Tune convex mixing strength and historical spatial range together on internal
source tasks. The 40U target update and fixed Tobit matcher are unchanged.
Test the saved model on official queries and pre-enrolled held-phone tasks.
"""
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from external_baselines import tpm_adapter as tpm
from .field_shape import basis, conditioned
from .forward import MATCH
from .protocol import BUDGETS, HASHES, SEEDS, read_floors, split_rows, history_floors, position_weights

STRENGTHS = (0., .5, 1.)
LENGTHS = (5., 10., 20., 40.)
GRID = tuple(product(STRENGTHS, LENGTHS))
FIXED = {'identity': (0.,10.), 'half10': (.5,10.), 'full10': (1.,10.), 'full20': (1.,20.)}


def local_rows(target, ids):
    ix = np.searchsorted(target['ids'], ids)
    if np.any(target['ids'][ix] != ids):
        raise ValueError('Device manifest row belongs to another target floor')
    return ix


def data_for_floor(train, official, device, b, f):
    target = train[b,f]
    hist = history_floors(train,b,f)
    if device is not None:
        frame, phone_by_row = device
        design = frame[b,f]
        dev = design['phone']
        hist = [{key:col[phone_by_row[h['ids']]!=dev] for key,col in h.items()} for h in hist]
        if [len(h['ids']) for h in hist] != design['retained_history_scans']:
            raise ValueError('Device-restricted H differs from frozen design')
        hist = [h for h in hist if len(h['ids'])]
        qrows = local_rows(target, np.asarray(design['query_ids']))
        q = {key:target[key][qrows] for key in ('xy','rss')}
        sr = local_rows(target, np.asarray(design['support_orders']))
        if (phone_by_row[target['ids'][sr]]==dev).any() or not (phone_by_row[target['ids'][qrows]]==dev).all():
            raise ValueError('Invalid device partition')
        return hist, [(draw,s,q) for draw,s in zip(SEEDS,sr)]
    episodes = []
    for draw in SEEDS:
        sr,qr = split_rows(target,b,f,draw,source=official is None)
        q = {key:target[key][qr] for key in ('xy','rss')} if official is None else official[b,f]
        episodes.append((draw,sr,q))
    return hist, episodes


def measure(train, *, frozen=None, official=None, device=None):
    rows = []
    for (b,f), target in train.items():
        if device is not None and device[0][b,f].get('unavailable'):
            continue
        hist, episodes = data_for_floor(train, official, device, b, f)
        full,xy = tpm.history_map(hist,None,None)
        hmask = np.logical_or.reduce([(h['rss']!=100).any(0) for h in hist])
        arms = {str(pair):pair for pair in GRID} if frozen is None else frozen[str(b)]
        lengths = sorted({ell for g,ell in arms.values()})
        maps = {ell:basis(full,xy,ell) for ell in lengths}
        for draw,srows,q in episodes:
            weights = position_weights(q['xy'])
            for k in BUDGETS:
                sr = srows[:k]
                mask = hmask | (target['rss'][sr]!=100).any(0)
                sy,sxy = tpm.position_map(tpm.dbm(target['rss'][sr][:,mask]), target['xy'][sr])
                qrss = q['rss'][:,mask]
                coords = np.vstack([xy,sxy]) - xy[0]
                parts = {ell:conditioned(full[:,mask],xy,weak[:,mask],strong[:,mask],sxy,sy)
                         for ell,(weak,strong) in maps.items()}
                cache = {}
                for method,(strength,ell) in arms.items():
                    key = (strength,ell if strength else None)
                    if key not in cache:
                        base,w,t = parts[ell]
                        field = np.clip(base-strength*w-strength*t, -110.,0.)
                        p = tpm.tobit_weights(qrss,np.vstack([field,sy]),*MATCH)
                        pred = p @ coords + xy[0]
                        mde = weights @ np.linalg.norm(pred-q['xy'],axis=1)
                        if not np.isfinite(mde):
                            raise ValueError((b,f,draw,k,method))
                        cache[key] = mde
                    rows.append((b,f,draw,k,method,strength,ell,cache[key]))
        print(f'completed B{b} F{f}', flush=True)
    return pd.DataFrame(rows, columns=['building','floor','draw','K','method','strength','ell','mde'])


def main(data, out):
    train = read_floors(data, 'trainingData.csv')
    out.mkdir(parents=True,exist_ok=True)
    source = measure(train)
    floor = source.groupby(['building','floor','K','strength','ell'],as_index=False).mde.mean()
    floor.to_csv(out/'smooth_scale_source.csv', index=False)
    mean = floor.groupby(['building','strength','ell']).mde.mean()
    models = {}
    risks = {}
    for b in sorted({b for b,_ in train}):
        risk = mean.drop(index=b,level='building').groupby(['strength','ell']).mean()
        models[b] = dict(**FIXED, selected=min(GRID,key=lambda pair:risk.loc[pair]))
        risks[b] = {str(pair):float(risk.loc[pair]) for pair in GRID}
    phone_path = out / 'device_stress_design.json'
    record = dict(choices=models, source_risks=risks, train_sha256=HASHES['trainingData.csv'],
                  fields='T(m)=(1-g)m+g P_ell m, rows of P_ell normalized Gaussian over unique H points including self',
                  source_objective='equal building, floor, draw and K on internal source Q; held entire evaluation building out',
                  grid=GRID, propagation_length=40., ridge=1., fixed_tobit=MATCH,
                  scope='Development follow-up to boundary solution; never select settings from official or held-phone errors.',
                  phone_manifest_sha256=hashlib.sha256(phone_path.read_bytes()).hexdigest())
    fp = out/'smooth_scale_selection.json'
    fp.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    models = json.loads(fp.read_text())['choices']
    official = read_floors(data, 'validationData.csv')
    measure(train, frozen=models, official=official).to_csv(out/'smooth_scale_official.csv', index=False)
    design = json.loads(phone_path.read_text())
    if design['train_sha256'] != HASHES['trainingData.csv']:
        raise ValueError('Wrong UJI file for the device design')
    jobs = {(e['building'],e['floor']):e for e in design['events']}
    phone = pd.read_csv(data/'trainingData.csv',usecols=['PHONEID']).PHONEID.to_numpy()
    measure(train, frozen=models, device=(jobs,phone)).to_csv(out/'smooth_scale_phone.csv', index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    main(args.data,args.out)
