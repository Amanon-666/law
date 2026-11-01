"""Stress-test frozen UJI rivals on query phones absent from deployment H and S.

python -m research.uji_core.device_stress DATA FROZEN_RESULTS OUTPUT
Select phones, query rows and support draws using coordinates/phone IDs only.
This internal protocol reuses the UCI training file, not the official queries.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from external_baselines import tpm_adapter as tpm
from residual_probe.episode_rows import nested_rows
from .forward import FIXED_RADIUS, MATCH, RIDGE
from .forward_match import empirical_weights
from .inverse import predict
from .protocol import BUDGETS, HASHES, SEEDS, history_floors, read_floors
from .wknn_reference import pool_floor


def subset(a, rows):
    return {k:v[rows] for k,v in a.items()}


def enrollment(b, f, target, phone_by_id, histories):
    phone = phone_by_id[target['ids']]
    xy = target['xy']
    grouped = {}
    for i,(dev, point) in enumerate(zip(phone,xy)):
        grouped.setdefault(int(dev), {}).setdefault(tuple(point), []).append(i)
    candidates = sorted(grouped, key=lambda dev:(-len(grouped[dev]),dev))
    skipped = []
    for dev in candidates:
        points = sorted(grouped[dev])
        key = f'device-stress-v1/{b}/{f}/{dev}'
        if len(points) < 12:
            skipped.append(dict(phone=dev, reason='fewer than 12 query positions'))
            continue
        qxy = random.Random(key+'/q-xy').sample(points,12)
        qset = set(qxy)
        rng = random.Random(key+'/q-rss')
        q = [rng.choice(grouped[dev][p]) for p in qxy]
        eligible = [i for i in range(len(xy)) if phone[i]!=dev and tuple(xy[i]) not in qset]
        ns = len({tuple(xy[i]) for i in eligible})
        counts = [int(np.count_nonzero(phone_by_id[h['ids']]!=dev)) for h in histories]
        if ns < max(BUDGETS) or sum(counts)==0:
            skipped.append(dict(phone=dev, reason='too few eligible S positions or no H',
                                remaining_support_positions=ns, history_scans=sum(counts)))
            continue
        orders = [nested_rows(xy,eligible,BUDGETS,0,key+f'/support/{seed}')['support_rows'][max(BUDGETS)]
                  for seed in SEEDS]
        if any(len(set(map(tuple,xy[s]))) != max(BUDGETS) or
               bool(set(map(tuple,xy[s])) & qset) or (phone[s]==dev).any() for s in orders):
            raise ValueError('Forbidden support row or repeated support coordinate')
        return dict(phone=dev,query_offsets=q,support_offsets=orders,
                    total_phone_positions=len(points),remaining_support_positions=ns,
                    retained_history_scans=counts,skipped=skipped)
    return dict(skipped=skipped, unavailable=True)


def main(data, frozen, output):
    output.mkdir(exist_ok=True,parents=True)
    path = data/'trainingData.csv'
    train = read_floors(data,'trainingData.csv')
    phone = pd.read_csv(path,usecols=['PHONEID'])['PHONEID'].to_numpy()
    paths = {k:frozen/file for k,file in [('wknn','wknn_parameters.json'),('match','forward_match_selection.json')]}
    config = {key:json.loads(file.read_text()) for key,file in paths.items()}
    jobs,records = {},[]
    for (b,f),target in train.items():
        histories = history_floors(train,b,f)
        design = enrollment(b,f,target,phone,histories)
        jobs[b,f] = design
        record = {key:v for key,v in design.items() if key not in ('query_offsets','support_offsets')}
        if not design.get('unavailable'):
            record['query_ids'] = target['ids'][design['query_offsets']].tolist()
            record['support_orders'] = [target['ids'][s].tolist() for s in design['support_offsets']]
        records.append(dict(building=b,floor=f,**record))
    manifest = dict(train_sha256=HASHES['trainingData.csv'],index='zero-based CSV data rows',
                    models_sha256={key:hashlib.sha256(file.read_bytes()).hexdigest() for key,file in paths.items()},
                    split='Phone with most distinct target XY first (phone ID ascending for ties). Each candidate gets '
                    'one deterministic 12-position, 1-scan query draw. Accept when other-device S disjoint from all '
                    'query XY has 20 distinct positions and some H remains. All other-floor H removes the query phone. '
                    'Ten nested S draws. No official rows, target scoring or RSS select split.',
                    events=records)
    (output/'device_stress_design.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results=[]
    for (b,f),design in jobs.items():
        if design.get('unavailable'):
            continue
        dev = design['phone']
        target=train[b,f]
        q=subset(target,design['query_offsets'])
        histories=[subset(h,phone[h['ids']]!=dev) for h in history_floors(train,b,f)]
        histories=[h for h in histories if len(h['ids'])]
        if any((phone[h['ids']]==dev).any() for h in histories) or not (phone[q['ids']]==dev).all():
            raise AssertionError('Phone separation failed')
        raw,hxy=tpm.history_map(histories,None,None)
        hmask=np.logical_or.reduce([(h['rss']!=100).any(0) for h in histories])
        choice=config['wknn']['selected'][str(b)]['joint_tuned']
        bank=histories if choice['reference']=='raw' else [pool_floor(h) for h in histories]
        law=config['match']['folds'][str(b)]
        for seed,order in zip(SEEDS,design['support_offsets']):
            for k in BUDGETS:
                s=subset(target,order[:k])
                mask=hmask|(s['rss']!=100).any(0)
                sr=dict(rss=s['rss'][:,mask],xy=s['xy'])
                proto,xy=tpm.calibrate((raw[:,mask],hxy),sr,(FIXED_RADIUS,RIDGE))
                qrss=q['rss'][:,mask]
                weights={'fixed_tobit':tpm.tobit_weights(qrss,proto,*MATCH),
                         'tuned_tobit':tpm.tobit_weights(qrss,proto,*law['tobit']),
                         'empirical':empirical_weights(qrss,proto,law['empirical'])}
                pred={key:w@(xy-xy[0])+xy[0] for key,w in weights.items()}
                pred['strong_wknn']=predict(bank,s,q['rss'],neighbors=choice['neighbors'],scale=1.,transport=0.)
                for method,p in pred.items():
                    errors=np.linalg.norm(p-q['xy'],axis=1)
                    if not np.isfinite(errors).all():
                        raise FloatingPointError((b,f,dev,seed,k,method))
                    results.append(dict(building=b,floor=f,query_phone=dev,seed=seed,K=k,method=method,
                                        mde=float(errors.mean()),p90_positions=float(np.quantile(errors,.9))))
    pd.DataFrame(results).to_csv(output/'device_stress_events.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','frozen','output'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    main(args.data,args.frozen,args.output)
