"""Transfer frozen source retention to one B1 development floor, without refitting rho.

B1 has appeared in previous development. This is a new grouped evaluation of
a source-fitted rule, not a blind new-building confirmation or a K-curve.
"""
import hashlib
import json
import socket
import time

import numpy as np
import torch

from acquisition_tasks import load_acquisition_rows,acquisitions,episode_scans,adapt_acquisition
from acquisition_transfer import prepare_acquisition_transfer
from acquisition_retention_trial import evaluate
from conditional_budget_trial import history_reader
from fields import InterceptBankPrior
from fingerprint_posterior import propose_support
from recomposition_multibudget_natural import wknn
from recomposition_trial import ROOT,OLD,PLAN,CSV,digest,write,now

SOURCE=ROOT/'results/acquisition_retention_trial_20261105'
OUT=ROOT/'results/acquisition_retention_transfer_20261105'
SEED=1150537


def make_events(rows,parent):
    """Two broadest R3 groups, chosen by metadata alone, then fixed random S/Q."""
    assert parent['floor']=='B1F1' and parent['role']=='validation'
    groups=acquisitions(rows,parent['floor'])
    positions={g:sorted(x for x,ids in p.items() if len(ids)>=3) for g,p in groups.items()}
    ranked=sorted(groups,key=lambda g:(-len(positions[g]),g))
    first=ranked[0]
    second=next(g for g in ranked[1:] if g[:2]!=first[:2])
    selected=[first,second];rng=np.random.default_rng(SEED)
    support={g:[positions[g][i] for i in rng.permutation(len(positions[g]))[:5]] for g in selected}
    excluded=set(support[first])|set(support[second])
    events=[]
    for j,(sg,qg) in enumerate([(first,second),(second,first)]):
        available=[x for x in positions[qg] if x not in excluded]
        assert len(support[sg])==5 and len(available)>=12
        qxy=[available[i] for i in rng.permutation(len(available))[:12]]
        event=dict(id=f'B1F1_acquisition_{j}',role='validation',floor=parent['floor'],
            parent_event=parent['id'],histories=parent['histories'],history_rows=parent['history_rows'],
            support_rows=[i for x in support[sg] for i in groups[sg][x][:3]],
            query_rows=[i for x in qxy for i in groups[qg][x][:3]],
            support_acquisition=list(sg),query_acquisition=list(qg),K=5,R=3,Q=12)
        assert len(set(event['support_rows']))==15 and len(set(event['query_rows']))==36
        assert all(rows[i]['xy'] not in excluded for i in event['query_rows'])
        assert all(rows[i]['acquisition']==sg for i in event['support_rows'])
        assert all(rows[i]['acquisition']==qg for i in event['query_rows'])
        events.append(event)
    catalog=[dict(group=g,positions=len(groups[g]),R3_positions=len(positions[g])) for g in ranked]
    return events,catalog


def run():
    started=time.monotonic();OUT.mkdir(exist_ok=False)
    source=json.loads((SOURCE/'complete.json').read_text())
    ref=json.loads((SOURCE/'freeze.json').read_text())
    parameters=json.loads((SOURCE/'parameters.json').read_text())
    assert source['complete'] and source['stationary']
    assert parameters==source['fits']
    assert all(digest(ROOT/n)==h for n,h in ref['code'].items())
    assert all(digest(p)==h for p,h in ref['inputs'].items())
    rho=parameters['1']['rho']
    assert len(rho)==2 and rho[0]==rho[1] and 0<=rho[0]<=1
    plan=json.loads(PLAN.read_text())
    parent=next(e for e in plan['events'] if e['id']=='validation_0')
    rows=load_acquisition_rows(CSV,{parent['floor'],*parent['histories']})
    events,catalog=make_events(rows,parent)
    names=[*ref['code'],'acquisition_retention_transfer.py']
    freeze=dict(created=now(),events=events,catalog=catalog,seed=SEED,retention=rho,
        code={n:digest(ROOT/n) for n in names},
        inputs={str(p):digest(p) for p in [PLAN,CSV,SOURCE/'complete.json',SOURCE/'parameters.json',SOURCE/'freeze.json']},
        learning='None. Retention frozen from B0F1, no target query loss optimization or checkpoint selection.',
        adaptation='Original H/S-only source weights, intercept and censored S posterior fitted afresh per direction.',
        metric='Position-balanced MDE using the frozen geometric-median decision; mean decision is secondary.',
        gate='Both directions no worse than original GP; equal-direction mean >=3% better than GP and same H+S WKNN.',
        limits=['B1 is previously used development, not a blind new-building test.',
                'One floor, two metadata-selected acquisition directions, one K5R3 S draw each.',
                'Query devices can occur in H on other floors; grouping is not causal device identification.',
                'rho=0 equals a fixed no-target-residual-transfer rule, not evidence a trained network is necessary.'],
        compute=dict(threads=2,GPU=0,wall_seconds=600,downloads=0))
    write(OUT/'freeze.json',freeze)
    for n in names:(OUT/n).write_bytes((ROOT/n).read_bytes())
    origin=np.asarray(plan['origin']);used={};records=[]
    for e in events:
        print(json.dumps(dict(stage='adapt',event=e['id'],seconds=time.monotonic()-started)),flush=True)
        packed=episode_scans(rows,e,origin)
        meta=json.loads((OLD/f"results/decision_local_pilot_20261030/{e['parent_event']}/meta.json").read_text())
        task=adapt_acquisition(packed['histories'],packed['support'],tuple(e['support_acquisition']),
                              plan['base'],plan['obs'],history_reader(meta,plan['base'],origin,used))
        write(OUT/f"{e['id']}_adaptation.json",task['state'])
        proposals={};numerical=[]
        with torch.no_grad():
            for ap,c in task['cases'].items():
                p=InterceptBankPrior(c['histories'],task['state']['weights'],plan['base'],task['state']['sigma']['intercept_std'])
                sm,sf=p.features(c['sx']);sk=p.covariance(c['sx'],c['sx'],sf,sf)
                seed=int.from_bytes(hashlib.sha256(f'{SEED}/{e["id"]}/{ap}'.encode()).digest()[:4],'little')
                proposals[ap],info=propose_support(dict(sm=sm,sk=sk),c['stats'],plan['obs'],seed)
                numerical.append(dict(ap=ap,**info))
            ep=prepare_acquisition_transfer(task,proposals)
        del proposals,task
        for c in ep['cases'].values():del c['proposal']
        del ep['geometry']
        torch.save(ep,OUT/f"{e['id']}_episode.pt")
        write(OUT/f"{e['id']}_numerical.json",numerical)
        q=packed['query']
        record=dict(event=e['id'],query_rows=e['query_rows'],truth=q['xy'].tolist(),
            group=q['group'].tolist(),position_counts=q['stats']['n'].tolist(),APs=len(ep['cases']),
            GP=evaluate([1.,1.],ep,q),retained=evaluate(rho,ep,q),
            WKNN=wknn(packed['support'],q,list(packed['histories'].values()),np.array(list(ep['cases'])),
                       torch.arange(5),ep['grid']))
        records.append(record);write(OUT/'evaluation.json',records)
        print(json.dumps(dict(stage='evaluated',event=e['id'],seconds=time.monotonic()-started)),flush=True)
        del ep
    write(OUT/'history_inputs.json',used)
    means={name:float(np.mean([r[name]['mde']['median'] for r in records])) for name in ('GP','retained')}
    means['WKNN']=float(np.mean([r['WKNN']['mde'] for r in records]))
    gate=(all(r['retained']['mde']['median']<=r['GP']['mde']['median'] for r in records)
          and all(means['retained']<=.97*means[name] for name in ('GP','WKNN')))
    assert all(digest(ROOT/n)==h for n,h in freeze['code'].items())
    assert all(digest(p)==h for p,h in freeze['inputs'].items())
    result=dict(complete=True,created=now(),seconds=time.monotonic()-started,means=means,
        gate=gate,retention=rho,training_updates=0,events=len(records),blind_test=False)
    write(OUT/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    try:run()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),error=repr(error)))
        raise
