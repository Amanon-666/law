"""One B0F0 K3 integration; reuse all historical fits, no shared updates."""
import gc
import json
import socket
import time
from functools import lru_cache
import numpy as np
import torch
from anchored_barycenter_assets import CACHE,MANIFEST
from building_source_assets import load_field
from fields import InterceptBankPrior
from neural_history_trial import metric
from observed_recomposition import prepare,build_field
from operator_proposals import propose_states,summarize_states,compare_scrambles
from paired_assets import read_allowed
from recomposition import Parameters
from recomposition_adaptation import fit,adapted_prior
from recomposition_coupled import evaluate
from recomposition_coupled_trial import episode
from recomposition_predict import locate
from refi_source_selection_numerical import attach_all_queries
from recomposition_trial import ROOT,CSV,digest,write,now,subset

NEG=ROOT/'results/observed_history_trial_20261107'
REFERENCE=ROOT/'results/anchored_barycenter_localization_20261107'
OUT=ROOT/'results/observed_recomposition_20261107'


def run():
    start=time.monotonic();torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    nf=json.loads((NEG/'freeze.json').read_text());nd=json.loads((NEG/'complete.json').read_text())
    rf=json.loads((REFERENCE/'freeze.json').read_text());rv=json.loads((REFERENCE/'verification.json').read_text())
    assert nd['complete'] and rv['passed'] and digest(CACHE)==rf['inputs'][str(CACHE)]
    assert digest(CSV)==rf['inputs'][str(CSV)]
    assert digest(REFERENCE/'comparison.json')==rv['comparison_sha256']
    data=torch.load(CACHE,weights_only=False);aps=list(data['cases']);del data['cases'];gc.collect()
    events=data['events'];e=events[0];origin=np.asarray(data['origin'])
    assert e['floor']=='B0F0' and e['history_rows']==nf['history_rows']
    assert data['base']==nf['base'] and data['obs']==nf['obs']
    assert all(digest(ROOT/n)==h for n,h in rf['code'].items())
    manifest=json.loads(MANIFEST.read_text());records={(r['floor'],r['ap']):r for r in manifest['fields']}
    negative={}
    for h in e['histories']:
        path=NEG/f'{h}_negative.npz';assert digest(path)==nd['artifacts'][path.name]
        negative[h]=dict(path=str(path),sha256=digest(path))
        assert manifest['history_rows'][h]==e['history_rows'][h]
    rows=set(e['support_rows']).union(*(set(r) for r in e['history_rows'].values()))
    assert all(not rows.intersection(v['query_rows']) for v in events)
    raw=read_allowed(CSV,rows)
    hs={h:subset(raw[h],r,origin) for h,r in e['history_rows'].items()}
    support=subset(raw[e['floor']],e['support_rows'],origin)
    @lru_cache(None)
    def positive(h,ap):
        r=records[h,ap];assert digest(ROOT/r['path'])==r['sha256']
        return load_field(r,origin,data['base'])
    @lru_cache(None)
    def negative_field(h):return load_field(negative[h],origin,data['base'])
    OUT.mkdir(exist_ok=False)
    names=('observed_recomposition.py','observed_recomposition_trial.py','observed_history.py',
        'recomposition.py','recomposition_adaptation.py','recomposition_coupled.py','recomposition_predict.py',
        'observations.py','posterior.py','operator_proposals.py','neural_history_trial.py','location_decision.py')
    paths=(CACHE,MANIFEST,CSV,NEG/'freeze.json',NEG/'complete.json',REFERENCE/'comparison.json',REFERENCE/'query.npz')
    write(OUT/'freeze.json',dict(created=now(),code={n:digest(ROOT/n) for n in names},
        inputs={str(p):digest(p) for p in paths},events=events,negative_fields=negative,
        criterion='Both AA/AB median MDE improve >=3% vs saved same-task ReFi',
        shared_updates=0,historical_fits=0,cpu=2,gpu=0,seconds=180,query_read=False))
    for n in names:(OUT/n).write_bytes((ROOT/n).read_bytes())
    delta=[-int(h.split('F')[1]) for h in e['histories']]
    task,cases=prepare(hs,support,data['base'],data['obs'],delta,positive,negative_field)
    assert list(cases)==aps and torch.equal(task['grid'],data['grid'])
    zero=Parameters(torch.zeros(9),torch.ones(9))
    state=fit(zero,cases,data['obs'],differentiable=False)
    write(OUT/'adaptation.json',dict(created=now(),weights=state['weights'].tolist(),
        intercept_variance=float(state['intercept_variance']),evidence=task['nondetection_evidence'],query_read=False))
    max_identity=0.;checks=[]
    with torch.no_grad():
        for ap,c in cases.items():
            p=adapted_prior(zero,c['blocks'],state)
            source=task['cases'][ap];direct=InterceptBankPrior(source['histories'],state['weights'].tolist(),data['base'],float(state['intercept_variance'].sqrt()))
            sm,sf=direct.features(source['sx']);gm,gf=direct.features(data['grid'])
            expected=dict(sm=sm,gm=gm,sk=direct.covariance(source['sx'],source['sx'],sf,sf),
                cross=direct.covariance(data['grid'],source['sx'],gf,sf),gv=direct.diagonal(data['grid'],gf))
            max_identity=max(max_identity,max(float((p[k]-v).abs().max()) for k,v in expected.items()))
            assert max_identity<1e-8
            summaries=[]
            for repeat in (0,1):
                nodes=propose_states([p],c['stats'],data['obs'],power=16,seed=117092+ap*100+repeat*100000)
                summaries.append(summarize_states([p],c['stats'],data['obs'],nodes))
                if repeat==0:first=nodes
            check=dict(AP=ap,**compare_scrambles(*summaries));checks.append(check)
            with (OUT/'numerical.jsonl').open('a') as f:f.write(json.dumps(check)+'\n')
            if not check['accepted']:raise RuntimeError(f'New posterior precision failed AP {ap}: {check}')
            c['proposal']=first
    data.update(cases=cases,state=state,nondetection_evidence=task['nondetection_evidence'])
    torch.save(data,OUT/'support.pt')
    cached=build_field(zero,cases,state,data['obs'],data['grid']);torch.save(cached,OUT/'field.pt')
    write(OUT/'field_frozen.json',dict(created=now(),query_read=False,field_sha256=digest(OUT/'field.pt'),support_sha256=digest(OUT/'support.pt'),identity_gap=max_identity))
    del task,raw,hs;positive.cache_clear();negative_field.cache_clear();gc.collect()
    item,coverage=attach_all_queries(data)
    with np.load(REFERENCE/'query.npz') as q:
        assert all(np.array_equal(q[k],item['q'][k]) for k in ('rss','xy','positions','group'))
    # Fixed nuisance fit is the original ReFi learning convention. One gradient,
    # no optimizer; this does not claim differentiation through the fit itself.
    ep=item['ep'];zero.zero_grad();value=evaluate(zero,ep,state,backward=True)
    assert torch.isfinite(zero.theta.grad).all()
    deployed=locate(cached,item['q']['rss'][:,aps])['scores'].log_softmax(1)
    path_gap=float((deployed-value['log_probability']).abs().max());assert path_gap<1e-9
    np.savez_compressed(OUT/'predictions.npz',log_probability=deployed.numpy(),grid=data['grid'].numpy())
    old=json.loads((REFERENCE/'comparison.json').read_text());cells={};row=0
    for event,q in zip(events,item['cells']):
        stop=row+len(q['rss']);cells[event['id']]=metric(deployed[row:stop],data['grid'],q);row=stop
    gains={e['id']:1-cells[e['id']]['mde']['median']/old['scores']['refi_adapted']['cells'][e['id']]['mde']['median'] for e in events}
    avg={r:float(np.mean([c['mde'][r] for c in cells.values()])) for r in ('mean','median')}
    result=dict(cells=cells,mde=avg,original=old['scores']['refi_adapted']['median_mde'],wknn=old['wknn']['mde'],gains=gains,
        continued_effect=all(v>=.03 for v in gains.values()),beats_wknn=avg['median']<old['wknn']['mde'],coverage=coverage,
        identity_gap=max_identity,training_deployment_gap=path_gap,source_NLL=value['nll'],gradient=zero.theta.grad.tolist(),
        gradient_is_not_learning=True,shared_updates=0)
    write(OUT/'comparison.json',result)
    assert all(digest(p)==h for p,h in json.loads((OUT/'freeze.json').read_text())['inputs'].items())
    write(OUT/'complete.json',dict(complete=True,created=now(),seconds=time.monotonic()-start,APs=len(aps),
        minimum_ESS=min(c['states'][0]['minimum_ESS'] for c in checks),comparison_sha256=digest(OUT/'comparison.json'),
        predictions_sha256=digest(OUT/'predictions.npz'),shared_updates=0,historical_fits=0))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    try:run()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),error=repr(error)))
        raise
