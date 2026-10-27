"""Evaluate only the new frozen source coefficients on the existing K rows.

All comparison scores are read from the completed original evaluation. Their
absence remains absence. No model selection, S refit, AP selection change or
replacement of an old numerical failure occurs here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from residual_probe import broad_source as source
from residual_probe.learning import prediction
from inference import observation_log_prob

curve = source.curve
OUT = source.OUT/'evaluation'
REFERENCES = ('H_base','H_support','WKNN_HS','H_source')


def freeze():
    source.verify()
    curve.verify()
    models = {b:source.OUT/'models'/b/'model.json' for b in source.data_module().CONFIGS}
    for b,p in models.items():
        m=source.read(p)
        assert m['steps']==512 and b not in m['training_buildings']
    inputs={curve.OUT/'manifest.json',source.OUT/'freeze.json',
            source.OUT/'label_metric_erratum.json',*models.values()}
    for e in curve.manifest()['events']:
        inputs.add(curve.OUT/e['id']/'adaptation.json')
        inputs.update(curve.OUT/e['id']/f'{m}.json' for m in REFERENCES
                      if (curve.OUT/e['id']/f'{m}.json').exists())
    source.write(OUT/'freeze.json',dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        code_sha256={str(Path(__file__)):source.digest(__file__)},
        input_sha256={str(p):source.digest(p) for p in sorted(inputs)},
        models={b:str(p) for b,p in models.items()},
        information='Exact original K-curve H/S/Q/AP/domain and same cached S-only w/sigma. New theta only.',
        numerical='Same bounded target and two-scramble consistency policy as source.broad_source; original common AP seed.',
        failure='One attempt/event. Record numerical failure and continue predetermined remaining events, never score a partial AP set.',
        scope='Development; no official validation labels. All four source models complete before first score.'))


def verify():
    source.verify()
    f=source.read(OUT/'freeze.json')
    for key in ('code_sha256','input_sha256'):
        for path,sha in f[key].items():assert source.digest(path)==sha,path
    return f


@torch.no_grad()
def event(meta,theta,base,obs):
    start=time.perf_counter()
    task=curve.episodes.build_task(meta)
    points=curve.grid(meta)
    cases=curve.support_cases(meta,task,base)
    adaptation=source.read(curve.OUT/meta['id']/'adaptation.json')
    scores=torch.zeros((len(task['Q']['rss']),len(points)))
    ess_values=[]
    directory=OUT/meta['id']
    directory.mkdir(parents=True,exist_ok=False)
    for column,(ap,case) in enumerate(cases.items()):
        prior=source.InterceptBankPrior(case['histories'],adaptation['weights'],base,
                                       adaptation['sigma']['intercept_std'])
        sm,sf=prior.features(case['sx'])
        qm,qf=prior.features(points)
        prepared=dict(sm=sm,sk=prior.covariance(case['sx'],case['sx'],sf,sf),
            qm=qm,qv=prior.diagonal(points,qf),cross=prior.covariance(points,case['sx'],qf,sf),
            phi_s=curve.phi(case,case['sx'],base),phi_q=curve.phi(case,points,base),**case['stats'])
        numeric,info=source.proposal(prepared,obs,theta,source.seed_for(f'kcurve|{meta["id"]}|{ap}'))
        prepared.update(numeric)
        value,ess=prediction(theta,prepared,obs,list(range(10)))
        if not all(torch.isfinite(v).all() for v in value.values()) or (value['variance']<0).any():
            raise FloatingPointError(f'Invalid field: {meta["id"]}, AP{ap}')
        curve.save_case(directory/f'ap{ap}.npz',dict(mean=value['mean'].numpy(),
            variance=value['variance'].numpy(),ess=np.array(float(ess)),sampler=np.array(info['sampler'])))
        ess_values.append(float(ess))
        scores += observation_log_prob(source.tensor(task['Q']['rss'][:,column])[:,None],
                    value['mean'][None],value['variance'][None],obs)
        if column%100==0:print(json.dumps(dict(stage='evaluation_AP',event=meta['id'],done=column+1,total=len(cases))),flush=True)
    if not torch.isfinite(scores).all():raise FloatingPointError('Invalid complete scan scores')
    pred=(scores.softmax(1)@points).numpy()
    q=task['Q']
    result=dict(**curve.metrics(pred,q),predictions=pred.tolist(),
        errors=np.linalg.norm(pred-q['xy'],axis=1).tolist(),min_ESS=min(ess_values),
        seconds=time.perf_counter()-start,query_scan_ids=meta['query_scan_ids'])
    curve.save_case(directory/'predictions.npz',dict(scores=scores.numpy(),predictions=pred,
        query_xy=q['xy'],query_scan_ids=q['scan_ids']))
    source.write(directory/'new_source.json',result)
    print(json.dumps(dict(stage='event_complete',event=meta['id'],mde=result['mde'])),flush=True)


def run(building):
    f=verify();m=curve.manifest()
    theta=source.tensor(source.read(f['models'][building])['theta'])
    completed,failures=[],[]
    for meta in m['events']:
        if meta['building']!=building:continue
        try:
            event(meta,theta,m['base'],m['obs']);completed.append(meta['id'])
        except Exception as error:
            failure=dict(event=meta['id'],error=repr(error),UTC=datetime.now(timezone.utc).isoformat())
            source.write(OUT/meta['id']/'failure.json',failure)
            failures.append(failure)
            print(json.dumps(dict(stage='event_failed',**failure)),flush=True)
    source.write(OUT/f'{building}_finished.json',dict(completed=completed,failures=failures))


def required_budget(points,threshold):
    """First measured passing K; an earlier missing point prevents a minimum claim."""
    missing=False
    for k,error in sorted(points):
        if error is None:
            missing=True
        elif error<=threshold:
            return dict(K=None if missing else k,
                        status='earlier_point_missing' if missing else 'reached')
    return dict(K=None,status='missing_points' if missing else 'not_reached')


def label_saving(points,reference):
    threshold=points[reference][5]
    if threshold is None:return dict(status='reference_K5_missing')
    base=required_budget(points[reference].items(),threshold)
    learned=required_budget(points['new_source'].items(),threshold)
    saving=None if base['K'] is None or learned['K'] is None else base['K']-learned['K']
    return dict(threshold=threshold,reference_minimum=base,learned_minimum=learned,
                saved_positions=saving,fraction_saved=None if saving is None else saving/base['K'])


def summary():
    manifest=curve.manifest();groups=[];events=[]
    for e in manifest['events']:
        methods={name:source.read(curve.OUT/e['id']/f'{name}.json') for name in REFERENCES
                 if (curve.OUT/e['id']/f'{name}.json').exists()}
        path=OUT/e['id']/'new_source.json'
        if path.exists():methods['new_source']=source.read(path)
        events.append(dict(**{k:e[k] for k in ('id','building','K','seed')},
            unit=e['coordinate_unit'],query_scan_ids=e['query_scan_ids'],results=methods))
    for building in source.data_module().CONFIGS:
        for k in manifest['K']:
            es=[e for e in events if e['building']==building and e['K']==k]
            methods={name:dict(available=sum(name in e['results'] for e in es),
                mde=float(np.mean([e['results'][name]['mde'] for e in es]))
                    if all(name in e['results'] for e in es) else None)
                for name in (*REFERENCES,'new_source')}
            groups.append(dict(building=building,K=k,unit=es[0]['unit'],methods=methods))
    ratios={};primary_complete=True
    for b in source.data_module().CONFIGS:
        rows=[g for g in groups if g['building']==b and g['K'] in (1,3,5)]
        values={name:[r['methods'][name]['mde'] for r in rows]
                for name in ('H_base','H_support','WKNN_HS','new_source')}
        if any(None in v for v in values.values()):primary_complete=False;continue
        ratios[b]={name:float(np.mean(values['new_source'])/np.mean(values[name]))
                   for name in ('H_base','H_support','WKNN_HS')}
    learning_pass=primary_complete and sum(all(r[m]<1 for m in ('H_base','H_support')) for r in ratios.values())>=3
    learning_pass=learning_pass and all(np.mean([r[m] for r in ratios.values()])<=.95
        and max(r[m] for r in ratios.values())<=1.10 for m in ('H_base','H_support'))
    system_pass=primary_complete and sum(r['WKNN_HS']<1 for r in ratios.values())>=3
    savings=[]
    for b in source.data_module().CONFIGS:
        for reference in ('H_base','WKNN_HS'):
            points={name:{g['K']:g['methods'][name]['mde'] for g in groups if g['building']==b}
                    for name in (reference,'new_source')}
            draws=[]
            for seed in manifest['seeds']:
                per_draw={name:{e['K']:e['results'].get(name,{}).get('mde') for e in events
                    if e['building']==b and e['seed']==seed} for name in (reference,'new_source')}
                draws.append(dict(seed=seed,**label_saving(per_draw,reference)))
            savings.append(dict(building=b,reference=reference,
                                **label_saving(points,reference),draws=draws))
    finished=all((OUT/f'{b}_finished.json').exists() for b in source.data_module().CONFIGS)
    result=dict(finished=finished,
        events=events,groups=groups,primary_complete=primary_complete,ratios=ratios,
        learning_gate_passed=bool(learning_pass),system_gate_passed=bool(system_pass),
        independent_validation_authorized_by_gates=bool(finished and learning_pass and system_pass),
        label_efficiency=savings,
        interpretation='Ratios use each building mean of K1/3/5. Missing primary pairs prevent a pass; high-K missing baselines remain unavailable. No cross-unit absolute averaging.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','groups')}),flush=True)


if __name__=='__main__':
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','summary'])
    parser.add_argument('--building');args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='run':run(args.building)
    else:summary()
