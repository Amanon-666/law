"""B1-only shared spatial GP calibration for a new, isolated evaluation.

Fit the existing mean/amplitude/length model by a weighted source ELBO.
Observations and target discrepancy are fixed from the completed B1 TPM fit.
No Q localization, target adaptation, or new learning architecture is involved.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
import torch

import inference
from run import ap_stats, tensor
from transfer import rbf
from uji_probability_study import load_raw, DATA
from external_baselines.tpm_source_calibration import FLOORS, sha, write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/source_base_B1_20261028'
TRANSFER=ROOT/'results/tpm_source_calibration_20261027/all_B1.json'


def prepare():
    raw=load_raw('trainingData.csv',set(FLOORS))
    union=np.logical_or.reduce([(raw[f]['stats']['k']>0).any(0) for f in FLOORS])
    catalogue=np.flatnonzero(union)
    cases=[]
    for floor in FLOORS:
        counts=raw[floor]['stats']['k'].sum(0)
        absent=catalogue[counts[catalogue]==0]
        present=catalogue[counts[catalogue]>0]
        ordered=present[np.argsort(counts[present],kind='stable')]
        strata=[absent,*np.array_split(ordered,5)]
        for index,aps in enumerate(strata):
            if not len(aps):continue
            ap=int(aps[(len(aps)-1)//2])
            cases.append(dict(floor=floor,ap=ap,stratum=index,represented_APs=len(aps),
                              weight=float(len(aps)/len(catalogue)/len(FLOORS))))
    assert abs(sum(c['weight'] for c in cases)-1)<1e-12
    transfer=json.loads(TRANSFER.read_text())
    assert transfer['usable'] and transfer['held_source_floor'] is None
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),ROOT/'inference.py',ROOT/'transfer.py',ROOT/'run.py',ROOT/'data.py',ROOT/'uji_probability_study.py']
    write(OUT/'freeze.json',dict(UTC=datetime.now(timezone.utc).isoformat(),purpose=__doc__,cases=cases,
        source_floors=FLOORS,source_csv_rows={f:raw[f]['row_ids'].tolist() for f in FLOORS},
        scope='Only B1 source data. B0/B2/UAH/Pisa development queries and independent data excluded.',
        panel='Per floor, all-absent AP stratum plus five detection-count quantiles; deterministic middle AP; weights preserve B1-visible AP proportions and equal floors.',
        objective='Weighted negative Gaussian-variational ELBO per acquired scan. Represents a stratified 24-case source panel, not exact optimization over every AP.',
        hyperparameters=['mean','amplitude','length'],fixed='obs and drift from all_B1 TPM source fit',
        initial=[0.,float(np.log(20.)),float(np.log(20.))],
        coordinates='mean=-95+20*x0, amplitude=exp(x1), length=exp(x2); UJI native coordinate units',
        bounds=[[-3.25,3.75],[float(np.log(.1)),float(np.log(100.))],[float(np.log(.25)),float(np.log(300.))]],
        bounds_reason='Broad source estimation box: mean -160..-20 dBm, amplitude .1..100 dB, length .25..300 source coordinate units. Boundary solutions are reported, not silently accepted as interior convergence.',
        outer=dict(maxiter=64,maxls=20,ftol=1e-10,gtol=1e-5),inner=dict(maxiter=640,max_projected_gradient=1e-3),
        outer_acceptance='Finite returned values and projected outer gradient below 1e-4; report optimizer flag and boundary separately.',
        review=['The inner optimized whitened Gaussian is held fixed only for ELBO envelope gradients, not for prediction gradients.',
                'Inner stationarity is checked before using its envelope gradient.',
                'Finite differences on one three-position source case precede full calibration.',
                'No target labels choose a start, bounds, case panel or checkpoint.',
                'This calibration must regenerate H under these same parameters; old B0 GP caches are not interchangeable.',
                'No RSS/ELBO decrease is claimed to be a positioning gain.'],
        code_sha256={str(p):sha(p) for p in paths},input_sha256={str(DATA/'trainingData.csv'):sha(DATA/'trainingData.csv'),str(TRANSFER):sha(TRANSFER)}))


def run():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    frozen=json.loads((OUT/'freeze.json').read_text())
    for section in ('code_sha256','input_sha256'):
        for name,digest in frozen[section].items():
            if sha(name)!=digest:raise ValueError(f'Frozen file changed: {name}')
    raw=load_raw('trainingData.csv',set(FLOORS))
    for name in FLOORS:assert raw[name]['row_ids'].tolist()==frozen['source_csv_rows'][name]
    law=json.loads(TRANSFER.read_text())['transfer']
    obs=dict(alpha=law['h'],beta=law['theta'],detect_scale=law['s'],noise_std=law['sigma_total'])
    cases=[]
    for meta in frozen['cases']:
        f=raw[meta['floor']]
        cases.append((meta,tensor(f['positions']-f['positions'].mean(0)),ap_stats(f,meta['ap'])))
    original=inference.minimize
    inner=[]
    def monitored(*args,**kwargs):
        result=original(*args,**kwargs)
        g=result.jac.copy()
        for j,(low,high) in enumerate(kwargs['bounds']):
            if low is not None and result.x[j]<=low+1e-10 and g[j]>0:g[j]=0
        maximum=float(np.abs(g).max())
        inner.append(dict(success=bool(result.success),iterations=int(result.nit),projected_gradient=maximum))
        if not np.isfinite(maximum) or maximum>frozen['inner']['max_projected_gradient']:
            raise RuntimeError(f'Unfinished inner solve: {inner[-1]}')
        return result

    def case_value(value,x,stats):
        mean=-95+20*value[0]
        amplitude,length=value[1:].exp().unbind()
        with patch.object(inference,'minimize',monitored):
            posterior=inference.infer(mean.expand(len(x)),rbf(x,x,amplitude,length),stats,obs,
                                      steps=frozen['inner']['maxiter'],hyperparameter_fit=True)
        return posterior['loss']/stats['n'].sum()

    # A small REAL source case checks the nontrivial envelope differentiation.
    _,x,stats=next(c for c in cases if c[0]['stratum']==3)
    chosen=torch.arange(min(3,len(x)))
    x=x[chosen];stats={k:v[chosen] for k,v in stats.items()}
    start=torch.tensor(frozen['initial'],requires_grad=True)
    loss=case_value(start,x,stats)
    gradient,=torch.autograd.grad(loss,start)
    numerical=[]
    for j in range(3):
        delta=torch.zeros(3);delta[j]=1e-3
        plus=case_value(start.detach()+delta,x,stats)
        minus=case_value(start.detach()-delta,x,stats)
        numerical.append(float((plus-minus)/.002))
    gap=float(np.max(np.abs(gradient.detach().numpy()-numerical)))
    check=dict(autograd=gradient.detach().tolist(),finite_difference=numerical,max_gap=gap,passed=gap<1e-3,
               scope='Three-position source-only envelope check; no performance evidence')
    write(OUT/'gradient_check.json',check)
    if not check['passed']:raise RuntimeError(f'Envelope check failed: {check}')
    inner.clear()
    evaluations=[]
    def objective(values):
        variable=torch.tensor(values,requires_grad=True)
        total=0.;grad=np.zeros(3);inner.clear()
        for meta,x,stats in cases:
            value=case_value(variable,x,stats)
            derivative,=torch.autograd.grad(value,variable)
            total+=meta['weight']*float(value.detach())
            grad+=meta['weight']*derivative.detach().numpy()
        row=dict(evaluation=len(evaluations),objective=total,parameters=list(map(float,values)),gradient=grad.tolist(),
                 max_inner_projected_gradient=max(v['projected_gradient'] for v in inner),inner_solves=len(inner))
        evaluations.append(row)
        with (OUT/'trajectory.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        return total,grad
    result=minimize(objective,np.array(frozen['initial']),method='L-BFGS-B',jac=True,
                    bounds=frozen['bounds'],options=frozen['outer'])
    projected=result.jac.copy();boundary=[]
    for i,(lo,hi) in enumerate(frozen['bounds']):
        if result.x[i]<=lo+1e-7:
            boundary.append([i,'lower']);projected[i]=min(projected[i],0.)
        if result.x[i]>=hi-1e-7:
            boundary.append([i,'upper']);projected[i]=max(projected[i],0.)
    accepted=bool(np.isfinite(result.fun) and np.isfinite(result.x).all() and np.max(np.abs(projected))<1e-4)
    record=dict(accepted=accepted,optimizer_success=bool(result.success),message=str(result.message),
                projected_gradient_max=float(np.max(np.abs(projected))),boundary=boundary,iterations=int(result.nit),
                evaluations=len(evaluations),objective=float(result.fun),initial_objective=evaluations[0]['objective'],
                base=dict(mean=float(-95+20*result.x[0]),amplitude=float(np.exp(result.x[1])),length=float(np.exp(result.x[2])),
                          drift=float(np.sqrt(np.median(np.asarray(law['sb2'])+law['s_w2'])))),
                obs=obs,source_building='UJI_B1',freeze_sha256=sha(OUT/'freeze.json'),
                scope='Local stationary source-panel ELBO solution only; not a localization or global-optimality claim')
    write(OUT/'parameters.json',record)
    write(OUT/'execution.json',dict(state='complete' if accepted else 'stopped_unaccepted',accepted=accepted,
        target_predictions=0,downloads=0))
    if not accepted:raise RuntimeError(f'Outer calibration not stationary: {record}')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:
        try:run()
        except Exception as error:
            write(OUT/'execution.json',dict(state='stopped_error',error=repr(error),target_predictions=0,downloads=0))
            raise
