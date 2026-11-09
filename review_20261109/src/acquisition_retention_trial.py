"""One source floor: spatial retention versus uniform retention, with held positions.

This is source development, not unseen-environment validation. All nuisance
parameters were previously fitted on B0; only the new retention fit excludes
the evaluation locations. No device label enters the prediction interface.
"""
import hashlib
import json
import socket
import time
from collections import Counter

import numpy as np
import torch
from scipy.optimize import minimize

from acquisition_tasks import load_acquisition_rows,acquisitions,episode_scans,adapt_acquisition
from acquisition_transfer import prepare_acquisition_transfer,evaluate_acquisition_transfer,build_acquisition_field
from conditional_budget_trial import history_reader
from conditional_history_field import predict
from fields import InterceptBankPrior
from fingerprint_posterior import propose_support
from neural_history_trial import metric
from recomposition_multibudget_natural import wknn
from recomposition_trial import ROOT,OLD,PLAN,CSV,digest,write,now
from uji import group_scans

STRUCTURE = ROOT/'results/acquisition_transfer_structure_20261105'
OUT = ROOT/'results/acquisition_retention_trial_20261105'
SEED = 1150536


def partition(rows, manifest):
    """One global spatial split, excluding BOTH support sets before sampling Q."""
    events = [dict(f['events'][-1]) for f in manifest['families'] if f['floor']=='B0F1']
    assert len(events)==2 and all(e['K']==5 and e['R']==3 for e in events)
    excluded = {rows[i]['xy'] for e in events for i in e['support_rows']}
    groups = acquisitions(rows,'B0F1')
    eligible = sorted({x for g in groups.values() for x,ids in g.items()
                       if len(ids)>=3 and x not in excluded})
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(eligible)); cut = len(eligible)//2
    pools = dict(fit={eligible[i] for i in perm[:cut]},evaluation={eligible[i] for i in perm[cut:]})
    for e in events:
        qp = groups[tuple(e['query_acquisition'])]
        for phase,pool in pools.items():
            xy = [x for x in sorted(pool) if len(qp.get(x,[]))>=3]
            assert len(xy)>=12,(e['id'],phase,len(xy))
            selected = [xy[i] for i in rng.permutation(len(xy))[:12]]
            e[phase+'_rows'] = [i for x in selected for i in qp[x][:3]]
        del e['query_rows']
    fit = {rows[i]['xy'] for e in events for i in e['fit_rows']}
    evaluation = {rows[i]['xy'] for e in events for i in e['evaluation_rows']}
    assert not fit.intersection(evaluation) and not (fit|evaluation).intersection(excluded)
    for e in events:
        for phase in pools:
            ids=e[phase+'_rows']
            assert len(ids)==len(set(ids))==36
            assert all(list(rows[i]['acquisition'])==e['query_acquisition'] for i in ids)
    return events


def collect_query(rows, event, phase, origin):
    ids=event[phase+'_rows']
    q=group_scans(np.asarray([rows[i]['rss'] for i in ids],dtype=np.int16),
                  np.asarray([rows[i]['xy'] for i in ids])-origin)
    return dict(q,acquisition_group=tuple(event['query_acquisition']))


def fit_retention(episodes, queries, width):
    """Bounded source optimization; no evaluation labels or selection on MDE."""
    history=[]
    def objective(x):
        rho=torch.tensor(np.repeat(x,2) if width==1 else x,requires_grad=True)
        values=[evaluate_acquisition_transfer(rho,ep,q,backward=True)['nll']
                for ep,q in zip(episodes,queries)]
        gradient=rho.grad.detach().numpy()/len(episodes)
        if width==1:gradient=np.asarray([gradient.sum()])
        value=float(np.mean(values))
        assert np.isfinite(value) and np.isfinite(gradient).all()
        history.append(dict(rho=rho.detach().tolist(),nll=value,gradient=gradient.tolist()))
        write(OUT/f'fit_{width}.json',history)
        return value,gradient
    result=minimize(objective,np.ones(width),jac=True,method='L-BFGS-B',bounds=[(0.,1.)]*width,
                    options=dict(maxiter=12,maxfun=24,maxls=10,ftol=1e-9,gtol=1e-5))
    gradient=result.jac.copy()
    gradient[(result.x<=1e-10)&(gradient>0)]=0
    gradient[(result.x>=1-1e-10)&(gradient<0)]=0
    return dict(rho=(np.repeat(result.x,2) if width==1 else result.x).tolist(),
                nll=float(result.fun),iterations=int(result.nit),evaluations=int(result.nfev),
                success=bool(result.success),message=str(result.message),
                projected_gradient=float(np.abs(gradient).max()))


@torch.no_grad()
def evaluate(rho, episode, query):
    # Deployment is exercised explicitly: no Q label is accepted by field/predict.
    field=build_acquisition_field(torch.tensor(rho),episode)
    logp=predict(field,query['rss'][:,field['APs']])['log_probability']
    result=metric(logp,field['grid'],query)
    nearest=torch.cdist(torch.tensor(query['xy']),field['grid']).argmin(1)
    weights=torch.tensor(1/(len(query['positions'])*query['stats']['n'][query['group']]))
    result['nll']=float(-(weights*logp[torch.arange(len(nearest)),nearest]).sum())
    return result


def run():
    OUT.mkdir(exist_ok=False);started=time.monotonic()
    structure=json.loads((STRUCTURE/'freeze.json').read_text())
    assert json.loads((STRUCTURE/'complete.json').read_text())['complete']
    assert all(digest(p)==h for p,h in structure['inputs'].items())
    assert all(digest(ROOT/n)==h for n,h in structure['code'].items())
    plan=json.loads(PLAN.read_text())
    rows=load_acquisition_rows(CSV,{'B0F0','B0F1','B0F2'})
    events=partition(rows,json.loads((STRUCTURE/'source_tasks.json').read_text()))
    names=['acquisition_retention_trial.py','acquisition_transfer.py','acquisition_tasks.py',
           'adaptation.py','predict.py','episodic_tasks.py','fields.py','fingerprint_change.py',
           'fingerprint_posterior.py','posterior.py','observations.py','conditional_history_field.py',
           'conditional_budget_trial.py','neural_history_trial.py','recomposition_multibudget_natural.py',
           'recomposition_trial.py','uji.py']
    freeze=dict(created=now(),events=events,seed=SEED,code={n:digest(ROOT/n) for n in names},
        inputs={str(p):digest(p) for p in [PLAN,CSV,STRUCTURE/'freeze.json',STRUCTURE/'source_tasks.json']},
        optimizer=dict(name='L-BFGS-B',initial=1.,bounds=[0.,1.],maxiter=12,maxfun=24,maxls=10,
                       ftol=1e-9,gtol=1e-5,objective='Equal-task complete-scan nearest-grid NLL'),
        scope='B0F1 two acquisition directions, K5R3, 12 fit and 12 evaluation positions each; global position separation',
        gate='Both learned fits successful with projected gradient <=1e-3; spatial beats uniform on each direction and mean by >=3%; also beats original GP and H+S WKNN mean by >=3%',
        selection='Final optimizer parameters only. No evaluation-based selection, restart or expansion.',
        limitations=['B0 nuisance fits used all B0 data: source development only.',
                     'Two user-phone pairs, confounded with dates; no device-causal or unseen-device claim.',
                     'Only one floor and one S draw per direction; no cross-environment or K-curve evidence.'],
        compute=dict(threads=2,GPU=0,wall_seconds=900,downloads=0))
    write(OUT/'freeze.json',freeze)
    for n in names:(OUT/n).write_bytes((ROOT/n).read_bytes())
    episodes=[];scans=[];used={};origin=np.asarray(plan['origin'])
    for e in events:
        print(json.dumps(dict(stage='adapt',event=e['id'],seconds=time.monotonic()-started)),flush=True)
        packed=episode_scans(rows,dict(e,query_rows=e['fit_rows']),origin)
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
            episode=prepare_acquisition_transfer(task,proposals)
        del proposals,task
        for c in episode['cases'].values():del c['proposal']
        del episode['geometry']
        torch.save(episode,OUT/f"{e['id']}_episode.pt")
        write(OUT/f"{e['id']}_numerical.json",numerical)
        episodes.append(episode);scans.append(packed)
        print(json.dumps(dict(stage='prepared',event=e['id'],APs=len(episode['cases']),
            rank_counts=dict(Counter(c['history_contrast_rank'] for c in episode['cases'].values())),
            seconds=time.monotonic()-started)),flush=True)
    write(OUT/'history_inputs.json',used)
    queries=[p['query'] for p in scans]
    fits={str(width):fit_retention(episodes,queries,width) for width in (1,2)}
    write(OUT/'parameters.json',fits)
    records=[]
    for e,ep,p in zip(events,episodes,scans):
        q=collect_query(rows,e,'evaluation',origin)
        record=dict(event=e['id'],query_rows=e['evaluation_rows'],truth=q['xy'].tolist(),
                    group=q['group'].tolist(),position_counts=q['stats']['n'].tolist(),
                    GP=evaluate([1.,1.],ep,q),uniform=evaluate(fits['1']['rho'],ep,q),
                    spatial=evaluate(fits['2']['rho'],ep,q),
                    WKNN=wknn(p['support'],q,list(p['histories'].values()),np.array(list(ep['cases'])),
                               torch.arange(5),ep['grid']))
        records.append(record);write(OUT/'evaluation.json',records)
    means={name:float(np.mean([r[name]['mde']['median'] for r in records])) for name in ('GP','uniform','spatial')}
    means['WKNN']=float(np.mean([r['WKNN']['mde'] for r in records]))
    stationary=all(v['success'] and v['projected_gradient']<=1e-3 for v in fits.values())
    gate=(stationary and all(r['spatial']['mde']['median']<r['uniform']['mde']['median'] for r in records)
          and all(means['spatial']<=.97*means[n] for n in ('GP','uniform','WKNN')))
    assert all(digest(ROOT/n)==h for n,h in freeze['code'].items())
    result=dict(complete=True,created=now(),seconds=time.monotonic()-started,fits=fits,means=means,
                gate=gate,stationary=stationary,events=len(records),new_environment_test=False)
    write(OUT/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    try:run()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),error=repr(error)))
        raise
