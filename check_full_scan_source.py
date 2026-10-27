"""Numerical sensitivity of source sharing-prior learning, same model/data.

Repair every ESS<1000 component using two independent defensive-importance
estimates. Original fields remain immutable. No target observations are read.
"""
import argparse
import json
import math
import socket

import numpy as np
import torch
from pypmc.density.gauss import Gauss
from pypmc.density.mixture import MixtureDensity
from scipy.special import logsumexp

from data import group_scans
from full_scan_pooling_source import OUT,SPARSE
from history_bank import BankPrior
from inference import gaussian_update,observation_log_prob,predict_from_posterior
from partial_source_pooling import conditional_pooling,partial_ap_log_prob,scan_log_prob,fit_probability
from reference_pmc import setup,sample_reference
from run import ap_stats,tensor
from run_multi_history import digest,write
from source_weight_mixture import mixture_log_prob
from uji_history_bank_study import cached_history,load_raw

DEST=OUT/'source_numerical_check'
SEEDS=[261626,261627]


@torch.no_grad()
def task_check(task):
    src=OUT/'source'/f'task{task:02}';assert (src/'completed.json').exists()
    dest=DEST/f'task{task:02}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completed.json').exists():return
    freeze=json.loads((OUT/'freeze.json').read_text());meta=json.loads((OUT/'source_manifest.json').read_text())
    e=meta['events'][task];base,obs=freeze['source_setting']['base'],freeze['source_setting']['obs']
    with np.load(src/'fields.npz') as a:values={k:a[k] for k in a.files}
    selection=np.argwhere(values['ess']<1000)
    write(dest/'selection.json',dict(post_hoc_numerical_check=True,original_sha256=digest(src/'fields.npz'),
        rule='Every coefficient component with ESS<1000; no target data or outcome selection',
        components=selection.tolist(),draws=131072,seeds=SEEDS,proposal='90% fitted mixture + 10% original whitened RSS Gaussian'))
    f=load_raw('trainingData.csv',{e['floor']})[e['floor']];lookup={int(r):i for i,r in enumerate(f['row_ids'])}
    sr=np.array([lookup[r] for r in e['support_rows']]);qr=np.array([lookup[r] for r in e['query_rows']]);origin=np.array(meta['origin'])
    s=group_scans(f['rss'][sr],f['xy'][sr]-origin);q=group_scans(f['rss'][qr],f['xy'][qr]-origin)
    sx=tensor(s['positions']);qx=tensor(q['positions']);group=torch.tensor(q['group'])
    old=json.loads((SPARSE/'source'/f'task{task:02}'/'completed.json').read_text())
    old_index={row['ap']:i for i,row in enumerate(old['episodes'])};rows=[]
    for i,j in selection:
        ap=e['ap_columns'][int(i)];path=dest/f'ap{ap}_node{j}.npz';record=path.with_suffix('.json')
        if path.exists():rows.extend(json.loads(record.read_text())['rows']);continue
        hh=[cached_history(OUT,n,ap,base) for n in e['histories']];w=float(freeze['nodes'][j])
        prior=BankPrior(hh,[w,1-w],base) if any(h is not None for h in hh) else BankPrior([],[],base)
        sm,sf=prior.features(sx);sk=prior.covariance(sx,sx,sf,sf);stats=ap_stats(s,ap)
        target,initial,center,root,_=setup(sm,sk,stats,obs)
        seed=261327+task*4+old_index[ap] if ap in old_index else 261527+task*520+ap
        _,_,info=sample_reference(target,initial,center,root,seed)
        proposal=MixtureDensity([Gauss(np.array(m),np.array(c)) for m,c in zip(info['mixture_means'],info['mixture_covariances'])]
            +[Gauss(np.zeros(len(sx)),np.eye(len(sx)))],np.r_[.9*np.array(info['mixture_weights']),.1]);proposal.normalize()
        qm,qf=prior.features(qx);qv=prior.diagonal(qx,qf);cross=prior.covariance(qx,sx,qf,sf)
        rss_loss=gaussian_update(sm,sk,stats,obs)['loss'].item();z=[];lp=[];records=[]
        for rep,seed in enumerate(SEEDS):
            u=proposal.propose(131072,rng=np.random.RandomState(seed));lw=target(u)-proposal.multi_evaluate(u)
            weights=tensor(np.exp(lw-logsumexp(lw)));samples=tensor(center+u@root.T);pm=weights@samples;c=samples-pm
            pred=predict_from_posterior(qm,qv,cross,sm,sk,dict(mean=pm,cov=(c.T*weights)@c))
            z.append(float(logsumexp(lw)-math.log(len(lw))-rss_loss))
            lp.append(observation_log_prob(tensor(q['rss'][:,ap]),pred['mean'][group],pred['variance'][group],obs).numpy())
            records.append(dict(task=task,ap=ap,node=int(j),rep=rep,ess=float(1/weights.square().sum()),old_ess=float(values['ess'][i,j]),
                evidence_change=z[-1]-float(values['log_z'][i,j]),max_query_logprob_change=float(np.abs(lp[-1]-values['log_prob'][i,j]).max())))
        np.savez_compressed(path,log_z=z,log_prob=lp);write(record,dict(rows=records));rows.extend(records)
    write(dest/'completed.json',dict(completed=True,task=task,components=len(selection),rows=rows))
    print(json.dumps(dict(stage='source_numerical_task',task=task,components=len(selection))),flush=True)


def fit_check():
    freeze=json.loads((OUT/'freeze.json').read_text());data=[[],[],[]];mass=tensor(freeze['prior_mass'])
    for task in range(freeze['source_tasks']):
        d=DEST/f'task{task:02}';assert (d/'completed.json').exists()
        with np.load(OUT/'source'/f'task{task:02}'/'fields.npz') as a:values={k:a[k] for k in a.files}
        indices=values['old_query_AP_indices'];aps=values['ap_columns'].tolist()
        for variant in range(3):
            z=values['log_z'].copy();lp=values['log_prob'].copy()
            if variant:
                for p in d.glob('ap*_node*.npz'):
                    ap,node=p.stem.split('_node');i=aps.index(int(ap[2:]));j=int(node)
                    with np.load(p) as a:z[i,j]=a['log_z'][variant-1];lp[i,j]=a['log_prob'][variant-1]
            data[variant].append((tensor(z),tensor(lp),tensor(values['weights']),indices))
    results=[]
    for variant,items in enumerate(data):
        for mode in ['full','context_only']:
            def loss(lam):
                scores=[]
                for z,lp,weights,indices in items:
                    if mode=='full':score=scan_log_prob(z,lp,mass,lam)
                    else:
                        gamma,shared,local,independent,_=conditional_pooling(z,mass,lam)
                        joint=sum(partial_ap_log_prob(lp[i],shared[i],local[i],independent[i]) for i in indices)
                        score=mixture_log_prob(joint,gamma)
                    scores.append(-(weights*score).sum())
                return float(torch.stack(scores).mean())
            fit=fit_probability(loss);fit.update(variant=variant,mode=mode,shared_loss=loss(0),half_loss=loss(.5),independent_loss=loss(1))
            if variant==0:
                original=json.loads((OUT/('source_fit.json' if mode=='full' else 'context_only_fit.json')).read_text())
                np.testing.assert_allclose(fit['objective'],original['objective'],rtol=0,atol=1e-9)
            results.append(fit)
            write(DEST/f'fit_{mode}_{variant}.json',fit)
            print(json.dumps({k:v for k,v in fit.items() if k not in ['grid','profile']}),flush=True)
    write(DEST/'completed.json',dict(completed=True,original_fields_unchanged=True,target_data_read=False,results=results))


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on server')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['task','fit']);p.add_argument('--task',type=int);args=p.parse_args()
    if args.stage=='task':task_check(args.task)
    else:fit_check()
