"""Post-hoc numerical sensitivity for low-ESS coefficient components.

Same prior/likelihood, defensive importance proposal already used in this
project. Frozen fields and predictions remain untouched. Select by ESS only.
"""
import argparse
import json
import math
from pathlib import Path
import socket
import time

import numpy as np
import torch
from pypmc.density.gauss import Gauss
from pypmc.density.mixture import MixtureDensity
from scipy.special import logsumexp

from data import group_scans
from full_ap_study import metrics
from history_bank import BankPrior
from inference import gaussian_update, observation_log_prob, predict_from_posterior
from reference_pmc import setup, sample_reference
from run import ap_stats,tensor
from run_multi_history import digest,write
from source_weight_mixture import coefficient_posteriors,mixture_log_prob
from run_source_weight_mixture import OUT,BANK,SOURCES,cached_history,load_raw


def evaluate(event_id):
    directory=OUT/event_id
    assert (directory/'completed.json').exists()
    dest=OUT/'numerical_check'/event_id;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completed.json').exists(): return
    selection=[]
    for p in sorted(directory.glob('ap*.json')):
        record=json.loads(p.read_text())
        for row in record['results']:
            if row['effective_samples']<1000: selection.append(dict(ap=record['ap'],**row))
    write(dest/'selection.json',dict(created_unix=time.time(),post_hoc=True,rule='Every component with ESS < 1000, independent of localization errors',
        draws=131072,seeds=[261226,261227],proposal='90% original fitted mixture + 10% original whitened RSS-updated Gaussian',
        model_changes=0,original_prediction_sha256=digest(directory/'predictions.npz'),components=selection))
    params=json.loads((OUT/'parameters.json').read_text());base,obs=params['base'],params['obs']
    freeze=json.loads((OUT/'freeze.json').read_text());nodes=np.array(freeze['coefficient_nodes']);prior_mass=tensor(freeze['prior_mass'])
    meta=next(e for e in json.loads((OUT/'manifest.json').read_text())['events'] if e['id']==event_id)
    f=load_raw('trainingData.csv',{meta['floor']})[meta['floor']];q=load_raw('validationData.csv',{meta['floor']})[meta['floor']]
    fi={int(r):i for i,r in enumerate(f['row_ids'])};qi={int(r):i for i,r in enumerate(q['row_ids'])}
    sr=np.array([fi[r] for r in meta['support_rows']]);ir=np.array([fi[r] for r in meta['internal_rows']]);orr=np.array([qi[r] for r in meta['official_rows']])
    origin=np.array(json.loads((BANK/'histories/completed.json').read_text())['origin'])
    support=group_scans(f['rss'][sr],f['xy'][sr]-origin);sx=tensor(support['positions'])
    with np.load(directory/'predictions.npz') as a: grid=tensor(a['grid']-origin)
    # Regenerate only unstable components with the same model. Store both
    # independent numerical repetitions; do not select the better prediction.
    replacement={};diagnostics=[]
    for ap in sorted({r['ap'] for r in selection}):
        path=dest/f'ap{ap}.npz'
        if path.exists():
            with np.load(path) as a: replacement[ap]={k:a[k] for k in a.files}
            diagnostics.extend(json.loads((dest/f'ap{ap}.json').read_text())['rows']);continue
        with np.load(directory/f'ap{ap}.npz') as a:
            replacement[ap]={f'{key}_{rep}':a[key].copy() for key in ['mean','variance','log_evidence'] for rep in range(2)}
        histories=[cached_history(BANK,name,ap,base) for name in SOURCES];rows=[]
        for selected in [r for r in selection if r['ap']==ap]:
            j=selected['node'];w=nodes[j]
            p=BankPrior(histories,[float(w),float(1-w)],base) if any(h is not None for h in histories) else BankPrior([],[],base)
            with torch.no_grad():
                sm,sf=p.features(sx);sk=p.covariance(sx,sx,sf,sf);stats=ap_stats(support,ap)
                target,initial,center,root,_=setup(sm,sk,stats,obs)
                _,_,info=sample_reference(target,initial,center,root,261026+meta['seed'])
                proposal=MixtureDensity([Gauss(np.array(m),np.array(c)) for m,c in zip(info['mixture_means'],info['mixture_covariances'])]
                    +[Gauss(np.zeros(len(sx)),np.eye(len(sx)))],np.r_[.9*np.array(info['mixture_weights']),.1])
                proposal.normalize();gm,gf=p.features(grid);gv=p.diagonal(grid,gf);cross=p.covariance(grid,sx,gf,sf)
                rss_loss=gaussian_update(sm,sk,stats,obs)['loss'].item()
                for rep,seed in enumerate([261226,261227]):
                    u=proposal.propose(131072,rng=np.random.RandomState(seed));lw=target(u)-proposal.multi_evaluate(u)
                    weights=tensor(np.exp(lw-logsumexp(lw)));samples=tensor(center+u@root.T);pm=weights@samples;c=samples-pm
                    pred=predict_from_posterior(gm,gv,cross,sm,sk,dict(mean=pm,cov=(c.T*weights)@c))
                    evidence=float(logsumexp(lw)-math.log(len(lw))-rss_loss)
                    replacement[ap][f'mean_{rep}'][j]=pred['mean'].numpy();replacement[ap][f'variance_{rep}'][j]=pred['variance'].numpy()
                    replacement[ap][f'log_evidence_{rep}'][j]=evidence
                    rows.append(dict(ap=ap,node=j,rep=rep,seed=seed,ess=float(1/weights.square().sum()),
                        old_ess=selected['effective_samples'],log_evidence=evidence,old_log_evidence=selected['log_evidence']))
        np.savez_compressed(path,**replacement[ap]);write(dest/f'ap{ap}.json',dict(rows=rows));diagnostics.extend(rows)
    aps=meta['ap_columns']; all_evidence=[]
    for ap in aps:
        with np.load(directory/f'ap{ap}.npz') as a: all_evidence.append(a['log_evidence'])
    evidence=[tensor(np.array(all_evidence)) for _ in range(3)]
    for ap,values in replacement.items():
        i=aps.index(ap)
        for rep in range(2): evidence[rep+1][i]=tensor(values[f'log_evidence_{rep}'])
    posterior=[coefficient_posteriors(e,prior_mass) for e in evidence]
    point=[np.array([np.lexsort((np.abs(nodes-.5),-row))[0] for row in e.numpy()]) for e in evidence]
    rss=tensor(np.concatenate((f['rss'][ir],q['rss'][orr])));truth=np.concatenate((f['xy'][ir],q['xy'][orr]))
    device=torch.device('cuda:0'); n=len(nodes);nq=len(rss);ng=len(grid)
    # Stream independent queries; the all-query tensor needlessly exceeds a
    # shared A4000's memory. The original-prediction replay below checks equivalence.
    predictions={v:{m:[] for m in ['shared_bma','ap_bma','ap_point']} for v in range(3)}
    all_fields=[]
    for ap in aps:
        with np.load(directory/f'ap{ap}.npz') as a: fields=[(tensor(a['mean']),tensor(a['variance']))]
        if ap in replacement: fields.extend((tensor(replacement[ap][f'mean_{r}']),tensor(replacement[ap][f'variance_{r}'])) for r in range(2))
        all_fields.append(fields)
    with torch.no_grad():
        for start in range(0,nq,16):
            sl=slice(start,min(start+16,nq));size=sl.stop-sl.start
            shared_scores=torch.zeros((3,n,size,ng),device=device);ap_scores=torch.zeros((3,size,ng),device=device);point_scores=torch.zeros_like(ap_scores)
            for i,ap in enumerate(aps):
                fields=all_fields[i];r=rss[sl,ap].to(device);lp0=None
                for variant in range(3):
                    if variant and ap not in replacement: lp=lp0
                    else:
                        m,v=(t.to(device) for t in fields[variant]);lp=observation_log_prob(r[None,:,None],m[:,None,:],v[:,None,:],obs)
                        if not variant: lp0=lp
                    shared_scores[variant]+=lp
                    ap_scores[variant]+=mixture_log_prob(lp,posterior[variant][1][i].to(device))
                    point_scores[variant]+=lp[point[variant][i]]
            for variant in range(3):
                scores=dict(shared_bma=mixture_log_prob(shared_scores[variant],posterior[variant][0].to(device)),ap_bma=ap_scores[variant],ap_point=point_scores[variant])
                for m,score in scores.items():predictions[variant][m].append((torch.softmax(score,1)@grid.to(device)).cpu().numpy()+origin)
    predictions={v:{m:np.concatenate(parts) for m,parts in results.items()} for v,results in predictions.items()}
    with np.load(directory/'predictions.npz') as old:
        for method,pred in predictions[0].items(): np.testing.assert_allclose(pred,old[method],rtol=0,atol=1e-7)
    rows=[]
    for variant,results in predictions.items():
        for split,sl in [('internal',slice(0,len(ir))),('official',slice(len(ir),None))]:
            xy=truth[sl];_,g,count=np.unique(xy,axis=0,return_inverse=True,return_counts=True);sw=1/(len(count)*count[g.ravel()])
            for method,pred in results.items(): rows.append(dict(variant=variant,method=method,split=split,**metrics(pred[sl],dict(xy=xy,scan_weights=sw))))
    np.savez_compressed(dest/'predictions.npz',**{f'{method}_{variant}':pred for variant,result in predictions.items() for method,pred in result.items()},query_xy=truth,internal_count=len(ir))
    result=dict(completed=True,event=event_id,post_hoc_numerical_check=True,original_unchanged=True,selected_components=len(selection),rows=rows,component_diagnostics=diagnostics)
    write(dest/'completed.json',result)
    print(json.dumps(dict(event=event_id,selected_components=len(selection),official=[r for r in rows if r['split']=='official'])),flush=True)


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Server only')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--event',required=True);args=p.parse_args();evaluate(args.event)
