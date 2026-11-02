"""B1 Table II: identical GP support posterior/means/grid, AP-wise mean vs spatial variance.
No hyperparameter grid or neural module. Training/source params and six event manifests are frozen.
Also verify an exact evaluation identity: baseline log P(all missing | x) + detected-only increments.
Run from root: python -m research.uji_core.round2.paper_replay.variance_ablation CSV_DIR OUT FLOOR SEED
The reconstruction is called numerically matched only if full-GP macro MDE matches both printed
columns (11.102/11.787, tolerance 0.00051); no per-query DeepSets outputs are assumed.
"""
import argparse, hashlib, json, time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import inference
from data import group_scans
from run import tensor, ap_stats
from multi_history import single_history
from history_bank import BankPrior, fit_two_source_weight
from transfer import Prior, Transfer, rbf
from reference_pmc import setup, sample_reference
from inference import predict_from_posterior, observation_log_prob
from .prior_bridge import load
from .replay import hashids
from research.uji_core.protocol import position_weights

HERE=Path(__file__).resolve().parent

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def run(data,out,f,seed):
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1); t0=time.monotonic()
    target=load(data,'trainingData.csv'); official=load(data,'validationData.csv')[f]; floor=target[f]
    manifest={x['key']:x for x in json.loads((HERE/'results/replay/replay.json').read_text())['manifest']}
    event=f'B1F{f}_{seed}'; meta=manifest[event+'_internal']; lookup={i:k for k,i in enumerate(floor['ids'])}
    sr=np.array([lookup[i] for i in meta['support_1based']]); hist=[target[i] for i in [0,3]]
    assert [hashids(h['ids']) for h in hist]==meta['H_sha256']
    origin=np.concatenate([h['positions'] for h in hist]).mean(0)
    support=group_scans(floor['rss'][sr],floor['xy'][sr]-origin);sx=tensor(support['positions']); assert len(sx)==5 and len(sr)==15
    hseen=np.logical_or.reduce([(h['rss']!=100).any(0) for h in hist]);aps=np.flatnonzero(hseen|(support['rss']!=100).any(0));assert len(aps)==meta['aps']
    hdata=[group_scans(h['rss'],h['xy']-origin) for h in hist]; known=np.vstack([h['positions'] for h in hist]+[floor['xy'][sr]])-origin
    axes=[np.linspace(c.min(),c.max(),int(np.ceil(np.ptp(c)/2.))+1) for c in known.T];grid=tensor(np.stack(np.meshgrid(*axes),-1).reshape(-1,2));G=len(grid)
    params=json.loads((HERE.parents[1]/'field_parameters.json').read_text())['0'];base,obs=params['base'],params['obs']
    real_min=inference.minimize
    def checked(*a,**kw):
        z=real_min(*a,**kw)
        if not z.success:raise RuntimeError('History fit did not converge: '+str(z.message))
        return z
    with patch.object(inference,'minimize',checked):
        banks=[[single_history({'scans':h},a,base,obs,steps=640) for h in hdata] for a in aps]
    def gather(hs):
        hm=[];hc=[]
        for h in hs:
            prior=Prior(Transfer('identity'),h,dict(base,drift=0.));m,fx=prior.features(sx);hm.append(m);hc.append(prior.covariance(sx,sx,fx,fx))
        absent=all(h is None for h in hs);kk=rbf(sx,sx,base['amplitude'] if absent else base['drift'],base['length'])
        return torch.stack(hm), torch.zeros_like(torch.stack(hc)) if absent else torch.stack(hc),kk
    means,cov,residual=[torch.stack(z).detach() for z in zip(*(gather(h) for h in banks))]
    counts=[ap_stats(support,a) for a in aps];allcounts={k:torch.stack([d[k] for d in counts]) for k in counts[0]}
    weight,trials=fit_two_source_weight(means,cov,residual,allcounts,obs)
    M=[]; V=[]; ess=[]
    with torch.no_grad():
        for i,(hs,stat) in enumerate(zip(banks,counts)):
            prior=BankPrior(hs,weight,base) if any(h is not None for h in hs) else BankPrior([],[],base)
            sm,sf=prior.features(sx);sk=prior.covariance(sx,sx,sf,sf);tg,pr,ctr,root,_=setup(sm,sk,stat,obs)
            sample,lw,info=sample_reference(tg,pr,ctr,root,261026+seed);pw=lw.exp();pm=pw@sample;d=sample-pm
            gm,gf=prior.features(grid);gv=prior.diagonal(grid,gf);cross=prior.covariance(grid,sx,gf,sf)
            v=predict_from_posterior(gm,gv,cross,sm,sk,{'mean':pm,'cov':(d.T*pw)@d});M.append(v['mean']);V.append(v['variance']);ess.append(info['effective_samples'])
    M=torch.stack(M);V=torch.stack(V);assert torch.isfinite(M).all() and torch.isfinite(V).all() and (V>=0).all()
    ii=np.flatnonzero(~np.isin(floor['group'],floor['group'][sr]));oi=np.flatnonzero(~(official['xy'][:,None]==floor['positions'][np.unique(floor['group'][sr])][None]).all(-1).any(-1));qsets=[('internal',floor,ii),('official',official,oi)]
    for name,q,loc in qsets:assert hashids(q['ids'][loc])==manifest[event+'_'+name]['query_ids_sha256']
    Q=np.vstack([q['rss'][loc][:,aps] for name,q,loc in qsets]); rows=len(Q);scores=[np.zeros((rows,G)) for _ in range(2)];sparse=np.zeros_like(scores[0]); base_miss=np.zeros(G)
    for i,a in enumerate(aps):
        vals,inv=np.unique(np.r_[Q[:,i],100],return_inverse=True);inv=inv.ravel();miss=inv[-1];inv=inv[:-1];seen=np.flatnonzero(Q[:,i]!=100)
        for b,var in enumerate([V[i],V[i].mean().expand_as(V[i])]):
            table=observation_log_prob(tensor(vals)[:,None],M[i],var,obs).detach().numpy();scores[b]+=table[inv]
            if b==0:
                base_miss+=table[miss];sparse[seen]+=table[inv[seen]]-table[miss]
    sparse+=base_miss;coord=lambda a:(torch.from_numpy(a).softmax(1)@grid).numpy()+origin
    p0,p1=map(coord,scores);ps=coord(sparse); err=float(np.max(np.abs(p0-ps))); assert err<1e-6
    outs={};pred={};off=0
    for name,q,ix in qsets:
        end=off+len(ix);w=position_weights(q['xy'][ix]);stat={};pred[name+'_ids']=q['ids'][ix];pred[name+'_xy']=q['xy'][ix]
        for lab,p in [('full',p0),('flat',p1)]:
            e=np.linalg.norm(p[off:end]-q['xy'][ix],axis=1);stat[lab]=float(w@e);pred[name+'_'+lab]=p[off:end]
        stat['flat_minus_full']=stat['flat']-stat['full'];outs[name]=stat;off=end
    out.mkdir(exist_ok=True,parents=True);np.savez_compressed(out/(event+'_pred.npz'),**pred)
    np.savez_compressed(out/(event+'_field.npz'),mu=M.numpy(),variance=V.numpy(),grid=grid.numpy(),origin=origin,aps=aps)
    rec=dict(event=event,weights=weight,grid=G,ap_count=len(aps),support_ids=meta['support_1based'],means_sha=digest(M.numpy()),variance_sha=digest(V.numpy()),grid_sha=digest(grid.numpy()),min_ess=float(min(ess)),max_dense_sparse_coordinate_diff=err,detected_ap_fraction=float((Q!=100).mean()),runtime_seconds=time.monotonic()-t0,scores=outs,backend='B0 parameters reconstructed; original single_history, shared EB, original M-PMC 6x4096+32768; CPU float64. No DeepSets weights.')
    (out/(event+'.json')).write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');print(json.dumps(rec),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);p.add_argument('floor',type=int,choices=(1,2));p.add_argument('seed',type=int,choices=(0,1,2));a=p.parse_args();run(a.data,a.out,a.floor,a.seed)
