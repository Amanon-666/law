"""Matched FIVE vs DENSE target intervention. Does not train or select a deployable candidate.
Identical target S/H, prior weight learned once from S5 only, APs, grid, observer and solver; dense arm adds R=3 at training-file sites outside the evaluated spatial block only. Use common full-Gaussian VI for both arms since original reference PMC stops at K<=10. Keep this bridge explicit. Query RSS/XY may only score and choose a held block, never train. Report failed numeric fits; no drops."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from unittest.mock import patch
import inference
from data import group_scans
from run import tensor,ap_stats
from multi_history import single_history
from history_bank import BankPrior
from inference import infer,predict_from_posterior
from research.uji_core.protocol import position_weights
from ..prior_bridge import load
from ..replay import hashids
from .headroom import folds
from .model_ceiling import score
from scipy.special import softmax

def run(data,out,f,seed):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);out.mkdir(exist_ok=True,parents=True);basep=Path(__file__).resolve().parents[1];meta={r['key']:r for r in json.loads((basep/'results/replay/replay.json').read_text())['manifest']};e=f'B1F{f}_{seed}';rec=next(r for r in json.loads((basep/'results/variance/summary.json').read_text())['events'] if r['event']==e);pp=json.loads((basep.parents[1]/'field_parameters.json').read_text())['0'];base,obs=pp['base'],pp['obs'];tr=load(data,'trainingData.csv');v=load(data,'validationData.csv')[f];t=tr[f];h=[tr[0],tr[3]];o=np.vstack([z['positions'] for z in h]).mean(0);lu=dict(zip(t['ids'],range(len(t['ids']))));sr=np.array([lu[i] for i in meta[e+'_internal']['support_1based']]);s=group_scans(t['rss'][sr],t['xy'][sr]-o);hd=[group_scans(z['rss'],z['xy']-o) for z in h];known=np.vstack([z['positions'] for z in [*hd,s]]);ax=[np.linspace(x.min(),x.max(),int(np.ceil(np.ptp(x)/2))+1) for x in known.T];gr=tensor(np.stack(np.meshgrid(*ax),-1).reshape(-1,2));aa=np.flatnonzero(np.logical_or.reduce([(z['rss']!=100).any(0) for z in [*h,s]]));assert len(aa)==rec['ap_count'] and len(gr)==rec['grid']
    real=inference.minimize;maxgrad=0.;calls=0
    def checked(*args,**kwargs):
        nonlocal maxgrad,calls
        a=real(*args,**kwargs);gg=a.jac.copy();
        for ii,(lo,hi) in enumerate(kwargs['bounds']):
            if lo is not None and a.x[ii]<=lo+1e-10 and gg[ii]>0:gg[ii]=0.
        m=np.max(abs(gg));calls+=1;maxgrad=max(maxgrad,float(m));
        if not a.success or not np.isfinite(m) or m>1e-3:raise ValueError(f'Unfinished fit {calls}: {a.message} {m}')
        return a
    orig_groups=np.unique(t['group'][sr]);it=np.flatnonzero(~np.isin(t['group'],orig_groups));iv=np.flatnonzero(~(v['xy'][:,None]==t['positions'][orig_groups][None]).all(-1).any(-1));qs=[('internal',t,it),('official',v,iv)];xy=np.vstack([z['xy'][idx] for _,z,idx in qs]);fold=folds(xy);rss=np.vstack([z['rss'][idx][:,aa] for _,z,idx in qs]);
    for sp,z,idx in qs:assert hashids(z['ids'][idx])==meta[e+'_'+sp]['query_ids_sha256']
    extra=[];rng=np.random.default_rng(811*f+seed);tile=folds(t['positions']);eligible=np.flatnonzero((t['stats']['n']>=3)&~np.isin(np.arange(len(tile)),orig_groups));rows3={g:rng.choice(np.flatnonzero(t['group']==g),3,replace=False) for g in eligible};fitsets=[('S5',sr,np.arange(len(xy)))];
    for i in np.unique(fold):
        grps=[g for g in eligible if tile[g]!=i];idx=np.concatenate([sr,*[rows3[g] for g in grps]]);sel=np.flatnonzero(fold==i);assert not np.intersect1d(t['ids'][idx],t['ids'][it[sel[sel<len(it)]]]).size;assert not ({tuple(z) for z in np.round(t['positions'][grps],3)}&{tuple(z) for z in np.round(xy[sel],3)});fitsets.append(('denseR3',idx,sel));extra.append({'fold':int(i),'sites':5+len(grps),'target_scans':len(idx),'target_ids_sha256':hashids(t['ids'][idx])})
    preds={k:np.empty_like(xy) for k in ['S5','denseR3']};w=rec['weights'];
    with patch.object(inference,'minimize',checked):
        banks=[[single_history({'scans':hh},a,base,obs,steps=640) for hh in hd] for a in aa];pri=[BankPrior(hs,w,base) if any(hh is not None for hh in hs) else BankPrior([],[],base) for hs in banks];
        for lab,idx,sel in fitsets:
            target=group_scans(t['rss'][idx],t['xy'][idx]-o);sx=tensor(target['positions']);assert np.all(target['stats']['n']==3);M=[];V=[]
            for a,p in zip(aa,pri):
                sm,sf=p.features(sx);sk=p.covariance(sx,sx,sf,sf);post=infer(sm,sk,ap_stats(target,a),obs,steps=800);gm,gf=p.features(gr);vout=predict_from_posterior(gm,p.diagonal(gr,gf),p.covariance(gr,sx,gf,sf),sm,sk,post);M.append(vout['mean'].detach().numpy());V.append(vout['variance'].detach().numpy())
            M=np.asarray(M);V=np.asarray(V);assert np.isfinite(M).all() and np.isfinite(V).all() and V.min()>=0;preds[lab][sel]=softmax(score(rss[sel],M,V,obs),axis=1)@gr.numpy()+o;print('finished',e,lab,len(sx),flush=True)
    store={};result={};start=0
    for sp,z,idx in qs:
        end=start+len(idx);txy=z['xy'][idx];wei=position_weights(txy);result[sp]={};store[sp+'_xy']=txy;store[sp+'_ids']=z['ids'][idx]
        for lab,p in preds.items():
            assert np.isfinite(p).all();store[sp+'_'+lab]=p[start:end];result[sp][lab]=float(wei@np.linalg.norm(p[start:end]-txy,axis=1))
        start=end
    rr={'event':e,'scores':result,'dense_sources':extra,'max_projected_numerical_gradient':maxgrad,'fits':calls,'weight':w};(out/(e+'.json')).write_text(json.dumps(rr,indent=2)+'\n');np.savez_compressed(out/(e+'.npz'),**store);print('DENSE',json.dumps(rr))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);p.add_argument('floor',type=int);p.add_argument('seed',type=int);a=p.parse_args();run(a.data,a.out,a.floor,a.seed)
