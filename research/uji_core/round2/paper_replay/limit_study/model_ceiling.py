"""Frozen-GP diagnostic. No fitting or tuning. Only the H-only coverage reference below is deployable at K5. The true site-list experiments and synthetic matched-model risk are deliberately privileged; do not use them to select the learned challenger. Input includes the six pre-existing field and prediction files from run 41370648408. History-coverage and reflected-history controls use the existing B0 GP length, no bandwidth grid. GPs, candidate grid, AP axis, observations and expectation fixed for the legal comparisons. 1024 simulated scans per event/split use the fitted observation model and UNIFORM true query-site list, not the distribution of an independently deployed building. No additional modules installed in the default model."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import log_ndtr,ndtr,softmax,logsumexp
from ..prior_bridge import load
from research.uji_core.protocol import position_weights
BASE=Path(__file__).resolve().parents[3];N_SIM=1024

def score(q,mu,v,obs):
    # Inputs AP x location. Accumulate exact scan log probability from miss-only baseline.
    a,b,t,n=[obs[k] for k in ('alpha','beta','detect_scale','noise_std')];n2=n*n; tv=v+n2; gain=v/tv
    miss=np.logaddexp(np.log1p(-a),np.log(a)+log_ndtr((b-mu)/np.sqrt(t*t+v)))
    L=np.broadcast_to(miss.sum(0),(len(q),mu.shape[1])).copy()
    for j in range(q.shape[1]):
        rows=np.flatnonzero(q[:,j]!=100)
        if not len(rows):continue
        r,inv=np.unique(q[rows,j],return_inverse=True);r=r[:,None];dif=r-mu[j];z=(mu[j]+gain[j]*dif-b)/np.sqrt(t*t+n2*gain[j]);inc=np.log(a)-.5*(np.log(2*np.pi*tv[j])+dif**2/tv[j])+log_ndtr(z)-miss[j];L[rows]+=inc[inv.ravel()]
    return L

def median_risk(p,x):
    # Return a feasible decision, with a primal and a dual bound on posterior minimal Euclidean loss.
    zz=p@x
    for _ in range(160):
        d=cdist(zz,x);ww=p/np.maximum(1e-10,d);new=ww@x/ww.sum(1)[:,None]
        if np.linalg.norm(zz-new,axis=1).max()<1e-8:zz=new;break
        zz=new
    diff=x[None]-zz[:,None];dd=np.linalg.norm(diff,axis=2);up=(p*dd).sum(1);lo=[]
    for i in range(len(p)):
        u=p[i,:,None]*diff[i]/np.maximum(dd[i,:,None],1e-12);hit=dd[i]<1e-7;u[hit]=0;v=u.sum(0)
        if hit.any(): u[hit]=-p[i,hit,None]*v/max(np.linalg.norm(v),p[i,hit].sum(),1e-300)
        u-=p[i,:,None]*u.sum(0);factor=max(1.,np.max(np.linalg.norm(u,axis=1)/np.maximum(p[i],1e-300)));lo.append(float(np.sum((u/factor)*x)))
    lo=np.maximum(np.asarray(lo),0.);assert (lo<=up+1e-7).all();return zz,lo,up

def run(d,fi,out):
    out.mkdir(exist_ok=True,parents=True);pp=json.loads((BASE/'field_parameters.json').read_text())['0'];obs=pp['obs'];ell=pp['base']['length'];tr=load(d,'trainingData.csv');va=load(d,'validationData.csv');history=np.unique(np.vstack([tr[h]['positions'] for h in (0,3)]),axis=0);res=[]; preds={}
    for f in [1,2]:
        for sd in range(3):
            eid=f'B1F{f}_{sd}';g=np.load(next(fi.rglob(eid+'_field.npz')));ref=np.load(next(fi.rglob(eid+'_pred.npz')));mu=g['mu'];var=g['variance'];grid=g['grid'];org=g['origin'];h=history-org
            def cover(h):return logsumexp(-cdist(grid,h,'sqeuclidean')/(2*ell**2),axis=1)-np.log(len(h))
            cov=cover(h);reflect=cover(grid.min(0)+grid.max(0)-h)
            for split,z in [('internal',tr[f]),('official',va[f])]:
                ids=ref[split+'_ids'];idx=np.searchsorted(z['ids'],ids);assert np.array_equal(z['ids'][idx],ids);q=z['rss'][idx][:,g['aps']];truth=z['xy'][idx];assert np.array_equal(truth,ref[split+'_xy']);ww=position_weights(truth);truth=truth-org;X,grp=np.unique(truth,axis=0,return_inverse=True);grp=grp.ravel();near=cdist(X,grid).argmin(1);L=score(q,mu,var,obs)
                pm=softmax(L[:,near],axis=1);gm,_,_=median_risk(pm,X);pr={k:softmax(L+off,axis=1)@grid for k,off in [('base',0.),('H_coverage',cov),('rotated_H',reflect)]};pr.update(known_sites_mean=pm@X,known_sites_median=gm);assert np.max(abs(pr['base']+org-ref[split+'_full']))<1e-6
                act={k:float(ww@np.linalg.norm(v-truth,axis=1)) for k,v in pr.items()};qkey=eid+'_'+split;preds[qkey+'_ids']=ids;preds[qkey+'_xy']=truth+org
                for k,vv in pr.items():preds[qkey+'_'+k]=vv+org
                # Simulated world exactly matches the Gaussian marginal/readout, independent APs, true sites equiprobable.
                rng=np.random.default_rng(639102+10000*f+100*sd+int(split=='official'));sidx=rng.integers(len(X),size=N_SIM);latent=mu[:,near[sidx]].T+np.sqrt(var[:,near[sidx]].T)*rng.normal(size=(N_SIM,len(g['aps'])));det=rng.random(latent.shape)<obs['alpha']*ndtr((latent-obs['beta'])/obs['detect_scale']);Y=np.where(det,latent+obs['noise_std']*rng.normal(size=latent.shape),100.);simprob=softmax(score(Y,mu[:,near],var[:,near],obs),axis=1);_,low,up=median_risk(simprob,X);sim={'empirical_mc_primal':float(up.mean()),'empirical_mc_dual':float(low.mean()),'mc_se_primal':float(up.std(ddof=1)/np.sqrt(len(up))),'n':N_SIM,'num_query_sites':len(X),'mean_real_site_grid_offset':float(np.linalg.norm(X-grid[near],axis=1).mean())};res.append(dict(event=eid,split=split,real_mde=act,matched_world=sim));print(eid,split,act,sim,flush=True)
    result={s:{**{k:float(np.mean([r['real_mde'][k] for r in res if r['split']==s])) for k in res[0]['real_mde']},**{k:float(np.mean([r['matched_world'][k] for r in res if r['split']==s])) for k in ['empirical_mc_primal','empirical_mc_dual']}} for s in ['internal','official']};(out/'model_ceiling.json').write_text(json.dumps(dict(summary=result,events=res),indent=2)+'\n');np.savez_compressed(out/'ceiling_predictions.npz',**preds);print('MODEL_CEILING',json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('fields',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.data,a.fields,a.out)
