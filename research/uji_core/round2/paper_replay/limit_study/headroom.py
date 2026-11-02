"""B1 original exact-XY location weighting. No model tuning. References using extra labels are NOT deployable at K=5. A deterministic 8-U tile to five partition map keeps near duplicates in the same validation fold (coordinates rounded to 3 decimals FOR ASSIGNMENT ONLY). Each held fold sees all 5 legal support sites and extra training sites outside the fold. No validation-file RSS/XY goes into any training bank. Only uses earlier rows for dense maps. Identical-RSS collision lower bound is for this finite sample and metric, not population Bayes risk."""
import json,argparse,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist,pdist,squareform
from ..prior_bridge import load
from ..replay import hashids,legacy_voter
from research.uji_core.protocol import position_weights
ROOT=Path(__file__).resolve().parents[1]
def folds(xy):
    b=np.floor(np.round(xy,3)/8.).astype(np.int64)
    return np.array([int.from_bytes(hashlib.sha256(f'b1-headroom-v1/{i}/{j}'.encode()).digest()[:4],'little')%5 for i,j in b])
def lbgroup(x,w):
    # Pairwise triangle + feasible dual vectors; candidate coordinate gives upper bracket.
    X,inv=np.unique(x,axis=0,return_inverse=True); w=np.bincount(inv.ravel(),weights=w);X=X-X[0]; W=w.sum()
    if len(X)==1:return 0.,0.
    dis=squareform(pdist(X));lb=max((w[:,None]*w[None,:]*dis).sum()/(2*W),(np.minimum(w[:,None],w[None,:])*dis).max())
    val=dis@w;j=val.argmin();z=X[j].copy();best=val[j];
    for it in range(300):
        diff=X-z;d=np.linalg.norm(diff,axis=1);ok=d>1e-10;u=np.zeros_like(X);u[ok]=(w[ok]/d[ok])[:,None]*diff[ok];s=u.sum(0);ws=w[~ok].sum();mag=np.linalg.norm(s)
        if ws: u[~ok]=-w[~ok,None]*s/max(mag,ws)
        u-=w[:,None]/W*u.sum(0);scale=max(1.,(np.linalg.norm(u,axis=1)/w).max());v=(u*X).sum()/scale;lb=max(lb,float(v));best=min(best,float(w@d))
        if best-lb<1e-8:break
        t=(X[ok]*(w[ok]/d[ok])[:,None]).sum(0)/(w[ok]/d[ok]).sum();beta=min(1.,ws/max(mag,1e-30));z=beta*z+(1-beta)*t
    assert lb<=best+1e-7;return float(lb),float(best)
def collision(y,x,wt):
    _,idx=np.unique(y,axis=0,return_inverse=True);idx=idx.ravel();lo=hi=amb=0.;groups=rows=0
    for c in np.flatnonzero(np.bincount(idx)>1):
        sel=idx==c
        if len(np.unique(x[sel],axis=0))<2:continue
        a,b=lbgroup(x[sel],wt[sel]);lo+=a;hi+=b;amb+=wt[sel].sum();groups+=1;rows+=int(sel.sum())
    return dict(lower=lo,upper=hi,ambiguous_weight=amb,groups=groups,rows=rows)
def write(d,out):
    out.mkdir(exist_ok=True,parents=True);tr=load(d,'trainingData.csv');val=load(d,'validationData.csv');wk=legacy_voter(); ms={e['key']:e for e in json.loads((ROOT/'results/replay/replay.json').read_text())['manifest']};outrows={};report=[]
    for f in (1,2):
        t=tr[f];v=val[f];h={k:np.concatenate([tr[0][k],tr[3][k]]) for k in ['rss','xy']};hfold=folds(t['xy']);hxy=np.unique(h['xy'],axis=0)
        for seed in range(3):
            eid=f'B1F{f}_{seed}'; lu=dict(zip(t['ids'],range(len(t['ids']))));sr=np.array([lu[i] for i in ms[eid+'_internal']['support_1based']]);s={k:t[k][sr] for k in ['rss','xy']};aps=np.flatnonzero((s['rss']!=100).any(0)|(h['rss']!=100).any(0));allowed=np.isin(t['group'],t['group'][sr]);pts=np.vstack([hxy,s['xy']]); origin=hxy.mean(0)
            def cat(a,b):return {k:np.vstack([a[k],b[k]]) for k in ['rss','xy']}
            def vote(b,Q):
                # Original integer arithmetic and original tie policy; keep memory bounded.
                bank={'rss':b['rss'].astype(np.int64),'xy':b['xy']-origin};return np.concatenate([wk(bank,{'rss':Q[i:i+8].astype(np.int64)},aps,False) for i in range(0,len(Q),8)])+origin
            sets=[('internal',t,np.flatnonzero(~allowed)),('official',v,np.flatnonzero(~(v['xy'][:,None]==s['xy'][None]).all(-1).any(-1)))];
            for split,q,ix in sets:
                key=eid+'_'+split;xy=q['xy'][ix];rss=q['rss'][ix];weights=position_weights(xy);assert hashids(q['ids'][ix])==ms[key]['query_ids_sha256']
                group=folds(xy);pred={'H':vote(h,rss),'HS5':vote(cat(h,s),rss),'S5':vote(s,rss),'dense_target':np.zeros_like(xy),'H_dense_target':np.zeros_like(xy)};ns=[]
                for fold in np.unique(group):
                    sel=group==fold;aux=(hfold!=fold)&~allowed;assert split!='internal' or not np.intersect1d(q['ids'][ix[sel]],t['ids'][aux]).size
                    dense=cat(s,{k:t[k][aux] for k in s});xy3=lambda arr:{tuple(v) for v in np.round(arr,3)};assert not xy3(t['xy'][aux])&xy3(xy[sel]);
                    pred['dense_target'][sel]=vote(dense,rss[sel]);pred['H_dense_target'][sel]=vote(cat(h,dense),rss[sel]);ns.append([int(fold),len(np.unique(dense['xy'],axis=0)),len(dense['rss'])]);
                geo=cdist(xy,np.unique(pts,axis=0)).min(1);outside=np.linalg.norm(np.maximum(np.maximum(pts.min(0)-xy,xy-pts.max(0)),0.),axis=1)
                col=collision(rss[:,aps],xy,weights);mde={k:float(weights@np.linalg.norm(x-xy,axis=1)) for k,x in pred.items()};p90={}
                for k,p in pred.items():
                    er=np.linalg.norm(p-xy,axis=1);o=np.argsort(er);p90[k]=float(er[o][np.searchsorted(weights[o].cumsum(),.9)]);outrows[key+'_'+k]=p
                outrows[key+'_ids']=q['ids'][ix];outrows[key+'_xy']=xy
                report.append(dict(event=eid,split=split,ap_count=len(aps),nq=len(xy),positions=len(np.unique(xy,axis=0)),extra_design=ns,mde=mde,p90=p90,ambiguity=col,nearest_legal_location=float(weights@geo),distance_outside_grid_rectangle=float(weights@outside)))
    summary={}
    for split in ['internal','official']:
        r=[e for e in report if e['split']==split];summ=lambda a:float(np.mean(a));summary[split]={k:summ([e['mde'][k] for e in r]) for k in r[0]['mde']};summary[split].update({k:summ([e[k] for e in r]) for k in ['nearest_legal_location','distance_outside_grid_rectangle']});summary[split].update({'collision_'+k:summ([e['ambiguity'][k] for e in r]) for k in ['lower','upper','ambiguous_weight']})
    (out/'headroom.json').write_text(json.dumps({'summary':summary,'events':report},indent=2)+'\n');np.savez_compressed(out/'headroom_preds.npz',**outrows);print('HEADROOM',json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);a=p.parse_args();write(a.data,a.out)
