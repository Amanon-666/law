"""Development fold (B0): reader variants on the translated map. No B1/B2 target scores."""
import sys, time, pickle; sys.path.insert(0,'/home/claude/work/vt')
import numpy as np, pandas as pd, core as C, vt_lib as V, translator as T
man=C.DEV
rng=np.random.default_rng(20261110)
Xh,Xs,y=V.training_set([1,2],rng,draws=8,single_acquisition=False)
mHS=V.Model(np.hstack([Xh,Xs]),y); print('trained',len(y),flush=True)
def knn_general(a,xy,q,bd,k=3,tau=None,offset=False):
    if offset:   # remove per-pair mean difference over APs seen in either (device/gain offset), then L2
        D=np.empty((len(q),len(a)))
        for i,row in enumerate(q):
            m=(row>T.ND)[None]|(a>T.ND); diff=np.where(m,row[None]-a,0); n=np.maximum(m.sum(1),1)
            mu=diff.sum(1)/n; mu=np.clip(mu,-10,10)
            D[i]=np.sqrt((np.where(m,(diff-mu[:,None])**2,0)).sum(1)/a.shape[1])
    else: D=V.dist(q,a)
    if tau is not None:
        w=np.exp(-(D**2-(D**2).min(1,keepdims=True))/tau); w/=w.sum(1,keepdims=True); return np.clip(w@xy,bd[0],bd[1])
    idx=np.argsort(D,1,kind='stable')[:,:k]; w=1/np.maximum(np.take_along_axis(D,idx,1),1e-8); w/=w.sum(1,keepdims=True)
    return np.clip((w[:,:,None]*xy[idx]).sum(1),bd[0],bd[1])
res={}
for e in man['events']:
    b=int(e['building'][1]); t=int(e['floor'].split('F')[1])
    H,S,Q=C.event_arrays_m(man,e); ctx=np.concatenate([H,S]); aps=np.flatnonzero(C.DET[ctx].any(0)); bd=C.bounds_of(C.XY[ctx])
    Xc,mp=V.translate(mHS,b,t,S,aps)
    q=T.encode(C.RSS[Q][:,aps]); s=T.encode(C.RSS[S][:,aps]); h=T.encode(C.RSS[H][:,aps])
    a=np.vstack([mp,s]); xy=np.vstack([Xc,C.XY[S]])
    out={'k1':knn_general(a,xy,q,bd,1),'k3':knn_general(a,xy,q,bd,3),'k5':knn_general(a,xy,q,bd,5),'k9':knn_general(a,xy,q,bd,9),
         'soft_tau2':knn_general(a,xy,q,bd,tau=2.),'soft_tau5':knn_general(a,xy,q,bd,tau=5.),'soft_tau10':knn_general(a,xy,q,bd,tau=10.),
         'k3_offset':knn_general(a,xy,q,bd,3,offset=True),'k5_offset':knn_general(a,xy,q,bd,5,offset=True),
         'map_only_k3':knn_general(mp,Xc,q,bd,3),
         'union_rawH_k3':knn_general(np.vstack([a,h]),np.vstack([xy,C.XY[H]]),q,bd,3)}
    raw=C.nearest3(C.RSS[ctx],C.XY[ctx],C.RSS[Q],aps,bd); out['raw3NN']=raw; out['avg_raw_translated']=(raw+out['k3'])/2
    for k,v in out.items(): res.setdefault(k,{})[e['id']]=C.event_error_m(man,e,v)
rows=[]
for k,v in res.items():
    s_=C.summarize_m(man,v); rows.append(dict(method=k,mean=s_['mean'],**{f'K{a}':x for a,x in s_['by_K'].items()},**s_['by_floor']))
print(pd.DataFrame(rows).round(2).to_string())
