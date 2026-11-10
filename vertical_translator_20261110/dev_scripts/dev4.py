"""Development fold (B0 targets, trained on B1+B2): what the translator uses. No B1/B2 target scores."""
import sys, time; sys.path.insert(0,'/home/claude/work/vt')
import numpy as np, pandas as pd, core as C, vt_lib as V, translator as T
man=C.DEV
rng=np.random.default_rng(20261110)
Xh,Xs,y=V.training_set([1,2],rng,draws=8,single_acquisition=False)
nh=Xh.shape[1]
slot=lambda k: list(range(k,40,5))            # 0 v, 1 smooth, 2 det, 3 fmax, 4 fmean ; 40 ref, 41 gap_dn, 42 gap_up
S=lambda *i: [nh+j for j in i]                # 0 d_mean 1 y_max 2 y_mean 3 pooled 4 grp 5 sgrp 6 d_near 7 y_near 8 r_near 9 dist 10 K 11 seen
allh=list(range(nh)); alls=S(*range(12))
variants={
 'H: nearest-floor value only (ref, gaps)':[40,41,42],
 'H: same-xy values on all floors':slot(0)+slot(2)+[40,41,42],
 'H: + neighbourhood means':slot(0)+slot(1)+slot(2)+[40,41,42],
 'H: + AP vertical profile (full H)':allh,
 'HS: per-AP S differences only':allh+S(0,1,2,6,7,8,9,10,11),
 'HS: pooled S differences only':allh+S(3,4,5,10),
 'HS: full':allh+alls}
X=np.hstack([Xh,Xs]); rows=[]
for name,cols in variants.items():
    cols=np.array(cols); m=V.Model(X[:,cols],y)
    res={}
    for e in man['events']:
        b=int(e['building'][1]); t=int(e['floor'].split('F')[1])
        H,Sr,Q=C.event_arrays_m(man,e); ctx=np.concatenate([H,Sr]); aps=np.flatnonzero(C.DET[ctx].any(0)); bd=C.bounds_of(C.XY[ctx])
        Xc,F,ref=V.hist_feats(b,t); Fe=F[:,aps]; n,A,_=Fe.shape
        Z=np.hstack([Fe.reshape(n*A,-1),V.supp_feats(b,t,Sr)[:,aps].reshape(n*A,-1)])[:,cols]
        mp=np.clip(m.predict(Z),T.ND,0).reshape(n,A)
        res[e['id']]=C.event_error_m(man,e,V.read_knn(Xc,mp,Sr,Q,aps,bd))
    s=C.summarize_m(man,res); rows.append(dict(variant=name,mean=s['mean'],**{f'K{a}':x for a,x in s['by_K'].items()},**s['by_floor'])); print(rows[-1]['variant'],round(s['mean'],2),flush=True)
print(pd.DataFrame(rows).round(2).to_string())
