"""Development fold (B0 targets): single-stage vs two-stage translator, trained on two buildings or one. No B1/B2 scores."""
import sys, time; sys.path.insert(0,'/home/claude/work/vt')
import numpy as np, pandas as pd, core as C, vt_lib as V, translator as T
man=C.DEV; rows=[]
for train in ([1,2],[1],[2]):
    t0=time.time(); rng=np.random.default_rng(20261110)
    Xh,Xs,y=V.training_set(train,rng,draws=8,single_acquisition=False)
    one=V.Model(np.hstack([Xh,Xs]),y)
    ts=V.fit_two_stage(train,np.random.default_rng(20261110))
    print('trained on',train,round(time.time()-t0),'s',flush=True)
    res={}
    for e in man['events']:
        b=int(e['building'][1]); t=int(e['floor'].split('F')[1])
        H,S,Q=C.event_arrays_m(man,e); ctx=np.concatenate([H,S]); aps=np.flatnonzero(C.DET[ctx].any(0)); bd=C.bounds_of(C.XY[ctx])
        Xc,m1=V.translate(one,b,t,S,aps); Xc,pH,m2=V.translate_two_stage(ts,b,t,S,aps)
        for k,mp in (('single_stage_HS',m1),('stage1_H',pH),('two_stage_HS',m2)):
            res.setdefault(k,{})[e['id']]=C.event_error_m(man,e,V.read_knn(Xc,mp,S,Q,aps,bd))
    for k,v in res.items():
        s=C.summarize_m(man,v); rows.append(dict(train=str(train),method=k,mean=s['mean'],**{f'K{a}':x for a,x in s['by_K'].items()},**s['by_floor']))
print(pd.DataFrame(rows).round(2).to_string())
