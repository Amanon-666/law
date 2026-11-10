"""Development fold (B0 targets, translator trained on B1+B2): choose training-set construction. No B1/B2 target scores."""
import sys, time; sys.path.insert(0,'/home/claude/work/vt')
import numpy as np, pandas as pd, core as C, vt_lib as V, translator as T
man=C.DEV
def evaluate(models):
    res={}
    for e in man['events']:
        b=int(e['building'][1]); t=int(e['floor'].split('F')[1])
        H,S,Q=C.event_arrays_m(man,e); ctx=np.concatenate([H,S]); aps=np.flatnonzero(C.DET[ctx].any(0)); bd=C.bounds_of(C.XY[ctx])
        for name,(m,ws) in models.items():
            Xc,mp=V.translate(m,b,t,S,aps,ws)
            res.setdefault(name,{})[e['id']]=C.event_error_m(man,e,V.read_knn(Xc,mp,S,Q,aps,bd))
        if 'raw3NN' not in res or e['id'] not in res['raw3NN']:
            res.setdefault('raw3NN',{})[e['id']]=C.event_error_m(man,e,C.nearest3(C.RSS[ctx],C.XY[ctx],C.RSS[Q],aps,bd))
    return res
models={}
for name,kw in (('H_only',dict(with_support=False,draws=1)),
                ('HS_d2_mixed',dict(draws=2,single_acquisition=False,max_rows=40000)),
                ('HS_d8_mixed',dict(draws=8,single_acquisition=False)),
                ('HS_d8_single',dict(draws=8,single_acquisition=True))):
    t0=time.time(); rng=np.random.default_rng(20261110)
    Xh,Xs,y=V.training_set([1,2],rng,**kw)
    ws=kw.get('with_support',True)
    models[name]=(V.Model(np.hstack([Xh,Xs]) if ws else Xh,y),ws)
    print(name,len(y),'rows',round(time.time()-t0),'s',flush=True)
res=evaluate(models)
rows=[]
for k,v in res.items():
    s=C.summarize_m(man,v); rows.append(dict(method=k,mean=s['mean'],**{f'K{a}':x for a,x in s['by_K'].items()},**s['by_floor']))
print(pd.DataFrame(rows).round(2).to_string())
