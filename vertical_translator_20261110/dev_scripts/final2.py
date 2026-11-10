"""Seed variability and ensembling on the 54 events (configuration unchanged from final1)."""
import sys, time, json; sys.path.insert(0,'/home/claude/work/vt')
import numpy as np, pandas as pd, core as C, vt_lib as V, translator as T
SEEDS=(1,2,3,4,5); out={}
for tag,train in (('lobo',{1:[0,2],2:[0,1]}),('b0only',{1:[0],2:[0]})):
    maps={}; res={}
    for seed in SEEDS:
        models={}
        for hb,tr in train.items():
            rng=np.random.default_rng(seed)
            Xh,Xs,y=V.training_set(tr,rng,draws=8,single_acquisition=False)
            models[hb]=(V.Model(Xh,y),V.Model(np.hstack([Xh,Xs]),y))
        for e in C.MAN['events']:
            b=int(e['building'][1]); t=int(e['floor'].split('F')[1])
            H,S,Q=C.event_arrays(e); ctx=np.concatenate([H,S]); aps=np.flatnonzero(C.DET[ctx].any(0)); bd=C.bounds_of(C.XY[ctx])
            Xc,mH=V.translate(models[b][0],b,t,S,aps,False); Xc,mHS=V.translate(models[b][1],b,t,S,aps,True)
            for k,mp in (('H',mH),('HS',mHS)):
                res.setdefault(f'{k}_seed{seed}',{})[e['id']]=C.event_error(e,V.read_knn(Xc,mp,S,Q,aps,bd))
                maps.setdefault((k,e['id']),[]).append(mp)
        print(tag,'seed',seed,{k:round(C.summarize(res[f'{k}_seed{seed}'])['mean'],3) for k in ('H','HS')},flush=True)
    for e in C.MAN['events']:
        b=int(e['building'][1]); t=int(e['floor'].split('F')[1])
        H,S,Q=C.event_arrays(e); ctx=np.concatenate([H,S]); aps=np.flatnonzero(C.DET[ctx].any(0)); bd=C.bounds_of(C.XY[ctx])
        Xc=V.candidates(b,t)
        for k in ('H','HS'):
            res.setdefault(f'{k}_ensemble5',{})[e['id']]=C.event_error(e,V.read_knn(Xc,np.mean(maps[(k,e['id'])],0),S,Q,aps,bd))
    rows=[]
    for k in ('H_ensemble5','HS_ensemble5'):
        s=C.summarize(res[k]); rows.append(dict(train=tag,method=k,mean=s['mean'],**{f'K{a}':x for a,x in s['by_K'].items()},**s['by_floor']))
    print(pd.DataFrame(rows).round(2).to_string(),flush=True)
    for k in ('H','HS'):
        v=[C.summarize(res[f'{k}_seed{s}'])['mean'] for s in SEEDS]; print(tag,k,'seeds mean %.3f sd %.3f min %.3f max %.3f'%(np.mean(v),np.std(v),min(v),max(v)),flush=True)
    out[tag]=res
json.dump(out,open('/home/claude/work/vt/final2.json','w'))
