"""Fresh, pre-declared seeds 11..20 (never run before): per-seed results, two disjoint 5-seed ensembles, 10-seed ensemble.
Configuration identical to final2 / run_nine_floor.py."""
import sys, json; sys.path.insert(0,'/home/claude/work/vt')
import numpy as np, pandas as pd, core as C, vt_lib as V
SEEDS=tuple(range(11,21)); out={}
for tag,train in (('b0only',{1:[0],2:[0]}),('lobo',{1:[0,2],2:[0,1]})):
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
                maps.setdefault((k,e['id']),[]).append(mp.astype(np.float32))
        s={k:C.summarize(res[f'{k}_seed{seed}']) for k in ('H','HS')}
        print(tag,'seed',seed,{k:round(v['mean'],3) for k,v in s.items()},'HS byK',{a:round(x,2) for a,x in s['HS']['by_K'].items()},flush=True)
    for name,sl in (('ensA_11_15',slice(0,5)),('ensB_16_20',slice(5,10)),('ens10',slice(0,10))):
        for e in C.MAN['events']:
            b=int(e['building'][1]); t=int(e['floor'].split('F')[1])
            H,S,Q=C.event_arrays(e); ctx=np.concatenate([H,S]); aps=np.flatnonzero(C.DET[ctx].any(0)); bd=C.bounds_of(C.XY[ctx])
            Xc=V.candidates(b,t)
            for k in ('H','HS'):
                res.setdefault(f'{k}_{name}',{})[e['id']]=C.event_error(e,V.read_knn(Xc,np.mean(maps[(k,e['id'])][sl],0),S,Q,aps,bd))
    rows=[]
    for name in ('ensA_11_15','ensB_16_20','ens10'):
        for k in ('H','HS'):
            s=C.summarize(res[f'{k}_{name}']); rows.append(dict(train=tag,method=f'{k}_{name}',mean=s['mean'],**{f'K{a}':x for a,x in s['by_K'].items()},**s['by_floor']))
    print(pd.DataFrame(rows).round(2).to_string(),flush=True)
    for k in ('H','HS'):
        v=[C.summarize(res[f'{k}_seed{s}'])['mean'] for s in SEEDS]; print(tag,k,'10 seeds mean %.3f sd %.3f min %.3f max %.3f'%(np.mean(v),np.std(v,ddof=1),min(v),max(v)),flush=True)
    out[tag]=res
    json.dump(out,open('/home/claude/work/vt/final3.json','w'))
