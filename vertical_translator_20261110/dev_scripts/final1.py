"""Final run on the project's 54 events with the configuration fixed on the B0 development fold:
single-stage translator, 8 support draws per K per source floor, project's 3NN reader."""
import sys, time, json; sys.path.insert(0,'/home/claude/work/vt')
import numpy as np, pandas as pd, core as C, vt_lib as V, translator as T
out={}
for tag,train in (('lobo',{1:[0,2],2:[0,1]}),('b0only',{1:[0],2:[0]})):
    models={}
    for hb,tr in train.items():
        rng=np.random.default_rng(20261110); t0=time.time()
        Xh,Xs,y=V.training_set(tr,rng,draws=8,single_acquisition=False)
        models[hb]=(V.Model(Xh,y),V.Model(np.hstack([Xh,Xs]),y)); print(tag,'held-out B%d'%hb,'train',tr,len(y),'rows',round(time.time()-t0),'s',flush=True)
    res={}; preds={}
    for e in C.MAN['events']:
        b=int(e['building'][1]); t=int(e['floor'].split('F')[1])
        H,S,Q=C.event_arrays(e); ctx=np.concatenate([H,S]); aps=np.flatnonzero(C.DET[ctx].any(0)); bd=C.bounds_of(C.XY[ctx])
        Xc,mH=V.translate(models[b][0],b,t,S,aps,False); Xc,mHS=V.translate(models[b][1],b,t,S,aps,True)
        for k,mp in (('translated_H',mH),('translated_HS',mHS)):
            p=V.read_knn(Xc,mp,S,Q,aps,bd); res.setdefault(k,{})[e['id']]=C.event_error(e,p); preds[(k,e['id'])]=p
    rows=[]
    for k,v in res.items():
        s=C.summarize(v); rows.append(dict(train=tag,method=k,mean=s['mean'],**{f'K{a}':x for a,x in s['by_K'].items()},**s['by_floor']))
    print(pd.DataFrame(rows).round(2).to_string(),flush=True)
    out[tag]=res
    np.savez_compressed(f'/home/claude/work/vt/final1_{tag}_pred.npz',**{f'{k}__{i}':p for (k,i),p in preds.items()})
json.dump(out,open('/home/claude/work/vt/final1.json','w'))
