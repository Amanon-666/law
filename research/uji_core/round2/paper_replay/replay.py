"""Replay manuscript Table II and the two original UJI K-curves on unchanged deployment code.
No model selection and no access to reserved datasets. Input: two checksum-verified UCI CSVs.
Run from root: python -m research.uji_core.round2.paper_replay.replay CSV_DIR OUTPUT_DIR
"""
import argparse, ast, hashlib, json
from pathlib import Path
import numpy as np
from research.uji_core.protocol import read_floors, position_weights, HASHES
from research.uji_core.legacy_wknn_audit import split_rows as legacy_prefix
from data import group_scans
from refi import fit_map


def legacy_voter():
    # Execute only the original pure numpy reference; avoid importing the unrelated GPU simulator.
    root=Path(__file__).resolve().parents[4]
    tree=ast.parse((root/'full_ap_study.py').read_text())
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='wknn')
    scope={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'legacy-wknn','exec'),scope)
    return scope['wknn']


def sub(t,i):return {k:v[i] for k,v in t.items()}
def hashids(data):return hashlib.sha256(np.asarray(data,dtype='<i8').tobytes()).hexdigest()

def evaluate(h,s,q,gaps,vote):
    mask=np.logical_or.reduce([(t['rss']!=100).any(0) for t in [*h,s]])
    merged={k:np.concatenate([t[k] for t in [*h,s]]) for k in ('xy','rss')}
    merged['rss']=merged['rss'].astype(np.int64)
    w=position_weights(q['xy']); aps=np.flatnonzero(mask)
    scores={}; preds={}
    # Match integer precision and original NumPy argsort tie rule.
    for label,bank in [('wknn_hs',merged),('wknn_s',s)]:
        pred=np.concatenate([vote(bank, {'rss':q['rss'][i:i+16].astype(np.int64)},aps,False) for i in range(0,len(w),16)])
        preds[label]=pred
    for label,cfg in [('core_equal',{}),('core_gap',{'floor_gaps':gaps})]:
        model=fit_map(h,s,**cfg);preds[label]=model.predict(q['rss'])
    for label,p in preds.items():
        e=np.linalg.norm(p-q['xy'],axis=1); assert np.isfinite(e).all()
        scores[label]=float(w@e)
    return scores,preds


def main(data,out):
    out.mkdir(exist_ok=True,parents=True);train=read_floors(data,'trainingData.csv');valid=read_floors(data,'validationData.csv');vote=legacy_voter();rows=[];ids=[];preds={}
    for f in (1,2):
        t=train[1,f];q=valid[1,f];h=[train[1,0],train[1,3]]; group=group_scans(t['rss'],t['xy']);g=group['group']
        for seed in (0,1,2):
            rng=np.random.default_rng(seed)
            loc=np.sort(rng.choice(np.flatnonzero(group['stats']['n']>=3),5,replace=False))
            sr=np.concatenate([rng.choice(np.flatnonzero(g==p),3,replace=False) for p in loc]);s=sub(t,sr)
            for name,qs in [('internal',sub(t,~np.isin(g,loc))),('official',sub(q,~(q['xy'][:,None]==group['positions'][loc][None]).all(-1).any(-1)))]:
                assert len(np.unique(s['xy'],axis=0))==5 and len(s['xy'])==15
                scores,pp=evaluate(h,s,qs,[f,3-f],vote)
                key=f'B1F{f}_{seed}_{name}';preds[key+'_query_ids']=qs['ids']+1
                for label,p in pp.items():preds[key+'_'+label]=p
                ids.append(dict(key=key,support_1based=(s['ids']+1).tolist(),query_ids_sha256=hashids(qs['ids']+1),H_sha256=[hashids(x['ids']+1) for x in h],aps=int(np.logical_or.reduce([(x['rss']!=100).any(0) for x in [*h,s]]).sum()),query_scans=len(qs['xy']),query_positions=len(np.unique(qs['xy'],axis=0))))
                rows.append(dict(floor=f,seed=seed,split=name,**scores))
    means={a:{key:float(np.mean([r[key] for r in rows if r['split']==a])) for key in ('wknn_hs','wknn_s','core_equal','core_gap')} for a in ('internal','official')}
    target={'internal':{'wknn_hs':14.202,'wknn_s':37.605,'deepsets':10.661},'official':{'wknn_hs':18.434,'wknn_s':34.706,'deepsets':11.586}}
    parity={a:{m:abs(means[a][m]-target[a][m])<.00051 for m in ('wknn_hs','wknn_s')} for a in target}
    curves=[]
    for b,f in [(0,1),(2,2)]:
        t=train[b,f];other=sorted(g for bb,g in train if bb==b and g!=f);h=[train[b,g] for g in other]
        for seed in (261026,261027,261028):
            order,qq=legacy_prefix(t['xy'],seed);q=sub(t,qq)
            for k in (1,3,5,10,20):
                s=sub(t,order[:k]);v,_=evaluate(h,s,q,[abs(f-g) for g in other],vote)
                curves.append(dict(building=b,seed=seed,K=k,Sid=hashids(s['ids']+1),Qid=hashids(q['ids']+1),**v))
    compact={str(b):{k:[round(float(np.mean([r[k] for r in curves if r['building']==b and r['K']==n])),5) for n in (1,3,5,10,20)] for k in ('wknn_hs','core_gap')} for b in (0,2)}
    record=dict(dataset_sha256=HASHES,version=np.__version__,seed_basis='uji_history_bank_study.prepare; legacy_wknn_audit.split_rows',tableII=means,paper=target,rounded_reference_parity=parity,events=rows,manifest=ids,curves=curves,curve_means=compact,scope='Existing estimator, development replay, no new training. Attribution of learning NOT tested. No per-event DeepSets predictions provided.')
    (out/'replay.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');np.savez_compressed(out/'predictions.npz',**preds)
    print('TABLEII',json.dumps(means));print('FOUR_CHECKS',json.dumps(parity));print('CURVES',json.dumps(compact));print('EVENTS',json.dumps(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.data,a.out)
