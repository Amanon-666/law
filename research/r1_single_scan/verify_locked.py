"""Read-only independent verifier for the original R1 single-scan draw-0 artifacts.

Run: python -m research.r1_single_scan.verify_locked /path/to/flat/artifact/directory
Requires NumPy only; never trains, tunes, selects winners, or accesses test data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

FLOORS=range(4)
REGIMES=('official','phone')
KS=(1,3,5)
ARMS=('gp','seven','nonlinear')
SHA={
    'model_source_sha256':'53061b01164f78def406e7b1395c5e62a295adb9a9a3570a2b91efed7b71e7c7',
    'affine_source_sha256':'cd5c4972f4ad418a58c9c73f88d9c65c245f24ecf5c35842bcdc8535114caf58',
    'manifest_sha256':'5f4fb0ceccc5a64aa7411cc287715e6fa9a39d593da6713910920a3fda67c613',
    'device_design_sha256':'76763a87569d885705dccc54cbd72e878f7a711cbb5343fbfee4517caf2482e6',
}
OFFICIAL_ROWS=(30,143,87,47)

def digest_ids(ids):
    return hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()

def metric(pred,true):
    unique,ix=np.unique(true,axis=0,return_inverse=True)
    d=np.linalg.norm(pred-true,axis=1)
    w=1/(len(unique)*np.bincount(ix)[ix])
    order=np.argsort(d)
    return float(w@d),float(d[order][np.searchsorted(np.cumsum(w[order]),.9)])

def load(path,regime,f):
    key=f'{regime}_F{f}_0'
    j=json.loads((path/(key+'.json')).read_text())
    with np.load(path/(key+'.npz'),allow_pickle=False) as z:
        n={k:z[k] for k in z.files}
    assert j['regime']==regime and j['floor']==f and j['building']==1 and j['draw']==0
    assert j['R']==1 and j['three_history']
    assert j['historical_floor_ids']==[a for a in FLOORS if a!=f]
    for k,v in SHA.items():assert j[k]==v,(key,k)
    assert j['query_count']==len(n['query_ids'])==len(n['query_xy'])
    assert j['query_rows_sha256']==digest_ids(n['query_ids'])
    if regime=='official':assert j['query_count']==OFFICIAL_ROWS[f]
    else:assert j['query_count']==j['query_positions']==12
    assert [e['K'] for e in j['results']]==list(KS)
    order=[]
    for e in j['results']:
        k=e['K']
        assert len(e['support_ids'])==k and e['support_hash']==digest_ids(e['support_ids'])
        order.append(e['support_ids'])
        assert len(e['weights'])==3 and min(e['weights'])>=0 and abs(sum(e['weights'])-1)<1e-7
        assert e['grid']>0 and e['ap_count']>0 and set(e['metrics'])==set(ARMS)
        assert min(e['min_ess'].values())>=1000
        if regime=='phone':assert set(e['support_ids']).isdisjoint(n['query_ids'].tolist())
        for a in ARMS:
            pred=n[f'K{k}_{a}']
            assert pred.shape==n['query_xy'].shape and np.isfinite(pred).all()
            md,p90=metric(pred,n['query_xy'])
            assert abs(md-e['metrics'][a]['mde'])<1e-8,(key,k,a,md,e['metrics'][a])
            assert abs(p90-e['metrics'][a]['p90'])<1e-8,(key,k,a,p90,e['metrics'][a])
    assert order[0]==order[1][:1] and order[1]==order[2][:3]
    return j,n

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    found={};missing=[];invalid=[]
    for regime in REGIMES:
        for f in FLOORS:
            key=f'{regime}_F{f}_0'
            if not all((args.directory/(key+ext)).exists() for ext in ('.json','.npz')):
                missing.append(key);continue
            try: found[key]=load(args.directory,regime,f)
            except Exception as e:invalid.append((key,repr(e)))
    result=dict(scope='Eight fixed draw-0 R1 events (24 K cells), three histories, R=1; a partial shard is not a full R1 curve',
                expected=8,verified=len(found),missing=missing,invalid=invalid,summary={})
    for regime in REGIMES:
        keys=[f'{regime}_F{f}_0' for f in FLOORS if f'{regime}_F{f}_0' in found]
        reg={}
        for k in KS:
            reg[f'K{k}']={
                'floor_count':len(keys),
                'mde':{a:float(np.mean([found[key][0]['results'][KS.index(k)]['metrics'][a]['mde'] for key in keys])) for a in ARMS} if keys else {},
                'p90':{a:float(np.mean([found[key][0]['results'][KS.index(k)]['metrics'][a]['p90'] for key in keys])) for a in ARMS} if keys else {},
                'per_floor':{key:{a:found[key][0]['results'][KS.index(k)]['metrics'][a]['mde'] for a in ARMS} for key in keys},
                'archived_wknn':{key:found[key][0]['results'][KS.index(k)]['wknn_archived'] for key in keys},
            }
        result['summary'][regime]=reg
    out=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    print(out)
    if args.output:args.output.write_text(out)
    if invalid:raise SystemExit('Invalid or misaligned outputs; do not report aggregate')
    if missing:raise SystemExit('Partial event set; do not report as complete 24-cell R1 result')

if __name__=='__main__':main()
