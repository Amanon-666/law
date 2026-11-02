"""Unzip results/<id>/replication.zip, then python verify.py <unpacked dir>.
Independent of the training backend; all confidence statements are conditional on these sites.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from collections import defaultdict
p=Path(sys.argv[1]);rng=np.random.default_rng(20261102);B=5000;model=['gp','seven','early','separate','wknn'];out={};bs=defaultdict(list)
for line in (p/'SHA256SUMS').read_text().splitlines():
    h,f=line.split(maxsplit=1);assert hashlib.sha256((p/f.strip().lstrip('*')).read_bytes()).hexdigest()==h
nfl=0
for site in ('TIE1','SAH1'):
    d=json.loads((p/(site+'.json')).read_text());fset=d['retained_floors'];assert len(d['events'])==3*len(fset);nfl+=len(fset);floors={};by=defaultdict(list);scores=[];qrows=[]
    for f in fset:
        es=[v for v in d['events'] if v['floor']==f];assert sorted(e['seed'] for e in es)==[0,1,2];blocks=sorted({tuple(i) for e in es for i in e['blocks']});bi={k:j for j,k in enumerate(blocks)};a=np.full((3,len(blocks),len(model)),np.nan)
        for e in es:
            z=np.load(p/f'{site}_F{f:g}_{e["seed"]}.npz',allow_pickle=False);qrows.extend(z['qids'].tolist());assert hashlib.sha256(z['qids'].astype('<i8').tobytes()).hexdigest()==e['qsha'];assert not set(z['qids']).intersection(e['sids']);assert len(set(e['sids']))==5
            _,inv,n=np.unique(z['xy'],axis=0,return_counts=True,return_inverse=True);base=1/n[inv.ravel()];bl,ib=np.unique(z['block'],axis=0,return_inverse=True);ib=ib.ravel();w=base/np.bincount(ib,weights=base)[ib]/len(bl);np.testing.assert_allclose(w,z['roww']);ix=[bi[tuple(b)] for b in bl];row={}
            for j,m in enumerate(model):
                err=np.linalg.norm(z[m]-z['xy'],axis=1);row[m]=float(w@err);np.testing.assert_allclose(row[m],e['metrics'][m]['mde']);v=np.bincount(ib,weights=w*err)*len(bl);np.testing.assert_allclose(v,e['metrics'][m]['blocks']);a[e['seed'],ix,j]=v
            scores.append(dict(floor=f,seed=e['seed'],mde=row))
        floors[str(f)]={m:float(np.mean([s['mde'][m] for s in scores if s['floor']==f])) for m in model};wgt=rng.multinomial(len(blocks),np.ones(len(blocks))/len(blocks),size=B);den=wgt@np.isfinite(a[:,:,0]).T;assert np.all(den>0);pick=rng.integers(3,size=(B,3))
        for cmp in ('early','seven','gp'):
            v=a[:,:,3]-a[:,:,model.index(cmp)];means=(wgt@np.nan_to_num(v).T)/den;by[cmp].append(np.take_along_axis(means,pick,axis=1).mean(1))
    ci={};mean={m:float(np.mean([v[m] for v in floors.values()])) for m in model};cr={m:float(np.mean(v)) for m,v in d['contrast_rmse'].items()}
    for k,v in by.items():
        a=np.mean(v,axis=0);ci[k]=np.quantile(a,[.025,.975]).tolist();bs[k].append(a)
    out[site]=dict(events=scores,floors=floors,mde_m=mean,contrast_rmse_db=cr,n_query_instances=len(qrows),unique_query_rows=len(set(qrows)),ci=ci,p90_m={m:float(np.mean([e['metrics'][m]['p90'] for e in d['events']])) for m in model},pass_site=all(cr['separate']<cr[m] and mean['separate']<=.98*mean[m] for m in ('early','seven')))
out['both_sites_ci']={m:np.quantile(np.mean(v,axis=0),[.025,.975]).tolist() for m,v in bs.items()};out['advance']=nfl>=2 and all(out[s]['pass_site'] for s in ('TIE1','SAH1')) and out['both_sites_ci']['early'][1]<0
print(json.dumps(out,indent=2,allow_nan=False))
