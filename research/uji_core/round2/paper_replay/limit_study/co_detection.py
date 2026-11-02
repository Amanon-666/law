"""Source-only count covariance; no parameter tuned in B1, no neural head or extra target data."""
import argparse,csv,json,itertools,hashlib
from pathlib import Path
import numpy as np
from scipy.special import ndtr,softmax
from .model_ceiling import BASE,score
from ..prior_bridge import load
from research.uji_core.protocol import HASHES,position_weights

def fit(d):
    path=d/'trainingData.csv';assert hashlib.sha256(path.read_bytes()).hexdigest()==HASHES['trainingData.csv'];groups={}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            b=int(row['BUILDINGID']);
            if b not in (0,2):continue
            key=(b,int(row['FLOOR']),float(row['LONGITUDE']),float(row['LATITUDE']),int(row['PHONEID']));stamp=int(row['TIMESTAMP']);g=groups.setdefault(key,{})
            if stamp not in g:g[stamp]=np.array([row[f'WAP{i:03}']!='100' for i in range(1,521)],dtype=bool)
    floors={}
    for key,sub in groups.items():
        n=len(sub)
        if n<5:continue
        a=np.stack(list(sub.values()));p=a.mean(0);v=p*(1-p)*n/(n-1);V=a.sum(1).var(ddof=1);floors.setdefault(key[:2],[]).append([V-v.sum(),max(0.,v.sum()**2-v@v)])
    m=np.mean([np.mean([np.mean(a,axis=0) for (bb,f),a in floors.items() if bb==b],axis=0) for b in (0,2)],axis=0); assert m[1]>0
    return dict(c=float(np.sqrt(np.clip(m[0]/m[1],0,1))),num=float(m[0]),den=float(m[1]),unbounded_ratio=float(m[0]/m[1]),cells={f'{b}:{f}':len(a) for (b,f),a in floors.items()},source_csv=HASHES['trainingData.csv'])

def adjust(D,p,c):
    if c==0:return 0.
    ell=[]
    for s in [-1,1]:
        miss=np.log1p(-s*c*p);hit=np.log1p(s*c*(1-p));ell.append(miss.sum(0)+D@(hit-miss))
    return np.logaddexp(*ell)-np.log(2.)

def unit():
    p=np.array([[.01,.3],[.4,.4],[.9,.7]]);D=np.array(list(itertools.product([0.,1.],repeat=3)));B=D@np.log(p)+(1-D)@np.log1p(-p)
    for c in [0,.5,1]:
        q=np.exp(B+adjust(D,p,c));np.testing.assert_allclose(q.sum(0),1.,atol=1e-12);np.testing.assert_allclose(D.T@q,p,atol=1e-12)
        d=p*(1-p);cv=d.sum(0)+c*c*(d.sum(0)**2-(d*d).sum(0));np.testing.assert_allclose(((D.sum(1)[:,None]-p.sum(0))**2*q).sum(0),cv,atol=1e-12)

def run(data,fi,out):
    out.mkdir(exist_ok=True,parents=True);unit();r=fit(data);(out/'codetect_source.json').write_text(json.dumps(r,indent=2));cs=dict(plain=0.,fixed_half=.5,fixed_one=1.,source_c=r['c']);obs=json.loads((BASE/'field_parameters.json').read_text())['0']['obs'];tr=load(data,'trainingData.csv');va=load(data,'validationData.csv');records=[];save={}
    for f in (1,2):
        for sd in range(3):
            eid=f'B1F{f}_{sd}';g=np.load(next(fi.rglob(eid+'_field.npz')));ref=np.load(next(fi.rglob(eid+'_pred.npz')));mu=g['mu'];v=g['variance'];ap=g['aps'];prob=obs['alpha']*ndtr((mu-obs['beta'])/np.sqrt(v+obs['detect_scale']**2));assert prob.min()>1e-15 and prob.max()<1.-1e-15
            for sp,t in [('internal',tr[f]),('official',va[f])]:
                ids=ref[sp+'_ids'];idx=np.searchsorted(t['ids'],ids);assert np.array_equal(t['ids'][idx],ids);xy=t['xy'][idx];assert np.array_equal(xy,ref[sp+'_xy']);q=t['rss'][idx][:,ap];L=score(q,mu,v,obs);w=position_weights(xy);D=(q!=100).astype(float);ev={};key=eid+'_'+sp;save[key+'_ids']=ids;save[key+'_xy']=xy
                for arm,c in cs.items():
                    pred=softmax(L+adjust(D,prob,c),axis=1)@g['grid']+g['origin'];assert np.isfinite(pred).all();
                    if arm=='plain':assert np.max(abs(pred-ref[sp+'_full']))<1e-6
                    save[key+'_'+arm]=pred;ev[arm]=float(w@np.linalg.norm(pred-xy,axis=1))
                records.append(dict(event=eid,split=sp,mde=ev))
    summary={sp:{a:float(np.mean([e['mde'][a] for e in records if e['split']==sp])) for a in cs} for sp in ['internal','official']};(out/'co_detection.json').write_text(json.dumps(dict(source=r,summary=summary,events=records),indent=2));np.savez_compressed(out/'codetect_pred.npz',**save);print('CODETECT',r,summary)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('fields',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.data,a.fields,a.out)
