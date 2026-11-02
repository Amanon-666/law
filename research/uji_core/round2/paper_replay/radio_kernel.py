"""Replace only the residual propagation kernel; unchanged history, anchors, likelihood, candidates and readout.
RBF in XY times an RBF of the normalized history spectrum. One gamma learned on B0/B2 INTERNAL data;
one same-source spatial length as matched competitor. No B1 query fits, new query data or held-out sites.
All candidate grids fixed in this file prior to the run. B1 is a previously inspected development benchmark.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from refi import historical_prior,fit_map
from data import group_scans
from external_baselines import tpm_adapter as tpm
from research.uji_core.protocol import HASHES,position_weights

GAMMA=(0.,1.,4.,16.);LENGTH=(10.,20.,40.,80.);REF='radio_0'

def load(path,filename,buildings):
    f=path/filename;assert hashlib.sha256(f.read_bytes()).hexdigest()==HASHES[filename];groups={}
    with f.open(newline='') as stream:
        for i,r in enumerate(csv.DictReader(stream),1):
            b=int(r['BUILDINGID']);k=(b,int(r['FLOOR']))
            if b in buildings:groups.setdefault(k,[]).append((i,[int(r[f'WAP{a:03}']) for a in range(1,521)],[float(r['LONGITUDE']),float(r['LATITUDE'])]))
    ans={}
    for key,values in groups.items():
        ids,rss,xy=map(np.array,zip(*values));ans[key]=dict(group_scans(rss,xy),ids=ids)
    return ans

def feat(p):
    a=p+110.;return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)

def episode(t,h,f,gaps,seed,official,config,tag):
    rng=np.random.default_rng(seed);loc=np.sort(rng.choice(np.flatnonzero(t['stats']['n']>=3),5,replace=False));sr=np.concatenate([rng.choice(np.flatnonzero(t['group']==i),3,replace=False) for i in loc]);s={k:t[k][sr] for k in ('xy','rss')};mask=np.logical_or.reduce([(x['rss']!=100).any(0) for x in [*h,s]]);hist=[dict(xy=x['xy'],rss=x['rss'][:,mask]) for x in h];pm,x=historical_prior(hist,10.,.5,gaps);sy,sx=tpm.position_map(tpm.dbm(s['rss'][:,mask]),s['xy']);ms=tpm.lookup(sx,x,pm);r=sy-ms;ph,ps=feat(pm),feat(ms);space_h=cdist(x,sx,'sqeuclidean');space_s=cdist(sx,sx,'sqeuclidean');radio_h=cdist(ph,ps,'sqeuclidean');radio_s=cdist(ps,ps,'sqeuclidean');grid=np.vstack((x,sx));c=grid[0];rows=[]
    # Sham preserves the history field itself; permutes only the location-to-profile relation used by the kernel.
    shuf=np.random.default_rng(15622).permutation(len(x));wrongh=feat(pm[shuf]);wrongs=feat(tpm.lookup(sx,x,pm[shuf]));sh=cdist(wrongh,wrongs,'sqeuclidean');ss=cdist(wrongs,wrongs,'sqeuclidean')
    arms={};cache={};refmap=fit_map(h,s,floor_gaps=gaps)
    for name,(l,g,perm) in config.items():
        key=l,g,perm
        if key not in cache:
            cross=np.exp(-.5*(space_h/l**2+g*(sh if perm else radio_h)));K=np.exp(-.5*(space_s/l**2+g*(ss if perm else radio_s)));assert np.linalg.eigvalsh(K).min()>-1e-8
            ym=pm+cross@np.linalg.solve(K+np.eye(len(sx)),r);cache[key]=np.vstack((np.clip(ym,-110.,0.),sy))
        arms[name]=cache[key]
    np.testing.assert_allclose(arms[REF],refmap.prototypes,atol=1e-10,rtol=0);np.testing.assert_array_equal(grid,refmap.xy)
    ii=np.flatnonzero(~np.isin(t['group'],loc));qs={'internal':(t,ii)}
    if official is not None:
        oi=np.flatnonzero(~(official['xy'][:,None]==sx[None]).all(-1).any(-1));qs['official']=(official,oi)
    for split,(q,idx) in qs.items():
        xy=q['xy'][idx];rss=q['rss'][idx][:,mask];w=position_weights(xy);errs={};
        for name,p in arms.items():
            pred=tpm.tobit_weights(rss,p,12.,-85.)@(grid-c)+c;errs[name]=float(w@np.linalg.norm(pred-xy,axis=1))
        assert np.isfinite(list(errs.values())).all()
        rows.append(dict(f=f,tag=tag,seed=seed,split=split,S=t['ids'][sr].tolist(),Q_sha256=hashlib.sha256(np.asarray(q['ids'][idx],dtype='<i8').tobytes()).hexdigest(),errors=errs))
    return rows

def main(d,out):
    out.mkdir(parents=True,exist_ok=True);source=load(d,'trainingData.csv',{0,2});src=[];config={f'radio_{x:g}':(40.,x,False) for x in GAMMA};config.update({f'xy_{x:g}':(x,0.,False) for x in LENGTH})
    for b in (0,2):
        floors=sorted(f for bb,f in source if bb==b);low,hi=floors[0],floors[-1];h=[source[b,low],source[b,hi]]
        for f in floors[1:-1]:
            for seed in range(3):src+=episode(source[b,f],h,f,[f-low,hi-f],seed,None,config,b)
    risk={n:float(np.mean([np.mean([a['errors'][n] for a in src if a['tag']==b]) for b in (0,2)])) for n in config};radio=min([n for n in config if n.startswith('radio')],key=lambda n:risk[n]);geo=min([n for n in config if n.startswith('xy')],key=lambda n:risk[n]);l,g,_=config[radio]
    freeze=dict(source='B0/B2 training-file internal only, equal building, then uniform interior floors and draws',train_hash=HASHES['trainingData.csv'],risk=risk,radio=radio,geo=geo,config=config,gate='Both B1 Table-II means strictly below DeepSets 10.661/11.586 AND learned radio below same-source spatial-only and base in both splits. No holdout if fail. Include fixed gamma 1/4/16 as descriptive controls, do not select on B1.')
    (out/'kernel_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');t=load(d,'trainingData.csv',{1});v=load(d,'validationData.csv',{1});h=[t[1,0],t[1,3]];res=[];config.update(selected_radio=(l,g,False),selected_xy=config[geo],permuted_radio=(l,g,True))
    for f in (1,2):
        for seed in range(3):res+=episode(t[1,f],h,f,[f,3-f],seed,v[1,f],config,1)
    means={sp:{n:float(np.mean([a['errors'][n] for a in res if a['split']==sp])) for n in config} for sp in ('internal','official')}
    gate=all(a['selected_radio']<min([paper,a['selected_xy'],a[REF]]) for a,paper in zip([means['internal'],means['official']],[10.661,11.586]));record=dict(freeze=freeze,source_rows=src,target_rows=res,means=means,gate=bool(gate));(out/'kernel.json').write_text(json.dumps(record,indent=2)+'\n');print('RADIO_LAW',radio,geo,means,gate)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.data,a.out)
