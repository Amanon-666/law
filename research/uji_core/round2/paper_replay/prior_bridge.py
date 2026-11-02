"""B1 Table II causal bridge, frozen H/S and fixed score for every empirical prototype.
Use original CSV float decoding, original 1-based row convention, equal-position score.
A literal 2-U manuscript rectangular grid inherits each nearest prototype's score; this tests
only the coordinate prior/quadrature, never claims to have reproduced a continuous GP.
"""
import argparse, ast, csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from data import group_scans
from external_baselines import tpm_adapter as tpm
from refi import fit_map
from .replay import hashids,legacy_voter
from research.uji_core.protocol import HASHES,position_weights

def load(data,filename):
    path=data/filename;assert hashlib.sha256(path.read_bytes()).hexdigest()==HASHES[filename];floors={}
    with path.open(newline='') as stream:
        for idx,row in enumerate(csv.DictReader(stream),1):
            if int(row['BUILDINGID'])!=1:continue
            f=int(row['FLOOR']);floors.setdefault(f,[]).append((idx,[int(row[f'WAP{a:03}']) for a in range(1,521)],[float(row['LONGITUDE']),float(row['LATITUDE'])]))
    ans={}
    for f,v in floors.items():
        ids,rss,xy=map(np.array,zip(*v));ans[f]=dict(group_scans(rss,xy),ids=ids)
    return ans

def calc(data,out):
    t=load(data,'trainingData.csv');v=load(data,'validationData.csv');h=[t[0],t[3]];wk=legacy_voter()
    origin=np.concatenate([t[i]['positions'] for i in (0,1,2,3)]).mean(0);events=[]
    for floor in (1,2):
        a,q=t[floor],v[floor]
        for seed in (0,1,2):
            rng=np.random.default_rng(seed);chosen=np.sort(rng.choice(np.flatnonzero(a['stats']['n']>=3),5,replace=False));sr=np.concatenate([rng.choice(np.flatnonzero(a['group']==i),3,replace=False) for i in chosen]);s={k:a[k][sr] for k in ('rss','xy')}
            model=fit_map(h,s,floor_gaps=[floor,3-floor]);xy=model.xy-origin;unique,inv,counts=np.unique(xy,axis=0,return_inverse=True,return_counts=True);inv=inv.ravel()
            axes=[np.linspace(u.min(),u.max(),int(np.ceil((u.max()-u.min())/2))+1) for u in unique.T];grid=np.stack(np.meshgrid(*axes),axis=-1).reshape(-1,2)
            # Voronoi ties to lexicographically first unique prototype; coincident fingerprints share its mass.
            assign=np.concatenate([cdist(b,unique,'sqeuclidean').argmin(1) for b in np.array_split(grid,16)]);cnt=np.bincount(assign,minlength=len(unique));cent=unique.copy();acc=np.zeros_like(cent);np.add.at(acc,assign,grid);cent[cnt>0]=acc[cnt>0]/cnt[cnt>0,None]
            bank={k:np.concatenate([z[k] for z in [*h,s]]) for k in ('rss','xy')};bank['xy']-=origin;aps=np.flatnonzero(model.visible)
            for name,tr,idx in [('internal',a,np.flatnonzero(~np.isin(a['group'],chosen))),('official',q,np.flatnonzero(~(q['xy'][:,None]==a['positions'][chosen][None]).all(-1).any(-1)))]:
                w=position_weights(tr['xy'][idx]-origin);Q=tr['rss'][idx];truth=tr['xy'][idx]-origin;prob=tpm.tobit_weights(Q[:,aps],model.prototypes,12.,-85.);score={}
                for key,mass,coords in [('count',np.ones(len(xy)),xy),('site',1./counts[inv],xy),('area',cnt[inv]/counts[inv],xy),('grid_centroid',cnt[inv]/counts[inv],cent[inv])]:
                    # Normalize in log-space independently, so a strongly peaked score with zero cell mass causes no underflow.
                    if np.min(mass)<=0:
                        keep=mass>0;pr=tpm.tobit_weights(Q[:,aps],model.prototypes[keep],12.,-85.);p=pr*mass[None,keep];pred=p@coords[keep]/p.sum(1,keepdims=True)
                    else:
                        p=prob*mass;pred=p@coords/p.sum(1,keepdims=True)
                    score[key]=float(w@np.linalg.norm(pred-truth,axis=1))
                pred=np.concatenate([wk(bank,{'rss':Q[i:i+16]},aps,False) for i in range(0,len(Q),16)]);score['original_wknn']=float(w@np.linalg.norm(pred-truth,axis=1));assert np.isfinite(list(score.values())).all()
                # Audit whether a different coordinate can occupy a tied boundary slot without changing distances.
                r=np.where(bank['rss'][:,aps]==100,-110,bank['rss'][:,aps]).astype(float);qq=np.where(Q[:,aps]==100,-110,Q[:,aps]);dist=cdist(qq,r,'sqeuclidean');cut=np.partition(dist,2,axis=1)[:,2];amb=[]
                for i,line in enumerate(dist):
                    ties=np.flatnonzero(line==cut[i]);free=3-int((line<cut[i]).sum());amb.append(len(ties)>free and len(np.unique(bank['xy'][ties],axis=0))>1)
                events.append(dict(floor=floor,seed=seed,split=name,S=hashids(a['ids'][sr]),Q=hashids(tr['ids'][idx]),grid=len(grid),unique=len(unique),max_cell_fraction=float(cnt.max()/len(grid)),tie_rows=int(sum(amb)),tie_position_mass=float(w@amb),**score))
    summary={m:{k:float(np.mean([r[k] for r in events if r['split']==m])) for k in ('original_wknn','count','site','area','grid_centroid','tie_position_mass')} for m in ('internal','official')};out.mkdir(exist_ok=True,parents=True);res=dict(numpy=np.__version__,summary=summary,events=events);(out/'bridge.json').write_text(json.dumps(res,indent=2,allow_nan=False));print('PRIOR_BRIDGE',json.dumps(summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);a=p.parse_args();calc(a.data,a.out)
