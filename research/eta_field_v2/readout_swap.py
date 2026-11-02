"""Explanatory, paired swap of C and D marginal moments. No new training or selection."""
import sys,json,io,itertools,time,hashlib,zipfile
from pathlib import Path
import numpy as np
import torch
from .study import (B,O,D,Law,banks,center,load,group_scans,hashids,tensor,ap_stats,field_from_prior,observation_log_prob,position_weights,write)
ARMS=('CC','DC','CD','DD')

def run(data,ref,out,f,seed):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);tic=time.perf_counter();out.mkdir(parents=True,exist_ok=True);eid=f'B1F{f}_{seed}';meta=json.loads((ref/(eid+'.json')).read_text());freeze=json.loads((ref/'freeze.json').read_text());e=freeze['selected']['full'];assert e==.5 and meta['eta']==e;r=np.load(ref/(eid+'.npz'),allow_pickle=False);ma={x['key']:x for x in json.loads((D/'results/replay/replay.json').read_text())['manifest']};assert ma[eid+'_internal']['support_1based']==meta['support'];w=meta['w'];source=load(data,'trainingData.csv',{1});t=source[1,f];v=load(data,'validationData.csv',{1})[1,f];hs=[source[1,0],source[1,3]];lu=dict(zip(t['ids'],range(len(t['ids']))));sr=np.array([lu[i] for i in meta['support']]);org,s,h=center(t,hs,sr);sx=tensor(s['positions']);aa=np.flatnonzero(np.logical_or.reduce([(z['rss']!=100).any(0) for z in [*hs,s]]));np.testing.assert_array_equal(aa,meta['aps']);bank=banks(h,aa,B);points=np.vstack([u['positions'] for u in h]+[s['positions']]);axes=[np.linspace(a.min(),a.max(),int(np.ceil(np.ptp(a)/2))+1) for a in points.T];gr=tensor(np.stack(np.meshgrid(*axes),-1).reshape(-1,2));
    qs=[];splits=('internal','official');arrays={}
    for sp,ds in zip(splits,(t,v)):
        ids=r[sp+'_ids'];assert hashids(ids)==ma[eid+'_'+sp]['query_ids_sha256'];ix=np.array([dict_index for dict_index in map(dict(zip(ds['ids'],range(len(ds['ids'])))).__getitem__,ids)]);np.testing.assert_array_equal(ds['xy'][ix],r[sp+'_xy']);qs.append(ds['rss'][ix][:,aa]);arrays[sp+'_ids']=ids;arrays[sp+'_xy']=r[sp+'_xy']
    q=np.vstack(qs);ll={a:np.zeros((len(q),len(gr))) for a in ARMS};ms={a:np.zeros(len(gr)) for a in ARMS};miness=np.inf
    for a,hp in zip(range(len(aa)),bank):
        fields={}
        for k,apply in (('C',False),('D',True)):
            res=field_from_prior(Law(hp,w,B,e,apply),O,sx,ap_stats(s,int(aa[a])),gr,261026+10000*seed+int(aa[a]),defensive_fraction=.1);miness=min(miness,float(res['effective_samples']));fields[k]=(res['mean'].detach(),res['variance'].detach())
        seen=np.flatnonzero(q[:,a]!=100);vals,inv=np.unique(q[seen,a],return_inverse=True)
        for k in ARMS:
            mu,va=fields[k[0]][0],fields[k[1]][1];miss=observation_log_prob(tensor(100.),mu,va,O).numpy();ms[k]+=miss
            if len(seen):ll[k][seen]+=(observation_log_prob(tensor(vals)[:,None],mu,va,O).numpy()-miss)[inv.ravel()]
    assert miness>=1000;parity={}
    for k in ARMS:
        xy=(tensor(ll[k]+ms[k]).softmax(1)@gr).numpy()+org;assert np.isfinite(xy).all();offset=0
        for sp,q0 in zip(splits,qs):
            pred=xy[offset:offset+len(q0)];arrays[sp+'_'+k]=pred;offset+=len(q0)
            if k in ('CC','DD'):
                par=float(np.abs(pred-r[sp+('_mean' if k=='CC' else '_full')]).max());assert par<1e-5,(eid,k,par);parity[k+'_'+sp]=par
    np.savez_compressed(out/(eid+'.npz'),**arrays);write(out/(eid+'.json'),dict(event=eid,parity_max_error=parity,minESS=miness,seconds=time.perf_counter()-tic))

def summary(out):
    res={};rows={};assert len(list(out.glob('B1F*.npz')))==6
    for sp in ('internal','official'):
        M=[];boot=[];dr=np.random.default_rng(20261102)
        for f in (1,2):
            cases=[]
            for i in range(3):
                with np.load(out/f'B1F{f}_{i}.npz') as z:
                    xy=z[sp+'_xy'];u,idx,counts=np.unique(xy,axis=0,return_inverse=True,return_counts=True);idx=idx.ravel();err=np.stack([np.linalg.norm(z[sp+'_'+k]-xy,axis=1) for k in ARMS]);per=np.stack([np.bincount(idx,weights=v)/counts for v in err],1);k,j,l,h=per.T;per=np.column_stack([per,.5*((j-k)+(h-l)),.5*((l-k)+(h-j)),h-j-l+k,h-k]);assert np.allclose(per[:,4]+per[:,5],per[:,7]);M.append(per.mean(0));cases.append((np.floor(np.round(u,3)/8).astype(int),per))
            tiles=np.unique(np.vstack([e for e,_ in cases]),axis=0);dd=dict(zip(map(tuple,tiles),range(len(tiles))));n=4000;w=dr.multinomial(len(tiles),np.repeat(1/len(tiles),len(tiles)),size=n);se=dr.integers(0,3,size=(n,3));bs=[]
            for cells,values in cases:
                ww=w[:,[dd[tuple(c)] for c in cells]];assert (ww.sum(1)>0).all();bs.append(ww@values/ww.sum(1)[:,None])
            boot.append(np.stack(bs,1)[np.arange(n)[:,None],se].mean(1))
        M=np.stack(M);boot=np.mean(boot,0);res[sp]={k:dict(mean=float(M[:,j].mean()),CI95=np.quantile(boot[:,j],[.025,.975]).tolist(),floor=M[:,j].reshape(2,3).mean(1).tolist(),events=M[:,j].tolist()) for j,k in enumerate((*ARMS,'mean_component','variance_component','interaction','total'))}
    write(out/'decomposition.json',res);print(json.dumps(res))
if __name__=='__main__':
    if sys.argv[1]=='combine':summary(Path(sys.argv[2]))
    else:run(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))
