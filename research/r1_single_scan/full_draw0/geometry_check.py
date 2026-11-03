"""Generate hash-checked geometry-only receipts, then analyze already frozen predictions. No model fit.
make: python geometry_check.py make /original_data evidence.zip geometry-dir
analyze: python geometry_check.py analyze rows draw1 remaining geometry-dir ten.json geo.json
"""
import hashlib,io,sys,json,zipfile
from pathlib import Path
import numpy as np
K=(1,3,5,10,20);REG=('official','phone');N=('gp','seven','nonlinear');H=lambda b:hashlib.sha256(b).hexdigest()
def make(data,ev,out):
    import pandas as pd
    z=zipfile.ZipFile(ev);m=json.loads(z.read('results/manifest.json'))['episodes'];ph=json.loads(z.read('results/device_stress_design.json'))['events'];checks=json.loads(z.read('results/manifest.json'))['checksums'];xy={};meta={};out.mkdir(exist_ok=True,parents=True)
    for file in ('trainingData.csv','validationData.csv'):
        p=data/file;assert H(p.read_bytes())==checks[file];frame=pd.read_csv(p,usecols=['LONGITUDE','LATITUDE','BUILDINGID','FLOOR']);xy[file]=frame[['LONGITUDE','LATITUDE']].to_numpy();meta[file]=frame[['BUILDINGID','FLOOR']].to_numpy()
    for r in REG:
        for f in range(4):
            arr={};row_ids=None
            for d in range(10):
                e=next(v for v in (m if r=='official' else ph) if (v['building'],v['floor'])==(1,f) and (r!='official' or v['draw']==d));S=np.asarray(e['support_order'] if r=='official' else e['support_orders'][d]);Q=np.asarray(e['query_ids']);src='validationData.csv' if r=='official' else 'trainingData.csv';assert (meta[src][Q]==np.array([1,f])).all() and (meta['trainingData.csv'][S]==np.array([1,f])).all();ss=xy['trainingData.csv'][S];truth=xy[src][Q];dist=((truth[:,None,:]-ss[None,:,:])**2).sum(2);assert len(S)==20
                if row_ids is not None:np.testing.assert_array_equal(Q,row_ids)
                row_ids=Q;arr[f'draw{d}_S']=S;arr[f'draw{d}_oldnearest_squared']=np.stack([dist[:,:k].min(1) for k in K]);arr['Q']=Q;arr['xy_hash']=np.array(H(np.asarray(truth,dtype='<f8').tobytes()))
            np.savez_compressed(out/f'{r}_F{f}.npz',**arr)
    print('original row IDs + min squared distances for all 400 cells; no raw RSS')

def analyze(d0,d1,others,geo,metrics):
    total=json.loads(metrics.read_text());ranges={'all10':list(range(10)),'later8':list(range(2,10))};answer={'verified_cells':total['verified'],'scopes':{}}
    for subset,draws in ranges.items():
        t=answer['scopes'][subset]={}
        for r in REG:
            t[r]={}
            for ix,(old,new) in enumerate(zip(K[:-1],K[1:])):
                tally={n:np.zeros((2,2)) for n in N};stable_count=0;stable=[];complete=True
                for f in range(4):
                    gg=np.load(geo/f'{r}_F{f}.npz',allow_pickle=False)
                    for d in draws:
                        root=(d0 if d==0 else d1 if d==1 else others)/(('gridpilot' if d==0 else 'draw'+str(d))+f'-{r}-F{f}');paths=[root/f'K{k}.npz' for k in (old,new)]
                        if not all(p.is_file() for p in paths):complete=False;continue
                        aa,bb=[np.load(p,allow_pickle=False) for p in paths];jj=[json.loads((root/f'K{k}.json').read_text()) for k in (old,new)];p=gg[f'draw{d}_oldnearest_squared'];closer=p[ix]-p[ix+1]>1e-8;assert np.all(p[ix+1]<=p[ix]+1e-8);y=aa['truth'];np.testing.assert_array_equal(aa['query_ids'],gg['Q']);np.testing.assert_array_equal(aa['query_ids'],bb['query_ids']);np.testing.assert_array_equal(aa['support_ids'],gg[f'draw{d}_S'][:old]);np.testing.assert_array_equal(bb['support_ids'],gg[f'draw{d}_S'][:new]);assert gg['xy_hash'].item()==H(np.asarray(y,dtype='<f8').tobytes());pos,g,ct=np.unique(y,axis=0,return_inverse=True,return_counts=True);wt=1/(ct[g.ravel()]*len(ct));is_stable=np.array_equal(aa['aps'],bb['aps']) and jj[0]['grid_hash']==jj[1]['grid_hash'] and np.linalg.norm(np.subtract(jj[0]['weights'],jj[1]['weights']),ord=1)<=1e-8
                        if is_stable:stable_count+=1
                        ds={}
                        for n in N:
                            de=np.linalg.norm(bb[n]-y,axis=1)-np.linalg.norm(aa[n]-y,axis=1);ds[n]=float(wt@de)
                            for j,mask in enumerate((closer,~closer)):tally[n][j,0]+=wt[mask].sum();tally[n][j,1]+=wt[mask]@de[mask]
                        if is_stable:stable.append(dict(floor=f,draw=d,delta=ds))
                        aa.close();bb.close()
                    gg.close()
                if not complete:t[r][f'{old}-{new}']={'complete':False};continue
                record=dict(complete=True,paired_events=4*len(draws),stable_count=stable_count,stable_events=stable,models={});
                for n,a in tally.items():
                    a/=4*len(draws);c=a[:,1];rc=dict(nearer_fraction=float(a[0,0]),nearer_mean_delta=float(c[0]/a[0,0]) if a[0,0]>0 else None,unchanged_mean_delta=float(c[1]/a[1,0]) if a[1,0]>0 else None,nearer_contribution=float(c[0]),unchanged_contribution=float(c[1]),total=float(c.sum()));record['models'][n]=rc
                    if subset=='all10':np.testing.assert_allclose(rc['total'],total['adjacent'][r][f'{old}-{new}'][n]['delta'],rtol=0,atol=1e-8)
                t[r][f'{old}-{new}']=record
    return answer
if __name__=='__main__':
    a=[Path(p) for p in sys.argv[2:]]
    if sys.argv[1]=='make':make(*a)
    elif sys.argv[1]=='analyze':a[-1].write_text(json.dumps(analyze(*a[:-1]),indent=2,allow_nan=False)+'\n')
    else:raise ValueError('make or analyze')
