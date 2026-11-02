"""Offline verification; no UJI raw data or training run required.
python -m research.eta_field_v2.verify RESULTS AFFINE_ZIP NONLINEAR_ZIP > verified.json
RESULTS is the extracted results directory or the original eta-six-verified ZIP.
"""
import io,sys,zipfile,json,itertools,hashlib
from pathlib import Path
import numpy as np

def reader(p):return (lambda n:(p/n).read_bytes()) if p.is_dir() else zipfile.ZipFile(p).read

def arrays(r,n):
    with np.load(io.BytesIO(r(n)),allow_pickle=False) as a:return {k:a[k] for k in a.files}

def verify(p,linear,network):
    read,aff,net=map(reader,[p,linear,network]);fr=read('freeze.json');freeze=json.loads(fr);rows={};info=[];ess=[];delta=[]
    for line in read('SHA256SUMS').decode().splitlines():
        h,n=line.split(maxsplit=1);assert hashlib.sha256(read(n)).hexdigest()==h
    for f,i in itertools.product((1,2),range(3)):
        eid=f'B1F{f}_{i}';j=json.loads(read(eid+'.json'));assert j['event']==eid and j['freeze_sha']==hashlib.sha256(fr).hexdigest() and j['eta']==freeze['selected']['full'];info.append(j['seconds']);ess.append(min(j['min_ess'].values()));a,l,n=[arrays(r,eid+'.npz') for r in (read,aff,net)];arms=list(j['metrics']);names=arms+['seven','nonlinear']
        for sp in ('internal','official'):
            xy,ids=a[sp+'_xy'],a[sp+'_ids'];assert len(np.unique(ids))==len(ids)
            for v in (l,n):np.testing.assert_array_equal(ids,v[sp+'_ids']);np.testing.assert_array_equal(xy,v[sp+'_xy'])
            delta.append(float(np.abs(a[sp+'_plain']-l[sp+'_plain']).max()));pred=[a[sp+'_'+arm] for arm in arms]+[l[sp+'_mean_only'],n[sp+'_mean_only']];err=np.stack([np.linalg.norm(pp-xy,axis=1) for pp in pred],1);assert np.isfinite(err).all();u,g,c=np.unique(xy,axis=0,return_inverse=True,return_counts=True);g=g.ravel();w=1/(len(c)*c[g]);per=np.stack([np.bincount(g,weights=v)/c for v in err.T],1);p90=[]
            for v in err.T:
                ix=v.argsort();p90.append(float(v[ix][min(len(ix)-1,np.searchsorted(w[ix].cumsum(),.9))]))
            np.testing.assert_allclose(per.mean(0)[:len(arms)],[j['metrics'][k][sp]['mde'] for k in arms],atol=1e-8,rtol=0);rows[sp,f,i]=(u,per,p90,len(ids))
    result=dict(selected=freeze['selected'],source_risk=freeze['risk'],identity_max_coordinate_diff=max(delta),min_ess=min(ess),seconds_per_event=info,splits={})
    for sp in ('internal','official'):
        m=[];p90=[];bvals=[];instances=0;rng=np.random.default_rng(20261102);nr=4000
        for f in (1,2):
            floor=[rows[sp,f,i] for i in range(3)];m.extend(x[1].mean(0) for x in floor);p90.extend(x[2] for x in floor);instances+=sum(x[3] for x in floor);cells=[np.floor(np.round(x[0],3)/8).astype(int) for x in floor];tiles=np.unique(np.vstack(cells),axis=0);look=dict(zip(map(tuple,tiles),range(len(tiles))));sample=rng.multinomial(len(tiles),np.repeat(1/len(tiles),len(tiles)),size=nr);draw=rng.integers(0,3,size=(nr,3));po=[]
            for row,cell in zip(floor,cells):
                ww=sample[:,[look[tuple(c)] for c in cell]];den=ww.sum(1);assert (den>0).all();po.append((ww@row[1])/den[:,None])
            bvals.append(np.stack(po,1)[np.arange(nr)[:,None],draw].mean(1))
        A=np.array(m);boot=np.mean(bvals,0);met={};pairs={}
        for z,name in enumerate(names):met[name]=dict(MDE=float(A[:,z].mean()),P90=float(np.mean(p90,0)[z]),floor=A[:,z].reshape(2,3).mean(1).tolist(),events=A[:,z].tolist())
        for left,right in itertools.product([k for k in names if k not in ('seven','nonlinear','plain')],names):
            if left==right:continue
            i,h=names.index(left),names.index(right);d=A[:,i]-A[:,h];v=boot[:,i]-boot[:,h];pairs[left+'-'+right]=dict(diff=float(d.mean()),relative_pct=float(-100*d.mean()/A[:,h].mean()),wins=int((d< -1e-9).sum()),CI95=np.quantile(v,[.025,.975]).tolist(),floor=d.reshape(2,3).mean(1).tolist())
        result['splits'][sp]=dict(instances=instances,metrics=met,pairs=pairs)
    return result
if __name__=='__main__':print(json.dumps(verify(*map(Path,sys.argv[1:4])),indent=2,allow_nan=False))
