"""Read only an original failed-event receipt; never tune to a query or replace a success.
Incomplete metrics stay missing if the KKT check still fails. No change to the deployed model.
"""
import argparse,ast,hashlib,inspect,itertools,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from scipy.optimize import minimize
import multi_source_adaptation as ms
from . import run as U

def loss_factory():
    src=inspect.getsource(ms.fit_source_weights);assert 'def objective(value):' in src
    prefix=src[:src.index('    def objective(value):')];scope=dict(ms.__dict__)
    exec(prefix.replace('def fit_source_weights(', 'def existing_loss(')+'    return loss, initial[prefix:].copy(), bounds[prefix:]\n',scope)
    return scope['existing_loss'],hashlib.sha256(prefix.encode()).hexdigest()

def numerical(means,covs,res,st,obs,prev):
    factory,dig=loss_factory();fn,start,bounds=factory(means,covs,res,st,obs);assert means.shape[1]==3
    def decode(t,order):
        u=t[0];zero=u*0
        raw=(u,1-u,zero) if len(order)==2 else (u,(1-u)*t[1],(1-u)*(1-t[1]));keys=list(order)+[a for a in range(3) if a not in order];return torch.stack([raw[keys.index(a)] for a in range(3)])
    trials=[];initial_best=min(float(t['objective']) for t in prev)
    for order in ((0,1),(0,2),(1,2),(0,1,2),(1,2,0),(2,0,1)):
        n=len(order)-1;b=[(0.,1.)]*n+bounds
        def objective(ar):
            t=torch.from_numpy(ar).requires_grad_(True);w=decode(t,order);f=fn(w,t[n:]);g,=torch.autograd.grad(f,t);return float(f.detach()),g.detach().numpy()
        seed=[.5] if n==1 else [1/3,.5];result=minimize(objective,np.r_[seed,start],jac=True,bounds=b,method='L-BFGS-B',options=dict(maxiter=1400,maxls=60,ftol=2e-15,gtol=2e-8));v=torch.from_numpy(result.x);w=decode(v,order).detach().requires_grad_(True);phi=v[n:].clone().requires_grad_(True);fv=fn(w,phi);wg,pg=torch.autograd.grad(fv,(w,phi));ww=w.detach().numpy();dw=wg.detach().numpy();rz=dw-ww@dw;viol=float(np.max(np.where(ww>1e-6,abs(rz),np.maximum(-rz,0))));pg=pg.detach().numpy().copy()
        for i,(lo,hi) in enumerate(bounds):
            if lo is not None and result.x[n+i]<=lo and pg[i]>0:pg[i]=0
            if hi is not None and result.x[n+i]>=hi and pg[i]<0:pg[i]=0
        vg=float(np.max(np.abs(pg)));fv=float(fv.detach());assert abs(fv-result.fun)<1e-9 and min(ww)>=0 and abs(sum(ww)-1)<1e-9
        trials.append(dict(start=list(order),weights=ww.tolist(),objective=fv,optimizer_success=bool(result.success),message=str(result.message),iterations=int(result.nit),simplex_kkt_residual=viol,variational_gradient=vg,stationary=bool(np.isfinite(fv) and fv<=initial_best+1e-7 and viol<.001 and vg<.001),same_loss_prefix=d ig if False else dig))
    ok=[t for t in trials if t['stationary']];assert ok,('No numerically valid continuation',trials);best=min(ok,key=lambda t:t['objective']);return best['weights'],trials

def main(data,folder,out):
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);t=time.perf_counter();R=U.R;head=json.loads((folder/'header.json').read_text());r,f,d=head['regime'],head['floor'],head['draw'];fail=sorted(folder.glob('failure_K*.json'));assert fail and r in ('phone','official') and 2<=d<10;c=R.validate_layout(R.EVIDENCE,R.read_raw(data),r,f,d);assert head['S_order']==c['support']['ids'].tolist() and head['Q_ids_hash']==c['query_rows_sha256'];fs=list(c['history']);org=np.vstack([v['xy'] for v in c['history'].values()]).mean(0);hg={ff:U.group_scans(v['rss'],v['xy']-org) for ff,v in c['history'].items()};ap=np.flatnonzero(np.logical_or.reduce([(v['rss']!=100).any(0) for v in c['history'].values()]));banks={int(a):[U.single_history({'scans':hg[ff]},int(a),R.B,R.O,steps=640) for ff in fs] for a in ap};beta,net=R.models();out=out/folder.name;out.mkdir(parents=True,exist_ok=False);hist_time=time.perf_counter()-t;bad=[]
    for fp in fail:
        k=int(fp.stem.split('K')[-1]);assert not (folder/f'K{k}.npz').exists() and not (folder/f'K{k}.json').exists();text=json.loads(fp.read_text())['error'];raw=ast.literal_eval(text[len('RuntimeError('):-1]);prefix='No stationary multi-source fit: ';assert raw.startswith(prefix),raw[:100];prev=ast.literal_eval(raw[len(prefix):]);assert prev and not any(s['stationary'] for s in prev);st=time.perf_counter()
        try:
            with patch.object(R,'fit_source_weights',lambda means,covs,res,stats,obs:numerical(means,covs,res,stats,obs,prev)):arr,rec=U.score_k(c,hg,banks,org,f,k,beta,net)
            rec.update(K=k,seconds=time.perf_counter()-st,extra_H_reconstruction_s=hist_time,archived_raw=R.cached_wknn(c,r,f,d,k),numeric_resume_only=True,original_failure_sha256=U.sha(fp.read_bytes()),original_rejected_trials=prev);hist_time=0.;np.savez_compressed(out/f'K{k}.npz',**arr);rec['npz_sha256']=U.sha((out/f'K{k}.npz').read_bytes());U.write(out/f'K{k}.json',rec);print('restored',folder.name,k,rec['weights'],flush=True)
        except Exception as e:bad.append(k);U.write(out/f'unresolved_K{k}.json',dict(error=repr(e)));print('NOT RESOLVED',k,repr(e),flush=True)
    if bad:raise RuntimeError(bad)
if __name__=='__main__':
    import sys;main(*map(Path,sys.argv[1:]))
