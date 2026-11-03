"""Continue only original failed support-weight fits. Do not alter any scored episode."""
import ast,hashlib,inspect,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from scipy.optimize import minimize
import multi_source_adaptation as ms
from . import run as U

def existing_loss():
    source=inspect.getsource(ms.fit_source_weights);prefix=source[:source.index('    def objective(value):')];scope=dict(ms.__dict__);exec(prefix.replace('def fit_source_weights(','def build(')+'    return loss,initial[prefix:].copy(),bounds[prefix:]\n',scope);return scope['build'],hashlib.sha256(prefix.encode()).hexdigest()

def numerical(means,covs,res,stats,obs,prev):
    factory,digest=existing_loss();fn,initial,bounds=factory(means,covs,res,stats,obs);assert means.shape[1]==3
    def decode(t,ids):
        u=t[0];vals=(u,1-u,0*u) if len(ids)==2 else (u,(1-u)*t[1],(1-u)*(1-t[1]));keys=list(ids)+[i for i in range(3) if i not in ids];return torch.stack([vals[keys.index(j)] for j in range(3)])
    trials=[];prevbest=min(v['objective'] for v in prev)
    for order in ((0,1),(0,2),(1,2),(0,1,2),(1,2,0),(2,0,1)):
        n=len(order)-1
        def obj(xx):
            t=torch.from_numpy(xx).requires_grad_();v=fn(decode(t,order),t[n:]);g,=torch.autograd.grad(v,t);return v.item(),g.detach().numpy()
        seed=[.5] if n==1 else [1/3,.5];fit=minimize(obj,np.r_[seed,initial],jac=True,method='L-BFGS-B',bounds=[(0.,1.)]*n+bounds,options=dict(maxiter=1400,maxls=60,ftol=2e-15,gtol=2e-8));xx=torch.from_numpy(fit.x);w=decode(xx,order).detach().requires_grad_();phi=xx[n:].clone().requires_grad_();v=fn(w,phi);gw,gp=torch.autograd.grad(v,(w,phi));ww=w.detach().numpy();gg=gw.detach().numpy();reduced=gg-ww@gg;violation=float(np.max(np.where(ww>1e-6,np.abs(reduced),np.maximum(-reduced,0))));grad=gp.detach().numpy().copy()
        for i,(lo,hi) in enumerate(bounds):
            if lo is not None and fit.x[n+i]<=lo and grad[i]>0:grad[i]=0
            if hi is not None and fit.x[n+i]>=hi and grad[i]<0:grad[i]=0
        pg=float(np.abs(grad).max());val=v.item();assert abs(val-fit.fun)<1e-9 and ww.min()>=0 and abs(ww.sum()-1)<1e-9
        trials.append(dict(start=list(order),weights=ww.tolist(),objective=val,optimizer_success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),simplex_kkt_residual=violation,variational_gradient=pg,stationary=bool(np.isfinite(val) and val<=prevbest+1e-7 and violation<1e-3 and pg<1e-3),same_loss_prefix=digest))
    valid=[v for v in trials if v['stationary']];assert valid,('No stationary numerical continuation',trials);return min(valid,key=lambda v:v['objective'])['weights'],trials

def main(data,folder,out):
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);t0=time.perf_counter();R=U.R;h=json.loads((folder/'header.json').read_text());r,f,d=h['regime'],h['floor'],h['draw'];failed=sorted(folder.glob('failure_K*.json'));assert failed and r in ('official','phone') and 2<=d<10;c=R.validate_layout(R.EVIDENCE,R.read_raw(data),r,f,d);assert h['S_order']==c['support']['ids'].tolist() and h['Q_ids_hash']==c['query_rows_sha256'];fs=list(c['history']);origin=np.vstack([v['xy'] for v in c['history'].values()]).mean(0);hg={ff:U.group_scans(v['rss'],v['xy']-origin) for ff,v in c['history'].items()};ap=np.flatnonzero(np.logical_or.reduce([(v['rss']!=100).any(0) for v in c['history'].values()]));banks={int(a):[U.single_history({'scans':hg[ff]},int(a),R.B,R.O,steps=640) for ff in fs] for a in ap};beta,net=R.models();dest=out/folder.name;dest.mkdir(parents=True,exist_ok=False);hsecs=time.perf_counter()-t0;bad=[]
    for fp in failed:
        k=int(fp.stem.split('K')[-1]);assert not (folder/f'K{k}.npz').exists() and not (folder/f'K{k}.json').exists();s=json.loads(fp.read_text())['error'];orig=ast.literal_eval(s[len('RuntimeError('):-1]);pr='No stationary multi-source fit: ';assert orig.startswith(pr),orig[:90];prev=ast.literal_eval(orig[len(pr):]);assert prev and not any(v['stationary'] for v in prev);start=time.perf_counter()
        try:
            with patch.object(R,'fit_source_weights',lambda m,cv,res,st,obs:numerical(m,cv,res,st,obs,prev)):arr,rec=U.score_k(c,hg,banks,origin,f,k,beta,net)
            rec.update(K=k,seconds=time.perf_counter()-start,extra_H_reconstruction_s=hsecs,archived_raw=R.cached_wknn(c,r,f,d,k),numeric_resume_only=True,original_failure_sha256=U.sha(fp.read_bytes()),original_rejected_trials=prev);hsecs=0.;np.savez_compressed(dest/f'K{k}.npz',**arr);rec['npz_sha256']=U.sha((dest/f'K{k}.npz').read_bytes());U.write(dest/f'K{k}.json',rec);print('restored',folder.name,k,rec['weights'],flush=True)
        except Exception as ex:bad.append(k);U.write(dest/f'unresolved_K{k}.json',dict(error=repr(ex)));print('UNRESOLVED',k,repr(ex),flush=True)
    if bad:raise RuntimeError(bad)
if __name__=='__main__':
    import sys;main(*map(Path,sys.argv[1:]))
