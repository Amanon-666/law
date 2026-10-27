"""AP-batched full-Gaussian detection VI using the existing ELBO.

The variational family and likelihood match inference.infer; this is not an
exact non-Gaussian posterior. Optimize the SUM over APs so a difficult AP's
gradient is not diluted by the AP count. Every AP must pass stationarity.
Detected-RSS conditioning uses the existing shared-weight fitter's batched
update and its documented numerical jitter.
"""
import numpy as np
from scipy.optimize import minimize
import torch

from history_bank import gaussian_batch
from inference import GH_NODES,GH_WEIGHTS,JITTER,_detection_logs


def infer_batch(mean,covariance,stats,obs,steps=640):
    mean,covariance=mean.detach(),covariance.detach()
    stats={k:v.detach() for k,v in stats.items()}
    aps,k=mean.shape
    if k==0:
        return dict(mean=mean,cov=covariance,loss=mean.new_zeros(aps),
                    projected_gradient=[0.]*aps,optimizer_success=True,iterations=0)
    center,cov,rss_loss=gaussian_batch(mean,covariance,stats,obs['noise_std'])
    root=torch.linalg.cholesky(cov+JITTER*torch.eye(k,dtype=mean.dtype))
    rows,cols=torch.tril_indices(k,k)
    diag=rows==cols;width=k+len(rows)
    initial=np.zeros((aps,width));initial[:,k+np.flatnonzero(diag.numpy())]=1.
    bounds=[(None,None)]*(aps*width)
    diagonals=[a*width+k+j for a in range(aps) for j in np.flatnonzero(diag.numpy())]
    for j in diagonals:bounds[j]=(1e-8,None)

    def values(flat):
        packed=flat.reshape(aps,width);eta,entries=packed[:,:k],packed[:,k:]
        lower=flat.new_zeros((aps,k,k));lower[:,rows,cols]=entries
        pm=center+(root@eta[...,None]).squeeze(-1)
        pc=root@lower
        pv=pc.square().sum(-1)
        signal=pm[...,None]+pv.sqrt()[...,None]*GH_NODES
        detected,missing=_detection_logs(signal,obs)
        expected=((stats['k'][...,None]*detected+(stats['n']-stats['k'])[...,None]*missing)*GH_WEIGHTS).sum((-1,-2))
        kl=.5*(eta.square().sum(-1)+lower.square().sum((-1,-2))-k-2*entries[:,diag].log().sum(-1))
        return kl-expected,pm,pc

    def objective(value):
        variable=torch.from_numpy(value).requires_grad_()
        loss=values(variable)[0].sum()
        grad,=torch.autograd.grad(loss,variable)
        return loss.item(),grad.detach().numpy()
    result=minimize(objective,initial.ravel(),jac=True,method='L-BFGS-B',bounds=bounds,
                    options=dict(maxiter=steps,ftol=1e-14,gtol=1e-8,maxls=40))
    gradient=result.jac.copy()
    for index in diagonals:
        if result.x[index]<=1e-8 and gradient[index]>0:gradient[index]=0.
    residual=np.abs(gradient.reshape(aps,width)).max(1)
    if not np.isfinite(result.fun) or not np.isfinite(residual).all() or residual.max()>1e-3:
        raise RuntimeError(f'Unfinished batch VI: {result.message}; max AP gradient={residual.max()}')
    with torch.no_grad():
        loss,pm,pc=values(torch.from_numpy(result.x))
    return dict(mean=pm,cov=pc@pc.transpose(-1,-2),loss=rss_loss+loss,
                projected_gradient=residual.tolist(),optimizer_success=bool(result.success),iterations=int(result.nit))


if __name__=='__main__':
    import hashlib,json
    from pathlib import Path
    from inference import infer
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    mean=torch.tensor([[-75.,-85.],[-60.,-65.],[-95.,-100.]])
    cov=torch.tensor([[100.,40.],[40.,100.]]).expand(3,-1,-1).clone()
    rss=torch.tensor([[-72.,100.],[-61.,-68.],[100.,100.]])
    detected=rss!=100;v=torch.where(detected,rss,0.)
    stats=dict(n=torch.ones_like(rss),k=detected.double(),s1=v,s2=v.square())
    obs=dict(alpha=.9,beta=-85.,detect_scale=4.,noise_std=4.)
    batch=infer_batch(mean,cov,stats,obs)
    gaps=[]
    for ap in range(3):
        reference=infer(mean[ap],cov[ap],{k:v[ap] for k,v in stats.items()},obs,steps=640)
        diff={k:float((batch[k][ap]-reference[k]).abs().max()) for k in ['mean','cov','loss']}
        for key in diff:torch.testing.assert_close(batch[key][ap],reference[key],atol=1e-4,rtol=1e-6)
        gaps.append(diff)
    # D=1 with effectively certain detection must reduce to Gaussian RSS update.
    full=dict(n=torch.ones_like(rss),k=torch.ones_like(rss),s1=mean,s2=mean.square())
    gaussian=infer_batch(mean,cov,full,dict(obs,alpha=1.,beta=-10000.))
    gm,gc,_=gaussian_batch(mean,cov,full,obs['noise_std'])
    torch.testing.assert_close(gaussian['mean'],gm,atol=1e-8,rtol=0)
    torch.testing.assert_close(gaussian['cov'],gc+JITTER*torch.eye(2),atol=1e-8,rtol=0)
    result=dict(passed=True,scope='Small consistency checks of the Gaussian VI approximation; not validation against an exact censored posterior',
                per_AP_gaps=gaps,max_projected_gradient=max(batch['projected_gradient']),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitation='Separable ELBOs can have multiple local optima; batching is not a guarantee of the same local solution for every future dataset.')
    out=Path(__file__).resolve().parent/'results/fusion_candidate_20261027/batch_posterior_check.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
