"""General-J extension of the existing shared Support-only EB weight fit.

One coefficient vector is shared by all APs. Mean/variance inputs already
include any frozen content correction. J=2 uses the original implementation.
This is mature simplex adaptation infrastructure, not a learning contribution.
"""
import numpy as np
import torch
from scipy.optimize import minimize

from history_bank import gaussian_batch, fit_two_source_weight
from inference import GH_NODES, GH_WEIGHTS, JITTER, _detection_logs


def fit_source_weights(means,covariances,residual,stats,obs):
    aps,sources,k=means.shape
    if sources==1:return [1.],[]
    if sources==2:return fit_two_source_weight(means,covariances,residual,stats,obs)
    rows,cols=torch.tril_indices(k,k);diagonal=rows==cols
    width=k+len(rows);prefix=sources-1
    initial=np.zeros(prefix+aps*width)
    initial[prefix:].reshape(aps,width)[:,k+np.flatnonzero(diagonal.numpy())]=1.
    bounds=[(None,None)]*len(initial)
    for a in range(aps):
        for j in np.flatnonzero(diagonal.numpy()):bounds[prefix+a*width+k+j]=(1e-8,None)

    def loss(weights,packed):
        mean=(means*weights[None,:,None]).sum(1)
        covariance=(covariances*weights[None,:,None,None].square()).sum(1)+residual
        center,cov,rss_loss=gaussian_batch(mean,covariance,stats,obs['noise_std'])
        root=torch.linalg.cholesky(cov+JITTER*torch.eye(k))
        phi=packed.reshape(aps,width);eta,values=phi[:,:k],phi[:,k:]
        lower=packed.new_zeros((aps,k,k));lower[:,rows,cols]=values
        pm=center+(root@eta[...,None]).squeeze(-1)
        pv=(root@lower).square().sum(-1)
        signal=pm[...,None]+pv.sqrt()[...,None]*GH_NODES
        detected,missing=_detection_logs(signal,obs)
        expected=((stats['k'][...,None]*detected+(stats['n']-stats['k'])[...,None]*missing)*GH_WEIGHTS).sum((-1,-2))
        kl=.5*(eta.square().sum(-1)+lower.square().sum((-1,-2))-k-2*values[:,diagonal].log().sum(-1))
        return (rss_loss+kl-expected).mean()

    def objective(value):
        variable=torch.from_numpy(value).requires_grad_()
        weights=torch.softmax(torch.cat((variable[:prefix],variable.new_zeros(1))),0)
        value=loss(weights,variable[prefix:]);gradient,=torch.autograd.grad(value,variable)
        return value.item(),gradient.detach().numpy()

    starts=[np.full(sources,1/sources)]
    for source in range(sources):
        w=np.full(sources,.2/(sources-1));w[source]=.8;starts.append(w)
    trials=[]
    for start in starts:
        initial[:prefix]=np.log(start[:-1]/start[-1])
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,
                        options=dict(maxiter=800,maxls=40,ftol=1e-12,gtol=1e-6))
        variable=torch.from_numpy(result.x)
        weights=torch.softmax(torch.cat((variable[:prefix],variable.new_zeros(1))),0).requires_grad_()
        gw,=torch.autograd.grad(loss(weights,variable[prefix:]),weights)
        # A saturated softmax can hide a non-optimal simplex boundary; inspect
        # constrained derivatives in weight coordinates as well as logit space.
        lagrange=(weights.detach()*gw).sum();reduced=gw-lagrange
        violation=torch.where(weights.detach()>1e-6,reduced.abs(),(-reduced).clamp_min(0)).max().item()
        projected=result.jac[prefix:].copy();packed=result.x[prefix:]
        for a in range(aps):
            for j in np.flatnonzero(diagonal.numpy()):
                at=a*width+k+j
                if packed[at]<=1e-8 and projected[at]>0:projected[at]=0.
        good=np.isfinite(result.fun) and violation<1e-3 and np.abs(projected).max()<1e-3
        trials.append(dict(start=start.tolist(),weights=weights.detach().tolist(),objective=float(result.fun),
            optimizer_success=bool(result.success),message=str(result.message),iterations=int(result.nit),
            simplex_kkt_residual=violation,variational_gradient=float(np.abs(projected).max()),stationary=bool(good)))
    valid=[t for t in trials if t['stationary']]
    if not valid:raise RuntimeError(f'No stationary multi-source fit: {trials}')
    best=min(valid,key=lambda t:t['objective'])
    return best['weights'],trials


if __name__=='__main__':
    import json,socket
    from pathlib import Path
    from run_multi_history import digest,write
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(261026)
    means=-65+8*torch.randn(5,3,3)
    cov=.01*torch.eye(3).expand(5,3,3,3)
    residual=.2*torch.eye(3).expand(5,3,3)
    truth=torch.tensor([.65,.25,.10]);rss=(means*truth[None,:,None]).sum(1)
    stats=dict(n=torch.full_like(rss,3),k=torch.full_like(rss,3),s1=3*rss,s2=3*rss.square())
    obs=dict(alpha=.9,beta=-90.,detect_scale=3.,noise_std=1.)
    weights,trials=fit_source_weights(means,cov,residual,stats,obs)
    np.testing.assert_allclose(weights,truth.numpy(),atol=.02,rtol=0)
    order=[2,0,1];reverse,other=fit_source_weights(means[:,order],cov[:,order],residual,stats,obs)
    np.testing.assert_allclose(reverse,np.array(weights)[order],atol=1e-4,rtol=0)
    result=dict(scope='Synthetic general-J shared adaptation audit; no training/localization',
                weights=weights,truth=truth.tolist(),trials=trials,source_permutation=True,
                script_sha256=digest(Path(__file__)))
    out=Path(__file__).resolve().parent/'results/multi_source_adaptation_audit_20261026.json'
    if out.exists():raise FileExistsError(out)
    write(out,result);print(json.dumps(result))
