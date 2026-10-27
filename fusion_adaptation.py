"""Joint S-only EB weights/intercept and full-Gaussian variational posterior.

The intercept is one existing shared variance, not a new field component.
Unlike marginal sigma selection, the ELBO uses full spatial covariance.
Continuous sigma in [0,20] replaces the legacy five-point grid, disclosed as
part of the estimation-rule intervention. No Q signal or coordinates enter.
"""
import numpy as np
from scipy.optimize import minimize
import torch

from fusion_prior import combine
from history_bank import gaussian_batch
from inference import GH_NODES,GH_WEIGHTS,JITTER,_detection_logs


def fit_joint(means,covariances,residual,stats,obs,*,local,transfer_variance):
    aps,sources,k=means.shape
    if sources<1:raise ValueError('Joint source fit requires a historical source list; use GP_S for an empty bank')
    if not k:
        return dict(weights=[1/sources]*sources,sigma=0.,trials=[],
                    sm=means.sum(1),sk=residual,posterior=dict(mean=means.sum(1),cov=residual,
                        projected_gradient=[0.]*aps,optimizer_success=True,iterations=0))
    rows,cols=torch.tril_indices(k,k);diagonal=rows==cols
    width=k+len(rows);prefix=sources-1;offset=prefix+1
    initial=np.zeros(offset+aps*width)
    initial[offset:].reshape(aps,width)[:,k+np.flatnonzero(diagonal.numpy())]=1.
    bounds=[(None,None)]*len(initial);bounds[prefix]=(0.,4.)
    diagonal_indexes=[a*width+k+j for a in range(aps) for j in np.flatnonzero(diagonal.numpy())]
    for j in diagonal_indexes:bounds[offset+j]=(1e-8,None)

    def values(weights,variance_unit,packed):
        sm,sk=combine(means,covariances,residual,weights,local,transfer_variance)
        sk=sk+100*variance_unit
        center,cov,rss_loss=gaussian_batch(sm,sk,stats,obs['noise_std'])
        root=torch.linalg.cholesky(cov+JITTER*torch.eye(k,dtype=sm.dtype))
        phi=packed.reshape(aps,width);eta,entries=phi[:,:k],phi[:,k:]
        lower=packed.new_zeros((aps,k,k));lower[:,rows,cols]=entries
        pm=center+(root@eta[...,None]).squeeze(-1);pc=root@lower
        signal=pm[...,None]+pc.square().sum(-1).sqrt()[...,None]*GH_NODES
        detected,missing=_detection_logs(signal,obs)
        expected=((stats['k'][...,None]*detected+(stats['n']-stats['k'])[...,None]*missing)*GH_WEIGHTS).sum((-1,-2))
        kl=.5*(eta.square().sum(-1)+lower.square().sum((-1,-2))-k-2*entries[:,diagonal].log().sum(-1))
        loss=rss_loss+kl-expected
        return loss,sm,sk,pm,pc

    def decode(variable):
        weights=torch.softmax(torch.cat((variable[:prefix],variable.new_zeros(1))),0)
        return weights,variable[prefix],variable[offset:]

    def objective(value):
        variable=torch.from_numpy(value).requires_grad_()
        loss=values(*decode(variable))[0].mean()
        gradient,=torch.autograd.grad(loss,variable)
        return loss.item(),gradient.detach().numpy()

    uniform=np.full(sources,1/sources)
    starts=[(uniform,0.),(uniform,10.)]
    if sources>1:
        for j in range(sources):
            w=np.full(sources,.2/(sources-1));w[j]=.8;starts.append((w,10.))
    trials=[];best=None
    for start,sigma in starts:
        initial[:prefix]=np.log(start[:-1]/start[-1]);initial[prefix]=sigma**2/100
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,
                        options=dict(maxiter=800,maxls=40,ftol=1e-14,gtol=1e-8))
        variable=torch.from_numpy(result.x)
        w,v,packed=decode(variable)
        w=w.detach().requires_grad_();v=v.detach().requires_grad_()
        loss=values(w,v,packed)[0].mean();gw,gv=torch.autograd.grad(loss,(w,v))
        reduced=gw-(w.detach()*gw).sum()
        w_res=float(torch.where(w.detach()>1e-6,reduced.abs(),(-reduced).clamp_min(0)).max())
        vg=float(gv)
        variance_unit=float(v.detach())
        if variance_unit<=1e-8 and vg>0:vg=0.
        if variance_unit>=4-1e-8 and vg<0:vg=0.
        # Undo the AP averaging before checking every posterior's stationarity.
        pg=result.jac[offset:].copy()*aps
        for j in diagonal_indexes:
            if result.x[offset+j]<=1e-8 and pg[j]>0:pg[j]=0.
        per_ap=np.abs(pg.reshape(aps,width)).max(1)
        good=bool(np.isfinite(result.fun) and np.isfinite(per_ap).all() and
                  w_res<1e-3 and abs(vg)<1e-4 and per_ap.max()<1e-3)
        trial=dict(start_weights=start.tolist(),start_sigma=sigma,weights=w.detach().tolist(),
            sigma=float(torch.sqrt(100*v.detach())),objective=float(result.fun),
            optimizer_success=bool(result.success),message=str(result.message),iterations=int(result.nit),
            simplex_kkt_residual=w_res,variance_projected_gradient=abs(vg),
            max_AP_variational_gradient=float(per_ap.max()),stationary=good)
        trials.append(trial)
        if good and (best is None or result.fun<best[0]):
            best=(float(result.fun),result.x.copy(),trial,per_ap.tolist())
    if best is None:raise RuntimeError(f'No stationary joint S fit: {trials}')
    _,point,trial,per_ap=best
    with torch.no_grad():
        w,v,packed=decode(torch.from_numpy(point));loss,sm,sk,pm,pc=values(w,v,packed)
    return dict(weights=w.tolist(),sigma=float(torch.sqrt(100*v)),trials=trials,sm=sm,sk=sk,
        posterior=dict(mean=pm,cov=pc@pc.transpose(-1,-2),loss=loss,projected_gradient=per_ap,
                       optimizer_success=trial['optimizer_success'],iterations=trial['iterations']))


if __name__=='__main__':
    import json,hashlib
    from pathlib import Path
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    means=torch.zeros(1,1,2);cov=torch.zeros(1,1,2,2);residual=torch.eye(2)[None]
    obs=dict(alpha=1.,beta=-10000.,detect_scale=1.,noise_std=1.)
    records=[]
    for rss in [torch.tensor([[10.,10.]]),torch.tensor([[10.,-10.]])]:
        stats=dict(n=torch.ones_like(rss),k=torch.ones_like(rss),s1=rss,s2=rss.square())
        fit=fit_joint(means,cov,residual,stats,obs,local=False,transfer_variance=1.)
        expected=99. if rss[0,0]==rss[0,1] else 0.
        assert abs(fit['sigma']**2-expected)<1e-3
        records.append(dict(residual=rss.tolist(),joint_variance=fit['sigma']**2,
            analytic_joint_variance=expected,analytic_marginal_variance=98.,trials=fit['trials']))
    result=dict(passed=True,scope='Analytic Gaussian counterexample and joint-fit check; not localization evidence',
        explanation='Both residual patterns have identical one-point marginals. Joint covariance distinguishes common shift from opposite changes.',
        records=records,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=Path(__file__).resolve().parent/'results/fusion_candidate_20261027/joint_adaptation_check.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
