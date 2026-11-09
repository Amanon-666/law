"""Original ReFi support-only task adaptation, extracted without changes.

Shared source weights precede composite-marginal intercept selection. These
are empirical-Bayes point estimates; this module does not integrate their
uncertainty or change the target field posterior solver.
"""
import math
import numpy as np
import torch
from scipy.optimize import minimize
from fields import Prior, Transfer, rbf, InterceptBankPrior
from observations import JITTER, _detection_logs

_nodes, _weights = np.polynomial.hermite.hermgauss(20)
GH_NODES = torch.tensor(_nodes * math.sqrt(2), dtype=torch.float64)
GH_WEIGHTS = torch.tensor(_weights / math.sqrt(math.pi), dtype=torch.float64)
SIGMA = [0., 5., 10., 15., 20.]


def gaussian_batch(mean, cov, stats, noise_std):
    """Exact detected-RSS update, including likelihood constants, for AP batches.

    Precision form handles zero detections without an artificial RSS value.
    Axes are [AP, position] and [AP, position, position].
    """
    n = mean.shape[-1]
    eye = torch.eye(n, dtype=mean.dtype)
    root = torch.linalg.cholesky(cov + JITTER * eye)
    k, s1, s2 = (stats[key] for key in ('k', 's1', 's2'))
    noise2 = noise_std ** 2
    precision = k / noise2
    a = eye + root.transpose(-1, -2) @ (precision[..., None] * root)
    achol = torch.linalg.cholesky(a)
    v = (root.transpose(-1, -2) @ ((s1 - k * mean) / noise2)[..., None])
    solved = torch.cholesky_solve(v, achol)
    center = mean + (root @ solved).squeeze(-1)
    covariance = root @ torch.cholesky_solve(root.transpose(-1, -2), achol)
    constant = (k * math.log(2 * math.pi * noise2) +
                (s2 - 2 * mean * s1 + k * mean.square()) / noise2).sum(-1)
    loss = .5 * (constant + 2 * achol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
                 - (v * solved).sum((-1, -2)))
    return center, covariance, loss


def fit_two_source_weight(means, covariances, residual, stats, obs, cross_covariance=None):
    """Joint full-Gaussian VI and one bounded source weight, using S only.

    For each AP: f = w*h0 + (1-w)*h1 + delta. Three fixed initializations
    compare the SAME support ELBO, never held-out query errors. We use the
    original RSS and detection factors, including nondetections. The final
    location calculation still uses the established M-PMC posterior solver.
    Optional cross_covariance is C01 + C10 at S, not a correlation coefficient.
    """
    aps, _, k = means.shape
    rows, cols = torch.tril_indices(k, k)
    diagonal = rows == cols
    width = k + len(rows)
    initial = np.zeros(1 + aps * width)
    packed = initial[1:].reshape(aps, width)
    packed[:, k + np.flatnonzero(diagonal.numpy())] = 1.
    bounds = [(0., 1.)] + [(None, None)] * (aps * width)
    for ap in range(aps):
        for j in np.flatnonzero(diagonal.numpy()):
            bounds[1 + ap * width + k + j] = (1e-8, None)

    def objective(value):
        variable = torch.from_numpy(value).requires_grad_()
        w = variable[0]
        mean = w * means[:, 0] + (1 - w) * means[:, 1]
        covariance = w.square() * covariances[:, 0] + (1 - w).square() * covariances[:, 1] + residual
        if cross_covariance is not None:
            covariance = covariance + w * (1 - w) * cross_covariance
        center, covariance, rss_loss = gaussian_batch(mean, covariance, stats, obs['noise_std'])
        root = torch.linalg.cholesky(covariance + JITTER * torch.eye(k))
        phi = variable[1:].reshape(aps, width)
        eta, values = phi[:, :k], phi[:, k:]
        b = variable.new_zeros((aps, k, k))
        b[:, rows, cols] = values
        pm = center + (root @ eta[..., None]).squeeze(-1)
        pv = (root @ b).square().sum(-1)
        signal = pm[..., None] + pv.sqrt()[..., None] * GH_NODES
        detected, missing = _detection_logs(signal, obs)
        expected = ((stats['k'][..., None] * detected +
                     (stats['n'] - stats['k'])[..., None] * missing) * GH_WEIGHTS).sum((-1, -2))
        kl = .5 * (eta.square().sum(-1) + b.square().sum((-1, -2)) - k
                   - 2 * values[:, diagonal].log().sum(-1))
        loss = (rss_loss + kl - expected).mean()
        gradient, = torch.autograd.grad(loss, variable)
        return loss.item(), gradient.detach().numpy()

    trials = []
    for start in (.25, .5, .75):
        initial[0] = start
        result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=800, maxls=40, ftol=1e-12, gtol=1e-6))
        projected = result.jac.copy()
        if result.x[0] == 0 and projected[0] > 0 or result.x[0] == 1 and projected[0] < 0:
            projected[0] = 0
        record = dict(start=start, weight=float(result.x[0]), objective=float(result.fun),
                      iterations=int(result.nit), success=bool(result.success), message=str(result.message),
                      projected_gradient=float(np.abs(projected).max()))
        trials.append(record)
        if not result.success or not np.isfinite(result.fun) or record['projected_gradient'] > 1e-3:
            raise RuntimeError(f'Unfinished support-weight fit: {record}')
    best = min(trials, key=lambda x: x['objective'])
    return [best['weight'], 1 - best['weight']], trials


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


def marginal_loglik(mean, var, n, k, s1, s2, obs):
    """Exact log p(all scans at a position) for f ~ N(mean, var), positions independent.

    RSS factors integrate in closed form; detection factors use the original GH rule
    on the resulting RSS posterior. Equals observation_log_prob for a single scan.
    """
    noise2 = float(obs['noise_std']) ** 2
    kk = k.clamp_min(1)
    rbar = s1 / kk
    spread = (s2 - s1.square() / kk).clamp_min(0)
    total = var + noise2 / kk
    logz = (-.5 * (k - 1) * math.log(2 * math.pi * noise2) - .5 * kk.log() - spread / (2 * noise2)
            - .5 * torch.log(2 * math.pi * total) - (rbar - mean).square() / (2 * total))
    logz = torch.where(k > 0, logz, torch.zeros_like(logz))
    post_var = 1 / (1 / var + k / noise2)
    post_mean = post_var * (mean / var + k * rbar / noise2)
    signal = post_mean[..., None] + post_var.sqrt()[..., None] * GH_NODES
    detected, missing = _detection_logs(signal, obs)
    logd = torch.logsumexp(GH_WEIGHTS.log() + k[..., None] * detected + (n - k)[..., None] * missing, -1)
    return logz + logd


def fit_episode(cases, base, obs):
    means, covs, residuals = [], [], []
    for c in cases.values():
        sx = c['sx']; mm, cc = [], []
        for h in c['histories']:
            p = Prior(Transfer('identity'), h, dict(base, drift=0.))
            m, f = p.features(sx)
            mm.append(m); cc.append(p.covariance(sx, sx, f, f))
        if any(h is not None for h in c['histories']):
            residual = rbf(sx, sx, base['drift'], base['length'])
        else:
            cc = [torch.zeros_like(v) for v in cc]
            residual = rbf(sx, sx, base['amplitude'], base['length'])
        means.append(torch.stack(mm)); covs.append(torch.stack(cc)); residuals.append(residual)
    stats = {k: torch.stack([c['stats'][k] for c in cases.values()]) for k in ('n', 'k', 's1', 's2')}
    return fit_source_weights(torch.stack(means), torch.stack(covs), torch.stack(residuals), stats, obs)


@torch.no_grad()
def choose_intercept(support_cases, weights, base, obs):
    scores = torch.zeros(len(SIGMA), dtype=torch.float64)
    for case in support_cases.values():
        prior = InterceptBankPrior(case['histories'], weights, base, 0.)
        mean, features = prior.features(case['sx'])
        variance = prior.diagonal(case['sx'], features)
        for i, sigma in enumerate(SIGMA):
            scores[i] += marginal_loglik(mean, variance + sigma ** 2,
                                        **case['stats'], obs=obs).sum()
    best = int(scores.argmax())
    return dict(intercept_std=SIGMA[best], grid=SIGMA,
                composite_log_likelihood=scores.tolist(), boundary_hit=best in [0, len(SIGMA) - 1],
                criterion='All AP/S-position composite marginal likelihood; not a calibrated posterior or exact joint EB',
                query_used=False)
