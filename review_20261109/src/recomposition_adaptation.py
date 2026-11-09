"""Joint S-only source weights/intercept under the current ReFi prior.

This is a new empirical adaptation rule, not a bug fix to v0.26. The inner
objective is summed censored VI negative ELBO plus a symmetric Dirichlet
prior, alpha_j=1+1/J. Variance is fitted continuously on the old [0,20]^2
range. Shared parameters still belong to the supplied ReFi model.

IFT differentiates a checked local stationary solution. Its AP-local VI
blocks are eliminated by a Schur complement, not a giant dense Hessian.
VI selects task parameters only: use adapted_prior with the existing
condition_field/M-PMC for localization. Refit after every shared update.
Reference: Chen et al., ADKF-IFT, ICLR 2023, Eqs. 1-5; not a reproduction.
"""
import numpy as np
import torch
from scipy.optimize import minimize

from adaptation import gaussian_batch, GH_NODES, GH_WEIGHTS
from observations import JITTER, _detection_logs
from recomposition import prior


def support_block(block):
    """Drop candidate-grid work from inner fitting, without changing S or H."""
    b = dict(block)
    for key in ('zg', 'gm', 'gd'):
        if key in b:
            b[key] = b[key][:, :0]
    for key in ('gs', 'gv'):
        if key in b:
            b[key] = b[key][:, :, :0]
    b['rg'], b['rv'] = b['rg'][:0], b['rv'][:0]
    return b


def decode(u):
    weights = torch.cat((u[:-1], u.new_zeros(1))).softmax(0)
    return weights, 100*u[-1]


def adapted_prior(model, block, state):
    """block must have been prepared with intercept=0, including at deployment."""
    p = prior(model, block, state['weights'])
    v = state['intercept_variance']
    return dict(p, sk=p['sk']+v, cross=p['cross']+v, gv=p['gv']+v)


class SupportObjective:
    def __init__(self, model, cases, obs):
        self.model, self.obs = model, obs
        self.blocks = [support_block(c['blocks']) for c in cases.values()]
        self.sources, self.k = self.blocks[0]['sm'].shape
        assert self.sources >= 1
        assert all(b['sm'].shape == (self.sources, self.k) for b in self.blocks)
        self.aps = len(self.blocks)
        # All APs in a task share source geometry and K. Batch the original
        # support-only ReFi prior; do not repeat its Python graph per AP.
        self.batch = {key: torch.stack([b[key] for b in self.blocks])
                      for key in ('za', 'zs', 'sm', 'sd', 'ss', 'rs')}
        self.active = torch.tensor([b['active'] for b in self.blocks])
        self.base_mean = self.batch['sm'].new_tensor([b['base_mean'] for b in self.blocks])
        self.stats = {key:torch.stack([c['stats'][key] for c in cases.values()])
                      for key in ('n', 'k', 's1', 's2')}
        self.rows, self.cols = torch.tril_indices(self.k, self.k)
        self.diagonal = self.rows == self.cols
        self.width = self.k+len(self.rows)

    def support_prior(self, w):
        b = self.batch
        phi, e, lg, ls = self.model.values()
        anchor = ((self.model.normalize(b['za'])@phi).tanh().mean(-1)[..., None]
                  if b['za'].shape[2] else 0.)
        r = (self.model.normalize(b['zs'])@phi).tanh()-anchor
        a = w[None, :, None]*r.exp()
        a = a/a.sum(1, keepdim=True)
        mean = -75+lg.exp()*((a*(b['sm']+e*b['sd'])).sum(1)+75)
        coef = torch.stack((e.new_ones(()), e, e.square()))
        historical = (b['ss']*coef[None, None, :, None, None]).sum(2)
        cov = (a[..., None]*a[:, :, None, :]*historical).sum(1)*ls.mul(2).exp()+b['rs']
        return (torch.where(self.active[:, None], mean, self.base_mean[:, None]),
                torch.where(self.active[:, None, None], cov, b['rs']))

    def value(self, u, phi):
        w, v = decode(u)
        mean, cov = self.support_prior(w)
        cov = cov+v
        center, rss_cov, rss_loss = gaussian_batch(mean, cov, self.stats, self.obs['noise_std'])
        root = torch.linalg.cholesky(rss_cov+JITTER*torch.eye(self.k))
        eta, packed = phi[:, :self.k], phi[:, self.k:]
        B = phi.new_zeros((self.aps, self.k, self.k))
        B[:, self.rows, self.cols] = packed
        pm = center+(root@eta[..., None]).squeeze(-1)
        pv = (root@B).square().sum(-1)
        detected, missing = _detection_logs(pm[..., None]+pv.sqrt()[..., None]*GH_NODES, self.obs)
        expected = ((self.stats['k'][..., None]*detected+
            (self.stats['n']-self.stats['k'])[..., None]*missing)*GH_WEIGHTS).sum((-1, -2))
        kl = .5*(eta.square().sum(-1)+B.square().sum((-1, -2))-self.k-
                  2*packed[:, self.diagonal].log().sum(-1))
        # Use log_softmax directly: the Dirichlet barrier must not disappear
        # when an ordinary softmax weight becomes numerically tiny.
        logw = torch.cat((u[:-1], u.new_zeros(1))).log_softmax(0)
        return (rss_loss+kl-expected).sum()-logw.mean()

    def implicit_global(self, optimum, free):
        """Return the optimum in forward and its local implicit Jacobian back."""
        u = optimum[:self.sources].clone().requires_grad_()
        phi = optimum[self.sources:].reshape(self.aps, self.width).clone().requires_grad_()
        if not len(free):
            return u.detach(), dict(free_global=0)
        F = self.value(u, phi)
        gu, gp = torch.autograd.grad(F, (u, phi), create_graph=True)
        uu, cross = [], []
        for i in free:
            a, b = torch.autograd.grad(gu[i], (u, phi), retain_graph=True)
            uu.append(a[free]); cross.append(b)
        Huu = torch.stack(uu).detach()
        Hpu = torch.stack(cross, -1).detach()  # AP x local x free-global
        Hpp = torch.stack([torch.autograd.grad(gp[:, i].sum(), phi,
            retain_graph=True)[0] for i in range(self.width)], 1).detach()
        Hpp = (Hpp+Hpp.transpose(-1, -2))/2
        chol = torch.linalg.cholesky(Hpp)
        solved_cross = torch.cholesky_solve(Hpu, chol)
        schur = Huu-torch.einsum('adu,adv->uv', Hpu, solved_cross)
        schur = (schur+schur.T)/2
        root = torch.linalg.cholesky(schur)
        local_rhs = torch.cholesky_solve(gp[..., None], chol)[..., 0]
        rhs = gu[free]-torch.einsum('adu,ad->u', Hpu, local_rhs)
        correction = torch.cholesky_solve(rhs[:, None], root)[:, 0]
        # The matrices are detached; differentiating the residual supplies
        # -H^{-1} F_{inner,shared}, including the AP-local response.
        increment = u.new_zeros(u.shape).index_copy(0, free,
            correction-correction.detach())
        return u.detach()-increment, dict(free_global=len(free), local_width=self.width,
            max_newton_correction=float(correction.detach().abs().max()),
            minimum_schur_eigenvalue=float(torch.linalg.eigvalsh(schur).min()))


@torch.enable_grad()
def fit(model, cases, obs, differentiable=False, maxiter=600):
    """Fit once using only current H/S. Caller supplies no query data.

    This CPU double-precision reference supports arbitrary source/AP counts;
    full-task cost depends on K, AP count and inner iterations. Failed
    stationarity, singular curvature and active-set transitions raise errors.
    No damping, dropped AP, cached old task state or uniform failure fallback.
    """
    obj = SupportObjective(model, cases, obs)
    j, a, d, k = obj.sources, obj.aps, obj.width, obj.k
    if not k or not bool((obj.stats['n'] > 0).any()):
        return dict(weights=torch.full((j,), 1/j, dtype=torch.float64),
                    intercept_variance=torch.tensor(0., dtype=torch.float64),
                    diagnostics=dict(no_support=True, variance_convention='zero, not identified'))
    initial = np.zeros(j+a*d)
    initial[j-1] = 1.  # variance=100, not an extra learned initialization
    diagonal = np.flatnonzero(obj.diagonal.numpy())
    initial[j:].reshape(a, d)[:, k+diagonal] = 1.
    bounds = [(None, None)]*(j-1)+[(0., 4.)]+[(None, None)]*(a*d)
    for ap in range(a):
        for index in diagonal:
            bounds[j+ap*d+k+index] = (1e-8, None)

    def objective(x):
        value = torch.from_numpy(x).requires_grad_()
        F = obj.value(value[:j], value[j:].reshape(a, d))
        gradient, = torch.autograd.grad(F, value)
        return float(F.detach()), gradient.detach().numpy()

    starts = [np.full(j, 1/j)]
    if j > 1:
        for index in range(j):
            w = np.full(j, .2/(j-1)); w[index] = .8; starts.append(w)
    records, solutions = [], []
    for w in starts:
        initial[:j-1] = np.log(w[:-1]/w[-1])
        result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=maxiter, maxls=40, gtol=1e-8, ftol=1e-14))
        projected = result.jac.copy()
        for i, (low, high) in enumerate(bounds):
            if low is not None and result.x[i] <= low+1e-10 and projected[i] > 0: projected[i] = 0
            if high is not None and result.x[i] >= high-1e-10 and projected[i] < 0: projected[i] = 0
        residual = float(np.abs(projected).max())
        stationary = bool(np.isfinite(result.fun) and np.isfinite(result.x).all() and residual < 1e-5)
        records.append(dict(start=w.tolist(), objective=float(result.fun),
            iterations=int(result.nit), optimizer_success=bool(result.success),
            message=str(result.message), projected_gradient=residual, stationary=stationary))
        if stationary:
            solutions.append(result)
    if not solutions:
        raise RuntimeError(f'No stationary joint adaptation: {records}')
    chosen = min(solutions, key=lambda r:r.fun)
    optimum = torch.from_numpy(chosen.x.copy())
    free = list(range(j))
    bound = 'interior'
    if chosen.x[j-1] <= 1e-10 or chosen.x[j-1] >= 4-1e-10:
        bound = 'lower' if chosen.x[j-1] <= 1e-10 else 'upper'
        if differentiable and abs(chosen.jac[j-1]) < 1e-5:
            raise RuntimeError('Variance is at an active-set transition; no ordinary IFT')
        free.remove(j-1)
    diagnostics = dict(trials=records, selected_objective=float(chosen.fun), variance_bound=bound,
                       regularizer='Dirichlet alpha=1+1/J; summed AP likelihood')
    if differentiable:
        packed = optimum[j:].reshape(a, d)[:, k:]
        if float(packed[:, obj.diagonal].min()) <= 1e-6:
            raise RuntimeError('VI Cholesky diagonal too near its bound for unconstrained IFT')
        u, details = obj.implicit_global(optimum, torch.tensor(free, dtype=torch.long))
        if details.get('max_newton_correction', 0) > 1e-3:
            raise RuntimeError(f'Joint VI is insufficiently stationary for IFT: {details}')
        diagnostics.update(details)
    else:
        u = optimum[:j]
    weights, variance = decode(u)
    return dict(weights=weights, intercept_variance=variance, diagnostics=diagnostics)
