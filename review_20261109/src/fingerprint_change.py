"""Whole-history fingerprint geometry for ReFi target-change propagation.

Learn a PSD mixture of physical, uncertain-fingerprint and product kernels.
The history mean/covariance, S likelihood and localization readout are retained.
This borrows standard GP kernel machinery; no novelty or accuracy is implied.
"""
import math
import torch
from torch import nn
from fields import InterceptBankPrior, rbf
from recomposition import condition_field
from observations import scan_log_prob


def expected_rbf(mean_difference, difference_variance, length):
    """E RBF of one joint Gaussian fingerprint function at two positions.

    Inputs have axes [AP, left, right]. Difference variance includes cross
    covariance; at identical positions it is zero, not twice marginal variance.
    AP independence is the same assumption as the historical field model.
    """
    scale = len(mean_difference) * length.square()
    return (-.5 * (torch.log1p(difference_variance / scale)
                   + mean_difference.square() / (scale + difference_variance)).sum(0)).exp()


class FingerprintChange(nn.Module):
    def __init__(self):
        super().__init__()
        initial = torch.tensor([math.log(8.), 0., 0.], dtype=torch.float64)
        self.theta = nn.Parameter(initial.clone())
        self.register_buffer('initial', initial)

    def values(self):
        weights = torch.cat((self.theta[:2], self.theta.new_zeros(1))).softmax(0)
        length_ratio = (math.log(10.) * self.theta[2].tanh()).exp()
        return weights, length_ratio

    def kernels(self, geometry):
        spatial = (geometry['spatial_ss'], geometry['spatial_gs'])
        if not geometry['history_APs']:
            return spatial
        weights, ratio = self.values()
        length = ratio * geometry['amplitude']
        result = []
        for name, r in zip(('ss', 'gs'), spatial):
            radio = expected_rbf(geometry['mean_'+name], geometry['variance_'+name], length)
            result.append(weights[0]*r + weights[1]*radio + weights[2]*r*radio)
        return tuple(result)

    def penalty(self):
        return .001 * (self.theta-self.initial).square().sum()


@torch.no_grad()
def prepare_fingerprint(task, proposals, device='cpu'):
    """Only H/S assets and original proposal densities enter this cache.

    Original identity moments are compact: no source/operator grid blocks or
    GxG covariance is retained. All APs share the new spatial relation only;
    their latent RSS values and observation updates remain separate.
    """
    base, state, grid = task['base'], task['state'], task['grid']
    sx = next(iter(task['cases'].values()))['sx']
    rss = rbf(sx,sx,1.,base['length']); rgs = rbf(grid,sx,1.,base['length'])
    means_ss, means_gs, vars_ss, vars_gs, active_APs = [], [], [], [], []
    cases = {}
    for ap, c in task['cases'].items():
        p = InterceptBankPrior(c['histories'],state['weights'],base,state['sigma']['intercept_std'])
        sm, sf = p.features(sx); gm, gf = p.features(grid)
        sk = p.covariance(sx,sx,sf,sf); cross = p.covariance(grid,sx,gf,sf)
        gv = p.diagonal(grid,gf)
        active = any(h is not None for h in c['histories'])
        amplitude = base['drift'] if active else base['amplitude']
        if active:
            offset = state['sigma']['intercept_std']**2
            hs = sk-amplitude**2*rss-offset
            hg = cross-amplitude**2*rgs-offset
            hv = gv-amplitude**2-offset
            vs = hs.diag()[:,None]+hs.diag()[None]-2*hs
            vg = hv[:,None]+hs.diag()[None]-2*hg
            # Difference variance is mathematically >=0; allow only roundoff.
            assert not vs.numel() or float(vs.min()) > -1e-7
            assert not vg.numel() or float(vg.min()) > -1e-7
            means_ss.append(sm[:,None]-sm[None]); means_gs.append(gm[:,None]-sm[None])
            vars_ss.append(vs.clamp_min(0)); vars_gs.append(vg.clamp_min(0)); active_APs.append(ap)
        prior = dict(sm=sm,sk=sk,gm=gm,cross=cross,gv=gv)
        cases[ap] = dict(prior={k:v.to(device) for k,v in prior.items()},amplitude=amplitude,
                         stats={k:v.to(device) for k,v in c['stats'].items()},
                         proposal=None if proposals[ap] is None else tuple(v.to(device) for v in proposals[ap]))
    geometry = dict(spatial_ss=rss.to(device),spatial_gs=rgs.to(device),
                    history_APs=active_APs,amplitude=base['amplitude'])
    if active_APs:
        geometry.update({k:torch.stack(v).to(device) for k,v in
            (('mean_ss',means_ss),('mean_gs',means_gs),('variance_ss',vars_ss),('variance_gs',vars_gs))})
    return dict(cases=cases,geometry=geometry,grid=grid.to(device),obs=task['obs'],
                support=sx.to(device))


def changed_prior(case, geometry, kernels):
    p = case['prior']; scale = case['amplitude']**2
    return dict(p,sk=p['sk']+scale*(kernels[0]-geometry['spatial_ss']),
                cross=p['cross']+scale*(kernels[1]-geometry['spatial_gs']))


def evaluate_fingerprint(model, episode, query, backward=False, identity=False):
    """Original full-scan NLL with a two-pass gradient through shared SS/GS.

    Summed kernel adjoints avoid retaining all AP posterior graphs at once.
    Shared parameters and proposal state never change between the two passes.
    """
    grid = episode['grid']; device = grid.device
    truth = torch.as_tensor(query['xy'],dtype=grid.dtype,device=device)
    assert not (truth[:,None] == episode['support'][None]).all(-1).any()
    nearest = torch.cdist(truth.cpu(),grid.cpu()).argmin(1).to(device)
    qw = torch.as_tensor(1/(len(query['positions'])*query['stats']['n'][query['group']]),
                         dtype=grid.dtype,device=device)
    geometry = episode['geometry']
    kernels = ((geometry['spatial_ss'],geometry['spatial_gs']) if identity else model.kernels(geometry))
    leaves = tuple(k.detach().requires_grad_(backward) for k in kernels)
    total = grid.new_zeros((len(truth),len(grid))); esses = []
    def score(ap, c):
        output, ess = condition_field(changed_prior(c,geometry,leaves),c,episode['obs'])
        return scan_log_prob(torch.as_tensor(query['rss'][:,ap],dtype=grid.dtype,device=device),
                             output['mean'],output['variance'],episode['obs']), ess
    with torch.no_grad():
        for ap, c in episode['cases'].items():
            s, ess = score(ap,c); total += s
            if ess is not None:esses.append(ess)
    total.requires_grad_(backward)
    logp = total.log_softmax(1)
    loss = -(qw*logp[torch.arange(len(nearest),device=device),nearest]).sum()
    xy = logp.exp()@grid; errors = (xy-truth).norm(dim=1)
    if backward:
        adjoint, = torch.autograd.grad(loss,total)
        for ap, c in episode['cases'].items():
            s, _ = score(ap,c)
            if s.requires_grad:(s*adjoint).sum().backward()
        active = [(k,l.grad) for k,l in zip(kernels,leaves) if k.requires_grad and l.grad is not None]
        if active:torch.autograd.backward([k for k,_ in active],[g for _,g in active])
    return dict(nll=float(loss.detach()),mde=float((qw@errors).detach()),errors=errors.detach().tolist(),
                minimum_ESS=min(esses) if esses else None,predictions=xy.detach().tolist())
