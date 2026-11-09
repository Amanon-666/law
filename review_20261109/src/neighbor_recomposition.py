"""Two-point, H/S-conditioned linear operators for the existing ReFi path.

Ten shared edge coefficients extend the original twelve parameters. Setting
them to zero recovers any original ReFi model, including nonzero eta. They are
trained across tasks, not fitted separately at target locations. S-conditioned
operators define a conditional predictor, not a fixed-prior generative model.
No localization gain is established by this implementation.
"""
import hashlib
import torch
from torch import nn
from fields import rbf
from recomposition import Parameters, Source, features, prepare, prior, condition_field
from observations import scan_log_prob


class NeighborParameters(Parameters):
    def __init__(self, center, scale):
        super().__init__(center, scale)
        self.neighbor = nn.Parameter(torch.zeros(10, dtype=torch.float64))

    def penalty(self):
        return super().penalty() + .001*self.neighbor.square().sum()


def support_summary(x, sx, stats, source_mean, length):
    """Only observed RSS enters this descriptor; censoring stays in posterior."""
    k = stats['k']
    fraction = k/stats['n']
    residual = torch.where(k > 0, (stats['s1']/k.clamp_min(1)-source_mean)/20, 0.)
    weights = rbf(x, sx, 1., length)*fraction[None]
    total = weights.sum(1)
    return weights@residual/(1+total), total/(1+total)


@torch.no_grad()
def prepare_neighbors(histories, positions, delta, sx, grid, base, intercept, stats, chunk=128):
    """Construct H/S-only geometry. Grid coordinates are candidate inputs."""
    if not any(h is not None for h in histories):
        return dict(active=False, legacy=prepare(histories, positions, delta, sx, grid, base, intercept))
    anchors = torch.unique(torch.cat(positions), dim=0)
    sources = [Source(h, anchors, base) for h in histories]

    def at(x):
        parts = [source.at(x) for source in sources]
        for j, (part, source) in enumerate(zip(parts, sources)):
            part['missing'] = source.missing
            part['coverage'] = -torch.expm1(-rbf(x, torch.unique(positions[j], dim=0), 1., base['length']).sum(1))
            sm = source.p.features(sx)[0]
            part['support_residual'], part['support_coverage'] = support_summary(x, sx, stats, sm, base['length'])
        return parts

    aa, ss = at(anchors), at(sx)
    return dict(active=True, sources=sources, anchors=anchors, aa=aa, ss=ss,
                za=features(anchors, aa, positions, delta, base),
                zs=features(sx, ss, positions, delta, base),
                chunks=[(x, at(x)) for x in grid.split(chunk)],
                sx=sx, grid=grid, base=base, intercept=intercept,
                positions=positions, delta=delta)


def edge_features(part, anchor, delta, length):
    dm = (anchor['m'][None]-part['m'][:, None])/20
    dv = (anchor['v'][None]-part['v'][:, None])/400
    distance = ((part['x'][:, None]-anchor['x'][None])**2).sum(-1)/length**2
    r, ar = part['support_residual'][:, None], anchor['support_residual'][None]
    return torch.stack((dm, (anchor['m'][None]+part['m'][:, None]+150)/40,
                        dv, (anchor['v'][None]+part['v'][:, None])/800,
                        distance, anchor['coverage'][None]-part['coverage'][:, None],
                        (0. if delta is None else delta/3)*dm, ar-r, (ar+r)/2,
                        anchor['support_coverage'][None]-part['support_coverage'][:, None]), -1)


def coefficients(model, part, anchor, delta, length):
    edge = edge_features(part, anchor, delta, length)
    eta = .5*(model.theta[9]*model.mask[9]+edge@model.neighbor).tanh()
    c = part['w']*eta
    return 1-c.sum(1), c


def operator_mean(source, part, coeff):
    d, c = coeff
    return d*part['m']+c@source.ma


def operator_covariance(source, left, right, lc, rc, diagonal=False):
    d, c = lc
    if diagonal:
        return d.square()*left['v']+2*d*(c*left['ca']).sum(1)+((c@source.aa)*c).sum(1)
    e, b = rc
    raw = source.p.covariance(left['x'], right['x'], left['f'], right['f'])
    return (d[:, None]*raw*e[None]+d[:, None]*(left['ca']@b.T)
            +(c@right['ca'].T)*e[None]+c@source.aa@b.T)


def neighbor_prior(model, block, weights, support_only=False):
    if not block['active']:
        return prior(model, block['legacy'], weights)
    sources, aa, ss = block['sources'], block['aa'], block['ss']
    base, sx, grid = block['base'], block['sx'], block['grid']
    length, intercept = base['length'], block['intercept']
    sc = [coefficients(model, s, a, dz, length) for s, a, dz in zip(ss, aa, block['delta'])]
    alpha = model.weights(block['zs'], block['za'], weights)
    _, _, lg, ls = model.values()
    sm = (alpha*torch.stack([operator_mean(p, s, c) for p, s, c in zip(sources, ss, sc)])).sum(0)
    sk = sum(alpha[j, :, None]*alpha[j, None]*operator_covariance(p, s, s, c, c)
             for j, (p, s, c) in enumerate(zip(sources, ss, sc)))
    sk = ls.mul(2).exp()*sk+rbf(sx, sx, base['drift'], length)+intercept**2
    if support_only:
        return dict(sm=-75+lg.exp()*(sm+75), sk=sk)
    chunks = []
    for x, gg in block['chunks']:
        gc = [coefficients(model, g, a, dz, length) for g, a, dz in zip(gg, aa, block['delta'])]
        z = features(x, gg, block['positions'], block['delta'], base)
        beta = model.weights(z, block['za'], weights)
        gm = (beta*torch.stack([operator_mean(p, g, c) for p, g, c in zip(sources, gg, gc)])).sum(0)
        cross = sum(beta[j, :, None]*alpha[j, None]*operator_covariance(p, g, s, c, d)
                    for j, (p, g, s, c, d) in enumerate(zip(sources, gg, ss, gc, sc)))
        gv = sum(beta[j].square()*operator_covariance(p, g, g, c, c, True)
                 for j, (p, g, c) in enumerate(zip(sources, gg, gc)))
        chunks.append(dict(gm=-75+lg.exp()*(gm+75),
            cross=ls.mul(2).exp()*cross+rbf(x, sx, base['drift'], length)+intercept**2,
            gv=ls.mul(2).exp()*gv+base['drift']**2+intercept**2))
    return dict(sm=-75+lg.exp()*(sm+75), sk=sk,
                **{key:torch.cat([c[key] for c in chunks]) for key in ('gm', 'cross', 'gv')})


def neighbor_scores(model, case, weights, obs):
    f, ess = condition_field(neighbor_prior(model, case['blocks'], weights), case, obs)
    return scan_log_prob(case['rss'], f['mean'], f['variance'], obs), ess


@torch.no_grad()
def refresh_case(model, case, weights, obs, seed):
    from posterior import proposal
    p = neighbor_prior(model, case['blocks'], weights, support_only=True)
    k = len(p['sm'])
    if not k:
        return dict(case, proposal=None), dict(sampler='no_support')
    numerical, info = proposal(dict(sm=p['sm'], sk=p['sk'], phi_s=p['sm'].new_zeros((k, 0)),
                                    **case['stats']), obs, p['sm'].new_empty(0), seed)
    return dict(case, proposal=(numerical['samples'], numerical['log_proposal'])), info


@torch.no_grad()
def build_field(model, cases, positions, delta, base, obs, state, grid, event_id):
    """Adapt then cache with H/S only; reuse the established locate function."""
    weights = grid.new_tensor(state['weights'])
    means, variances, diagnostics = [], [], []
    for ap, c in cases.items():
        b = prepare_neighbors(c['histories'], positions, delta, c['sx'], grid, base,
                              state['sigma']['intercept_std'], c['stats'])
        seed = int.from_bytes(hashlib.sha256(f'neighbor-refi/{event_id}/{ap}'.encode()).digest()[:4], 'little')
        case, info = refresh_case(model, dict(blocks=b, stats=c['stats']), weights, obs, seed)
        f, ess = condition_field(neighbor_prior(model, b, weights), case, obs)
        means.append(f['mean']); variances.append(f['variance'])
        diagnostics.append(dict(ap=ap, ESS=ess, sampler=info['sampler']))
    return dict(APs=list(cases), grid=grid, mean=torch.stack(means),
                variance=torch.stack(variances), obs=obs, diagnostics=diagnostics)


def load_parameters(path):
    state = torch.load(path, weights_only=True)['model']
    model = NeighborParameters(state['center'], state['scale'])
    model.load_state_dict(state)
    return model.eval()
