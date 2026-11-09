"""Separate residual variation shared across acquisitions from batch variation.

The H-only residual split follows the 2026-11-03 ReFi proposal. This is an
implementation of hierarchical GP conditioning, not a new performance claim.
One homogeneous acquisition supplies S; predictions describe a new acquisition.
Mixed-acquisition S needs a different support covariance and is NOT supported.
"""
import torch
from fields import rbf
from fingerprint_change import prepare_fingerprint
from fingerprint_posterior import support_summary
from observations import scan_log_prob,predict_from_posterior


class HistoricalResidualSplit:
    """D_parallel = D L' (L D L')^dagger L D, with sourcewise H contrasts.

    L is constructed solely from the separate historical posterior means at
    their historical locations. Redundant directions are removed by numerical
    rank, not treated as independent extra evidence. D_parallel + D_perp = D.
    """
    def __init__(self, histories, amplitude, length, intercept_std):
        self.amplitude, self.length, self.intercept_std = amplitude, length, intercept_std
        active = []
        for h in histories:
            if h is None:
                continue
            contrast = h.m-h.m.mean()
            contrast = contrast-contrast.mean()
            norm = torch.linalg.vector_norm(contrast)
            if float(norm) > 0:
                active.append((h.z, contrast/norm))
        self.active = active
        if not active:
            self.rank = 0
            return
        self.z = torch.cat([z for z,_ in active])
        linear = self.z.new_zeros((len(active),len(self.z)))
        start = 0
        for i,(z,c) in enumerate(active):
            linear[i,start:start+len(z)] = c
            start += len(z)
        gram = linear@self.kernel(self.z,self.z)@linear.T
        values,vectors = torch.linalg.eigh((gram+gram.T)/2)
        tolerance = len(active)*torch.finfo(gram.dtype).eps*values.abs().max()
        assert bool((values >= -10*tolerance).all())
        keep = values > tolerance
        self.rank = int(keep.sum())
        self.projector = linear.T@(vectors[:,keep]/values[keep].sqrt())

    def kernel(self, x, y):
        return rbf(x,y,self.amplitude,self.length)+self.intercept_std**2

    def features(self, x):
        if not self.active:
            return x.new_zeros((len(x),0))
        return self.kernel(x,self.z)@self.projector

    def components(self, x, y):
        parallel = self.features(x)@self.features(y).T
        return parallel, self.kernel(x,y)-parallel

    def covariance(self, x, y, groups_x, groups_y, retention):
        """Residual joint covariance for known group labels (including new ones).

        Each component is sqrt(rho)*shared + sqrt(1-rho)*group-specific GP.
        The caller supplies fixed retention fractions in [0,1].
        """
        assert bool(((retention>=0)&(retention<=1)).all())
        parallel,perpendicular = self.components(x,y)
        same = (groups_x[:,None] == groups_y[None]).to(x.dtype)
        return ((retention[0]+(1-retention[0])*same)*parallel+
                (retention[1]+(1-retention[1])*same)*perpendicular)


@torch.no_grad()
def prepare_acquisition_transfer(task, proposals):
    """Use only a single declared calibration acquisition, plus existing H.

    Group IDs are data permissions, never predictive features. This explicit
    check prevents silently reusing old mixed-acquisition S posterior caches.
    """
    assert len(set(task['support_acquisition_groups'])) == 1
    episode = prepare_fingerprint(task,proposals)
    cases = {}
    for ap,case in episode['cases'].items():
        histories = task['cases'][ap]['histories']
        amplitude = task['base']['drift'] if any(h is not None for h in histories) else task['base']['amplitude']
        split = HistoricalResidualSplit(histories,amplitude,task['base']['length'],
                                         task['state']['sigma']['intercept_std'])
        parallel,perpendicular = split.components(episode['grid'],episode['support'])
        summary = support_summary(case['prior'],case,episode['obs']) if len(episode['support']) else None
        if summary is not None and summary['ESS'] < 1000:
            raise RuntimeError('Fixed single-acquisition support posterior ESS below 1000')
        cases[ap] = dict(case,parallel_cross=parallel,perpendicular_cross=perpendicular,
                        history_contrast_rank=split.rank,support_posterior=summary)
    return dict(episode,cases=cases,calibration_acquisition=task['support_acquisition_groups'][0])


def acquisition_prior(retention, case):
    """Same S marginal and new-user variance; only between-acquisition covariance changes.

    rho=(1,1) is exactly the original prior. rho=(0,0) retains historical
    uncertainty's cross-covariance, so it does not mean ignoring S entirely.
    """
    assert bool(((retention>=0)&(retention<=1)).all())
    p = case['prior']
    return dict(p,cross=p['cross']+(retention[0]-1)*case['parallel_cross']+
                (retention[1]-1)*case['perpendicular_cross'])


def acquisition_field(retention, case):
    """Reuse the unchanged, censored S posterior; retain gradients through cross covariance."""
    p = acquisition_prior(retention,case)
    summary = case['support_posterior']
    field = predict_from_posterior(p['gm'],p['gv'],p['cross'],p['sm'],p['sk'],summary)
    if not bool(torch.isfinite(field['mean']).all() and torch.isfinite(field['variance']).all()
                and (field['variance']>=0).all()):
        raise RuntimeError('Invalid cross-acquisition field moments')
    return field,summary['ESS'] if summary is not None else None


def evaluate_acquisition_transfer(retention, episode, query, backward=False):
    """Complete-scan source location objective through the existing censored S posterior.

    Query labels enter only this supervised objective. The support marginal
    does not depend on retention, so its original-density proposal stays valid.
    """
    grid = episode['grid']; truth = grid.new_tensor(query['xy'])
    assert query['acquisition_group'] != episode['calibration_acquisition']
    assert not (truth[:,None] == episode['support'][None]).all(-1).any()
    nearest = torch.cdist(truth,grid).argmin(1)
    weights = grid.new_tensor(1/(len(query['positions'])*query['stats']['n'][query['group']]))
    def score(ap,case):
        field,ess = acquisition_field(retention,case)
        return scan_log_prob(grid.new_tensor(query['rss'][:,ap]),field['mean'],field['variance'],episode['obs']),ess
    total = grid.new_zeros((len(truth),len(grid))); esses = []
    with torch.no_grad():
        for ap,case in episode['cases'].items():
            value,ess = score(ap,case); total += value
            if ess is not None: esses.append(ess)
    total.requires_grad_(backward)
    logp = total.log_softmax(1)
    loss = -(weights*logp[torch.arange(len(truth)),nearest]).sum()
    if backward:
        adjoint, = torch.autograd.grad(loss,total)
        for ap,case in episode['cases'].items():
            value,_ = score(ap,case)
            if value.requires_grad: (value*adjoint).sum().backward()
    return dict(nll=float(loss.detach()),log_probability=logp.detach(),
                minimum_ESS=min(esses) if esses else None)


@torch.no_grad()
def build_acquisition_field(retention, episode):
    """H/S-only deployment field for subsequent independent acquisitions."""
    fields = []; diagnostics = []
    for ap,case in episode['cases'].items():
        field,ess = acquisition_field(retention,case)
        fields.append(field); diagnostics.append(dict(ap=ap,ESS=ess,rank=case['history_contrast_rank']))
    return dict(APs=list(episode['cases']),grid=episode['grid'],obs=episode['obs'],
                mean=torch.stack([f['mean'] for f in fields]),
                variance=torch.stack([f['variance'] for f in fields]),diagnostics=diagnostics)
