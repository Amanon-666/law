"""S-only numerical proposals for a fixed fingerprint-change model.

These functions do not select model parameters or authorize a training retry.
They preserve the exact proposal density used in importance weighting.
"""
import torch
from posterior import proposal, ImportanceSupport, SupportIntegral
from observations import _chol


@torch.no_grad()
def support_summary(prior, case, obs):
    samples, logq = case['proposal']
    post = ImportanceSupport(prior['sm'],prior['sk'],case['stats'],obs,samples,logq)
    weights = post.log_weights.exp()
    mean = weights@post.samples
    centered = post.samples-mean
    return dict(ESS=float(weights.square().sum().reciprocal()),mean=mean,
                cov=(centered.T*weights)@centered)


@torch.no_grad()
def propose_support(prior, stats, obs, seed):
    """Original CPU M-PMC at the current S prior, without any query fields."""
    sm, sk = prior['sm'].detach().cpu(), prior['sk'].detach().cpu()
    data = {k:v.detach().cpu() for k,v in stats.items()}
    numerical, info = proposal(dict(sm=sm,sk=sk,phi_s=sm.new_zeros((len(sm),0)),**data),
                              obs,sm.new_empty(0),seed)
    device = prior['sm'].device
    return tuple(numerical[k].to(device) for k in ('samples','log_proposal')), info


@torch.no_grad()
def propose_qmc_support(prior, stats, obs, seed):
    """Existing RSS-Gaussian Sobol rule at fixed power18; no adaptive budget."""
    p = SupportIntegral(prior['sm'],prior['sk'],stats,obs,power=18,seed=seed,proposal='rss_gaussian')
    density = torch.distributions.MultivariateNormal(p.rss['mean'],scale_tril=_chol(p.rss['cov']),validate_args=False)
    return (p.samples,density.log_prob(p.samples)),dict(sampler='rss_gaussian_qmc',power=18,seed=seed)
