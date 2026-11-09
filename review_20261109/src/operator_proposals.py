"""Equal-stratum importance proposals covering every current operator state.

Numerical nodes only: no change to the prior, state probabilities or likelihood.
Every node uses the SAME mixture density, irrespective of its generating state.
Nodes and density are frozen for subsequent differentiable importance weights.
"""
import math
import torch
from observations import gaussian_update, _chol
from posterior import ImportanceSupport


@torch.no_grad()
def propose_states(priors, stats, obs, power, seed):
    """Take 2**power scrambled Sobol nodes from each RSS-updated Gaussian.

    Call only for a nonempty support. Allocate equally across numerical
    components; this is unrelated to the probabilistic model's state masses.
    """
    gaussians=[];parts=[]
    for index,p in enumerate(priors):
        rss=gaussian_update(p['sm'],p['sk'],stats,obs)
        density=torch.distributions.MultivariateNormal(rss['mean'],scale_tril=_chol(rss['cov']),
                                                      validate_args=False)
        engine=torch.quasirandom.SobolEngine(len(p['sm']),scramble=True,seed=seed+index)
        u=engine.draw_base2(power,dtype=p['sm'].dtype).to(p['sm'].device)
        # Same open digital-cell mapping used by the existing SupportIntegral.
        u=(u*(2**32-1)+.5)/2**32
        parts.append(density.loc+torch.special.ndtri(u)@density.scale_tril.T)
        gaussians.append(density)
    samples=torch.cat(parts)
    logq=torch.logsumexp(torch.stack([g.log_prob(samples) for g in gaussians]),0)-math.log(len(gaussians))
    return (samples,logq)


@torch.no_grad()
def summarize_states(priors, stats, obs, numerical):
    """Retain evidence as well as moments: mixture selection needs both."""
    summaries=[]
    for p in priors:
        post=ImportanceSupport(p['sm'],p['sk'],stats,obs,*numerical)
        mass=post.log_weights.exp();mean=mass@post.samples;d=post.samples-mean
        summaries.append(dict(ESS=float(mass.square().sum().reciprocal()),mean=mean,
            covariance=(d.T*mass)@d,log_evidence=float(post.log_detection_evidence-post.rss['loss'])))
    return summaries


def compare_scrambles(first,second):
    """Fixed accuracy gates; a caller must not install a rejected proposal."""
    differences=[]
    for a,b in zip(first,second):
        mean_gap=float((a['mean']-b['mean']).abs().max())
        cov_gap=float(torch.linalg.matrix_norm(a['covariance']-b['covariance'])/
                      torch.linalg.matrix_norm((a['covariance']+b['covariance'])/2))
        evidence_gap=abs(a['log_evidence']-b['log_evidence'])
        minimum=min(a['ESS'],b['ESS'])
        differences.append(dict(minimum_ESS=minimum,mean_gap=mean_gap,
            relative_covariance_gap=cov_gap,log_evidence_gap=evidence_gap,
            accepted=minimum>=1000 and mean_gap<=.1 and cov_gap<=.01 and evidence_gap<=.01))
    return dict(accepted=all(d['accepted'] for d in differences),states=differences)
