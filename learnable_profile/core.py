"""A fixed GP prior + an external differentiable mean correction g(x).

Only g is differentiated. Each floor uses the existing Gaussian-moment
Support readout, with differentiable evidence, then probabilities are mixed.
No training loop, new encoder, variance head, or shared-module changes.
"""
import torch

from inference import predict_from_posterior
from reference_pmc import setup, sample_reference, log_proposal
from support_integral import ImportanceSupport


class MeanCorrectedPrior:
    def __init__(self, prior, correction):
        self.prior, self.correction = prior, correction

    def features(self, x):
        with torch.no_grad():
            mean, features = self.prior.features(x)
        return mean + self.correction(x), features

    @torch.no_grad()
    def covariance(self, x, y, fx, fy):
        return self.prior.covariance(x, y, fx, fy)

    @torch.no_grad()
    def diagonal(self, x, fx):
        return self.prior.diagonal(x, fx)


@torch.no_grad()
def fixed_components(profile, attenuation):
    attenuation = float(torch.as_tensor(attenuation).detach())
    shifts = attenuation * profile.offset_units
    if bool((shifts == 0).all()):
        # All components are the SAME Bank+g, so summing their prior masses
        # gives one exactly. Positive/unequal shifts are never merged.
        return [profile.bank], shifts.new_zeros(1), ['identity']
    return ([profile.component(attenuation, i) for i in range(len(profile.labels))],
            profile.history_log_mass(attenuation).detach(), list(profile.labels))


def _check_ess(ess, minimum, label):
    value = float(ess.detach()) if torch.is_tensor(ess) else float(ess)
    if not value >= minimum:
        raise RuntimeError(f'Insufficient ESS for component {label}: {value:.6g} < {minimum}')


@torch.no_grad()
def prepare_prior_proposal(prior, obs, sx, stats, correction, seed,
                           min_ess=1000, label='prior'):
    """M-PMC .1 defensive proposal for any existing prior, including intercepts.

    Reuse this frozen proposal for finite differences; no automatic retry.
    """
    if not len(sx) or not bool((stats['n'] > 0).any()):
        return None
    wrapped = MeanCorrectedPrior(prior, correction)
    mean, features = wrapped.features(sx)
    cov = wrapped.covariance(sx, sx, features, features)
    target, initial, center, root, _ = setup(mean, cov, stats, obs)
    samples, logw, info = sample_reference(target, initial, center, root, seed,
                                          defensive_fraction=.1)
    _check_ess(info['effective_samples'], min_ess, label)
    return dict(samples=samples.detach(),
                log_proposal=log_proposal(samples, center, root, info).detach(),
                reference_log_weights=logw.detach(), info=info)


def prepare_proposals(profile, attenuation, obs, sx, stats, correction, seed,
                      min_ess=1000):
    """Floor wrapper: one frozen proposal per retained component."""
    priors, _, labels = fixed_components(profile, attenuation)
    return [prepare_prior_proposal(prior, obs, sx, stats, correction, seed, min_ess, label)
            for prior, label in zip(priors, labels)]


def _component_field(prior, obs, sx, stats, points, proposal, min_ess, label):
    qm, qf = prior.features(points)
    qv = prior.diagonal(points, qf)
    if not len(sx) or not bool((stats['n'] > 0).any()):
        return dict(mean=qm, variance=qv, log_evidence=qm.new_zeros(()),
                    effective_samples=None)
    sm, sf = prior.features(sx)
    sk = prior.covariance(sx, sx, sf, sf)
    posterior = ImportanceSupport(sm, sk, stats, obs,
                                  proposal['samples'], proposal['log_proposal'])
    weights = posterior.log_weights.exp()
    ess = weights.square().sum().reciprocal()
    _check_ess(ess, min_ess, label)
    mean = weights @ posterior.samples
    centered = posterior.samples - mean
    moments = dict(mean=mean, cov=(centered.T * weights) @ centered)
    cross = prior.covariance(points, sx, qf, sf)
    prediction = predict_from_posterior(qm, qv, cross, sm, sk, moments)
    # RSS factors are already absorbed in posterior.rss. Multiplying its
    # evidence by the detection integral uses each Support factor once.
    logz = posterior.log_detection_evidence - posterior.rss['loss']
    return dict(**prediction, log_evidence=logz, effective_samples=ess)


def field_from_prior(prior, obs, sx, stats, points, correction, proposal,
                     min_ess=1000, label='prior'):
    """Any fixed prior -> differentiable mean/variance/log_evidence/ESS.

    Does not require FloorProfile or a floor attenuation parameter. The
    caller scores these Gaussian moments with observation_log_prob.
    """
    return _component_field(MeanCorrectedPrior(prior, correction), obs, sx, stats,
                            points, proposal, min_ess, label)


def field_from_profile(profile, attenuation, obs, sx, stats, points, correction,
                       proposals, min_ess=1000):
    """Differentiable g -> component moments/logZ_S -> q_HS -> scan mixture.

    w, attenuation, q_H, base/obs and H are fixed inputs. At attenuation=0
    this is Bank+g; at g=0 as well it recovers the original Bank path.
    """
    priors, logq_h, labels = fixed_components(profile, attenuation)
    if len(proposals) != len(priors):
        raise ValueError('Proposal count must match every retained floor/null component')
    values = [field_from_prior(prior, obs, sx, stats, points, correction,
                               proposal, min_ess, label)
              for prior, proposal, label in zip(priors, proposals, labels)]
    evidence = torch.stack([v['log_evidence'] for v in values])
    joint = logq_h + evidence
    log_mass = torch.log_softmax(joint, 0)
    means = torch.stack([v['mean'] for v in values])
    variances = torch.stack([v['variance'] for v in values])
    mean = log_mass.exp() @ means
    variance = log_mass.exp() @ (variances + (means - mean).square())
    return dict(mean=mean, variance=variance, component_means=means,
                component_variances=variances, component_log_evidence=evidence,
                history_log_mass=logq_h, log_mass=log_mass,
                log_evidence=torch.logsumexp(joint, 0), labels=labels,
                effective_samples=[v['effective_samples'] for v in values])


def scan_scores(rss, fields, obs):
    """rss [scans, AP]; first mix each AP's floors, then sum AP log scores.

    Fields share the same candidate coordinates. For a shared environment
    latent integrated rather than fixed, its integral belongs OUTSIDE this sum.
    """
    if rss.shape[1] != len(fields):
        raise ValueError('Every supplied scan AP must have exactly one field')
    from physical_prior_candidate.floor_profile import scan_log_prob
    return torch.stack([scan_log_prob(rss[:, a, None], field, obs)
                        for a, field in enumerate(fields)]).sum(0)
