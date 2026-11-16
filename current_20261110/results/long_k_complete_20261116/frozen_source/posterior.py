"""Frozen ReFi numerical support policy, without old experiment imports.

proposal retains the legacy phi_s/theta compatibility signature. The identity
baseline supplies zeros; retaining the signature is not activating a learner.
"""
import math
import torch
from scipy.special import gammaincinv
from observations import _chol, _detection_logs, gaussian_update, observation_log_prob, predict_from_posterior
from sampling import setup, sample_reference, log_proposal


class SupportIntegral:
    def __init__(self, mean, covariance, stats, obs, power=13, seed=231024,
                 proposal="rss_gaussian"):
        self.mean, self.covariance, self.obs = mean, covariance, obs
        self.rss = gaussian_update(mean, covariance, stats, obs)
        self.analytic = not len(mean) or not bool((stats["n"] > 0).any())
        if self.analytic:
            return
        self.chol = _chol(covariance)
        if proposal not in ("rss_gaussian", "defensive_student"):
            raise ValueError(proposal)
        defensive = proposal == "defensive_student"
        dimension = len(mean)
        engine = torch.quasirandom.SobolEngine(dimension + int(defensive), scramble=True, seed=seed)
        uniform = engine.draw_base2(power - int(defensive), dtype=mean.dtype).to(mean.device)
        # Finite digital nodes -> open-cell midpoints, without tail clipping.
        uniform = (uniform * (2 ** 32 - 1) + .5) / 2 ** 32
        normals = torch.special.ndtri(uniform[:, :dimension])
        log_ratio = normals.new_zeros(len(normals))
        if defensive:
            # Fixed equal allocation to N(0,I) and multivariate t_3(0,I).
            # Balance weights use the mixture density for BOTH strata.
            df = 3.
            chi2 = torch.as_tensor(2 * gammaincinv(df / 2, uniform[:, -1].cpu().numpy()),
                                   dtype=mean.dtype, device=mean.device)
            normals = torch.cat((normals, normals * torch.sqrt(df / chi2)[:, None]))
            radius2 = normals.square().sum(-1)
            log_normal = -.5 * (dimension * math.log(2 * math.pi) + radius2)
            log_student = (math.lgamma((df + dimension) / 2) - math.lgamma(df / 2)
                           - dimension / 2 * math.log(df * math.pi)
                           - (df + dimension) / 2 * torch.log1p(radius2 / df))
            log_ratio = log_normal - torch.logsumexp(torch.stack((log_normal, log_student)), 0) + math.log(2)
        self.samples = self.rss["mean"] + normals @ _chol(self.rss["cov"]).T
        detected, missing = _detection_logs(self.samples, obs)
        log_likelihood = (stats["k"] * detected +
                          (stats["n"] - stats["k"]) * missing).sum(-1)
        log_importance = log_likelihood + log_ratio
        self.log_weights = torch.log_softmax(log_importance, dim=0)
        self.log_detection_evidence = torch.logsumexp(log_importance, 0) - math.log(len(normals))

    def components(self, query_mean, query_variance, cross):
        """Conditional marginals sharing ONE set of Support weights/nodes."""
        if self.analytic:
            prediction = predict_from_posterior(query_mean, query_variance, cross,
                                                self.mean, self.covariance, self.rss)
            return prediction["mean"][None], prediction["variance"], query_mean.new_zeros(1)
        A = torch.cholesky_solve(cross.T, self.chol).T
        variance = query_variance - (A * cross).sum(-1)
        if bool((variance < -1e-8).any()):
            raise RuntimeError("Prior joint covariance has negative conditional variance")
        # Schur complements are nonnegative; remove roundoff only.
        variance = variance.clamp_min(0)
        mean = query_mean[None] + (self.samples - self.mean) @ A.T
        return mean, variance, self.log_weights

    def log_prob(self, rss, query_mean, query_variance, cross, chunk_size=64):
        """One query marginal per RSS item; preserve the Gaussian mixture.

        Query RSS enters this readout only, never the Support weights. Chunking
        changes neither nodes nor normalization. AP likelihoods stay separate.
        """
        values = []
        for start in range(0, len(rss), chunk_size):
            sl = slice(start, start + chunk_size)
            mean, variance, log_weights = self.components(query_mean[sl], query_variance[sl], cross[sl])
            component_logp = observation_log_prob(rss[sl][None], mean, variance[None], self.obs)
            values.append(torch.logsumexp(log_weights[:, None] + component_logp, dim=0))
        return torch.cat(values)

    def diagnostics(self):
        if self.analytic:
            return dict(analytic=True, samples=0, support_mean=self.rss["mean"].detach().tolist(),
                        support_variance=self.rss["cov"].diag().detach().tolist())
        weights = self.log_weights.exp()
        mean = weights @ self.samples
        variance = weights @ (self.samples - mean).square()
        return dict(analytic=False, samples=len(weights),
                    effective_samples=(1 / weights.square().sum()).detach().item(),
                    max_weight=weights.max().detach().item(),
                    log_detection_evidence=self.log_detection_evidence.detach().item(),
                    support_mean=mean.detach().tolist(), support_variance=variance.detach().tolist())


class ImportanceSupport(SupportIntegral):
    """Fixed external proposal, with differentiable ORIGINAL posterior weights.

    Samples and proposal log-density are numerical auxiliaries, detached from
    model parameters. The RSS-updated Gaussian, normalized detection weights,
    and query conditional keep their model dependence. No IFT is used here.
    """
    def __init__(self, mean, covariance, stats, obs, samples, log_proposal):
        self.mean, self.covariance, self.obs = mean, covariance, obs
        self.rss = gaussian_update(mean, covariance, stats, obs)
        self.analytic = not len(mean) or not bool((stats["n"] > 0).any())
        if self.analytic:
            return
        self.chol = _chol(covariance)
        self.samples = samples.detach()
        gaussian = torch.distributions.MultivariateNormal(
            self.rss["mean"], scale_tril=_chol(self.rss["cov"]), validate_args=False)
        detected, missing = _detection_logs(self.samples, obs)
        log_likelihood = (stats["k"] * detected + (stats["n"] - stats["k"]) * missing).sum(-1)
        log_importance = gaussian.log_prob(self.samples) + log_likelihood - log_proposal.detach()
        self.log_weights = torch.log_softmax(log_importance, dim=0)
        self.log_detection_evidence = torch.logsumexp(log_importance, 0) - math.log(len(samples))


@torch.no_grad()
def proposal(case, obs, theta, seed):
    sm = case['sm'] + 20*(case['phi_s'] @ theta)
    stats = {k: case[k] for k in ('n', 'k', 's1', 's2')}
    target, initial, center, root, _ = setup(sm, case['sk'], stats, obs)
    samples, _, info = sample_reference(target, initial, center, root, seed, defensive_fraction=.1)
    if info['effective_samples'] >= 1000:
        return dict(samples=samples, log_proposal=log_proposal(samples, center, root, info)), dict(
            sampler='bounded_pmc', ESS=info['effective_samples'], samples=len(samples), seed=seed)
    # Same fixed two-scramble policy as the completed K-curve continuation.
    batches, means, covs, evidence, ess = [], [], [], [], []
    for mask in (0x13579BDF, 0x2468ACE0):
        p = SupportIntegral(sm, case['sk'], stats, obs, power=18,
                            seed=seed ^ mask, proposal='rss_gaussian')
        w = p.log_weights.exp()
        m = w @ p.samples
        centered = p.samples-m
        batches.append(p.samples)
        means.append(m)
        covs.append((centered.T*w)@centered)
        evidence.append(p.log_detection_evidence)
        ess.append(float(w.square().sum().reciprocal()))
    scale = ((covs[0].diag()+covs[1].diag())/2).sqrt()
    gaps = dict(mean=float((means[0]-means[1]).abs().max()),
                covariance=float(((covs[0]-covs[1]).abs()/(scale[:,None]*scale[None])).max()),
                log_evidence=float((evidence[0]-evidence[1]).abs()))
    record = dict(sampler='rss_gaussian_two_scrambles', seed=seed,
        original_ESS=info['effective_samples'], ESS=ess, gaps=gaps, samples=2**19,
        accepted=min(ess)>=1000 and gaps['mean']<=.1*obs['noise_std']
                 and gaps['covariance']<=.1 and gaps['log_evidence']<=.05)
    if not record['accepted']:
        raise RuntimeError(f'Fixed-budget source QMC failed: {record}')
    samples = torch.cat(batches)
    gaussian = torch.distributions.MultivariateNormal(
        p.rss['mean'], scale_tril=_chol(p.rss['cov']), validate_args=False)
    return dict(samples=samples, log_proposal=gaussian.log_prob(samples)), record
