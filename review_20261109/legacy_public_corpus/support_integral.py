"""Small-S posterior integration under the existing detection/RSS model.

This is a numerical candidate, not a new observation model or a replacement
for historical-H inference. Gaussian RSS factors enter the proposal exactly
once; only detection factors enter its normalized weights. Finite-sample
estimates are biased and can miss rare posterior mass even with high ESS.
"""
import math

import torch
from scipy.special import gammaincinv

from inference import (_chol, _detection_logs, gaussian_update,
                       observation_log_prob, predict_from_posterior)


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


def prepare(prior, sx, stats, obs, power=13, seed=231024, proposal="rss_gaussian"):
    mean, features = prior.features(sx)
    covariance = prior.covariance(sx, sx, features, features)
    return SupportIntegral(mean, covariance, stats, obs, power, seed, proposal), features


def conditional_loss(prior, case, obs, power=13, seed=231024, chunk_size=64,
                     proposal="rss_gaussian"):
    """Tensor-valued source loss; gradients include nodes AND normalized weights."""
    posterior, sf = prepare(prior, case["sx"], case["stats"], obs, power, seed, proposal)
    return _readout_loss(prior, case, obs, posterior, sf, chunk_size), posterior


def importance_loss(prior, case, obs, samples, log_proposal, chunk_size=64):
    sm, sf = prior.features(case["sx"])
    sk = prior.covariance(case["sx"], case["sx"], sf, sf)
    posterior = ImportanceSupport(sm, sk, case["stats"], obs, samples, log_proposal)
    return _readout_loss(prior, case, obs, posterior, sf, chunk_size), posterior


def _readout_loss(prior, case, obs, posterior, sf, chunk_size):
    qm, qf = prior.features(case["qx"])
    cross = prior.covariance(case["qx"], case["sx"], qf, sf)
    diagonal = prior.diagonal(case["qx"], qf)
    group = case["group"]
    logp = posterior.log_prob(case["rss"], qm[group], diagonal[group], cross[group], chunk_size)
    return -(case["weights"] * logp).sum()
