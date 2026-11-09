"""ReFi observation likelihood and conditional moment propagation.

Extracted verbatim from the existing inference.py functions. No posterior
solver, shared learning rule, or localization readout is changed here.
"""
import math
import torch

JITTER = 1e-6


def _double(value, like=None):
    return torch.as_tensor(value, dtype=torch.float64,
                           device=None if like is None else like.device)


def _chol(cov):
    return torch.linalg.cholesky(
        cov + JITTER * torch.eye(len(cov), dtype=cov.dtype, device=cov.device))


def _detection_logs(signal, obs):
    alpha = _double(obs["alpha"], signal)
    beta = _double(obs["beta"], signal)
    scale = _double(obs["detect_scale"], signal)
    z = (signal - beta) / scale
    detected = alpha.log() + torch.special.log_ndtr(z)
    # logaddexp's double backward can overflow for widely separated terms.
    # The same sum via logsumexp keeps backward exponentials at most one.
    missing = torch.logsumexp(torch.stack(torch.broadcast_tensors(
        torch.log1p(-alpha), alpha.log() + torch.special.log_ndtr(-z))), dim=0)
    return detected, missing


def observation_log_prob(rss, mean, variance, obs):
    """Broadcast elementwise log p(D=1,R=r) or log p(D=0) for RSS=100.

    This integrates the Gaussian latent signal exactly.  AP axes, if any,
    are not reduced; callers sum the relevant AP likelihoods for a scan.
    """
    mean = _double(mean)
    rss, variance = _double(rss, mean), _double(variance, mean)
    alpha, beta = _double(obs["alpha"], mean), _double(obs["beta"], mean)
    scale2 = _double(obs["detect_scale"], mean).square()
    noise2 = _double(obs["noise_std"], mean).square()
    total = variance + noise2
    gain = variance / total
    conditional_mean = mean + gain * (rss - mean)
    conditional_variance = variance * noise2 / total
    detected = (alpha.log() - 0.5 * (math.log(2 * math.pi) + total.log()
                                    + (rss - mean).square() / total)
                + torch.special.log_ndtr(
                    (conditional_mean - beta) / torch.sqrt(scale2 + conditional_variance)))
    missing = torch.logsumexp(torch.stack(torch.broadcast_tensors(
        torch.log1p(-alpha), alpha.log() + torch.special.log_ndtr(
            (beta - mean) / torch.sqrt(scale2 + variance)))), dim=0)
    return torch.where(rss == 100, missing, detected)


def gaussian_update(prior_mean, prior_cov, stats, obs):
    """Exactly absorb detected RSS factors using means and noise variance/k.

    Returns mean, cov, and negative log evidence of all detected RSS values.
    s2 affects the evidence constant, not the posterior.  The only numerical
    perturbation is JITTER in the observation covariance Cholesky.
    """
    mean = _double(prior_mean)
    cov = _double(prior_cov, mean)
    k, s1, s2 = (_double(stats[key], mean) for key in ("k", "s1", "s2"))
    keep = k > 0
    if not bool(keep.any()):
        return dict(mean=mean, cov=cov, loss=mean.new_zeros(()))

    count = k[keep]
    noise2 = torch.broadcast_to(_double(obs["noise_std"], mean).square(), mean.shape)[keep]
    average = s1[keep] / count
    cross = cov[:, keep]
    chol = _chol(cov[keep][:, keep] + torch.diag(noise2 / count))
    residual = average - mean[keep]
    weights = torch.cholesky_solve(residual[:, None], chol)[:, 0]
    updated_mean = mean + cross @ weights
    updated_cov = cov - cross @ torch.cholesky_solve(cross.T, chol)
    updated_cov = (updated_cov + updated_cov.T) / 2

    average_nll = 0.5 * (residual @ weights + 2 * chol.diag().log().sum()
                         + len(count) * math.log(2 * math.pi))
    within = s2[keep] - s1[keep].square() / count
    constant_nll = 0.5 * (((count - 1) * (math.log(2 * math.pi) + noise2.log())
                           + count.log() + within / noise2).sum())
    return dict(mean=updated_mean, cov=updated_cov,
                loss=average_nll + constant_nll)


def predict_from_posterior(query_mean, query_covdiag, cross, support_mean,
                           support_cov, posterior):
    """Propagate q(f_S) through the same prior conditional; return mean/variance."""
    mean = _double(query_mean)
    variance = _double(query_covdiag, mean)
    support_mean, support_cov = _double(support_mean, mean), _double(support_cov, mean)
    if not len(support_mean):
        return dict(mean=mean, variance=variance)
    cross = _double(cross, mean)
    A = torch.cholesky_solve(cross.T, _chol(support_cov)).T
    mean = mean + A @ (posterior["mean"] - support_mean)
    variance = variance + (A @ (posterior["cov"] - support_cov) * A).sum(dim=1)
    return dict(mean=mean, variance=variance)


def scan_log_prob(rss, mean, variance, obs):
    """Exact [scan, grid] scores, evaluating repeated RSS values only once.

    Non-detections and repeated integer RSS values share the same likelihood
    for a fixed target field. Reindexing restores every original scan and
    accumulates its gradient; no scans, APs or probability terms are dropped.
    """
    values, inverse = torch.unique(rss, sorted=True, return_inverse=True)
    return observation_log_prob(values[:, None], mean[None], variance[None], obs)[inverse]
