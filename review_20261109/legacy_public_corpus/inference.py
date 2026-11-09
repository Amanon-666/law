"""Scalar RSS/detection observations and full spatial Gaussian inference.

All values use raw dBm and torch.float64.  Given a latent signal f,
D ~ Bernoulli(alpha * Phi((f-beta)/detect_scale)); detected RSS is
N(f, noise_std**2).  Counts summarize original scans, without a binomial
coefficient: every acquired scan contributes its own likelihood factor.
"""

import math

import numpy as np
import torch
from scipy.optimize import minimize


JITTER = 1e-6  # dBm squared, used only inside Cholesky factorizations.
_nodes, _weights = np.polynomial.hermite.hermgauss(20)
GH_NODES = torch.tensor(_nodes * math.sqrt(2), dtype=torch.float64)
GH_WEIGHTS = torch.tensor(_weights / math.sqrt(math.pi), dtype=torch.float64)


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


def infer(prior_mean, prior_cov, stats, obs, steps, differentiable=False,
          hyperparameter_fit=False):
    """RSS update followed by a CPU L-BFGS-B full-Gaussian detection VI solve.

    q(z)=N(eta, B B.T), with packed lower-triangular B and positive diagonals.
    B's diagonals have a floating-point positive-domain bound of 1e-8.
    steps is the maximum number of SciPy L-BFGS-B iterations.
    The optimizer always sees detached base/observation parameters.
    differentiable=True uses the implicit gradient of a stationary local VI
    solution (at most 10 support positions), never the optimizer trajectory.
    This is the standard IFT cross derivative (cf. ADKF, arXiv:2205.02708,
    Eqs. 4-5); the inner variables here are VI coordinates, not ADKF kernel
    hyperparameters.  This module is not an ADKF reproduction.
    hyperparameter_fit=True holds the solved whitened q(z) fixed when taking
    the final ELBO's base/observation derivative; it does not build a Hessian.
    The two modes are mutually exclusive.  Default inference is detached.
    loss is the negative ELBO for the original RSS and detection factors.
    """
    if differentiable and hyperparameter_fit:
        raise ValueError("choose implicit prediction or ELBO hyperparameter fitting")
    track_model = differentiable or hyperparameter_fit
    mean = _double(prior_mean)
    cov = _double(prior_cov, mean)
    stats = {key: _double(stats[key], mean) for key in ("n", "k", "s1", "s2")}
    obs = {key: _double(obs[key], mean)
           for key in ("alpha", "beta", "detect_scale", "noise_std")}
    if not track_model:
        mean, cov = mean.detach(), cov.detach()
        stats = {key: value.detach() for key, value in stats.items()}
        obs = {key: value.detach() for key, value in obs.items()}

    baseline = gaussian_update(mean, cov, stats, obs)
    n = len(mean)
    if not n or not bool((stats["n"] > 0).any()):
        return baseline
    if differentiable and n > 10:
        raise ValueError("implicit full-Hessian VI is limited to 10 support positions")

    chol = _chol(baseline["cov"])
    rows, columns = torch.tril_indices(n, n, device=mean.device)
    diagonal = rows == columns
    nodes, weights = GH_NODES.to(mean.device), GH_WEIGHTS.to(mean.device)

    def variational(phi, center, root, likelihood):
        eta, packed = phi[:n], phi[n:]
        B = mean.new_zeros((n, n)).index_put((rows, columns), packed)
        posterior_mean = center + root @ eta
        posterior_chol = root @ B
        marginal_var = posterior_chol.square().sum(dim=1)
        signal = posterior_mean[None] + nodes[:, None] * marginal_var.sqrt()[None]
        log_detected, log_missing = _detection_logs(signal, likelihood)
        expected = (weights[:, None] * (
            stats["k"] * log_detected + (stats["n"] - stats["k"]) * log_missing)).sum()
        kl = 0.5 * (eta.square().sum() + B.square().sum()
                    - n - 2 * packed[diagonal].log().sum())
        return kl - expected, posterior_mean, posterior_chol

    center, root = baseline["mean"].detach(), chol.detach()
    likelihood = {key: value.detach() for key, value in obs.items()}
    diagonal_indices = n + torch.where(diagonal)[0].cpu().numpy()
    initial = np.zeros(n + len(rows), dtype=np.float64)
    initial[diagonal_indices] = 1.0
    lower_bound = 1e-8
    bounds = [(None, None)] * len(initial)
    for index in diagonal_indices:
        bounds[index] = (lower_bound, None)

    def value_and_gradient(value):
        phi = torch.from_numpy(value).requires_grad_()
        objective, _, _ = variational(phi, center, root, likelihood)
        gradient, = torch.autograd.grad(objective, phi)
        return objective.item(), gradient.detach().numpy()

    with torch.enable_grad():
        solved = minimize(value_and_gradient, initial, jac=True, method="L-BFGS-B",
                          bounds=bounds, options=dict(maxiter=steps, gtol=1e-8,
                                                     ftol=1e-14, maxls=40))
        if not np.isfinite(solved.fun) or not np.isfinite(solved.jac).all():
            raise RuntimeError("VI solve returned a nonfinite objective or gradient")
        optimum = torch.from_numpy(solved.x.copy())
        min_diagonal = optimum[n:][diagonal].min().detach()
        if differentiable:
            if min_diagonal.item() <= 100 * lower_bound:
                raise RuntimeError("VI solution is too close to the positive-diagonal bound for unconstrained IFT")
            point = optimum.requires_grad_()
            objective, _, _ = variational(point, baseline["mean"], chol, obs)
            gradient, = torch.autograd.grad(objective, point, create_graph=True)
            residual = gradient.detach().abs().max()
            if not bool(torch.isfinite(residual)) or residual.item() > 1e-5:
                raise RuntimeError(
                    f"VI is not stationary: max|gradient|={residual.item():.6g}; "
                    "implicit differentiation is undefined for an unfinished solve")
            hessian = torch.autograd.functional.hessian(
                lambda value: variational(value, center, root, likelihood)[0], optimum)
            hessian = ((hessian + hessian.T) / 2).detach()
            if not bool(torch.isfinite(hessian).all()):
                raise RuntimeError("VI Hessian is nonfinite")
            hessian_chol, info = torch.linalg.cholesky_ex(hessian)
            if info.item() != 0:
                raise RuntimeError("VI Hessian is not positive definite")
            correction = torch.cholesky_solve(gradient[:, None], hessian_chol)[:, 0]
            newton_step = correction.detach().abs().max()
            if not bool(torch.isfinite(newton_step)) or newton_step.item() > 1e-3:
                raise RuntimeError(
                    f"VI Newton correction is too large: {newton_step.item():.6g}")
            # Exactly the solved value in forward, -H^-1 F_phi,theta in backward.
            solution = optimum.detach() - (correction - correction.detach())
        else:
            solution = optimum

        objective, posterior_mean, posterior_chol = variational(
            solution, baseline["mean"], chol, obs)
        result = dict(mean=posterior_mean, cov=posterior_chol @ posterior_chol.T,
                      loss=baseline["loss"] + objective, min_diagonal=min_diagonal)
        if differentiable:
            result.update(stationarity=residual, newton_step=newton_step)
    if not track_model:
        result = {key: value.detach() for key, value in result.items()}
    return result


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


def _check():
    obs = dict(alpha=0.8, beta=-78.0, detect_scale=3.0, noise_std=4.0)
    mean = torch.tensor([-72.0], dtype=torch.float64)
    cov = torch.tensor([[25.0]], dtype=torch.float64)
    stats = dict(n=[2.0], k=[2.0], s1=[-142.0], s2=[10084.0])
    result = gaussian_update(mean, cov, stats, obs)
    denominator = 25 + 16 / 2 + JITTER
    torch.testing.assert_close(result["mean"], mean + 25 / denominator * (-71 - mean))
    torch.testing.assert_close(result["cov"], cov - 25 ** 2 / denominator)

    x, w = np.polynomial.hermite.hermgauss(100)
    latent = mean + math.sqrt(50) * torch.tensor(x, dtype=torch.float64)
    logs, _ = _detection_logs(latent, obs)
    rss = torch.tensor(-70.0, dtype=torch.float64)
    rss_log = -0.5 * (math.log(2 * math.pi * 16) + (rss - latent).square() / 16)
    numerical = torch.logsumexp(logs + rss_log + torch.tensor(w / math.sqrt(math.pi)).log(), dim=0)
    closed = observation_log_prob(rss, mean, cov.diag(), obs)[0]
    torch.testing.assert_close(closed, numerical, rtol=1e-10, atol=1e-10)
    missing = observation_log_prob(100, mean, cov.diag(), obs).exp()
    detected = 0.8 * torch.special.ndtr((mean + 78) / math.sqrt(34))
    torch.testing.assert_close(missing + detected, torch.ones_like(mean))
    endpoint = observation_log_prob(100, mean, cov.diag(), dict(obs, alpha=1.0))
    assert torch.isfinite(endpoint).all()

    # Extreme GH nodes must retain finite curvature even for zero counts.
    for z in (-50.0, 0.0, 50.0):
        signal = mean.new_tensor(obs["beta"] + z * obs["detect_scale"], requires_grad=True)
        for index in (0, 1):
            logp = _detection_logs(signal, obs)[index]
            gradient, = torch.autograd.grad(logp, signal, create_graph=True)
            curvature, = torch.autograd.grad(gradient, signal)
            nearby = []
            for offset in (1e-3, -1e-3):
                point = (signal.detach() + offset).requires_grad_()
                derivative, = torch.autograd.grad(_detection_logs(point, obs)[index], point)
                nearby.append(derivative)
            torch.testing.assert_close(curvature, (nearby[0] - nearby[1]) / 2e-3,
                                       rtol=1e-6, atol=1e-8)

    # Saturated detection previously left the forward finite but broke Adam
    # trajectory hypergradients.  The stationary VI solution is near q(z)=p(z).
    saturated = dict(alpha=0.9244192490906509, beta=-91.22905268124195,
                     detect_scale=2.3254537974881906, noise_std=2.690949295110751)
    counts = dict(n=[7.0], k=[7.0], s1=[-245.0], s2=[8575.0])
    shift = mean.new_zeros((), requires_grad=True)
    prior = mean.new_tensor([-35.0])
    covariance = mean.new_tensor([[4.0]])
    implicit = infer(prior + shift, covariance, counts, saturated, 160,
                     differentiable=True)
    derivative, = torch.autograd.grad(implicit["mean"].sum(), shift)
    epsilon = 1e-3
    plus = infer(prior + epsilon, covariance, counts, saturated, 160)
    minus = infer(prior - epsilon, covariance, counts, saturated, 160)
    finite_difference = (plus["mean"].sum() - minus["mean"].sum()) / (2 * epsilon)
    torch.testing.assert_close(derivative, finite_difference, rtol=1e-7, atol=1e-9)
    solved = infer(prior, covariance, counts, saturated, 160)
    torch.testing.assert_close(implicit["mean"], solved["mean"], rtol=0, atol=0)
    torch.testing.assert_close(implicit["cov"], solved["cov"], rtol=0, atol=0)


if __name__ == "__main__":
    _check()
    print("Gaussian update, exact scan density and saturated implicit-VI checks passed")
