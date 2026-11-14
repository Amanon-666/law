"""Historical GP fitting with the original ReFi availability rule.

Use only to fill genuinely missing history assets under a frozen data/parameter
manifest. Not a replacement of the target M-PMC posterior path.
"""
import numpy as np
import torch
import history_fit

def minimize(*args, **kwargs):
    result = history_fit.minimize(*args, **kwargs)
    del result.hess_inv
    return result
from observations import _double, _chol, _detection_logs, gaussian_update
from adaptation import GH_NODES, GH_WEIGHTS


def infer(prior_mean, prior_cov, stats, obs, steps, differentiable=False,
          hyperparameter_fit=False, stationarity_tolerance=None):
    """RSS update followed by a CUDA float64 objective with CPU L-BFGS-B full-Gaussian detection VI solve.

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
    stationarity_tolerance optionally rejects an unfinished or boundary
    inner solve before using its hyperparameter envelope derivative.
    The two modes are mutually exclusive.  Default inference is detached.
    loss is the negative ELBO for the original RSS and detection factors.
    """
    if differentiable and hyperparameter_fit:
        raise ValueError("choose implicit prediction or ELBO hyperparameter fitting")
    track_model = differentiable or hyperparameter_fit
    mean = _double(prior_mean).to("cuda")
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
        return {key: value.cpu() for key, value in baseline.items()}
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
        phi = torch.from_numpy(value).to(mean.device).requires_grad_()
        objective, _, _ = variational(phi, center, root, likelihood)
        gradient, = torch.autograd.grad(objective, phi)
        return objective.item(), gradient.detach().cpu().numpy()

    with torch.enable_grad():
        solved = minimize(value_and_gradient, initial, jac=True, method="L-BFGS-B",
                          bounds=bounds, options=dict(maxiter=steps, gtol=1e-8,
                                                     ftol=1e-14, maxls=40))
        if not np.isfinite(solved.fun) or not np.isfinite(solved.jac).all():
            raise RuntimeError("VI solve returned a nonfinite objective or gradient")
        optimum = torch.from_numpy(solved.x.copy()).to(mean.device)
        min_diagonal = optimum[n:][diagonal].min().detach()
        if hyperparameter_fit and stationarity_tolerance is not None:
            residual = float(np.abs(solved.jac).max())
            if residual > stationarity_tolerance or min_diagonal.item() <= 100*lower_bound:
                raise RuntimeError(
                    f'Unfinished/interior VI envelope: gradient={residual:.6g}, '
                    f'minimum diagonal={min_diagonal.item():.6g}')
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
        if hyperparameter_fit:
            result.update(inner_stationarity=mean.new_tensor(float(np.abs(solved.jac).max())),
                          inner_iterations=mean.new_tensor(solved.nit),
                          solver_success=mean.new_tensor(bool(solved.success)))
        if differentiable:
            result.update(stationarity=residual, newton_step=newton_step)
    if not track_model:
        result = {key: value.detach() for key, value in result.items()}
    return {key: value.cpu() for key, value in result.items()}
