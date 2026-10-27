"""TC_MSStack Eqs 1/15/16 on the existing LAW history-bank inputs.

Reuses the project's ICML2017 formulas and TNKN precursor's exact SPD solve.
The source Schur complement is algebraically the same dense GP, not a sparse
approximation. Group LOO removes ALL repeated scans at one target location.
Direct bounded lambda replaces the non-identifiable mu,b parameterization;
the kernel and stacking-weight families are unchanged.
"""
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize


SERVER = Path('/home/panyushuo/projects/panyushuo')
ORIGINAL = SERVER / 'icml17_replication/core.py'
SOLVE = SERVER / 'TNKN-UJI-20261013/gp.py'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


original = load_module('law_icml17_original', ORIGINAL)
precursor = load_module('law_transfer_precursor', SOLVE)


def weights(parameters):
    coefficients = torch.stack([p[-1] for p in parameters])
    # Same smoothed absolute value as the paper and existing implementation.
    # log(cosh z)=log1p(2*sinh(z/2)^2), avoiding subtraction near lambda=0.
    g = .01 * torch.log1p(2 * torch.sinh(coefficients / .02).square())
    if g.sum().detach().item() <= 0:
        raise ValueError('TC_MSStack Eq15 is undefined when every source coefficient is zero')
    return g / g.sum()


class SourcePair:
    def __init__(self, xs, xt, ys, yt, groups):
        self.xs, self.xt, self.ys, self.yt = xs, xt, ys, yt
        self.ss = original.distances(xs, xs)
        self.ts = original.distances(xt, xs)
        self.tt = original.distances(xt, xt)
        self.isource = torch.eye(len(xs), device=xs.device)
        self.itarget = torch.eye(len(xt), device=xs.device)
        self.groups = [torch.as_tensor(np.flatnonzero(groups == g), device=xs.device) for g in np.unique(groups)]

    @staticmethod
    def kernel(distance, p):
        return torch.exp(2 * p[1] - .5 * distance / torch.exp(2 * p[0]))

    def state(self, p):
        a = self.kernel(self.ss, p) + (torch.exp(2 * p[2]) + 1e-8) * self.isource
        b = p[4] * self.kernel(self.ts, p)
        u = precursor.SPDSolve.apply(a, torch.cat((self.ys[:, None], b.T), 1))
        mean = b @ u[:, 0]
        covariance = self.kernel(self.tt, p) + (torch.exp(2 * p[3]) + 1e-8) * self.itarget - b @ u[:, 1:]
        chol = torch.linalg.cholesky((covariance + covariance.T) / 2)
        alpha = torch.cholesky_solve((self.yt - mean)[:, None], chol)[:, 0]
        return u, chol, alpha

    def target_prediction(self, p, kind='group_loo'):
        _, chol, alpha = self.state(p)
        if kind == 'in_sample':
            return self.yt - (torch.exp(2 * p[3]) + 1e-8) * alpha
        precision = torch.cholesky_solve(self.itarget, chol)
        prediction = torch.empty_like(self.yt)
        for ids in self.groups:
            block = precision[ids[:, None], ids[None]]
            prediction[ids] = self.yt[ids] - torch.linalg.solve(block, alpha[ids])
        return prediction

    def predict(self, p, xq):
        u, chol, alpha = self.state(p)
        b = p[4] * self.kernel(original.distances(xq, self.xs), p)
        residual = self.kernel(original.distances(xq, self.xt), p) - b @ u[:, 1:]
        return b @ u[:, 0] + residual @ alpha

    def initial(self, rng):
        distances = torch.cat((self.ss.flatten(), self.ts.flatten(), self.tt.flatten()))
        positive = distances[distances > 0]
        length = float((positive.median() / 2).sqrt()) if len(positive) else 1.
        return np.r_[np.log(length), 0., np.log(.1), np.log(.1), .5] + rng.normal(0., .15, 5)


class EvaluationBudget(Exception): pass


def fit(pairs, seed, restarts=5, evaluations=200):
    rng = np.random.default_rng(seed); device = pairs[0].xs.device
    # Broad finite numerical domain in normalized feature/coordinate units.
    per_pair = [(math.log(1e-3), math.log(1e3)), (-5., 5.),
                (math.log(1e-4), math.log(10)), (math.log(1e-4), math.log(10)), (-1., 1.)]
    bounds = per_pair * len(pairs)
    runs = []; overall = None
    for restart in range(restarts):
        initial = np.concatenate([p.initial(rng) for p in pairs])
        trace = []; best = dict(loss=float('inf'))
        def objective(value):
            if len(trace) >= evaluations: raise EvaluationBudget()
            p = torch.as_tensor(value, dtype=torch.float64, device=device).requires_grad_()
            ps = p.reshape(len(pairs), 5)
            pred = weights(ps) @ torch.stack([g.target_prediction(v) for g, v in zip(pairs, ps)])
            loss = (pred - pairs[0].yt).square().mean()
            grad, = torch.autograd.grad(loss, p)
            if not torch.isfinite(loss) or not torch.isfinite(grad).all():
                raise FloatingPointError('Nonfinite TC_MSStack objective/gradient')
            loss_value = float(loss.detach()); gradient = grad.detach().cpu().numpy()
            trace.append(loss_value)
            if loss_value < best['loss']:
                best.update(loss=loss_value, parameters=value.copy(), gradient=gradient.copy())
            return loss_value, gradient
        started = time.perf_counter()
        try:
            result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                options=dict(maxiter=200, maxls=30, ftol=1e-12, gtol=1e-6))
            status = str(result.message); success = bool(result.success)
        except EvaluationBudget:
            status = '200 function-evaluation budget reached'; success = False
        record = dict(restart=restart, objective=best['loss'], evaluations=len(trace), status=status,
            optimizer_converged=success, parameters=best['parameters'].tolist(), trace=trace,
            gradient_max=float(np.abs(best['gradient']).max()),
            near_bounds=[i for i, (x, (lo, hi)) in enumerate(zip(best['parameters'], bounds)) if min(x-lo, hi-x)<1e-6],
            seconds=time.perf_counter()-started)
        runs.append(record)
        if overall is None or record['objective'] < overall['objective']: overall = record
    parameters = torch.tensor(overall['parameters'], device=device).reshape(len(pairs), 5)
    return parameters, weights(parameters).detach(), dict(chosen_restart=overall['restart'], runs=runs)


def check():
    """Validate the optimization shortcut and location-group exclusion, not performance."""
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(20261026)
    x = rng.normal(size=(14, 4)); y = rng.normal(size=14); groups = np.repeat(np.arange(3), 2)
    pair = SourcePair(torch.tensor(x[:8]), torch.tensor(x[8:]), torch.tensor(y[:8]), torch.tensor(y[8:]), groups)
    old = original.GP(x, y, [0]*8+[1]*6, 8)
    old_p = torch.tensor([.1, -.2, -1., -.7, math.log(1/3), 0.])
    p = torch.cat((old_p[:4], original.lam(old_p)[None])).requires_grad_()
    xq = torch.tensor(rng.normal(size=(4, 4)))
    torch.testing.assert_close(pair.predict(p, xq), old.predict(old_p, xq.numpy()), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(pair.target_prediction(p, 'in_sample'), old.target_prediction(old_p), rtol=1e-8, atol=1e-8)
    group_prediction = pair.target_prediction(p)
    for g in range(3):
        take = np.flatnonzero(groups != g); held = np.flatnonzero(groups == g)
        sub = SourcePair(pair.xs, pair.xt[take], pair.ys, pair.yt[take], groups[take])
        torch.testing.assert_close(group_prediction[held], sub.predict(p, pair.xt[held]), rtol=1e-8, atol=1e-8)
    analytic, = torch.autograd.grad(pair.target_prediction(p).square().mean(), p)
    finite = []
    for i in range(len(p)):
        d = torch.zeros_like(p); d[i] = 1e-5
        finite.append(float((pair.target_prediction(p+d).square().mean()-pair.target_prediction(p-d).square().mean())/2e-5))
    torch.testing.assert_close(analytic, torch.tensor(finite), rtol=1e-4, atol=1e-6)
    return dict(original_prediction_agreement=True, whole_position_exclusion=True,
                group_loo_gradient_max_error=float((analytic-torch.tensor(finite)).abs().max()))
