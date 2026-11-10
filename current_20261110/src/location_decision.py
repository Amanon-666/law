"""Distance-risk action, extracted unchanged from the 2026-10-19 audit.

The smooth objective gives an explicit bound on the unsmoothed risk gap.
This minimizes model risk; it does not promise smaller true location error.
Source: FeMLoc-Decision-Audit-20261019/scripts/audit_spatial_decision.py.
"""
import numpy as np
from scipy.optimize import minimize

EPS = 1e-3


def geometric_median(candidates, weights):
    """Return approximate median and certified gap for the *unsmoothed* risk."""
    def quantities(action):
        d = action - candidates
        radius = np.sqrt((d * d).sum(1) + EPS ** 2)
        gradient = (weights[:, None] * d / radius[:, None]).sum(0)
        curvature = weights / radius ** 3
        hxx = np.sum(curvature * (d[:, 1] ** 2 + EPS ** 2))
        hyy = np.sum(curvature * (d[:, 0] ** 2 + EPS ** 2))
        hxy = -np.sum(curvature * d[:, 0] * d[:, 1])
        return weights @ radius, gradient, np.array([[hxx, hxy], [hxy, hyy]])
    result = minimize(lambda a: quantities(a)[:2], weights @ candidates, jac=True,
                      hess=lambda a: quantities(a)[2], method='trust-exact',
                      options=dict(gtol=1e-9, maxiter=200))
    action = result.x
    smooth, gradient, _ = quantities(action)
    risk = weights @ np.linalg.norm(action - candidates, axis=1)
    # True optimum lies in conv(C). Linearize the smooth convex objective there,
    # then subtract its maximum possible smoothing error eps * sum(weights).
    lower = smooth + np.min(candidates @ gradient) - action @ gradient - EPS * weights.sum()
    gap = max(0., float(risk - lower))
    if not np.isfinite(gap) or gap > 2 * EPS + 1e-10:
        raise RuntimeError(f'Geometric median not certified: gap={gap}, solver={result.message}')
    return action, gap
