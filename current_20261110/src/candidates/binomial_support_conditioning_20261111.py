"""Use actual support detection counts in the existing logit-field prior.

The legacy C step regresses logit((d+.5)/(n+1)) on a learned prior. With
n=1,d=0 this pseudo-target is .25, so a missed AP can INCREASE a prior
below .25. That is not the likelihood of the observed missed scan.

Keep the original spatial logit covariance K=g*exp(-distance/ell). At the
known support positions, z has mean logit(p_S) and covariance K; conditional
d_i ~ Binomial(n_i, sigmoid(z_i)). In whitened coordinates z=mu+L*u, minimize
  .5*u.T*u + sum_i[n_i*softplus(z_i)-d_i*z_i].
Its Hessian I+L.T*diag(n*p*(1-p))*L is positive definite. The MAP is unique.
Predict the latent-mode correction K_xS*(d-n*p_MAP), then apply sigmoid.
This is a mode plug-in, NOT an integrated posterior probability or an
exact Bayesian localization guarantee. It does not identify acquisition
changes between S and Q. No source fit, new field, kernel tuning or reader.

Mechanism source: Rasmussen & Williams, GPML (2006), chapter 3, section 3.4,
https://gaussianprocess.org/gpml/chapters/RW3.pdf . Here the binomial counts,
nonzero learned prior mean and batched whitened solve are project choices.
Only use H-derived priors evaluated AT S, with the det_stats position order.
"""
import numpy as np
from scipy.special import expit


def condition_detection(X, p_H, S_xy, p_S, n, detected, *, ell=16., g=1.):
    """Return H detection probabilities and the support-mode solve record.

    n[K], detected[K,A], p_S[K,A], p_H[N,A]; S_xy contains unique positions.
    Empty S or g=0 leaves the field unchanged. All-missed observations lower
    the latent mode everywhere for this positive kernel; all hits raise it.
    Zero likelihood gradient at the prior leaves its mode unchanged. These
    are properties of this conditional model, not of cross-device truth.
    """
    if not len(S_xy) or g == 0:
        return p_H.copy(), dict(iterations=0, max_gradient=0., changed=False)
    K = g*np.exp(-np.linalg.norm(S_xy[:, None]-S_xy[None], axis=2)/ell)
    L = np.linalg.cholesky(K)
    mu = np.log(np.clip(p_S, 1e-4, 1-1e-4)) - np.log1p(-np.clip(p_S, 1e-4, 1-1e-4))
    n = np.asarray(n)[:, None]
    u = np.zeros_like(mu)

    def objective(value):
        z = mu+L@value
        return .5*(value*value).sum(0)+(n*np.logaddexp(0., z)-detected*z).sum(0)

    for iteration in range(30):
        z = mu+L@u
        prob = expit(z)
        gradient = u+L.T@(n*prob-detected)
        max_gradient = float(np.max(np.abs(gradient)))
        if max_gradient <= 1e-8:
            break
        weight = n*prob*(1-prob)
        hessian = np.eye(len(S_xy))[None]+np.einsum('ij,ia,ik->ajk', L, weight, L)
        step = np.linalg.solve(hessian, gradient.T[..., None])[..., 0].T
        # Newton's direction is a descent direction; line search is part of
        # the convex solve, not a retry or fallback to the legacy estimator.
        scale = np.ones(mu.shape[1])
        before = objective(u)
        decrease = (gradient*step).sum(0)
        for _ in range(30):
            proposed = u-step*scale
            rejected = objective(proposed) > before-1e-4*scale*decrease+1e-12
            if not rejected.any():
                break
            scale[rejected] *= .5
        else:
            raise RuntimeError('Binomial support Newton line search did not converge')
        u = proposed
    else:
        raise RuntimeError(f'Binomial support mode did not converge: gradient={max_gradient}')

    alpha = detected-n*prob
    cross = g*np.exp(-np.linalg.norm(X[:, None]-S_xy[None], axis=2)/ell)
    prior = np.clip(p_H, 1e-4, 1-1e-4)
    prediction = expit(np.log(prior)-np.log1p(-prior)+cross@alpha)
    return prediction, dict(iterations=iteration, max_gradient=max_gradient,
                            support_probability=prob, changed=True)


def mechanism_check():
    """One bounded server-only check of the specific missing-scan inversion.

    Also check the two implementation-sensitive symmetries of joint S use.
    This is a numerical mechanism check, never a localization experiment.
    """
    S = np.array([[0., 0.]])
    prior = np.array([[.02, .2, .6, .9]])
    missed, miss_info = condition_detection(S, prior, S, prior, np.ones(1), np.zeros_like(prior))
    hit, _ = condition_detection(S, prior, S, prior, np.ones(1), np.ones_like(prior))
    assert np.all(missed < prior) and np.all(hit > prior)
    z0 = np.log(prior)-np.log1p(-prior)
    old_miss = expit(z0+(np.log(.25/.75)-z0)/(1+1/(2*.25*.75)))
    assert old_miss[0, 0] > prior[0, 0]
    unchanged, _ = condition_detection(S, prior, S, prior, np.ones(1), prior)
    np.testing.assert_allclose(unchanged, prior, rtol=0, atol=1e-12)
    empty, _ = condition_detection(S, prior, np.empty((0, 2)), np.empty((0, 4)), np.empty(0), np.empty((0, 4)))
    np.testing.assert_array_equal(empty, prior)
    S2 = np.array([[0., 0.], [8., 3.]])
    X = np.array([[2., 1.], [6., -2.]])
    pH, pS = np.array([[.1, .6], [.2, .8]]), np.array([[.02, .9], [.4, .3]])
    n, d = np.array([1., 3.]), np.array([[0., 1.], [2., 0.]])
    joint, info = condition_detection(X, pH, S2, pS, n, d)
    swapped, _ = condition_detection(X, pH, S2[::-1], pS[::-1], n[::-1], d[::-1])
    shifted, _ = condition_detection(X+[20., -11.], pH, S2+[20., -11.], pS, n, d)
    np.testing.assert_allclose(joint, swapped, rtol=0, atol=1e-9)
    np.testing.assert_allclose(joint, shifted, rtol=0, atol=1e-12)
    return dict(passed=True, prior=prior.tolist(), missed=missed.tolist(), hit=hit.tolist(),
                legacy_missed=old_miss.tolist(), max_gradient=max(info['max_gradient'], miss_info['max_gradient']),
                fits=0, localization_predictions=0)


if __name__ == '__main__':
    import json
    import socket
    assert socket.gethostname() == 'ray-web', 'Run the bounded mechanism check on the server'
    print(json.dumps(mechanism_check()))
