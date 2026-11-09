"""Gaussian q/p evidence recycling, following Modular GP (NeurIPS 2021), eq. 4.

This is finite-dimensional Gaussian algebra, not that paper's variational
training algorithm or a replacement ReFi predictor. Correlations must be
supplied by an explicit joint prior. Each acquisition module is used once.
Non-Gaussian cached q gives approximate evidence, not its original likelihood.
"""
import torch


class GaussianEvidence:
    def __init__(self, prior_mean, prior_chol, posterior_mean, posterior_cov):
        self.prior_mean, self.prior_chol = prior_mean, prior_chol
        self.mean = torch.linalg.solve_triangular(
            prior_chol, (posterior_mean-prior_mean)[:, None], upper=False)[:, 0]
        left = torch.linalg.solve_triangular(prior_chol, posterior_cov, upper=False)
        cov = torch.linalg.solve_triangular(prior_chol, left.T, upper=False).T
        self.root = torch.linalg.cholesky((cov+cov.T)/2)

    def log_ratio(self, z):
        """log q_z(z) - log N(z;0,I), including its Gaussian constant."""
        v = torch.linalg.solve_triangular(self.root, (z-self.mean).T, upper=False).T
        return -.5*(v.square().sum(-1)-z.square().sum(-1))-self.root.diag().log().sum()

    def natural(self):
        inv = torch.cholesky_inverse(self.root)
        eta = inv@self.mean
        constant = -self.root.diag().log().sum()-.5*self.mean@eta
        return inv-torch.eye(len(inv), dtype=inv.dtype, device=inv.device), eta, constant


def combine_whitened(modules, joint_prior_cov):
    """Combine distinct q_j/p_j factors under zero-mean joint prior B.

    B refers to the stacked original whitened source variables, not source
    labels or a correlation guessed from posterior means. A non-PD combined
    precision raises: clipping it would silently change the statistical model.
    A negative individual site precision alone is NOT an error.
    """
    sites = [m.natural() for m in modules]
    prior_root = torch.linalg.cholesky(joint_prior_cov)
    precision = torch.cholesky_inverse(prior_root)+torch.block_diag(*[s[0] for s in sites])
    root = torch.linalg.cholesky((precision+precision.T)/2)
    eta = torch.cat([s[1] for s in sites])
    mean = torch.cholesky_solve(eta[:, None], root)[:, 0]
    log_normalizer = (sum(s[2] for s in sites)-prior_root.diag().log().sum()
                      -root.diag().log().sum()+.5*eta@mean)
    return dict(mean=mean, cov=torch.cholesky_inverse(root), log_normalizer=log_normalizer)
