"""One-position SupportIntegral using deterministic Gaussian quadrature.

Same censored posterior and complete query mixture as SupportIntegral. This is
only for K=1, not a replacement for multidimensional QMC. Accuracy must be
checked by order comparison; the concentration of node weights is not MC ESS.
"""
import math
import numpy as np
import torch

from observations import _chol, _detection_logs, gaussian_update, observation_log_prob
from posterior import SupportIntegral


class ScalarSupportIntegral(SupportIntegral):
    def __init__(self, mean, covariance, stats, obs, order):
        assert len(mean) == 1 and covariance.shape == (1, 1)
        self.mean, self.covariance, self.obs = mean, covariance, obs
        self.stats = stats
        self.rss = gaussian_update(mean, covariance, stats, obs)
        self.analytic = not bool((stats['n'] > 0).any())
        if self.analytic:
            return
        self.chol = _chol(covariance)
        x, w = np.polynomial.hermite.hermgauss(order)
        nodes = mean.new_tensor(x*np.sqrt(2.))[:, None]
        log_mass = mean.new_tensor(w/np.sqrt(np.pi)).log()
        self.nodes, self.node_log_mass = nodes[:, 0], log_mass
        self.samples = self.rss['mean'] + nodes @ _chol(self.rss['cov']).T
        detected, missing = _detection_logs(self.samples, obs)
        factors = (stats['k']*detected + (stats['n']-stats['k'])*missing).sum(-1)
        log_integrand = log_mass+factors
        self.log_detection_evidence = torch.logsumexp(log_integrand, 0)
        self.log_weights = log_integrand-self.log_detection_evidence

    def conditioned_log_prob(self, values, query_mean, query_variance, cross):
        """[RSS value, grid] likelihood; condition Gaussian on detected Q RSS.

        Query-dependent quadrature is local to this likelihood evaluation. It
        never modifies S posterior weights or learns a query calibration.
        """
        if self.analytic:
            means, variance, _ = self.components(query_mean, query_variance, cross)
            return observation_log_prob(values[:, None], means, variance[None], self.obs)
        A = torch.cholesky_solve(cross.T, self.chol).T[:, 0]
        conditional_variance = query_variance-A*cross[:, 0]
        if bool((conditional_variance < -1e-8).any()):
            raise RuntimeError('Negative query conditional variance')
        conditional_variance = conditional_variance.clamp_min(0)
        mu = self.rss['mean'][0]
        variance = _chol(self.rss['cov'])[0, 0].square()
        intercept = query_mean-A*self.mean[0]
        query_mu = intercept+A*mu
        noise2 = torch.as_tensor(self.obs['noise_std'], dtype=mu.dtype, device=mu.device).square()
        total = conditional_variance+A.square()*variance+noise2
        seen = values != 100
        output = query_mean.new_empty((len(values), len(query_mean)))
        if bool((~seen).any()):
            means, var, mass = self.components(query_mean, query_variance, cross)
            lp = observation_log_prob(100., means, var[None], self.obs)
            output[~seen] = torch.logsumexp(mass[:, None]+lp, 0)
        if bool(seen.any()):
            rss = values[seen, None]
            center = mu + (variance*A/total)[None]*(rss-query_mu[None])
            spread = (variance*(conditional_variance+noise2)/total).sqrt()
            s = center[:, None, :] + self.nodes[None, :, None]*spread[None, None, :]
            detected, missing = _detection_logs(s, self.obs)
            support_log = self.stats['k'][0]*detected + (self.stats['n'][0]-self.stats['k'][0])*missing
            before = intercept[None, None, :]+A[None, None, :]*s
            gain = conditional_variance/(conditional_variance+noise2)
            after = before+gain[None, None, :]*(rss[:, None, :]-before)
            after_variance = conditional_variance*noise2/(conditional_variance+noise2)
            alpha = torch.as_tensor(self.obs['alpha'], dtype=mu.dtype, device=mu.device)
            beta = torch.as_tensor(self.obs['beta'], dtype=mu.dtype, device=mu.device)
            scale2 = torch.as_tensor(self.obs['detect_scale'], dtype=mu.dtype, device=mu.device).square()
            query_detection = alpha.log()+torch.special.log_ndtr(
                (after-beta)/(scale2+after_variance[None, None, :]).sqrt())
            integral = torch.logsumexp(self.node_log_mass[None, :, None]+support_log+query_detection, 1)
            rss_log = -.5*(math.log(2*math.pi)+total[None].log()+(rss-query_mu[None]).square()/total[None])
            output[seen] = rss_log+integral-self.log_detection_evidence
        return output
