"""Fixed historical representations with one GP observation per calibration site.

The mean/features must be functions of H and the individual RSS scan only.
This borrows deep-kernel regression, not a new GP principle or ReFi RSS-field
transform. Coordinates are in metres; sigma is inverse-model discrepancy,
not survey-label noise. Only three source-shared log scales are learned.
"""
import math
import torch
from torch import nn


def squared_distance(left, right):
    """RMS-feature squared distance; may be cached when encodings are frozen."""
    left, right = left.flatten(1), right.flatten(1)
    return (left.square().mean(1)[:,None] + right.square().mean(1)[None]
            - 2*(left@right.T)/left.shape[1]).clamp_min(0)


def site_average(xy):
    """One linear functional per labelled site, retaining every scan equally."""
    locations, group, counts = torch.unique(xy, dim=0, sorted=True,
                                           return_inverse=True, return_counts=True)
    average = xy.new_zeros((len(locations), len(xy)))
    average[group, torch.arange(len(xy), device=xy.device)] = 1/counts[group].to(xy.dtype)
    return locations, average


class HistoryFeatureGP(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scales = nn.Parameter(torch.tensor([math.log(10.), 0., math.log(5.)],
                                                   dtype=torch.float64))

    def kernel(self, left, right):
        return self.kernel_from_distance(squared_distance(left, right))

    def kernel_from_distance(self, distance):
        amplitude, length, _ = self.log_scales.exp()
        return amplitude.square()*(-distance/(2*length.square())).exp()

    def posterior(self, support_mean, support_phi, support_xy, query_mean, query_phi):
        """S coordinates occur only in the site grouping and observation update.

        Mean/features are frozen H encodings. This returns marginal predictions;
        it does not approximate the learned kernel by a mean feature vector.
        Empty S needs no invented target fit. The Gaussian is not clipped to
        a candidate box here; a declared final decision rule may impose one.
        """
        return self.posterior_from_distance(support_mean, support_xy, query_mean,
            squared_distance(support_phi,support_phi), squared_distance(query_phi,support_phi))

    def posterior_from_distance(self, support_mean, support_xy, query_mean, dss, dqs):
        """The same update with cached distances, never cached learned kernels."""
        amplitude, _, noise = self.log_scales.exp()
        prior_variance = amplitude.square().expand(len(query_mean))
        if not len(support_xy):
            return dict(mean=query_mean, latent_variance=prior_variance,
                        variance=prior_variance+noise.square())
        locations, average = site_average(support_xy)
        covariance = average@self.kernel_from_distance(dss)@average.T
        covariance = covariance + noise.square()*torch.eye(len(locations),
            dtype=covariance.dtype, device=covariance.device)
        cross = self.kernel_from_distance(dqs)@average.T
        root = torch.linalg.cholesky(covariance)
        residual = locations - average@support_mean
        mean = query_mean + cross@torch.cholesky_solve(residual,root)
        latent_variance = prior_variance - (cross*torch.cholesky_solve(cross.T,root).T).sum(1)
        if not bool(torch.isfinite(latent_variance).all()) or bool((latent_variance < -1e-8).any()):
            raise RuntimeError('Invalid conditional GP variance')
        latent_variance = latent_variance.clamp_min(0)  # Schur-complement roundoff only.
        return dict(mean=mean,latent_variance=latent_variance,
                    variance=latent_variance+noise.square())


def position_balanced_nll(prediction, query_xy):
    """Marginal two-coordinate Gaussian score, averaged equally across sites."""
    _, group, counts = torch.unique(query_xy, dim=0, sorted=True,
                                   return_inverse=True, return_counts=True)
    weights = 1/(len(counts)*counts[group].to(query_xy.dtype))
    error = (prediction['mean']-query_xy).square().sum(1)
    variance = prediction['variance']
    return (weights*(torch.log(2*math.pi*variance)+error/(2*variance))).sum()
