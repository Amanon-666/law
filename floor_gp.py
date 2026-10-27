"""Ordinary floor-aware spatial GP, retaining LAW's observation/readout.

Independent APs; this is not the full MOGPAug LMC/augmentation/RNN method.
Floor is an ordinal input with a learned source-side length, not metric height.
Raw historical observations are conditioned jointly, not pooled in XY alone.
"""
import torch

from inference import _chol, infer
from run import ap_stats, tensor
from transfer import rbf


def floor_kernel(x, y, base, floor_length):
    spatial = rbf(x[:, :2], y[:, :2], base['amplitude'], base['length'])
    return spatial * torch.exp(-.5 * ((x[:, None, 2]-y[None, :, 2])/floor_length).square())


class FloorGPPrior:
    """Target-floor slice of a GP conditioned on all raw historical floors."""
    def __init__(self, positions, posterior, base, floor_length, target_floor):
        self.z, self.base = positions, base
        self.floor_length, self.target_floor = floor_length, target_floor
        self.prior_cov = floor_kernel(positions, positions, base, floor_length)
        self.root = _chol(self.prior_cov)
        self.mean = posterior['mean'].detach()
        self.delta = posterior['cov'].detach()-self.prior_cov

    def features(self, x):
        xyz = torch.cat((x, x.new_full((len(x), 1), self.target_floor)), 1)
        cross = floor_kernel(xyz, self.z, self.base, self.floor_length)
        projection = torch.cholesky_solve(cross.T, self.root).T
        mean = self.base['mean'] + projection @ (self.mean-self.base['mean'])
        return mean, projection

    def covariance(self, x, y, fx, fy):
        return rbf(x, y, self.base['amplitude'], self.base['length']) + fx @ self.delta @ fy.T

    def diagonal(self, x, fx):
        return self.base['amplitude']**2 + ((fx @ self.delta)*fx).sum(-1)


def fit_floor_history(records, ap, base, obs, floor_length, target_floor, steps=640):
    """Records contain floor index and grouped raw scans with one common XY origin.

    Each acquisition belongs to one record; callers must not alias the same
    history twice. Missing RSS=100 enters the existing detection likelihood.
    """
    positions = []; counts = []
    for record in records:
        xy = tensor(record['scans']['positions'])
        positions.append(torch.cat((xy, xy.new_full((len(xy), 1), record['floor'])), 1))
        counts.append(ap_stats(record['scans'], ap))
    xyz = torch.cat(positions)
    stats = {key: torch.cat([c[key] for c in counts]) for key in counts[0]}
    cov = floor_kernel(xyz, xyz, base, floor_length)
    posterior = infer(xyz.new_full((len(xyz),), base['mean']), cov, stats, obs, steps=steps)
    return FloorGPPrior(xyz, posterior, base, floor_length, target_floor), posterior
