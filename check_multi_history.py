"""Mechanism checks: complementary evidence, correlated error, source identity.

No real-data training or performance selection takes place here.
"""
import json

import numpy as np
import torch

from data import group_scans
from multi_history import (GaussianStrengthRelation, MultiHistory, collect_histories, project_posterior,
                           single_history, temporal_covariance)
from transfer import rbf


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    base = dict(mean=0., amplitude=1., length=.5, drift=.2)
    x = torch.tensor([[0., 0.], [5., 0.]])
    times = torch.tensor([1., 0.])
    target = torch.ones(2)
    cov = temporal_covariance(x, times, x, times, base, 10.)
    cross = temporal_covariance(x, target, x, times, base, 10.)
    noise = .1 * torch.eye(2)
    y = torch.tensor([2., 8.])
    post = dict(mean=cov @ torch.linalg.solve(cov + noise, y),
                cov=cov - cov @ torch.linalg.solve(cov + noise, cov))
    projected = project_posterior(cov, cross, post, rbf(x, x, 1., .5), 0.)
    expected_mean = cross @ torch.linalg.solve(cov + noise, y)
    expected_cov = rbf(x, x, 1., .5) - cross @ torch.linalg.solve(cov + noise, cross.T)
    assert torch.allclose(projected["mean"], expected_mean, atol=1e-5)
    assert torch.allclose(projected["cov"], expected_cov, atol=2e-6)
    assert expected_mean[1] > 6 and expected_cov[1, 1] < .3
    # One known-current-point alone tells essentially nothing about x=5.
    single_cross = cross[:, :1]
    single_mean = single_cross[:, 0] * y[0] / 1.1
    assert single_mean[1].abs() < 1e-10

    independent = 1 / (1 + 2.)
    correlated = 1 / (1 + torch.ones(2) @ torch.linalg.solve(
        torch.tensor([[1., .9], [.9, 1.]]), torch.ones(2)))
    assert correlated > independent

    scans = group_scans(np.array([[-70], [-70], [-80], [100]]),
                        np.array([[0., 0.], [0., 0.], [1., 0.], [1., 0.]]))
    records = collect_histories({1: scans}, [(1, [0, 1, 2]), (1, [1, 2, 3])])
    assert records[0]["row_ids"].tolist() == [0, 1, 2, 3]
    assert records[0]["scans"]["stats"]["n"].tolist() == [2, 2]
    real_base = dict(mean=-90., amplitude=15., length=2., drift=3.)
    obs = dict(alpha=.95, beta=-95., detect_scale=3., noise_std=3.)
    calendar = dict(days=[0.], temporal_length_days=10.)
    old = single_history(records[0], 0, real_base, obs)
    new = MultiHistory(records, 0, real_base, obs, calendar)
    assert torch.allclose(old.m, new.m, atol=2e-6)
    assert torch.allclose(old.c, new.c, atol=3e-6)
    relation=GaussianStrengthRelation("strength_rbf",real_base["length"])
    points=torch.tensor([[0.,0.],[.5,0.],[2.,0.]])
    kernel=relation.kernel(points,points,new,real_base)
    torch.linalg.cholesky(kernel)
    assert torch.allclose(kernel.diag(),torch.full((3,),real_base["drift"]**2))
    # Verify the Gaussian strength expectation against deterministic quadrature.
    nodes,weights=np.polynomial.hermite.hermgauss(64)
    mu,var,ell=4.,9.,5.
    numerical=float(weights@np.exp(-.5*(mu+np.sqrt(2*var)*nodes)**2/ell**2)/np.sqrt(np.pi))
    analytic=np.sqrt(ell**2/(ell**2+var))*np.exp(-.5*mu**2/(ell**2+var))
    assert abs(numerical-analytic)<1e-12
    print(json.dumps(dict(passed=True, complementary_single_mean=single_mean.tolist(),
                         complementary_multi_mean=expected_mean.tolist(),
                         complementary_multi_variance=expected_cov.diag().tolist(),
                         independent_error_variance=independent,
                         correlated_error_variance=correlated.item(),
                         unique_acquisitions=len(records[0]["row_ids"]),
                         gaussian_strength_quadrature_error=abs(numerical-analytic),
                         single_history_max_mean_difference=(old.m-new.m).abs().max().item())))


if __name__ == "__main__":
    main()
