"""Observed scans from the existing retained-GP single-scan working model.

Generated rows approximate a predictive distribution. They are not additional
measurements, a coherent map draw, or a recovered cross-AP copula.
"""
import math

import numpy as np
import torch
from scipy.special import ndtr

from acquisition_bank import prepare_bank_blocks
from observations import observation_log_prob, predict_from_posterior
from recomposition import Parameters, prior
from scalar_support import ScalarSupportIntegral
from uji import group_scans


def sample_observations(mean, variance, obs, rng):
    """D and measured RSS share the same latent F; no clipping or rounding."""
    mean, variance = np.asarray(mean), np.asarray(variance)
    assert mean.shape == variance.shape
    assert np.isfinite(mean).all() and np.isfinite(variance).all() and (variance >= 0).all()
    latent = mean + np.sqrt(variance)*rng.standard_normal(mean.shape)
    detected = ((rng.random(mean.shape) < obs['alpha']) &
                (latent + obs['detect_scale']*rng.standard_normal(mean.shape) > obs['beta']))
    measured = latent + obs['noise_std']*rng.standard_normal(mean.shape)
    assert not np.any(detected & (measured == 100.))
    return np.where(detected, measured, 100.)


def check_sampler():
    """One bounded check of selection moments, including deterministic F.

Independent analytic truncated-normal moments catch sampling detection and
RSS independently. This is not a localization or calibration experiment.
"""
    obs = dict(alpha=.85, beta=-85., detect_scale=5., noise_std=3.)
    m = np.array([-100., -85., -70., -85.])
    v = np.array([0., 1., 100., 400.])
    count = 80000
    rss = sample_observations(np.broadcast_to(m, (count, 4)),
                             np.broadcast_to(v, (count, 4)), obs,
                             np.random.default_rng(115081470))
    c = np.sqrt(v+obs['detect_scale']**2)
    t = (m-obs['beta'])/c
    phi = np.exp(-t*t/2)/math.sqrt(2*math.pi)
    expected = obs['alpha']*np.stack((ndtr(t), v/c*phi,
        (v+obs['noise_std']**2)*ndtr(t)-v*v*t/(c*c)*phi), axis=1)
    detected = rss != 100
    centered = np.where(detected, rss-m, 0.)
    samples = np.stack((detected, centered, centered**2), axis=-1)
    actual = samples.mean(0)
    standard_error = samples.std(0, ddof=1)/np.sqrt(count)
    standardized = np.abs(actual-expected)/standard_error
    assert np.isfinite(standardized).all() and standardized.max() <= 7., standardized
    missing = observation_log_prob(torch.full((4,), 100.), torch.tensor(m),
                                   torch.tensor(v), obs).exp().numpy()
    probability_error = float(np.abs(missing-(1-expected[:, 0])).max())
    assert probability_error <= 1e-12
    return dict(passed=True, draws_per_case=count, maximum_standard_errors=float(standardized.max()),
                threshold_standard_errors=7., detection_probability_error=probability_error,
                expected=expected.tolist(), actual=actual.tolist(),
                quantities=['D', 'D*(R-m)', 'D*(R-m)^2'])


@torch.no_grad()
def retained_fields(task, selected, target_floor):
    """Exactly the same Gaussian moment field as gp_predictions(retained)."""
    grid = task['grid']
    blocks = prepare_bank_blocks(task,
        [int(target_floor.split('F')[1])-int(name.split('F')[1]) for name in task['names']])
    model = Parameters(torch.zeros(9), torch.ones(9)).requires_grad_(False)
    weights = grid.new_tensor([selected['weight'], 1-selected['weight']])
    sigma2 = selected['sigma']**2
    means, variances = [], []
    for ap, block in blocks.items():
        p = prior(model, block, weights)
        sk = p['sk']+sigma2
        post = ScalarSupportIntegral(p['sm'], sk, task['cases'][ap]['stats'], task['obs'], 256)
        mass = post.log_weights.exp()
        mean = mass@post.samples
        centered = post.samples-mean
        posterior = dict(mean=mean, cov=(centered.T*mass)@centered)
        field = predict_from_posterior(p['gm'], p['gv']+sigma2,
            p['cross']-block['rg'], p['sm'], sk, posterior)
        means.append(field['mean'].numpy())
        variances.append(field['variance'].numpy())
    means, variances = np.stack(means, axis=1), np.stack(variances, axis=1)
    assert np.isfinite(means).all() and np.isfinite(variances).all() and (variances >= 0).all()
    assert list(blocks) == list(task['cases'])
    return dict(mean=means, variance=variances, aps=np.array(list(blocks)), grid=grid.numpy())


def generate_context(field, obs, rows, columns, seed):
    """Uniform original-grid location prior; no query input or rejection."""
    rng = np.random.default_rng(seed)
    grid_indices = rng.integers(len(field['grid']), size=rows)
    rss = np.full((rows, columns), 100.)
    rss[:, field['aps']] = sample_observations(field['mean'][grid_indices],
        field['variance'][grid_indices], obs, rng)
    scans = group_scans(rss, field['grid'][grid_indices])
    return scans, grid_indices


def concatenate_context(parts, floor):
    """Context rows only. The unchanged consumer adds real S once."""
    return {floor:group_scans(np.concatenate([p['rss'] for p in parts]),
                             np.concatenate([p['xy'] for p in parts]))}
