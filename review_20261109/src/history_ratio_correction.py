"""Conditional historical discrimination as a direct GP posterior correction.

Finite fitted logits define a discriminative location model. Only ideal,
conditionally correct density ratios have the exact observation-mixture
interpretation; no probability-calibration or risk-preservation claim.
"""
import numpy as np
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 11509148


def historical_proxy(histories, grid, indices, length, noise_std, seed=SEED):
    """RBF conditional empirical distribution with Gaussian RSS kernels.

The RSS bandwidth smooths an empirical proxy; it is not a second physical
measurement. Non-detection remains an atom. No query values enter here.
The same seed with row-aligned raw/calibrated H gives matched source rows/noise.
"""
    rss = np.concatenate([h['rss'] for h in histories.values()])
    xy = np.concatenate([h['xy'] for h in histories.values()])
    locations, groups = np.unique(xy, axis=0, return_inverse=True)
    grouped_rows = [np.flatnonzero(groups == i) for i in range(len(locations))]
    counts = np.array([len(rows) for rows in grouped_rows])
    distances = np.square(grid[:, None, :]-locations[None, :, :]).sum(-1)
    logits = -.5*distances/length**2+np.log(counts)
    weights = np.exp(logits-logits.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    cdf = weights.cumsum(1)
    cdf[:, -1] = 1.
    rng = np.random.default_rng(seed)
    selected_groups = (rng.random(len(indices))[:, None] > cdf[indices]).sum(1)
    selected_rows = np.array([rows[rng.integers(len(rows))]
                             for rows in (grouped_rows[i] for i in selected_groups)])
    positive = rss[selected_rows].astype(float, copy=True)
    detected = positive != 100
    noise = noise_std*rng.standard_normal(positive.shape)
    positive[detected] += noise[detected]
    assert np.array_equal(positive != 100, detected)
    return dict(rss=positive, indices=np.asarray(indices).copy(), source_rows=selected_rows)


def features(rss, indices, field, obs, bounds):
    """Fixed GP-relative representation, shared by fit and candidate scoring."""
    mean = field['mean'][indices]
    variance = field['variance'][indices]
    detected = rss != 100
    residual = np.where(detected, (rss-mean)/np.sqrt(variance+obs['noise_std']**2), 0.)
    detection_probability = obs['alpha']*ndtr((mean-obs['beta']) /
        np.sqrt(variance+obs['detect_scale']**2))
    xy = (field['grid'][indices]-bounds[0])/(bounds[1]-bounds[0])
    return np.concatenate((detected, residual, detection_probability, xy), axis=1).astype(np.float32)


def fit_correction(positive, negative, indices, field, obs, bounds):
    """Equal class priors and exactly paired candidate positions."""
    x = np.concatenate((features(negative, indices, field, obs, bounds),
                        features(positive, indices, field, obs, bounds)))
    y = np.concatenate((np.zeros(len(indices), dtype=int), np.ones(len(indices), dtype=int)))
    model = HistGradientBoostingClassifier(max_iter=100, early_stopping=False,
                                          random_state=SEED)
    model.fit(x, y)
    assert model.n_iter_ == 100 and np.array_equal(model.classes_, [0, 1])
    return model


def correction_scores(model, rss, field, obs, bounds):
    """All candidate positions; no candidate pruning or coordinate labels."""
    grid_size = len(field['grid'])
    flat_size = len(rss)*grid_size
    logits = np.empty(flat_size)
    for start in range(0, flat_size, 4096):
        flat = np.arange(start, min(start+4096, flat_size))
        query_indices, grid_indices = divmod(flat, grid_size)
        x = features(rss[query_indices], grid_indices, field, obs, bounds)
        logits[start:start+len(flat)] = model.decision_function(x)
    assert np.isfinite(logits).all()
    logits = logits.reshape(len(rss), grid_size)
    return np.logaddexp(0., logits)-np.log(2.)
