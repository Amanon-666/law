"""Finite historical position ranking with an explicit GP score offset.

This is not density-ratio estimation or a calibrated observation model.
Even a GP offset can be cancelled by a sufficiently flexible residual.
"""
import numpy as np
import torch
import xgboost as xgb

from history_ratio_correction import features
from observations import observation_log_prob

SEED = 11509152
ROUNDS = 100
PARAMS = dict(objective='rank:pairwise', tree_method='hist', device='cpu',
    max_depth=6, eta=.1, nthread=2, seed=SEED, base_score=0.,
    lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=5,
    lambdarank_normalization=False, lambdarank_score_normalization=False)


def candidate_groups(positive, size, seed):
    rng = np.random.default_rng(seed)
    candidates = np.empty((len(positive), 5), dtype=int)
    candidates[:, 0] = positive
    for i, position in enumerate(positive):
        negative = rng.choice(size-1, size=4, replace=False)
        candidates[i, 1:] = negative+(negative >= position)
    assert np.all(np.diff(np.sort(candidates, axis=1), axis=1) != 0)
    return candidates


def rank_features(rss, indices, field, obs, bounds, gp_score):
    # All arms, including the no-offset control, see the complete GP score.
    return np.column_stack((features(rss, indices, field, obs, bounds),
                            gp_score.astype(np.float32)))


@torch.no_grad()
def training_matrix(rss, candidates, field, obs, bounds):
    indices = candidates.ravel()
    repeated = np.repeat(rss, 5, axis=0)
    score = observation_log_prob(repeated, field['mean'][indices],
                                field['variance'][indices], obs).numpy().sum(1)
    x = rank_features(repeated, indices, field, obs, bounds, score)
    score = score.reshape(-1, 5)
    offset = (score-score.max(1, keepdims=True)).astype(np.float32).ravel()
    y = np.tile([1., 0., 0., 0., 0.], len(rss))
    matrix = xgb.DMatrix(x, label=y, nthread=2)
    matrix.set_group(np.full(len(rss), 5, dtype=np.uint32))
    return matrix, offset


def fit_ranker(matrix, offset, use_offset):
    matrix.set_base_margin(offset if use_offset else np.zeros_like(offset))
    model = xgb.train(PARAMS, matrix, num_boost_round=ROUNDS, verbose_eval=False)
    assert model.num_boosted_rounds() == ROUNDS
    return model


def margin_contract(model, matrix, offset):
    """Check the installed API on the real fitted model, before opening Q."""
    subset = matrix.slice(np.arange(25), allow_groups=True)
    margin = offset[:25]
    subset.set_base_margin(margin)
    with_margin = model.predict(subset, output_margin=True).astype(float)
    # Prediction caches belong to DMatrix handles; changing margin does not reset them.
    zero_subset = matrix.slice(np.arange(25), allow_groups=True)
    zero_subset.set_base_margin(np.zeros(25, dtype=np.float32))
    residual = model.predict(zero_subset, output_margin=True).astype(float)
    error = np.abs(with_margin-residual-margin)
    allowance = 8*np.finfo(np.float32).eps*(1+np.abs(margin)+np.abs(residual))
    assert (error <= allowance).all(), (float(error.max()), float(allowance.max()))
    return dict(passed=True, rows=25, maximum_absolute_error=float(error.max()))


def residual_scores(model, rss, score, field, obs, bounds):
    grid_size = len(field['grid'])
    flat_size = len(rss)*grid_size
    result = np.empty(flat_size)
    for start in range(0, flat_size, 4096):
        flat = np.arange(start, min(start+4096, flat_size))
        rows, positions = divmod(flat, grid_size)
        x = rank_features(rss[rows], positions, field, obs, bounds, score[rows, positions])
        matrix = xgb.DMatrix(x, nthread=2)
        # Explicit zero margin returns only trees. The caller adds float64 GP once.
        matrix.set_base_margin(np.zeros(len(flat), dtype=np.float32))
        result[start:start+len(flat)] = model.predict(matrix, output_margin=True)
    assert np.isfinite(result).all()
    return result.reshape(len(rss), grid_size)
