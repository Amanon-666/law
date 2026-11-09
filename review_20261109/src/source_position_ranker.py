"""Anonymous historical descriptors for GP-offset position discrimination.

Canonical sorting retains a multiset of the selected AP fields, not AP IDs
or the complete historical joint distribution. No target labels are needed.
"""
import numpy as np
import torch
import xgboost as xgb

from history_position_ranking import PARAMS, ROUNDS
from history_ratio_correction import features
from observations import observation_log_prob
from tabicl_adaptation_examples import AdaptationFeatures

SEED = 11509153
MAX_APS = 520


def descriptor(histories, support, floor, base, field):
    value = AdaptationFeatures(histories, support, floor, base)
    assert np.array_equal(value.aps, field['aps']) and len(value.aps) <= MAX_APS
    return dict(at_grid=value.history.at(field['grid']),
        at_support=np.concatenate((value.at_support, value.local_support), axis=1),
        common=value.common_support, support_position=value.support_position,
        length=value.length)


def anonymous(tokens):
    """Sort whole tokens lexicographically; equal-token order is irrelevant."""
    order = np.lexsort(tuple(tokens[:, :, j] for j in range(tokens.shape[2]-1, -1, -1)), axis=1)
    sorted_tokens = np.take_along_axis(tokens, order[:, :, None], axis=1)
    rows, aps, width = tokens.shape
    padded = np.zeros((rows, MAX_APS, width+1), dtype=np.float32)
    padded[:, :aps, 0] = 1.
    padded[:, :aps, 1:] = sorted_tokens
    return padded.reshape(rows, -1)


def make_features(rss, indices, field, obs, desc, gp_score, history=True):
    aps = len(field['aps'])
    bounds = np.stack((field['grid'].min(0), field['grid'].max(0)))
    relative = features(rss, indices, field, obs, bounds)[:, :3*aps]
    triples = relative.reshape(len(rss), 3, aps).transpose(0, 2, 1)
    common = np.column_stack((gp_score, (field['grid'][indices]-desc['support_position'])/desc['length'],
                              np.full(len(rss), aps)))
    if history:
        tokens = np.concatenate((np.broadcast_to(desc['at_support'], (len(rss), aps, 21)),
                                 desc['at_grid'][indices], triples), axis=2)
        assert tokens.shape[2] == 40
        common = np.column_stack((common, np.broadcast_to(desc['common'], (len(rss), 73))))
    else:
        # This control sorts its own triples: H descriptions cannot leak via order.
        tokens = triples
    return np.column_stack((anonymous(tokens), common)).astype(np.float32)


@torch.no_grad()
def candidate_features(rss, candidates, field, obs, desc):
    indices = candidates.ravel()
    repeated = np.repeat(rss, 5, axis=0)
    score = observation_log_prob(repeated, field['mean'][indices],
                                field['variance'][indices], obs).numpy().sum(1)
    full = make_features(repeated, indices, field, obs, desc, score)
    gp_only = make_features(repeated, indices, field, obs, desc, score, history=False)
    offset = score.reshape(-1, 5)
    offset = (offset-offset.max(1, keepdims=True)).astype(np.float32).ravel()
    return full, gp_only, offset


def fit_shared(x, offset, group_weights, use_offset):
    groups = len(group_weights)
    matrix = xgb.DMatrix(x, label=np.tile([1., 0., 0., 0., 0.], groups), nthread=2)
    matrix.set_group(np.full(groups, 5, dtype=np.uint32))
    matrix.set_weight(group_weights)
    matrix.set_base_margin(offset if use_offset else np.zeros_like(offset))
    params = dict(PARAMS, seed=SEED)
    model = xgb.train(params, matrix, num_boost_round=ROUNDS, verbose_eval=False)
    assert model.num_boosted_rounds() == ROUNDS
    return model


def score_shared(models, rss, score, field, obs, desc):
    grid_size = len(field['grid'])
    total = len(rss)*grid_size
    result = {name:np.empty(total) for name in models}
    for start in range(0, total, 256):
        flat = np.arange(start, min(start+256, total))
        rows, indices = divmod(flat, grid_size)
        for kind, names in ((True, ('primary', 'no_offset')), (False, ('gp_features',))):
            x = make_features(rss[rows], indices, field, obs, desc, score[rows, indices], history=kind)
            matrix = xgb.DMatrix(x, nthread=2)
            matrix.set_base_margin(np.zeros(len(flat), dtype=np.float32))
            for name in names:
                result[name][start:start+len(flat)] = models[name].predict(matrix, output_margin=True)
    assert all(np.isfinite(v).all() for v in result.values())
    return {name:value.reshape(len(rss), grid_size) for name, value in result.items()}
