"""Shared finite-observation statistics and 55+20 features; no corpus imports.

Exposure summaries describe equal-position averages, not independent sample
sizes. They neither identify query acquisition shifts nor calibrate Q from S.
"""
import numpy as np

from vertical_translator import translator as T


SCHEMA_ID = 'finite_observation_55_plus_20_v1'
SCHEMA = (
    [f'H{d:+d}_{name}' for d in T.SLOTS for name in
     ('nearest_encoded', 'smooth_encoded', 'nearest_detection', 'floor_max', 'floor_mean')]
    + ['history_reference', 'floor_gap_below', 'floor_gap_above']
    + ['S_delta_mean', 'S_max', 'S_mean', 'S_pooled_delta', 'S_home_delta', 'S_strength_delta',
       'S_nearest_delta', 'S_nearest_encoded', 'S_nearest_reference', 'S_nearest_distance',
       'S_position_count', 'S_informative_position_fraction']
    + [f'H{d:+d}_{name}' for d in T.SLOTS for name in ('log1p_n_nearest', 'log1p_n_eff_smooth')]
    + ['S_log1p_n_nearest', 'S_nearest_detection_rate', 'S_log1p_n_eff_all', 'S_mean_detection_rate'])
WIDTH = 75


def position_stats(rss, xy):
    """Keep raw scan sufficient statistics on every position and full AP axis."""
    rss, xy = np.asarray(rss), np.asarray(xy, dtype=np.float64)
    positions, inverse = np.unique(xy, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n = np.bincount(inverse, minlength=len(positions)).astype(float)
    detected = rss != 100
    values = np.where(detected, rss, 0.)
    nd = np.zeros((len(positions), rss.shape[1]))
    total, square = np.zeros_like(nd), np.zeros_like(nd)
    np.add.at(nd, inverse, detected)
    np.add.at(total, inverse, values)
    np.add.at(square, inverse, values**2)
    return dict(xy=positions, n=n, nd=nd, rss_sum=total, rss_square_sum=square)


def build_context(histories, support, target_floor):
    """histories: physical floor -> raw {xy,rss}; S uses the same AP columns.

Return features on the visible H/S AP axis, with all raw-position statistics
retained for provenance. Original 43 H + 12 S columns stay in original order.
"""
    srss, sxy = np.asarray(support['rss']), np.asarray(support['xy'], dtype=np.float64)
    width = srss.shape[1]
    assert histories and target_floor not in histories
    assert srss.ndim == 2 and sxy.shape == (len(srss), 2)
    maps, stats = {}, {}
    visible = (srss != 100).any(0)
    for floor, h in histories.items():
        rss, xy = np.asarray(h['rss']), np.asarray(h['xy'], dtype=np.float64)
        assert rss.shape == (len(xy), width) and xy.shape[1:] == (2,)
        maps[floor] = T.position_map(rss, xy)
        stats[floor] = position_stats(rss, xy)
        visible |= (rss != 100).any(0)
    aps = np.flatnonzero(visible)
    assert len(aps), 'No AP detected in the visible H/S scans'
    candidates = np.unique(np.concatenate([m[0] for m in maps.values()]), axis=0)
    F, ref = T.history_features(candidates, target_floor, maps)
    F = F.astype(np.float32)
    assert F.shape[2] == 43
    if len(sxy):
        sref = T.naive_ref_at(sxy, target_floor, maps)
        S = T.support_features(candidates, ref, sxy, srss, sref, F).astype(np.float32)
    else:
        S = np.full((len(candidates), width, 12), np.nan, dtype=np.float32)
    extra = []
    for offset in T.SLOTS:
        floor = target_floor + offset
        if floor not in stats:
            extra.extend([np.full((len(candidates), width), np.nan)] * 2)
            continue
        st = stats[floor]
        D = np.linalg.norm(candidates[:, None] - st['xy'][None], axis=2)
        j = D.argmin(1)
        nearest = np.where(D.min(1) <= T.R_SAME, st['n'][j], np.nan)
        W = D <= T.R_SMOOTH
        count = W.sum(1)
        effective = np.full(len(candidates), np.nan)
        present = count > 0
        effective[present] = count[present]**2 / (W[present] @ (1./st['n']))
        extra.extend([np.broadcast_to(np.log1p(v)[:, None], (len(candidates), width))
                      for v in (nearest, effective)])
    sstats = position_stats(srss, sxy)
    if len(sstats['xy']):
        D = np.linalg.norm(candidates[:, None] - sstats['xy'][None], axis=2)
        j = D.argmin(1)
        rate = sstats['nd']/sstats['n'][:, None]
        effective = len(sstats['n'])**2 / np.sum(1./sstats['n'])
        extra.extend([
            np.broadcast_to(np.log1p(sstats['n'][j])[:, None], (len(candidates), width)),
            rate[j], np.full((len(candidates), width), np.log1p(effective)),
            np.broadcast_to(rate.mean(0), (len(candidates), width))])
    else:
        extra.extend([np.full((len(candidates), width), np.nan)] * 4)
    features = np.concatenate([F[:, aps], S[:, aps], np.stack(extra, 2)[:, aps]], axis=2).astype(np.float32)
    assert len(SCHEMA) == features.shape[2] == WIDTH
    return dict(cand_xy=candidates, aps=aps, features=features, ap_count=width,
                history_stats=stats, support_stats=sstats)
