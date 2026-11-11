"""Replace only empirical S detection priors in an original prepared field.

H rows have equal weight, including repeated rows; floors are not reweighted.
The pooled H frequency is a prior mean, not a target spatial probability.
S prior concentration stays one regardless of H size. No model or query use.
"""
import hashlib

import numpy as np

from frozen_hurdle_translator import PreparedFrozenHurdle, _support_fields


def replace_support_prior(field, histories, support):
    """Aligned raw H/S arrays use missing=100 and the original full AP axis."""
    srss, sxy = np.asarray(support['rss']), np.asarray(support['xy'], dtype=float)
    assert srss.shape == (len(sxy), field.ap_count)
    seen = (srss != 100).any(0)
    dH = np.zeros(len(field.aps), dtype=np.int64)
    rows = {}
    for name, h in histories.items():
        rss = np.asarray(h['rss'])
        assert rss.shape == (len(h['xy']), field.ap_count)
        rows[name] = len(rss)
        dH += (rss[:, field.aps] != 100).sum(0)
        seen |= (rss != 100).any(0)
    assert np.array_equal(field.aps, np.flatnonzero(seen))
    nH = sum(rows.values())
    p0 = (dH + .5)/(nH + 1.)
    positions, old_p, old_m, old_s2 = _support_fields(srss[:, field.aps], sxy)
    history_xy = np.unique(np.concatenate([h['xy'] for h in histories.values()]), axis=0)
    assert np.array_equal(field.cand_xy, np.vstack([history_xy, positions]))
    start = len(history_xy)
    for actual, expected in ((field.p[start:], np.clip(old_p, 1e-4, 1-1e-4)),
                             (field.m[start:], old_m), (field.s2[start:], old_s2)):
        assert np.array_equal(actual, expected), 'Cached empirical S field does not match supplied S'
    # Use the same unique-location grouping as the original support constructor.
    _, inverse = np.unique(sxy, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    nS = np.bincount(inverse, minlength=len(positions))
    ndS = np.zeros_like(old_p)
    np.add.at(ndS, inverse, srss[:, field.aps] != 100)
    p = field.p.copy()
    p[start:] = np.clip((ndS + p0)/(nS[:, None] + 1.), 1e-4, 1-1e-4)
    transformed = PreparedFrozenHurdle(field.cand_xy, field.aps, p, field.m,
                                       field.s2, field.bounds, field.ap_count)
    return transformed, dict(nH=nH, H_rows_by_source=rows, support_rows=len(srss),
        support_candidates=len(positions), support_rows_per_location=nS.tolist(),
        candidate_start=start, AP_count=len(field.aps), prior_expected_detections=float(p0.sum()),
        max_p_change=float(np.max(np.abs(p[start:]-field.p[start:]))) if len(positions) else 0.,
        p_sha256=hashlib.sha256(p.tobytes()).hexdigest(),
        original_S_verified=True, learned_H_p_unchanged=True,
        same_m_s2_candidates_AP_bounds=True)
