"""Represent a translated mean fingerprint in the existing Tab context.

The supplied translator learns the position mean of RSS encoded with -105
for nondetection. The existing tabicl_history.features maps nondetection to
zero and detected RSS to (RSS + 105) / 105. Thus (mean_encoded + 105) / 105
is exactly the mean of that existing per-scan Tab feature. Keep this value
continuous: thresholding at -100 or rounding would change the representation.

Candidate coordinates and APs come from the original H/S context. The
translated rows carry the target floor; original S is passed separately to
FrozenHistoryRegressor.fit, once. Original bounds, APs, seed and weights stay
with the caller. This is a Tab feature context, not measured scans for GP
likelihood fitting. No training or prediction runs in this module.
"""
import numpy as np


def translated_history_context(candidate_xy, mean_encoded_rss, aps, target_floor, n_ap=520):
    """Return the history argument accepted by FrozenHistoryRegressor.fit.

    mean_encoded_rss has columns in aps order and includes -105 nondetection
    in its position averages. Other columns retain the raw-layout sentinel.
    """
    rss = np.full((len(candidate_xy), n_ap), 100., dtype=np.float64)
    rss[:, np.asarray(aps)] = mean_encoded_rss
    return {target_floor: dict(rss=rss, xy=np.asarray(candidate_xy, dtype=np.float64))}
