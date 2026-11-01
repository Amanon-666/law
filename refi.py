"""Fit one target-floor fingerprint map and locate subsequent Wi-Fi scans."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from external_baselines import tpm_adapter as tpm


def _scans(sample, width):
    rss = np.asarray(sample["rss"], dtype=float)
    xy = np.asarray(sample["xy"], dtype=float)
    if rss.ndim != 2 or rss.shape[1] != width or xy.shape != (len(rss), 2):
        raise ValueError("Expected aligned rss [scans, APs] and xy [scans, 2]")
    if not np.isfinite(rss).all() or not np.isfinite(xy).all():
        raise ValueError("Use 100 for an undetected AP; RSS and coordinates must be finite")
    return {"rss": rss, "xy": xy}


@dataclass(frozen=True)
class FingerprintMap:
    prototypes: np.ndarray
    xy: np.ndarray
    visible: np.ndarray

    def predict(self, query_rss):
        """Return [queries, 2] coordinates in the input coordinate system."""
        q = np.asarray(query_rss, dtype=float)
        if q.ndim != 2 or q.shape[1] != len(self.visible) or not np.isfinite(q).all():
            raise ValueError("Expected finite query RSS [queries, aligned APs]")
        probabilities = tpm.tobit_weights(q[:, self.visible], self.prototypes, 12., -85.)
        return probabilities @ (self.xy - self.xy[0]) + self.xy[0]


def smooth_history(prior, length, strength):
    """Shrink local peaks/troughs in the observed-space historical map."""
    values, xy = prior
    if strength == 0:
        return prior
    weights = np.exp(-cdist(xy, xy, "sqeuclidean") / (2 * length**2))
    weights /= weights.sum(1, keepdims=True)
    return (1. - strength) * values + strength * (weights @ values), xy


def fit_map(histories, support, *, history_length=10., smoothing=.5):
    """Use aligned, same-frame H and S. For K=0, pass empty 2-D S arrays.

    Both the historical smoothing range and 40-unit support influence range
    are expressed in UJI projected units.
    Multiple scans at one position are averaged. Each historical source first
    contributes one prototype per coordinate, then sources at identical
    coordinates receive equal weight. The target anchors stay in the map.
    """
    s = np.asarray(support["rss"])
    if s.ndim != 2:
        raise ValueError("Expected a 2-D support RSS matrix")
    if not np.isfinite(history_length) or history_length <= 0 or not 0. <= smoothing <= 1.:
        raise ValueError("Expected positive finite history_length and smoothing in [0,1]")
    width = s.shape[1]
    support = _scans(support, width)
    floors = [_scans(h, width) for h in histories]
    floors = [h for h in floors if len(h["rss"])]
    if not floors:
        raise ValueError("This history-conditioned model needs at least one historical scan")
    visible = np.logical_or.reduce([(h["rss"] != 100).any(0) for h in [*floors, support]])
    hist = [dict(rss=h["rss"][:, visible], xy=h["xy"]) for h in floors]
    anchors = dict(rss=support["rss"][:, visible], xy=support["xy"])
    prior = tpm.history_map(hist, transfer=None, cross_floor=None)
    prior = smooth_history(prior, history_length, smoothing)
    prototypes, xy = tpm.calibrate(prior, anchors, (40., 1.))
    return FingerprintMap(prototypes, xy, visible)
