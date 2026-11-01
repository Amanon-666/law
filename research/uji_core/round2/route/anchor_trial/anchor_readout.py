"""Optional readout for frozen anchor mixtures; no training or fitted defaults."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from external_baselines import tpm_adapter as tpm
from refi import FingerprintMap, _scans, historical_prior


@dataclass(frozen=True)
class AnchorReadout:
    historical: np.ndarray
    raw: np.ndarray
    field: np.ndarray
    xy: np.ndarray
    visible: np.ndarray

    def map_for(self, field_fraction):
        """g=0 appends empirical support rows; g=1 appends field predictions."""
        g = float(field_fraction)
        if not np.isfinite(g) or not 0.0 <= g <= 1.0:
            raise ValueError("Expected field fraction in [0, 1]")
        anchor_rows = self.field if g == 1.0 else self.raw + g * (self.field - self.raw)
        return FingerprintMap(np.vstack((self.historical, anchor_rows)), self.xy, self.visible)


def fit_anchor_readout(histories, support, *, floor_gaps=None, coordinate_units_per_uji=1.0):
    """Construct the fixed public historical field and both anchor templates.

    Coordinates are returned in the input frame. All probes share the same
    support interpolation, ridge, spatial update, candidate and AP ordering.
    An exact H/S location tie copies the calibrated historical prototype for
    that XY, following the enrolled anchor_measure convention.
    """
    unit = float(coordinate_units_per_uji)
    if not np.isfinite(unit) or unit <= 0.0:
        raise ValueError("Invalid coordinate scale")
    rss = np.asarray(support["rss"])
    if rss.ndim != 2:
        raise ValueError("Expected two-dimensional RSS")
    width = rss.shape[1]

    def normalized(sample):
        scan = _scans(sample, width)
        return {"rss": scan["rss"], "xy": scan["xy"] / unit}

    h = [normalized(x) for x in histories]
    s = normalized(support)
    keep = [len(x["rss"]) > 0 for x in h]
    gaps = None
    if floor_gaps is not None:
        gaps = np.asarray(floor_gaps, float)
        if gaps.shape != (len(h),) or not np.isfinite(gaps).all():
            raise ValueError("Expected a finite gap for each history")
        gaps = gaps[keep]
    h = [x for x, nonempty in zip(h, keep) if nonempty]
    if not h:
        raise ValueError("At least one historical scan is required")
    mask = np.logical_or.reduce([(x["rss"] != 100).any(0) for x in [*h, s]])
    h = [dict(rss=x["rss"][:, mask], xy=x["xy"]) for x in h]
    s = dict(rss=s["rss"][:, mask], xy=s["xy"])

    prior = historical_prior(h, 10.0, 0.5, gaps)
    h_mean, h_xy = prior
    prototypes, xy = tpm.calibrate(prior, s, (40.0, 1.0))
    n_history = len(h_xy)
    post_h, raw = prototypes[:n_history], prototypes[n_history:]
    s_xy = xy[n_history:]
    if not len(s_xy):
        return AnchorReadout(post_h, raw, raw, xy * unit, mask)

    local = tpm.lookup(s_xy, h_xy, h_mean)
    kernel = np.exp(-cdist(s_xy, s_xy, "sqeuclidean") / (2 * 40.0 ** 2))
    alpha = np.linalg.solve(kernel + np.eye(len(s_xy)), raw - local)
    field = np.clip(local + kernel @ alpha, -110.0, 0.0)
    index = {tuple(point): i for i, point in enumerate(h_xy)}
    for i, point in enumerate(s_xy):
        j = index.get(tuple(point))
        if j is not None:
            field[i] = post_h[j]
    return AnchorReadout(post_h, raw, field, xy * unit, mask)
