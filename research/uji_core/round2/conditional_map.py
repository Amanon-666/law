"""Research readout fitted on aligned historical and support fingerprints.

The fitted map returns XY in the input coordinate system. Use
coordinate_units_per_uji=1 for UJI-equivalent inputs and 0.766 for raw metres.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr, softmax

from external_baselines import tpm_adapter as tpm
from refi import FingerprintMap, _scans, historical_prior


EXTERNAL_GATE = (0.0, 0.21941321218335094, 0.0)
EXTERNAL_SCALAR = (0.04228933469458107,) * 3
BAND_EDGES = (0.2, 0.8)


def historical_moments(histories, gaps):
    """Return filled RSS means, XY, detection probabilities and detected-RSS means."""
    mean, xy = historical_prior(histories, 10.0, 0.5, gaps)
    detections = []
    strengths = []
    for history in histories:
        rss = np.asarray(history["rss"], dtype=float)
        seen = rss != 100
        detections.append({"rss": seen.astype(float), "xy": history["xy"]})
        # These moments pass through the same historical averaging operation as
        # RSS. Keep their numerical encoding distinct from its sentinel 100.
        strength = np.where(seen, (rss + 110.0) / 110.0, 0.0)
        strengths.append({"rss": strength, "xy": history["xy"]})

    probability, detection_xy = historical_prior(detections, 10.0, 0.5, gaps)
    numerator, strength_xy = historical_prior(strengths, 10.0, 0.5, gaps)
    if not np.array_equal(xy, detection_xy) or not np.array_equal(xy, strength_xy):
        raise ValueError("Historical moments have different candidate positions")

    observed = probability > 0.0
    conditional_strength = np.divide(
        numerator, probability, out=np.zeros_like(numerator), where=observed
    )
    detected_mean = np.where(observed, -110.0 + 110.0 * conditional_strength, mean)
    return mean, xy, probability, detected_mean


@dataclass(frozen=True)
class ConditionalMeanMap:
    base: FingerprintMap
    h_difference: np.ndarray
    h_bands: np.ndarray
    coefficients: tuple

    def predict(self, query_rss, coefficients=None):
        """Predict XY, optionally substituting one of the frozen control curves."""
        gate = np.asarray(self.coefficients if coefficients is None else coefficients, dtype=float)
        if gate.shape != (3,) or not np.isfinite(gate).all() or np.any((gate < 0) | (gate > 1)):
            raise ValueError("Expected three finite weights between zero and one")

        query = np.asarray(query_rss, dtype=float)
        if query.ndim != 2 or query.shape[1] != len(self.base.visible) or not np.isfinite(query).all():
            raise ValueError("Expected aligned, finite query RSS")
        query = query[:, self.base.visible]

        prototypes = self.base.prototypes
        n_history = len(self.h_difference)
        history = prototypes[:n_history]
        corrected = np.clip(history + self.h_difference, -110.0, 0.0)
        detected_mean = prototypes.copy()
        # tpm.calibrate appends the measured support anchors after the H rows.
        # The historical correction applies only to the H part of the map.
        detected_mean[:n_history] += gate[self.h_bands] * (corrected - history)

        seen = (query != 100).astype(float)
        observed_rss = np.where(query != 100, query + 75.0, 0.0)
        centered_mean = detected_mean + 75.0
        detected_score = (
            observed_rss @ centered_mean.T - (0.5 * seen) @ (centered_mean ** 2).T
        ) / 144.0
        missed_score = (1.0 - seen) @ log_ndtr((-85.0 - prototypes) / 12.0).T
        weights = softmax(detected_score + missed_score, axis=1)
        xy = self.base.xy
        return weights @ (xy - xy[0]) + xy[0]


def fit_conditional_map(
    histories, support, *, floor_gaps=None, coordinate_units_per_uji=1.0, coefficients=EXTERNAL_GATE
):
    """Fit the gap-weighted historical map and store its detected-RSS moments.

    RSS columns must align across all inputs. An empty support uses RSS with
    shape (0, number of APs) and XY with shape (0, 2). Signed or absolute
    floor-index differences follow the original order of the historical scans.
    """
    scale = float(coordinate_units_per_uji)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Invalid XY unit conversion")
    support_rss = np.asarray(support["rss"])
    if support_rss.ndim != 2:
        raise ValueError("Expected a two-dimensional support RSS matrix")

    def scaled_scans(sample):
        scans = _scans(sample, support_rss.shape[1])
        return {"rss": scans["rss"], "xy": scans["xy"] / scale}

    floors = [scaled_scans(history) for history in histories]
    anchors = scaled_scans(support)
    keep = [len(floor["rss"]) > 0 for floor in floors]
    gaps = None
    if floor_gaps is not None:
        gaps = np.asarray(floor_gaps, dtype=float)
        if gaps.shape != (len(floors),) or not np.isfinite(gaps).all():
            raise ValueError("Expected one finite floor gap per historical source")
        gaps = gaps[keep]
    floors = [floor for floor, use in zip(floors, keep) if use]
    if not floors:
        raise ValueError("Need historical scans")

    visible = np.logical_or.reduce([(s["rss"] != 100).any(axis=0) for s in [*floors, anchors]])
    mean, xy, probability, detected_mean = historical_moments(floors, gaps)
    visible_anchors = {"rss": anchors["rss"][:, visible], "xy": anchors["xy"]}
    prototypes, positions = tpm.calibrate((mean[:, visible], xy), visible_anchors, (40.0, 1.0))
    return ConditionalMeanMap(
        base=FingerprintMap(prototypes, positions * scale, visible),
        h_difference=detected_mean[:, visible] - mean[:, visible],
        h_bands=np.searchsorted(BAND_EDGES, probability[:, visible], side="right"),
        coefficients=tuple(coefficients),
    )
