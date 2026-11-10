"""Descriptive evidence alongside the original history position map.

Counts describe raw records, not independent effective sample sizes. Empirical
population variances describe recorded RSS dispersion, not uncertainty of a
known mean. No variance/count uncertainty estimate is made here.
"""
import numpy as np

from vertical_translator import translator as T


def position_evidence(rss, xy):
    """Return original P/M/R plus counts and empirical population variances.

    Arrays follow the original sorted position/AP order. n_scans is [P];
    n_detected and all additional moments are [P,A]. Detected means with no
    detections and variances with fewer than two relevant records are NaN.
    """
    rss, xy = np.asarray(rss), np.asarray(xy)
    positions, mean, rate = T.position_map(rss, xy)
    _, inverse = np.unique(xy, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_scans = np.bincount(inverse, minlength=len(positions))
    detected = rss != 100
    n_detected = np.zeros(mean.shape, dtype=np.int64)
    detected_sum = np.zeros_like(mean)
    np.add.at(n_detected, inverse, detected)
    np.add.at(detected_sum, inverse, np.where(detected, rss, 0.))
    detected_mean = np.divide(detected_sum, n_detected,
        out=np.full_like(mean, np.nan), where=n_detected > 0)

    encoded_squared_deviation = np.zeros_like(mean)
    np.add.at(encoded_squared_deviation, inverse,
              (T.encode(rss) - mean[inverse]) ** 2)
    encoded_variance = np.divide(encoded_squared_deviation, n_scans[:, None],
        out=np.full_like(mean, np.nan), where=n_scans[:, None] >= 2)

    detected_squared_deviation = np.zeros_like(mean)
    residual = np.where(detected, rss - detected_mean[inverse], 0.)
    np.add.at(detected_squared_deviation, inverse, residual ** 2)
    detected_variance = np.divide(detected_squared_deviation, n_detected,
        out=np.full_like(mean, np.nan), where=n_detected >= 2)
    return dict(P=positions, M=mean, R=rate, n_scans=n_scans,
                n_detected=n_detected, detected_mean=detected_mean,
                encoded_variance=encoded_variance,
                detected_variance=detected_variance)


def neighborhood_evidence(X, evidence):
    """Count evidence within the original inclusive T.R_SMOOTH radius.

    X must contain permitted candidates or source-supervision positions.
    Callers must not pass deployment Query truth coordinates. Empty
    neighborhoods have zero counts; no spatial or detection sampling occurs.
    """
    X = np.asarray(X)
    distance = np.linalg.norm(X[:, None] - evidence['P'][None], axis=2)
    membership = (distance <= T.R_SMOOTH).astype(np.int64)
    return dict(n_positions=membership.sum(axis=1),
                n_scans=membership @ evidence['n_scans'],
                n_detected=membership @ evidence['n_detected'])
