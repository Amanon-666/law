"""Protocol adaptation of the project's existing TPM/SCM-T pipeline.

Preserves TPM prior_map, observed-space anchor residual GP, and Tobit matching.
Multiple histories are transformed separately, then equal-averaged at exactly
coincident H coordinates before calibration. This pooling is an adaptation,
not the original single-reference method or an author-code reproduction.

Ports of evaluate_signal_calibrated_map.py and evaluate_scm_tobit.py from
FeMLoc-TPM-Learned-Map; the source-fit prior_map is called directly. No target
Q coordinates are accepted. Columns/coordinates must already be aligned.
"""
import numpy as np
from scipy.special import log_ndtr

from data import group_scans


def dbm(rss):
    return np.where(rss == 100, -110., rss).astype(float)


def position_map(values, xy):
    positions, inverse = np.unique(xy, axis=0, return_inverse=True)
    sums = np.zeros((len(positions), values.shape[1]))
    np.add.at(sums, inverse, values)
    return sums / np.bincount(inverse)[:, None], positions


def lookup(xy, positions, prototypes):
    distance = np.linalg.norm(xy[:, None]-positions[None], axis=-1)
    indices = np.argsort(distance, axis=1)[:, :3]
    weight = 1 / (np.take_along_axis(distance, indices, axis=1) + 1)
    weight /= weight.sum(1, keepdims=True)
    return (weight[..., None] * prototypes[indices]).sum(1)


def calibrate(prior, support, field):
    prototypes, positions = prior
    if not len(support['rss']):
        return prior
    anchor, anchor_xy = position_map(dbm(support['rss']), support['xy'])
    residual = anchor - lookup(anchor_xy, positions, prototypes)
    length, noise = field
    def kernel(a, b):
        return np.exp(-((a[:, None]-b[None])**2).sum(-1)/(2*length**2))
    coefficients = np.linalg.solve(kernel(anchor_xy,anchor_xy)+noise*np.eye(len(anchor_xy)),residual)
    virtual = np.clip(prototypes+kernel(positions,anchor_xy)@coefficients,-110.,0.)
    return np.vstack((virtual,anchor)), np.vstack((positions,anchor_xy))


def history_map(histories, transfer, cross_floor):
    maps = []
    for h in histories:
        if transfer is None:
            maps.append(position_map(dbm(h['rss']),h['xy']))
        else:
            grouped = group_scans(h['rss'],h['xy'])
            cells = dict(xy=grouped['positions'],**grouped['stats'])
            maps.append(cross_floor.prior_map(cells,transfer))
    return position_map(np.concatenate([m for m,_ in maps]),np.concatenate([xy for _,xy in maps]))


def tobit_weights(query_rss, prototypes, sigma, threshold):
    detected = (query_rss != 100).astype(float)
    x = dbm(query_rss) * detected
    squared = (x*x).sum(1,keepdims=True)+detected@(prototypes*prototypes).T-2*x@prototypes.T
    scores = -.5/sigma**2*squared+(1-detected)@log_ndtr((threshold-prototypes)/sigma).T
    probability = np.exp(scores-scores.max(1,keepdims=True))
    return probability/probability.sum(1,keepdims=True)


def predict(history_bank, support, query_rss, transfer, field, match, cross_floor):
    if history_bank:
        prototypes, positions = calibrate(history_map(history_bank,transfer,cross_floor),support,field)
    else:
        if not len(support['rss']):
            raise ValueError('TPM needs H or at least one labelled S position')
        prototypes, positions = position_map(dbm(support['rss']),support['xy'])
    probability = tobit_weights(query_rss,prototypes,*match)
    return probability@positions
