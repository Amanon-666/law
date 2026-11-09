"""Within-source/location observed interactions on fixed GP marginals.

Only M/L kernel coordinates are centered explicitly. H=1-M-L supplies the
third coordinate, so the interaction has zero node margins by construction.
"""
import itertools

import numpy as np
import torch
from scipy.special import ndtr

from marginal_relation import gp_bins


def source_groups(proxy, event):
    sizes = [len(rows) for rows in event['history_rows'].values()]
    assert sum(sizes) == len(proxy['rss'])
    # Section149 concatenated histories in the frozen event dictionary order.
    source = np.repeat(np.arange(len(sizes)), sizes)
    _, first, groups, counts = np.unique(np.column_stack((source, proxy['xy'])),
        axis=0, return_index=True, return_inverse=True, return_counts=True)
    return groups.ravel(), first, counts


def complete_interaction(small):
    result = np.empty((*small.shape[:-2], 3, 3))
    result[..., :2, :2] = small
    result[..., :2, 2] = -small.sum(-1)
    result[..., 2, :2] = -small.sum(-2)
    result[..., 2, 2] = small.sum(axis=(-2, -1))
    return result


@torch.no_grad()
def interactions(proxy, mean, edges, event, between=True):
    groups, first, counts = source_groups(proxy, event)
    device = torch.device('cuda:0')
    rss = torch.as_tensor(proxy['rss'], dtype=torch.float64, device=device)
    cuts = torch.as_tensor(mean, dtype=torch.float64, device=device)
    spatial = torch.as_tensor(proxy['spatial'], dtype=torch.float64, device=device)
    locations = torch.as_tensor(proxy['group'], dtype=torch.long, device=device)
    group = torch.as_tensor(groups, dtype=torch.long, device=device)
    first = torch.as_tensor(first, dtype=torch.long, device=device)
    count = torch.as_tensor(counts, dtype=torch.float64, device=device)
    edge = torch.as_tensor(edges, dtype=torch.long, device=device)
    a, b = edge.T
    detected = (rss != 100).to(torch.float64)
    missing = 1-detected
    gm = torch.zeros((len(count), rss.shape[1]), dtype=torch.float64, device=device)
    gm.index_add_(0, group, missing)
    gm /= count[:, None]
    cm = missing-gm[group]
    within = np.empty((len(mean), len(edges), 2, 2))
    other = np.empty_like(within) if between else None
    for start in range(0, len(mean), 32):
        sl = slice(start, start+32)
        batch = len(cuts[sl])
        weights = spatial[sl, locations]
        group_weights = weights[:, first]*count[None, :]
        z = (cuts[sl, None, :]-rss[None, :, :])/proxy['bandwidth']
        low = .5*torch.special.erfc(-z/(2.**.5))*detected
        gl = torch.zeros((batch, len(count), rss.shape[1]), dtype=torch.float64, device=device)
        gl.index_add_(1, group, low)
        gl /= count[None, :, None]
        centered = (cm.expand_as(low), low-gl[:, group])
        for i in range(2):
            for j in range(2):
                within[sl, :, i, j] = torch.einsum('gn,gne,gne->ge', weights,
                    centered[i][:, :, a], centered[j][:, :, b]).cpu().numpy()
        if between:
            group_nodes = (gm.expand_as(gl), gl)
            group_centered = tuple(values-torch.einsum('gs,gsa->ga', group_weights, values)[:, None, :]
                                   for values in group_nodes)
            for i in range(2):
                for j in range(2):
                    other[sl, :, i, j] = torch.einsum('gs,gse,gse->ge', group_weights,
                        group_centered[i][:, :, a], group_centered[j][:, :, b]).cpu().numpy()
    stats = dict(groups=len(counts), single_scan_groups=int((counts == 1).sum()),
                 minimum_group_rows=int(counts.min()), maximum_group_rows=int(counts.max()))
    return complete_interaction(within), None if other is None else complete_interaction(other), stats


def relation_table(delta, probabilities, edges):
    a, b = edges.T
    row, column = probabilities[:, a], probabilities[:, b]
    independent = row[..., :, None]*column[..., None, :]
    relative = delta/independent
    scale = .5/np.maximum(1., -relative.min(axis=(-2, -1)))
    factor = 1+scale[..., None, None]*relative
    assert np.isfinite(factor).all() and factor.min() >= .5-1e-12
    table = independent*factor
    error = max(float(np.max(np.abs(table.sum(-1)/row-1))),
                float(np.max(np.abs(table.sum(-2)/column-1))))
    assert error <= 1e-8, error
    return np.log1p(scale[..., None, None]*relative), dict(
        maximum_relative_marginal_error=error, minimum_edge_GP_fraction=float(factor.min()),
        scale_minimum=float(scale.min()), scale_median=float(np.median(scale)),
        scale_maximum=float(scale.max()), maximum_absolute_interaction=float(np.abs(delta).max()))


def finite_check():
    p = np.array([[[.2, .3, .5], [.4, .35, .25], [.1, .7, .2]]])
    edges = np.array([[0, 1], [1, 2]])
    # Same XY, different sources: merging solely by XY would fail this check.
    rss = np.array([[100., -80., -75.]]*2+[[-90., 100., -95.]]*2)
    mean = np.array([[-85., -85., -85.]])
    proxy = dict(rss=rss, xy=np.zeros((4, 2)), group=np.zeros(4, dtype=int),
                 spatial=np.full((1, 1), .25), bandwidth=3.)
    event = dict(history_rows={'A':[0, 1], 'B':[2, 3]})
    within, between, groups = interactions(proxy, mean, edges, event)
    detected = rss != 100
    low = ndtr((mean[0]-rss)/3.)*detected
    kernels = np.stack((1-detected, low, detected-low), axis=-1)
    average = kernels.mean(0)
    pooled = np.stack([np.einsum('ni,nj->ij', kernels[:, a], kernels[:, b])/4
        -np.outer(average[a], average[b]) for a, b in edges])[None]
    assert groups['groups'] == 2 and np.abs(within).max() == 0
    identity, _ = relation_table(within, p, edges)
    ratio, diagnostics = relation_table(between, p, edges)
    labels = np.array(list(itertools.product(range(3), repeat=3)))
    mass = np.array([np.prod(p[0, np.arange(3), row])
        *np.exp(sum(ratio[0, e, row[a], row[b]] for e, (a, b) in enumerate(edges)))
        for row in labels])
    error = max(float(np.abs(pooled-between-within).max()), float(np.abs(identity).max()),
                float(np.abs(complete_interaction(between[..., :2, :2])-between).max()), abs(mass.sum()-1))
    for a in range(3):
        for state in range(3):
            error = max(error, abs(mass[labels[:, a] == state].sum()-p[0, a, state]))
    assert np.abs(between).max() > 0 and error <= 1e-12, error
    return dict(passed=True, enumerated_states=27, maximum_error=float(error),
                deterministic_groups_have_no_within_interaction=True, projection=diagnostics)
