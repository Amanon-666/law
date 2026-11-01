"""Compare shared score laws on identical, unit-gain forward maps.

Fit from the training CSV only. Official evaluation calls no selection code.
Usage: python -m research.uji_core.forward_match path/to/UJIndoorLoc
"""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import softmax

from external_baselines import tpm_adapter as tpm
from research.uji_core.forward import FIXED_RADIUS, MATCH, RIDGE, scores
from research.uji_core.protocol import BUDGETS, HASHES, SEEDS, history_floors, read_floors, split_rows

CENTERS = np.arange(-110., 1., 5.)
SMOOTH_DB = 5.
SIGMA_FLOOR = 3.
P_FLOOR = 1e-6
GRID = tuple(product((8., 12., 16., 24.), (-95., -85., -75.)))


def field_episodes(train, official, building, floor):
    target = train[building, floor]
    histories = history_floors(train, building, floor)
    prototypes, hxy = tpm.history_map(histories, None, None)
    visible = np.logical_or.reduce([(h['rss'] != 100).any(0) for h in histories])
    for seed in SEEDS:
        sr, qr = split_rows(target, building, floor, seed, source=official is None)
        if official is None:
            q = {key: target[key][qr] for key in ('xy','rss','ids')}
        else:
            q = official[building, floor]
        for k in BUDGETS:
            selected = sr[:k]
            mask = visible | (target['rss'][selected] != 100).any(0)
            s = dict(rss=target['rss'][selected][:,mask], xy=target['xy'][selected])
            m, xy = tpm.calibrate((prototypes[:,mask], hxy), s, (FIXED_RADIUS, RIDGE))
            yield seed, k, m, xy, q['rss'][:,mask], q['xy'], q['ids']


def observation_stats(m, xy, q, truth):
    """Return [total, detected, detected sum, detected sum of squares] per 5 dB bin.

    Ground-truth coordinates are source supervision here. Equal mass per source
    query, split among ties for the closest prototype and its visible APs.
    """
    dist = cdist(truth, xy, metric='sqeuclidean')
    nearest = np.isclose(dist, dist.min(1, keepdims=True), atol=1e-9, rtol=0.)
    iq, im = np.nonzero(nearest)
    w = np.broadcast_to(1. / nearest.sum(1)[iq,None], (len(iq), m.shape[1])).ravel()
    bins = np.clip(np.floor((m[im].ravel()-CENTERS[0])/5.+.5).astype(int), 0, len(CENTERS)-1)
    rss = q[iq].ravel().astype(float)
    seen = rss != 100
    weight = w * seen
    return np.array([np.bincount(bins, weights=v, minlength=len(CENTERS))
                     for v in (w, weight, weight*rss, weight*rss*rss)]) / (len(q)*m.shape[1])


def fit_law(stats):
    kernel = np.exp(-.5*((CENTERS[:,None]-CENTERS[None,:])/SMOOTH_DB)**2)
    total, observed, sums, squares = stats @ kernel
    prob = np.clip(observed/total, P_FLOOR, 1.-P_FLOOR)
    mu = sums/observed
    var = np.maximum(squares/observed - mu**2, SIGMA_FLOOR**2)
    if not np.isfinite([prob, mu, var]).all():
        raise ValueError('Source law has empty or nonfinite bins')
    return dict(prob=prob.tolist(), mu=mu.tolist(), var=var.tolist())


def empirical_weights(q, m, law):
    """P(detected|m), then Gaussian RSS given detection, shared across APs."""
    p, mu, var = [np.interp(m, CENTERS, law[key]) for key in ('prob','mu','var')]
    seen = (q != 100).astype(float)
    rss = q * seen
    logp = ((1.-seen) @ np.log1p(-p).T
            + seen @ (np.log(p)-.5*np.log(2*np.pi*var)-.5*mu*mu/var).T
            + rss @ (mu/var).T - .5*(rss*rss) @ (1./var).T)
    return softmax(logp, axis=1)


def score_weights(weights, xy, truth):
    _, grp, counts = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
    pred = weights @ (xy-xy[0]) + xy[0]
    return scores(pred, truth, grp.ravel(), counts)


def source_fit(train, output):
    stats = {}
    rows = []
    for building, floor in train:
        stats[building, floor] = np.zeros((4, len(CENTERS)))
        for seed, k, m, xy, q, truth, _ in field_episodes(train, None, building, floor):
            stats[building,floor] += observation_stats(m, xy, q, truth) / (len(SEEDS)*len(BUDGETS))
            for sigma, threshold in GRID:
                w = tpm.tobit_weights(q, m, sigma, threshold)
                rows.append(dict(building=building, floor=floor, K=k, seed=seed, sigma=sigma,
                                 threshold=threshold, mde=score_weights(w, xy, truth)['mde']))
        print(f'source B{building} F{floor} complete', flush=True)
    source = pd.DataFrame(rows)
    source.groupby(['building','floor','K','sigma','threshold'], as_index=False).mde.mean().to_csv(
        output/'forward_match_source.csv', index=False)
    risks = source.groupby(['building','floor','sigma','threshold']).mde.mean().groupby(['building','sigma','threshold']).mean()
    buildings = sorted({b for b,_ in train})
    bstats = {b: np.mean([s for (sb,_), s in stats.items() if sb == b], axis=0) for b in buildings}
    record = dict(centers=CENTERS.tolist(), bin_assignment='nearest, half-bin ties upward', smooth_db=SMOOTH_DB,
                  prob_clip=P_FLOOR, min_conditional_sd_db=SIGMA_FLOOR, map_radius=FIXED_RADIUS,
                  ridge=RIDGE, fixed_tobit=MATCH, tuner_grid=GRID, sha_train=HASHES['trainingData.csv'],
                  source_objectives='Empirical law: smooth moments of detected RSS and detection rate matched to source Q coordinates; tuned Tobit: source localization MDE.',
                  weighting='equal source buildings, floors, draws, budgets, source Q positions and permitted APs; ties in nearest XY split',
                  scope='Follow-up after earlier aggregate official outcomes; parameters selected before evaluating either new matcher on official data.',
                  pi_by_source_building={b: s.tolist() for b,s in bstats.items()}, folds={})
    for b in buildings:
        risk = risks.drop(index=b, level='building').groupby(['sigma','threshold']).mean()
        choice = min(GRID, key=lambda pair: risk.loc[pair])
        law = fit_law(np.mean([s for sb,s in bstats.items() if sb!=b], axis=0))
        record['folds'][b] = dict(tobit=choice, empirical=law, source_buildings=[sb for sb in buildings if sb!=b])
    target = output/'forward_match_selection.json'
    target.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return json.loads(target.read_text())['folds']


def main(directory, output):
    output.mkdir(parents=True, exist_ok=True)
    train = read_floors(directory, 'trainingData.csv')
    selection = source_fit(train, output)
    official = read_floors(directory, 'validationData.csv')
    rows = []
    for b, f in train:
        selected = selection[str(b)]
        for seed,k,m,xy,q,truth,_ in field_episodes(train, official, b, f):
            arms = {'fixed_tobit': tpm.tobit_weights(q,m,*MATCH),
                    'tuned_tobit': tpm.tobit_weights(q,m,*selected['tobit']),
                    'empirical': empirical_weights(q,m,selected['empirical'])}
            for method, w in arms.items():
                rows.append(dict(building=b, floor=f, seed=seed, K=k, method=method, **score_weights(w, xy, truth)))
        print(f'official B{b} F{f} complete', flush=True)
    pd.DataFrame(rows).to_csv(output/'forward_match_official.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    main(args.data, args.output)
