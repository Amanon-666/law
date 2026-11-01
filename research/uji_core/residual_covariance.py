"""Source-fitted second moments for a single observed-space residual update.

From repo root: python -m research.uji_core.residual_covariance DATA OUT
All fits use training.csv alone and hold out the entire evaluation building.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from external_baselines import tpm_adapter as tpm
from .forward import MATCH, scores
from .protocol import BUDGETS, HASHES, SEEDS, history_floors, read_floors, split_rows

SCALES = (10., 40., 80.)
FIXED = (0., 0., 1., 0.)
UNIFORM = (.25,) * 4


def basis(squared_distance):
    return np.stack([np.ones_like(squared_distance)] +
                    [np.exp(-squared_distance/(2.*s**2)) for s in SCALES], axis=-1)


def floor_moments(target, histories):
    hist, hxy = tpm.history_map(histories, None, None)
    mask = (target['rss'] != 100).any(0) | np.logical_or.reduce([(h['rss'] != 100).any(0) for h in histories])
    raw = tpm.dbm(target['rss'][:, mask])
    xy, g, counts = np.unique(target['xy'], axis=0, return_inverse=True, return_counts=True)
    g = g.ravel()
    mean, _ = tpm.position_map(raw, target['xy'])
    residual = mean - tpm.lookup(xy, hxy, hist[:, mask])
    # Across APs the uncentered second moment includes an environment-specific
    # spatially constant offset, which the constant covariance term can transfer.
    covariance = residual @ residual.T / raw.shape[1]
    scan_sq = ((raw - mean[g])**2).mean(1)
    ss = np.bincount(g, weights=scan_sq, minlength=len(xy))
    repeated = counts > 1
    if not repeated.any():
        raise ValueError('No within-position repeats to identify observation noise')
    variances = np.empty(len(xy))
    variances[repeated] = ss[repeated] / (counts[repeated] - 1)
    variances[~repeated] = variances[repeated].mean()
    # For a mean of n independent same-position measurements the noise energy
    # on the diagonal is v/n. Cross-position products do not require this correction.
    nugget = variances.mean()
    latent = np.mean(np.diag(covariance) - variances/counts)
    if latent <= 0 or nugget <= 0:
        raise ValueError(f'Unidentifiable dispersion: latent={latent}, nugget={nugget}')
    i, j = np.triu_indices(len(xy), 1)
    d2 = np.sum((xy[i]-xy[j])**2, axis=1)
    x, y = basis(d2), covariance[i, j] / latent
    # Sufficient statistics for each floor's mean normalized pairwise squared error.
    return dict(gram=x.T@x/len(x), cross=x.T@y/len(x), y2=float(y@y/len(y)),
                noise_fraction=float(nugget/(latent+nugget)), latent=float(latent), nugget=float(nugget),
                aps=int(mask.sum()), positions=len(xy), singleton=int((~repeated).sum()), pairs=len(x))


def fit(moments, held):
    source = sorted({b for b, _ in moments if b != held})

    def average(key):
        return np.mean([np.mean([m[key] for (b, _), m in moments.items() if b == source_b], axis=0)
                        for source_b in source], axis=0)

    gram, cross, y2 = (average(k) for k in ('gram','cross','y2'))

    def criterion(w):
        return float(w @ gram @ w - 2*w @ cross + y2)

    # Nonnegative constant and RBF coefficients summing to one enforce a PSD
    # unit-diagonal kernel. The learned nugget is expressed on that same scale.
    result = minimize(criterion, UNIFORM, jac=lambda w: 2*(gram@w-cross), method='SLSQP',
                      bounds=[(0.,1.)]*4, constraints=[dict(type='eq',fun=lambda w: w.sum()-1.,
                                                             jac=lambda w: np.ones(4))],
                      options={'ftol':1e-12, 'maxiter':500})
    if not result.success or np.min(result.x) < -1e-8 or abs(result.x.sum()-1)>1e-8:
        raise RuntimeError(result.message)
    weights = np.maximum(result.x, 0.)
    weights /= weights.sum()
    noise_fraction = float(average('noise_fraction'))
    return dict(source_buildings=source, weights=weights.tolist(), nugget=noise_fraction/(1-noise_fraction),
                normalized_pair_loss=dict(fitted=criterion(weights), uniform=criterion(np.array(UNIFORM)),
                                          fixed40=criterion(np.array(FIXED))))


def evaluate(train, valid, fits):
    rows = []
    for (b,f), target in train.items():
        histories = history_floors(train,b,f)
        hist, hxy = tpm.history_map(histories,None,None)
        hmask = np.logical_or.reduce([(h['rss'] != 100).any(0) for h in histories])
        fitted = fits[str(b)]
        arms = {'fixed40': (FIXED,1.), 'fixed_uniform': (UNIFORM,1.),
                'shape_only': (fitted['weights'],1.), 'nugget_only': (FIXED,fitted['nugget']),
                'learned': (fitted['weights'],fitted['nugget'])}
        query = valid[b,f]
        truth = query['xy']
        _, g, counts = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
        for seed in SEEDS:
            selected, _ = split_rows(target,b,f,seed)
            for k in BUDGETS:
                sr = selected[:k]
                mask = hmask | (target['rss'][sr] != 100).any(0)
                anchor,sxy = tpm.position_map(tpm.dbm(target['rss'][sr][:,mask]),target['xy'][sr])
                prior = hist[:,mask]
                innovation = anchor - tpm.lookup(sxy,hxy,prior)
                ss = basis(cdist(sxy,sxy,'sqeuclidean'))
                hs = basis(cdist(hxy,sxy,'sqeuclidean'))
                xy = np.vstack((hxy,sxy))
                for name,(weights,nugget) in arms.items():
                    alpha = np.linalg.solve(ss@weights+nugget*np.eye(k),innovation)
                    prototypes = np.vstack((np.clip(prior+(hs@weights)@alpha,-110.,0.),anchor))
                    probabilities = tpm.tobit_weights(query['rss'][:,mask],prototypes,*MATCH)
                    predicted = probabilities @ (xy-xy[0]) + xy[0]
                    score = scores(predicted,truth,g.ravel(),counts)
                    rows.append(dict(building=b,floor=f,seed=seed,K=k,method=name,**score))
    return pd.DataFrame(rows)


def main(data, output):
    output.mkdir(parents=True, exist_ok=True)
    train = read_floors(data,'trainingData.csv')
    moments = {(b,f):floor_moments(target,history_floors(train,b,f)) for (b,f),target in train.items()}
    selected = {str(b):fit(moments,b) for b in sorted({b for b,_ in train})}
    stats = {f'B{b}F{f}':{k:v for k,v in row.items() if k not in ('gram','cross','y2')}
             for (b,f),row in moments.items()}
    freeze = output/'covariance_parameters.json'
    record = dict(training_sha256=HASHES['trainingData.csv'], fixed_scales=SCALES,
                  basis_order=['constant','rbf10','rbf40','rbf80'], coordinates='UJI projected units',
                  observation='Single scan 100->-110; average before estimating second moments. '
                  'Within-position variance includes time/device variability; diagonal correction assumes independent repeats.',
                  selection='Fit normalized off-diagonal AP-averaged residual products, equal source buildings '
                  'and floors, equal unordered spatial pairs within floor. Single-scan noise fraction averaged '
                  'with the same hierarchy. Weight simplex gives a PSD field for every target.',
                  scope='Exploratory follow-up to previous UJI comparisons; the official CSV is not used for fitting.',
                  floor_statistics=stats, selected=selected)
    freeze.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    valid = read_floors(data,'validationData.csv')
    evaluate(train,valid,json.loads(freeze.read_text())['selected']).to_csv(output/'covariance_events.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('output',type=Path)
    opt=parser.parse_args()
    main(opt.data,opt.output)
