"""Paired UJI field controls using the existing history/posterior code.

OPENBLAS_NUM_THREADS=1 python -m research.uji_core.field_control DATA PARAMS OUTPUT --building 0
PARAMS is the archived source calibration mapping (by source building). Source
B0 serves B1/B2, source B1 serves B0. No query data select these constants.
This uses the later Gaussian-VI backend, not the original K-curve PMC backend.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import inference
from data import group_scans
from fusion_moments import fuse, source_moments, support_matrices
from fusion_posterior import infer_batch
from fusion_prior import combine
from inference import observation_log_prob, predict_from_posterior
from multi_source_adaptation import fit_source_weights
from operator_prior_v1.operator_prior import marginal_loglik
from research.uji_core.protocol import BUDGETS, history_floors, position_weights, read_floors, split_rows, SEEDS
from run import ap_stats, tensor
from transfer import History, rbf

TARGETS = {0: 1, 1: 1, 2: 2}
SIGMAS = (0., 5., 10., 15., 20.)


def fit_histories(floors, origin, base, obs):
    """Existing full-Gaussian censored history fits, with stationarity check."""
    real_minimize = inference.minimize

    def checked(*args, **kwargs):
        result = real_minimize(*args, **kwargs)
        grad = result.jac.copy()
        for i, (lo, hi) in enumerate(kwargs['bounds']):
            if lo is not None and result.x[i] <= lo + 1e-10 and grad[i] > 0:
                grad[i] = 0.
        if not np.isfinite(grad).all() or np.abs(grad).max() > 1e-3:
            raise RuntimeError(f'Unfinished historical VI: gradient={np.abs(grad).max():.3g}')
        return result

    banks = []
    with patch.object(inference, 'minimize', checked):
        for floor in floors:
            scans = group_scans(floor['rss'], floor['xy'] - origin)
            z = tensor(scans['positions'])
            mean = z.new_full((len(z),), base['mean'])
            cov = rbf(z, z, base['amplitude'], base['length'])
            bank = {}
            for ap in np.flatnonzero(scans['stats']['k'].sum(0)):
                posterior = inference.infer(mean, cov, ap_stats(scans, ap), obs, steps=640)
                bank[int(ap)] = History(z, posterior, base)
            banks.append(bank)
            print(f'History complete: {len(z)} positions, {len(bank)} APs', flush=True)
    return banks


def support_states(histories, support, aps, base, obs):
    sx = tensor(support['positions'])
    transforms = [(1., 0.)] * len(histories)
    matrices, stats = [], []
    for col, ap in enumerate(aps):
        moments = source_moments([h.get(ap) for h in histories], sx, sx[:0], base)
        matrices.append(support_matrices(moments, transforms))
        stats.append(ap_stats(support, col))
    means, covs, residual = (torch.stack(v) for v in zip(*matrices))
    stats = {key: torch.stack([row[key] for row in stats]) for key in stats[0]}
    states, failures = {}, {}
    k = len(sx)
    methods = ('GP_S', 'H_equal', 'H_fitted') if k else ('H_prior',)
    for name in methods:
        try:
            weights = None
            sigma = 0.
            if name == 'GP_S':
                sm = sx.new_full((len(aps), k), base['mean'])
                sk = rbf(sx, sx, base['amplitude'], base['length']).expand(len(aps), -1, -1)
            else:
                weights = [1. / len(histories)] * len(histories)
                if k and name == 'H_fitted':
                    weights, _ = fit_source_weights(means, covs, residual, stats, obs)
                sm, sk = combine(means, covs, residual, weights, False, base['drift'] ** 2)
                if k:
                    scores = [marginal_loglik(sm, sk.diagonal(dim1=-2, dim2=-1) + sd**2,
                                              **stats, obs=obs).sum().item() for sd in SIGMAS]
                    sigma = SIGMAS[int(np.argmax(scores))]
                    sk = sk + sigma**2
            # Independent AP blocks optimize the same objective without a giant
            # joint L-BFGS state. Every block must meet infer_batch's own gate.
            pm, pc = [], []
            for start in range(0, len(aps), 12):
                sl = slice(start, start + 12)
                posterior = infer_batch(sm[sl], sk[sl], {key: v[sl] for key, v in stats.items()}, obs)
                pm.append(posterior['mean'])
                pc.append(posterior['cov'])
            states[name] = dict(sm=sm, sk=sk, pm=torch.cat(pm), pc=torch.cat(pc),
                                weights=weights, sigma=sigma)
        except (RuntimeError, ValueError, FloatingPointError) as err:
            failures[name] = str(err)
    return states, failures


def score(histories, support, aps, grid, query_rss, states, base, obs):
    scores = {name: grid.new_zeros((len(query_rss), len(grid))) for name in states}
    sx = tensor(support['positions'])
    identity = [(1., 0.)] * len(histories)
    for col, ap in enumerate(aps):
        moment = source_moments([h.get(ap) for h in histories], sx, grid, base)
        # A single AP has few distinct RSS values. Compute each likelihood once.
        values, groups = np.unique(query_rss[:, ap], return_inverse=True)
        for name, state in states.items():
            if name == 'GP_S':
                field = dict(qm=grid.new_full((len(grid),), base['mean']),
                             qv=grid.new_full((len(grid),), base['amplitude']**2),
                             cross=rbf(grid, sx, base['amplitude'], base['length']))
            else:
                field = fuse(moment, identity, state['weights'], local=False,
                             transfer_variance=base['drift']**2, intercept_std=state['sigma'])
            posterior = dict(mean=state['pm'][col], cov=state['pc'][col])
            pred = predict_from_posterior(field['qm'], field['qv'], field['cross'],
                                          state['sm'][col], state['sk'][col], posterior)
            if not torch.isfinite(pred['variance']).all() or (pred['variance'] < 0).any():
                raise FloatingPointError(f'Invalid {name} predicted variance at AP {ap}')
            table = observation_log_prob(tensor(values)[:, None], pred['mean'], pred['variance'], obs)
            scores[name] += table[groups]
    if any(not torch.isfinite(v).all() for v in scores.values()):
        raise FloatingPointError('Nonfinite full scan score')
    return {name: (v.softmax(1) @ grid).numpy() for name, v in scores.items()}


def run(data, parameters, out, building, budgets, seed):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    target = TARGETS[building]
    source = 1 if building == 0 else 0
    params = json.loads(Path(parameters).read_text())[str(source)]
    base, obs = params['base'], params['obs']
    train = read_floors(data, 'trainingData.csv')
    floors = history_floors(train, building, target)
    target_rows = train[building, target]
    order, _ = split_rows(target_rows, building, target, seed)
    hxy = np.concatenate([f['xy'] for f in floors])
    origin = hxy.mean(0)
    hxy = hxy - origin
    lo, hi = np.floor(hxy.min(0)/2)*2, np.ceil(hxy.max(0)/2)*2
    axes = [np.arange(lo[d], hi[d]+1, 2.) for d in range(2)]
    grid = tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1, 2))
    banks = fit_histories(floors, origin, base, obs)
    hseen = np.logical_or.reduce([(f['rss'] != 100).any(0) for f in floors])
    query = read_floors(data, 'validationData.csv')[building, target]
    result = dict(building=building, floor=target, source_building=source, seed=seed,
                  query_rows=query['ids'].tolist(), grid_size=len(grid), events=[])
    for k in budgets:
        indices = order[:k]
        rss = target_rows['rss'][indices]
        aps = np.flatnonzero(hseen | (rss != 100).any(0))
        support = group_scans(rss[:, aps], target_rows['xy'][indices] - origin)
        print(f'B{building} K{k}: fitting {len(aps)} APs', flush=True)
        states, failures = support_states(banks, support, aps, base, obs)
        predictions = score(banks, support, aps, grid, query['rss'], states, base, obs)
        methods = {}
        for name, pred in predictions.items():
            xy = pred + origin
            distance = np.linalg.norm(xy - query['xy'], axis=1)
            state = states[name]
            methods[name] = dict(mde=float(position_weights(query['xy']) @ distance),
                                 weights=state['weights'], sigma=state['sigma'],
                                 predictions=xy.tolist())
        result['events'].append(dict(k=k, support_rows=target_rows['ids'][indices].tolist(),
                                     aps=len(aps), methods=methods, failures=failures))
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, allow_nan=False)+'\n')
        tmp.replace(path)
        print(f'B{building} K{k}: { {n:round(v["mde"],3) for n,v in methods.items()} }, failures={list(failures)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data')
    parser.add_argument('parameters')
    parser.add_argument('out')
    parser.add_argument('--building', type=int, choices=TARGETS, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', choices=(0, *BUDGETS), default=(0, *BUDGETS))
    parser.add_argument('--seed', type=int, choices=SEEDS, default=0)
    args = parser.parse_args()
    run(args.data, args.parameters, args.out, args.building, args.budgets, args.seed)
