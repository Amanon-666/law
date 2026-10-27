"""Post-hoc numerical check of the observed low-ESS AP, not a model change.

Frozen predictions remain untouched. A defensive importance proposal includes
the original whitened prior, so its unnormalized importance ratio is bounded.
Only this AP's posterior moments are replaced when checking location sensitivity.
"""
import json
import math
import socket

import numpy as np
import torch
from pypmc.density.gauss import Gauss
from pypmc.density.mixture import MixtureDensity
from scipy.special import logsumexp

from data import group_scans
from history_bank import BankPrior
from inference import observation_log_prob, predict_from_posterior
from reference_pmc import setup, sample_reference
from run import ap_stats, tensor
from run_multi_history import write
from uji_history_bank_study import OUT, SOURCES, cached_history, load_raw


def main():
    if socket.gethostname() != 'ray-web': raise RuntimeError('Server only')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    name, ap = 'B1F1_s1', 371
    directory = OUT / name
    params = json.loads((OUT / 'parameters.json').read_text()); base, obs = params['base'], params['obs']
    meta = next(e for e in json.loads((OUT / 'manifest.json').read_text())['events'] if e['id'] == name)
    origin = np.array(json.loads((OUT / 'histories/completed.json').read_text())['origin'])
    floor = load_raw('trainingData.csv', {meta['floor']})[meta['floor']]
    official = load_raw('validationData.csv', {meta['floor']})[meta['floor']]
    lookup = {int(r): i for i, r in enumerate(floor['row_ids'])}
    qlookup = {int(r): i for i, r in enumerate(official['row_ids'])}
    sr = np.array([lookup[r] for r in meta['support_rows']])
    qo = np.array([qlookup[r] for r in meta['official_rows']])
    support = group_scans(floor['rss'][sr], floor['xy'][sr] - origin)
    query = group_scans(official['rss'][qo], official['xy'][qo] - origin)
    with np.load(directory / 'predictions.npz') as old:
        grid = tensor(old['grid'] - origin)
    prior = BankPrior([cached_history(OUT, n, ap, base) for n in SOURCES], [.5, .5], base)
    sx = tensor(support['positions']); sm, sf = prior.features(sx); sk = prior.covariance(sx, sx, sf, sf)
    target, initial, center, root, _ = setup(sm, sk, ap_stats(support, ap), obs)
    _, _, info = sample_reference(target, initial, center, root, 261027)
    proposal = MixtureDensity([Gauss(np.array(m), np.array(c)) for m, c in
        zip(info['mixture_means'], info['mixture_covariances'])] + [Gauss(np.zeros(len(sx)), np.eye(len(sx)))],
        np.r_[.9 * np.array(info['mixture_weights']), .1])
    proposal.normalize()
    gm, gf = prior.features(grid); gv = prior.diagonal(grid, gf); cross = prior.covariance(grid, sx, gf, sf)
    device = torch.device('cuda:0')
    scores = torch.zeros((len(query['rss']), len(grid)), device=device)
    with torch.no_grad():
        for column in meta['ap_columns']:
            if column == ap: continue
            with np.load(directory / f'ap{column}.npz') as a:
                m, v = tensor(a['equal_bank_mean']).to(device), tensor(a['equal_bank_variance']).to(device)
            scores += observation_log_prob(tensor(query['rss'][:, column]).to(device)[:, None], m[None], v[None], obs)
        rows = []
        for seed in (261126, 261127):
            u = proposal.propose(131072, rng=np.random.RandomState(seed))
            logw = target(u) - proposal.multi_evaluate(u)
            w = torch.tensor(np.exp(logw - logsumexp(logw)))
            samples = tensor(center + u @ root.T); pm = w @ samples; c = samples - pm
            pred = predict_from_posterior(gm, gv, cross, sm, sk, dict(mean=pm, cov=(c.T * w) @ c))
            new_score = scores + observation_log_prob(tensor(query['rss'][:, ap]).to(device)[:, None],
                pred['mean'].to(device)[None], pred['variance'].to(device)[None], obs)
            location = (torch.softmax(new_score, 1) @ grid.to(device)).cpu().numpy()
            error = np.linalg.norm(location - query['xy'], axis=1)
            sw = 1 / (len(query['positions']) * query['stats']['n'][query['group']])
            rows.append(dict(seed=seed, ess=float(1 / w.square().sum()),
                log_detection_evidence=float(logsumexp(logw) - math.log(len(logw))),
                official_position_mde=float(sw @ error), support_mean=pm.tolist(),
                support_variance=(w[:, None] * c.square()).sum(0).tolist()))
    original = json.loads((directory / 'completed.json').read_text())
    original_mde = next(r['mde'] for r in original['results'] if r['method']=='equal_bank' and
                        r['split']=='official' and r['weighting']=='position')
    result = dict(post_hoc=True, event=name, ap=ap, method='equal_bank', original_mde=original_mde,
        proposal='90% fitted Gaussian mixture + 10% original whitened prior', draws=131072,
        frozen_outputs_modified=False, repeats=rows)
    write(OUT / 'sampling_sensitivity.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
