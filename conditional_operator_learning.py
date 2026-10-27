"""Few-S, variable-history episodes for the existing LAW operator.

Source weights see all eligible APs in S, never Q. They are fitted with the
mature g-free prior and remain fixed while g/s are trained. This is an explicit
two-stage model, not an assertion that joint weight adaptation is inferior.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from data import group_scans
from history_bank import BankPrior
from inference import observation_log_prob, predict_from_posterior
from multi_source_adaptation import fit_source_weights
from operator_prior import OperatorBankPrior, OperatorNet
from reference_pmc import setup, sample_reference, log_proposal
from run import ap_stats, tensor
from run_multi_history import digest, write
from run_operator_prior import load_history, TRAIN_FLOORS, ZeroNet
from source_weight_mixture import field_from_prior
from support_integral import ImportanceSupport
from transfer import Prior, Transfer, rbf
from uji_probability_study import load_raw, DATA

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/uji_operator_prior_v1_20261026'
BANK = SOURCE / 'train_histories'
OUT = ROOT / 'results/uji_conditional_operator_v1_20261026'
PARAMETERS = ROOT / 'results/uji_history_bank_v1_20261026/parameters.json'


def parameters():
    d = json.loads(PARAMETERS.read_text())
    return d['base'], d['obs']


@lru_cache(maxsize=1)
def raw_floors():
    return load_raw('trainingData.csv', {f'B{b}F{f}' for b, floors in TRAIN_FLOORS.items() for f in floors})


@lru_cache(maxsize=None)
def history(name, ap):
    return load_history(BANK / name / f'ap{ap}.npz', parameters()[0])


def episode(meta):
    raw = raw_floors(); f = raw[meta['target']]
    lookup = {int(r): i for i, r in enumerate(f['row_ids'])}
    sr = [lookup[r] for r in meta['support_rows']]
    qr = [lookup[r] for r in meta['query_rows']]
    with np.load(BANK / meta['target'] / 'target.npz') as d:
        origin = d['origin']
    s = group_scans(f['rss'][sr], f['xy'][sr] - origin)
    q = group_scans(f['rss'][qr], f['xy'][qr] - origin)
    assert len(s['positions']) == 5 and len(q['positions']) == 12
    assert not (s['positions'][:, None] == q['positions'][None]).all(-1).any()
    seen = (s['rss'] != 100).any(0)
    for name in meta['histories']:
        seen |= (raw[name]['rss'] != 100).any(0)
    aps = np.flatnonzero(seen).tolist()
    cases = {a: dict(histories=[history(n, a) for n in meta['histories']],
        dz=[int(n.split('F')[1]) - int(meta['target'].split('F')[1]) for n in meta['histories']],
        sx=tensor(s['positions']), qx=tensor(q['positions']), stats=ap_stats(s, a),
        rss=tensor(q['rss'][:, a]), group=torch.tensor(q['group']),
        query_weights=tensor(1 / (len(q['positions']) * q['stats']['n'][q['group']]))) for a in aps}
    return cases


def fit_episode(cases, base, obs):
    means, covs, residuals = [], [], []
    for c in cases.values():
        sx = c['sx']; mm, cc = [], []
        for h in c['histories']:
            p = Prior(Transfer('identity'), h, dict(base, drift=0.))
            m, f = p.features(sx)
            mm.append(m); cc.append(p.covariance(sx, sx, f, f))
        if any(h is not None for h in c['histories']):
            residual = rbf(sx, sx, base['drift'], base['length'])
        else:
            cc = [torch.zeros_like(v) for v in cc]
            residual = rbf(sx, sx, base['amplitude'], base['length'])
        means.append(torch.stack(mm)); covs.append(torch.stack(cc)); residuals.append(residual)
    stats = {k: torch.stack([c['stats'][k] for c in cases.values()]) for k in ('n', 'k', 's1', 's2')}
    return fit_source_weights(torch.stack(means), torch.stack(covs), torch.stack(residuals), stats, obs)


def prior_for(net, case, weights, base):
    if not any(h is not None for h in case['histories']):
        return BankPrior([], [], base)
    return OperatorBankPrior(case['histories'], weights, base, net, case['dz'])


@torch.no_grad()
def proposal_for(net, case, weights, base, obs, seed):
    # Separate prior instance: its operator cache must not detach the loss graph.
    p = prior_for(net, case, weights, base)
    sm, sf = p.features(case['sx']); sk = p.covariance(case['sx'], case['sx'], sf, sf)
    target, proposal, center, root, _ = setup(sm, sk, case['stats'], obs)
    samples, _, info = sample_reference(target, proposal, center, root, seed, defensive_fraction=.1)
    if info['effective_samples'] < 1000:
        raise RuntimeError(f'Insufficient posterior samples: {info["effective_samples"]}')
    return samples, log_proposal(samples, center, root, info), info


def prediction_for(net, case, weights, base, obs, samples=None, logq=None):
    p = prior_for(net, case, weights, base)
    qm, qf = p.features(case['qx']); qv = p.diagonal(case['qx'], qf)
    if samples is None or not len(case['sx']):
        return dict(mean=qm, variance=qv), None
    sm, sf = p.features(case['sx']); sk = p.covariance(case['sx'], case['sx'], sf, sf)
    posterior = ImportanceSupport(sm, sk, case['stats'], obs, samples, logq)
    w = posterior.log_weights.exp(); mean = w @ samples; centered = samples - mean
    pm = dict(mean=mean, cov=(centered.T * w) @ centered)
    cross = p.covariance(case['qx'], case['sx'], qf, sf)
    return predict_from_posterior(qm, qv, cross, sm, sk, pm), posterior


def loss_for(net, case, weights, base, obs, samples=None, logq=None):
    prediction, posterior = prediction_for(net, case, weights, base, obs, samples, logq)
    g = case['group']
    lp = observation_log_prob(case['rss'], prediction['mean'][g], prediction['variance'][g], obs)
    return -(case['query_weights'] * lp).sum(), posterior


def prepare():
    OUT.mkdir(exist_ok=False)
    files = [Path(__file__), ROOT / 'multi_source_adaptation.py', ROOT / 'operator_prior.py',
        ROOT / 'history_bank.py', ROOT / 'support_integral.py', ROOT / 'reference_pmc.py',
        ROOT / 'inference.py', ROOT / 'transfer.py', ROOT / 'source_weight_mixture.py',
        ROOT / 'run_operator_prior.py', ROOT / 'data.py', ROOT / 'uji_probability_study.py']
    inputs = [PARAMETERS, SOURCE / 'operator.pt', DATA / 'trainingData.csv', *sorted(BANK.glob('B*/*.npz'))]
    write(OUT / 'freeze.json', dict(created_unix=time.time(), scope=__doc__,
        source_floors=TRAIN_FLOORS, K=5, R=3, Q_positions=12, Q_scans_per_position=3,
        support_policy='Full H-or-S-visible AP union; g-free shared weights estimated on S only; Q absent from weight/AP selection',
        source_sets='Two seeds per target floor; random order of other floors, prefixes J=1..all; common S/Q within each seed',
        objective='Position/AP-balanced individual scan NLL, using deployed Gaussian posterior moments; not a localization metric',
        comparison='Same pretrained checkpoint and episodes; prior Q score versus S-conditioned Q score; training budget not yet frozen',
        code_sha256={str(p): digest(p) for p in files}, inputs={str(p): digest(p) for p in inputs}))
    (OUT / 'code').mkdir()
    for p in files:
        shutil.copy2(p, OUT / 'code' / p.name)
    events = []; raw = raw_floors()
    for b, floors in TRAIN_FLOORS.items():
        for t in floors:
            name = f'B{b}F{t}'; f = raw[name]
            for seed in range(2):
                rng = np.random.default_rng(261028 + b * 1000 + t * 10 + seed)
                positions = rng.permutation(np.flatnonzero(f['stats']['n'] >= 3))[:17]
                assert len(positions) == 17
                rows = [rng.choice(np.flatnonzero(f['group'] == g), 3, replace=False) for g in positions]
                sr = f['row_ids'][np.concatenate(rows[:5])].tolist()
                qr = f['row_ids'][np.concatenate(rows[5:])].tolist()
                sources = rng.permutation([j for j in floors if j != t]).tolist()
                for j in range(1, len(sources) + 1):
                    events.append(dict(id=f'{name}_s{seed}_J{j}', target=name, seed=seed,
                        histories=[f'B{b}F{v}' for v in sources[:j]], support_rows=sr, query_rows=qr))
    assert len(events) == 64
    write(OUT / 'manifest.json', dict(events=events))
    adapt()


def adapt():
    base, obs = parameters()
    freeze = json.loads((OUT / 'freeze.json').read_text())
    for group in ['code_sha256', 'inputs']:
        for path, expected in freeze[group].items():
            assert digest(Path(path)) == expected, path
    for meta in json.loads((OUT / 'manifest.json').read_text())['events']:
        dest = OUT / meta['id']; dest.mkdir(exist_ok=True)
        if (dest / 'adaptation.json').exists():
            continue
        cases = episode(meta); started = time.perf_counter()
        weights, trials = fit_episode(cases, base, obs)
        write(dest / 'adaptation.json', dict(weights=weights, trials=trials, APs=list(cases),
            seconds=time.perf_counter()-started, query_used_for_adaptation=False))
        print(json.dumps(dict(event=meta['id'], APs=len(cases), weights=weights)), flush=True)
    write(OUT / 'prepared.json', dict(completed=True, episodes=64, model_updates=0,
        manifest_sha256=digest(OUT / 'manifest.json')))


def audit():
    assert (OUT / 'prepared.json').exists()
    base, obs = parameters(); net = OperatorNet().double()
    net.load_state_dict(torch.load(SOURCE / 'operator.pt', weights_only=True)); net.eval()
    meta = json.loads((OUT / 'manifest.json').read_text())['events']; rows = []
    names = ['B0F0_s0_J1', 'B0F0_s0_J2', 'B0F0_s0_J3', 'B2F0_s0_J4']
    for index, name in enumerate(names):
        event = next(e for e in meta if e['id'] == name); cases = episode(event)
        adaptation = json.loads((OUT / name / 'adaptation.json').read_text())
        assert list(cases) == adaptation['APs']
        weights = adaptation['weights']
        candidates = [a for a, c in cases.items() if any(h is not None for h in c['histories'])]
        # Pick informative audit AP from S alone; Q never selects the case.
        ap = max(candidates, key=lambda a: float(cases[a]['stats']['k'].sum()))
        case = cases[ap]; seed = 261028 + index
        samples, logq, info = proposal_for(net, case, weights, base, obs, seed)
        prediction, posterior = prediction_for(net, case, weights, base, obs, samples, logq)
        reference = field_from_prior(prior_for(net, case, weights, base), obs, case['sx'], case['stats'], case['qx'], seed, defensive_fraction=.1)
        errors = {k: float((prediction[k].detach()-reference[k]).abs().max()) for k in ['mean', 'variance']}
        for key in errors:
            torch.testing.assert_close(prediction[key], reference[key], atol=1e-4, rtol=1e-7)
        permuted = dict(case, histories=case['histories'][::-1], dz=case['dz'][::-1])
        reverse, _ = prediction_for(net, permuted, weights[::-1], base, obs, samples, logq)
        for key in errors:
            torch.testing.assert_close(prediction[key], reverse[key], atol=1e-7, rtol=1e-9)
        empty = dict(case, sx=case['sx'][:0])
        prior_prediction, _ = prediction_for(net, case, weights, base, obs)
        empty_prediction, _ = prediction_for(net, empty, weights, base, obs, samples, logq)
        for key in errors:
            torch.testing.assert_close(prior_prediction[key], empty_prediction[key], rtol=0, atol=0)
        p = prior_for(ZeroNet(), case, weights, base); old = BankPrior(case['histories'], weights, base)
        pm, pf = p.features(case['qx']); om, of = old.features(case['qx'])
        torch.testing.assert_close(pm, om, atol=0, rtol=0)
        torch.testing.assert_close(p.covariance(case['qx'],case['qx'],pf,pf), old.covariance(case['qx'],case['qx'],of,of), atol=0, rtol=0)
        value, _ = loss_for(net, case, weights, base, obs, samples, logq)
        gradient, = torch.autograd.grad(value, net.rho[-1].bias)
        fd = []
        for column in range(2):
            original = net.rho[-1].bias[column].item(); epsilon = 1e-5
            with torch.no_grad():
                net.rho[-1].bias[column] = original+epsilon
                plus = loss_for(net,case,weights,base,obs,samples,logq)[0].item()
                net.rho[-1].bias[column] = original-epsilon
                minus = loss_for(net,case,weights,base,obs,samples,logq)[0].item()
                net.rho[-1].bias[column] = original
            fd.append((plus-minus)/(2*epsilon))
        np.testing.assert_allclose(gradient.detach().numpy(), fd, rtol=2e-3, atol=2e-5)
        assert torch.isfinite(gradient).all() and gradient.abs().max() > 1e-8
        row = dict(event=name, AP=ap, all_APs=len(cases), weights=weights, source_loss=value.item(),
            ess=info['effective_samples'], deployment_max_error=errors, gradient=gradient.tolist(), finite_difference=fd,
            source_permutation=True, zero_operator=True, empty_support=True)
        rows.append(row); print(json.dumps(row), flush=True)
    write(OUT / 'audit.json', dict(completed=True, scope='Four source-only conditional-gradient/readout checks; no training or localization', rows=rows))


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'adapt', 'audit'])
    args = parser.parse_args()
    {'prepare': prepare, 'adapt': adapt, 'audit': audit}[args.stage]()
