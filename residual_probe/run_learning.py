"""Frozen leave-building-out linear probes; held-out scores never select models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from residual_probe.learning import FEATURES, fit_linear, evaluate, load_case, prediction, observation_loss
from residual_probe.prepare import ROOT, data_module, InterceptBankPrior, history_features
from residual_probe.adaptive_proposal import AdaptivePrediction, check as check_refresh

SOURCE = ROOT / 'results/source_residual_probe_v1_20261026'
OUT = SOURCE / 'learning_v2'
KINDS = ('floor', 'content', 'paired')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    manifest = read(SOURCE / 'prepared/manifest.json')
    assert manifest['complete']
    events = manifest['events']
    assert len(events) == 24
    return events, read(SOURCE / 'prepared/parameters.json')['obs']


def check_paired():
    """New paired cache path against original prior, using artificial RSS only."""
    from learnable_profile.core import field_from_prior
    from residual_probe.paired_features import features
    pilot = read(SOURCE / 'prepared/pilot.json')
    name, ap = pilot['episode'], pilot['AP']
    path = SOURCE / 'prepared' / name / f'ap{ap}.npz'
    case = load_case(path, SOURCE / 'paired_features')
    case['rss'] = torch.where(torch.arange(len(case['rss'])) % 2 == 0, -70., 100.).double()
    raw_cases, meta = data_module().load_probe_episode(name)
    raw = raw_cases[ap]
    prior = InterceptBankPrior(raw['histories'], meta['adaptation']['weights'],
                               meta['base'], pilot['intercept_std'])
    theta = torch.linspace(-.02, .03, 10, requires_grad=True)
    def correction(x):
        phi = torch.cat((history_features(raw, x, meta['base']), features(raw, x, meta['base'])), -1)
        return 20 * (phi @ theta)
    generic = field_from_prior(prior, meta['obs'], raw['sx'], raw['stats'], raw['qx'], correction,
                               {k: case[k] for k in ('samples', 'log_proposal')})
    predicted, ess = prediction(theta, case, meta['obs'], FEATURES['paired'])
    errors = {}
    for key in ('mean', 'variance'):
        torch.testing.assert_close(predicted[key], generic[key], atol=1e-7, rtol=1e-8)
        errors[key] = float((predicted[key] - generic[key]).detach().abs().max())
    loss, _ = observation_loss(theta, case, meta['obs'], FEATURES['paired'])
    gradient, = torch.autograd.grad(loss, theta)
    finite = []
    for j in range(10):
        shift = torch.zeros(10); shift[j] = 1e-5
        plus, _ = observation_loss(theta.detach() + shift, case, meta['obs'], FEATURES['paired'])
        minus, _ = observation_loss(theta.detach() - shift, case, meta['obs'], FEATURES['paired'])
        finite.append(float((plus - minus) / 2e-5))
    torch.testing.assert_close(gradient, torch.tensor(finite), atol=2e-6, rtol=2e-5)
    return dict(event=name, AP=ap, prediction_errors=errors, ESS=float(ess.detach()),
                gradient_max_error=float((gradient - torch.tensor(finite)).abs().max()),
                actual_query_RSS_used=False, training_updates=0)


def freeze():
    events, _ = inputs()
    check = check_paired()
    OUT.mkdir(exist_ok=False)
    write(OUT / 'paired_check.json', check)
    refresh_check = check_refresh(SOURCE, OUT)
    source_files = [Path(__file__), ROOT / 'residual_probe/learning.py', ROOT / 'residual_probe/prepare.py',
                    ROOT / 'residual_probe/paired_features.py', ROOT / 'support_integral.py',
                    ROOT / 'inference.py', ROOT / 'learnable_profile/core.py',
                    ROOT / 'residual_probe/adaptive_proposal.py', ROOT / 'reference_pmc.py']
    records = [SOURCE / folder / name for folder, names in {
        'prepared': ['manifest.json', 'parameters.json', 'freeze.json', 'summary.json'],
        'paired_features': ['freeze.json', 'summary.json'],
    }.items() for name in names]
    buildings = sorted({e['building'] for e in events})
    write(OUT / 'freeze.json', dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        code_sha256={str(p): digest(p) for p in source_files},
        input_metadata_sha256={str(p): digest(p) for p in records},
        heldout_buildings=buildings, kinds=KINDS, steps=512, seed=261026,
        optimizer='Adam', learning_rate=.01, ridge=.001, batch_APs=4,
        parameters='g=20*Phi*theta; zero initialization; fixed g-free w and S-chosen intercept scale',
        objective='Post-S per-AP observation NLL; building then event then AP then Q-position balanced',
        hypothesis='Paired H content can predict post-S residuals in a held-out building beyond floor-only; localization must be scored separately',
        decision_rule='Report every fold and arm; do not select checkpoint, hyperparameters, or proposal from held-out scores. No claimed localization gain from NLL.',
        reverse_audit='Training receives only other-building events. H/S sets AP/grid; w and intercept scale use S only. g modifies S and Q means; importance weights and posterior moments are recomputed at every theta. Held-out Q only scores fixed model. Full-scan localization is separate.',
        counterexamples='g=0 gives frozen ordinary base; absent-H features give g=0; paired feature preserves source permutation and distinguishes equal averages with different RSS/dz pairs. Constant g may be absorbed by S. Wrong-history and source-shift can cause negative transfer. No variance adaptation or arbitrary-building registration.',
        numerical_boundary='ESS <1000 triggers exactly one current-theta H/S-only M-PMC proposal refresh; 6x4096 adaptation,32768 final,defensive .1 unchanged. Reuse refreshed proposal per AP/model; fail if still below 1000. Original prepared cache unchanged. Samples/densities detached; gradients include original posterior importance weights.',
        chronological_revision='v1 content stopped at step10 on ESS969.74 before held-out scoring. Preserve all v1 records. Change numerical proposal only, retain model/optimizer/loss/budgets; v1 successful Pisa floor is reused unchanged.',
        refresh_check=refresh_check,
        scope='4 development buildings, 3 datasets, K5/R1/Q12. B0-trained base/obs remain fixed: g-only leave-building-out, not pipeline-blind confirmation.',
        observation_rows='12 Q positions and all eligible APs per held-out event; J/seed repeats are correlated, not independent buildings',
        paired_check=check))


def verify():
    record = read(OUT / 'freeze.json')
    for key in ('code_sha256', 'input_metadata_sha256'):
        for path, expected in record[key].items():
            if digest(Path(path)) != expected:
                raise ValueError(f'Frozen file changed: {path}')
    return record


def run(heldout, kind):
    frozen = verify()
    assert heldout in frozen['heldout_buildings'] and kind in frozen['kinds']
    events, obs = inputs()
    train = [e for e in events if e['building'] != heldout]
    test = [e for e in events if e['building'] == heldout]
    assert train and test and not ({e['id'] for e in train} & {e['id'] for e in test})
    start = time.perf_counter()
    output = OUT / heldout / kind
    train_predictor = AdaptivePrediction(f'probe-v2/{heldout}/{kind}/train', output/'train_proposals.jsonl')
    model = fit_linear(train, obs, kind, output, paired_root=SOURCE / 'paired_features',
                       predictor=train_predictor, numerical_policy='S_only_ESS_refresh')
    rows = evaluate(model, test, obs, paired_root=SOURCE / 'paired_features',
                    predictor=AdaptivePrediction(f'probe-v2/{heldout}/{kind}/eval', output/'eval_proposals.jsonl'))
    result = dict(heldout=heldout, kind=kind, events=rows, seconds=time.perf_counter()-start,
                  observation_nll_per_AP=float(np.mean([e['observation_nll_per_AP'] for e in rows])),
                  min_ESS=min(e['min_ESS'] for e in rows), model_sha256=digest(output / 'model.json'))
    write(output / 'evaluation.json', result)
    print(json.dumps(result), flush=True)


def zero():
    verify()
    events, obs = inputs()
    model = dict(theta=[0.] * 8, feature_indices=list(range(8)), kind='content')
    rows = evaluate(model, events, obs)
    write(OUT / 'zero_observation.json', dict(events=rows))
    print(json.dumps(dict(stage='zero_complete', events=len(rows))), flush=True)


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'run', 'zero'])
    parser.add_argument('--heldout')
    parser.add_argument('--kind', choices=KINDS)
    args = parser.parse_args()
    if args.stage == 'run':
        run(args.heldout, args.kind)
    else:
        globals()[args.stage]()
