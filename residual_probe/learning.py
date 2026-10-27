"""Small linear mean probe on frozen multi-history, K=5 source episodes.

This reuses LAW's exact same fixed-proposal importance update and Gaussian
posterior-moment readout. It is not a new network or a location benchmark.
All APs already contributed to frozen S-only w/intercept-scale adaptation.
AP minibatches below are valid only for the additive observation objective;
they must not be reused as an estimator of the full-scan position softmax.
"""
import json
from pathlib import Path

import numpy as np
import torch

from inference import observation_log_prob, predict_from_posterior
from support_integral import ImportanceSupport


FEATURES = {'content': list(range(8)), 'floor': [2, 6, 7], 'paired': list(range(10))}
FLOAT_KEYS = ['sm', 'sk', 'qm', 'qv', 'cross', 'phi_s', 'phi_q',
              'samples', 'log_proposal', 'n', 'k', 's1', 's2', 'rss', 'query_weights']


def load_case(path, paired_root=None):
    with np.load(path, allow_pickle=False) as a:
        c = {k: torch.from_numpy(a[k]).to(torch.float64) for k in FLOAT_KEYS}
        c['group'] = torch.from_numpy(a['group']).long()
    if paired_root is not None:
        feature_path = Path(paired_root) / Path(path).parent.name / Path(path).name
        with np.load(feature_path, allow_pickle=False) as a:
            c['phi_s'] = torch.cat((c['phi_s'], torch.from_numpy(a['cross_s']).double()), dim=1)
            c['phi_q'] = torch.cat((c['phi_q'], torch.from_numpy(a['cross_q']).double()), dim=1)
    c['path'] = str(path)
    return c


def prediction(theta, case, obs, feature_indices):
    """Mean correction changes S posterior moments as well as the Q mean."""
    sm = case['sm'] + 20 * (case['phi_s'][:, feature_indices] @ theta)
    qm = case['qm'] + 20 * (case['phi_q'][:, feature_indices] @ theta)
    stats = {k: case[k] for k in ('n', 'k', 's1', 's2')}
    posterior = ImportanceSupport(sm, case['sk'], stats, obs,
                                  case['samples'], case['log_proposal'])
    weights = posterior.log_weights.exp()
    ess = weights.square().sum().reciprocal()
    if float(ess.detach()) < 1000:
        raise RuntimeError(f'Fixed source proposal has insufficient ESS: {float(ess.detach()):.6g}')
    mean = weights @ posterior.samples
    centered = posterior.samples - mean
    moments = dict(mean=mean, cov=(centered.T * weights) @ centered)
    value = predict_from_posterior(qm, case['qv'], case['cross'], sm, case['sk'], moments)
    return value, ess


def observation_loss(theta, case, obs, feature_indices, predictor=prediction):
    value, ess = predictor(theta, case, obs, feature_indices)
    group = case['group']
    logp = observation_log_prob(case['rss'], value['mean'][group], value['variance'][group], obs)
    return -(case['query_weights'] * logp).sum(), ess


def fit_linear(events, obs, kind, output, *, steps=512, seed=261026, paired_root=None,
               predictor=prediction, numerical_policy='fixed_proposal'):
    """Caller supplies TRAINING events only, each with building and case_paths.

    Sampling balances buildings, then their episode configurations, then APs.
    Hyperparameters are fixed before any held-out source score is computed.
    No early stopping or validation-based checkpoint choice. The caller fixes
    the numerical proposal policy independently of the learned model.
    """
    indices = FEATURES[kind]
    theta = torch.nn.Parameter(torch.zeros(len(indices), dtype=torch.float64))
    optimizer = torch.optim.Adam([theta], lr=.01)
    rng = np.random.default_rng(seed)
    buildings = sorted({e['building'] for e in events})
    by_building = {b: [e for e in events if e['building'] == b] for b in buildings}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'trace.jsonl').open('x') as trace:
        for step in range(steps):
            building = buildings[int(rng.integers(len(buildings)))]
            candidates = by_building[building]
            event = candidates[int(rng.integers(len(candidates)))]
            selected = rng.choice(len(event['case_paths']), size=4, replace=True)
            optimizer.zero_grad()
            losses, ess_values = [], []
            for i in selected:
                path = event['case_paths'][int(i)]
                try:
                    loss, ess = observation_loss(theta, load_case(path, paired_root if kind == 'paired' else None), obs, indices, predictor)
                except RuntimeError as error:
                    raise RuntimeError(f'{kind}, step {step}, case {path}: {error}') from error
                losses.append(loss)
                ess_values.append(float(ess.detach()))
            data_loss = torch.stack(losses).mean()
            objective = data_loss + .001 * theta.square().sum()
            objective.backward()
            if not torch.isfinite(objective) or not torch.isfinite(theta.grad).all():
                raise FloatingPointError(f'Nonfinite source loss/gradient at step {step}')
            record = dict(step=step, building=building, event=event['id'],
                sampled_AP_indices=selected.tolist(), observation_loss=float(data_loss.detach()),
                objective=float(objective.detach()), gradient_max=float(theta.grad.abs().max()),
                min_ESS=min(ess_values))
            optimizer.step()
            record['theta_after_update'] = theta.detach().tolist()
            trace.write(json.dumps(record) + '\n')
            if step % 32 == 0:
                trace.flush()
    result = dict(kind=kind, theta=theta.detach().tolist(), feature_indices=indices,
                  steps=steps, seed=seed, training_buildings=buildings,
                  optimizer='Adam', learning_rate=.01, ridge=.001, numerical_policy=numerical_policy,
                  objective='Building/episode/AP/position balanced post-S observation NLL',
                  target_access='No held-out building labels supplied to this function',
                  scope='Low-capacity mean learning only; no location loss or variance-head training')
    (output / 'model.json').write_text(json.dumps(result, indent=2))
    return result


@torch.no_grad()
def evaluate(model, events, obs, paired_root=None, predictor=prediction):
    """Full eligible AP evaluation of observation loss, not localization."""
    theta = torch.tensor(model['theta'], dtype=torch.float64)
    rows = []
    for event in events:
        losses, effective = [], []
        for path in event['case_paths']:
            case = load_case(path, paired_root if model['kind'] == 'paired' else None)
            value, ess = observation_loss(theta, case, obs, model['feature_indices'], predictor)
            losses.append(float(value))
            effective.append(float(ess))
        rows.append(dict(event=event['id'], building=event['building'], APs=len(losses),
            observation_nll_per_AP=float(np.mean(losses)), min_ESS=min(effective)))
    return rows


def check_prepared(prepared):
    """One real source case: cached matrices versus the generic prior path.

    Uses prescribed diagnostic RSS, never scores the actual held-out Q scans.
    This checks the new cache-to-learning boundary, not model performance.
    """
    import hashlib
    import time
    from learnable_profile.core import field_from_prior
    from residual_probe.prepare import InterceptBankPrior, history_features, data_module

    prepared = Path(prepared)
    pilot = json.loads((prepared / 'pilot.json').read_text())
    path = prepared / pilot['episode'] / f"ap{pilot['AP']}.npz"
    case = load_case(path)
    case['rss'] = torch.where(torch.arange(len(case['rss'])) % 2 == 0, -70., 100.).double()
    original, meta = data_module().load_probe_episode(pilot['episode'])
    raw = original[pilot['AP']]
    prior = InterceptBankPrior(raw['histories'], meta['adaptation']['weights'],
                               meta['base'], pilot['intercept_std'])
    proposal = {k: case[k] for k in ('samples', 'log_proposal')}
    rows = []
    for kind in ['content', 'floor']:
        ids = FEATURES[kind]
        theta = torch.linspace(-.02, .03, len(ids), dtype=torch.float64, requires_grad=True)
        def correction(x):
            return 20 * (history_features(raw, x, meta['base'])[:, ids] @ theta)
        generic = field_from_prior(prior, meta['obs'], raw['sx'], raw['stats'],
                                   raw['qx'], correction, proposal)
        value, ess = prediction(theta, case, meta['obs'], ids)
        errors = {k: float((value[k] - generic[k]).detach().abs().max()) for k in ('mean', 'variance')}
        for key in errors:
            torch.testing.assert_close(value[key], generic[key], atol=1e-7, rtol=1e-8)
        loss, _ = observation_loss(theta, case, meta['obs'], ids)
        gradient, = torch.autograd.grad(loss, theta)
        finite = []
        for j in range(len(theta)):
            shift = torch.zeros_like(theta); shift[j] = 1e-5
            plus, _ = observation_loss(theta.detach() + shift, case, meta['obs'], ids)
            minus, _ = observation_loss(theta.detach() - shift, case, meta['obs'], ids)
            finite.append(float(((plus - minus) / 2e-5).detach()))
        finite = torch.tensor(finite, dtype=torch.float64)
        torch.testing.assert_close(gradient, finite, atol=2e-6, rtol=2e-5)
        rows.append(dict(kind=kind, generic_prediction_max_errors=errors,
            gradient_max_error=float((gradient - finite).abs().max()),
            gradient=gradient.tolist(), ESS=float(ess.detach())))
    return dict(created_unix=time.time(), episode=pilot['episode'], AP=pilot['AP'],
        cases=rows, training_updates=0, real_Q_scored=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prepared_case_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    import socket
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Source numerical work runs on ray-web')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1] / 'results/source_residual_probe_v1_20261026'
    result = check_prepared(root / 'prepared')
    with (root / 'learning_check.json').open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))
