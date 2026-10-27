"""Refresh only the numerical S proposal when its current-model ESS is low.

H/S and theta determine refresh; Q coordinates/signals/labels are not inputs to
the sampler. Model, objective, sample count and the original posterior density
are unchanged. The refreshed samples/density remain detached auxiliaries.
"""
import hashlib
import json
from pathlib import Path

import torch

from reference_pmc import setup, sample_reference, log_proposal
from residual_probe.learning import prediction


class AdaptivePrediction:
    def __init__(self, namespace, log_path):
        self.namespace, self.log_path = namespace, Path(log_path)
        self.proposals, self.counts = {}, {}

    def __call__(self, theta, case, obs, indices):
        key = case['path']
        current = dict(case, **self.proposals.get(key, {}))
        try:
            return prediction(theta, current, obs, indices)
        except RuntimeError as error:
            if 'proposal has insufficient ESS:' not in str(error):
                raise
            previous_error = str(error)
        count = self.counts.get(key, 0) + 1
        seed = int.from_bytes(hashlib.sha256(f'{self.namespace}|{key}|{count}'.encode()).digest()[:4], 'little')
        with torch.no_grad():
            sm = case['sm'] + 20 * (case['phi_s'][:, indices] @ theta)
            stats = {k: case[k] for k in ('n', 'k', 's1', 's2')}
            target, initial, center, root, _ = setup(sm, case['sk'], stats, obs)
            samples, _, info = sample_reference(target, initial, center, root, seed, defensive_fraction=.1)
            proposal = dict(samples=samples.detach(), log_proposal=log_proposal(samples, center, root, info).detach())
        self.proposals[key], self.counts[key] = proposal, count
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a') as stream:
            stream.write(json.dumps(dict(case=key, refresh=count, seed=seed,
                trigger=previous_error, sampler_ESS=info['effective_samples'], theta=theta.detach().tolist()))+'\n')
        # Exactly one refresh at this evaluation. If still inadequate, fail.
        return prediction(theta, dict(case, **proposal), obs, indices)


def check(source, output):
    """The actual v1 failing case, artificial Q RSS, fixed refreshed proposal FD."""
    from residual_probe.learning import load_case, observation_loss
    source, output = Path(source), Path(output)
    old = source/'learning_v1/Pisa2019-b60/content/trace.jsonl'
    record = json.loads(old.read_text().splitlines()[-1])
    theta = torch.tensor(record['theta_after_update'], requires_grad=True)
    path = source/'prepared/UJI_B2_F2_s261026_J4/ap337.npz'
    case = load_case(path)
    case['rss'] = torch.where(torch.arange(len(case['rss'])) % 2 == 0, -70., 100.).double()
    obs = json.loads((source/'prepared/parameters.json').read_text())['obs']
    predictor = AdaptivePrediction('source-probe-v2/check', output/'refresh_check_trace.jsonl')
    _, ess = predictor(theta, case, obs, list(range(8)))
    assert predictor.counts[str(path)] == 1
    fixed = dict(case, **predictor.proposals[str(path)])
    loss, _ = observation_loss(theta, fixed, obs, list(range(8)))
    gradient, = torch.autograd.grad(loss, theta)
    finite = []
    for i in range(8):
        shift = torch.zeros(8); shift[i] = 1e-5
        plus, _ = observation_loss(theta.detach()+shift, fixed, obs, list(range(8)))
        minus, _ = observation_loss(theta.detach()-shift, fixed, obs, list(range(8)))
        finite.append(float((plus-minus)/2e-5))
    finite = torch.tensor(finite)
    torch.testing.assert_close(gradient, finite, atol=2e-6, rtol=2e-5)
    result = dict(case=str(path), original_step=record['step']+1, refreshed_ESS=float(ess.detach()),
                  gradient_max_error=float((gradient-finite).abs().max()), actual_Q_RSS_used=False,
                  sampler_reads='sm(theta,H), sk, n/k/s1/s2, obs only', model_updates=0)
    (output/'refresh_check.json').write_text(json.dumps(result, indent=2))
    return result
