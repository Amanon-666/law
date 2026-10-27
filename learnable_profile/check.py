"""One source AP and one two-component counterexample; no training/performance.

Before execution: zero g must replay the original fixed-proposal moments and
evidence; full gradients must match FD, while detaching logZ must lose a real
gradient term. All probes are H-only; probe RSS is diagnostic, not held-out Q.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'operator_prior_v1'))

import numpy as np
import torch

import conditional_operator_learning as C
from data import group_scans
from inference import observation_log_prob
from learnable_profile.core import field_from_profile, prepare_proposals, scan_scores
from learnable_profile.core import (MeanCorrectedPrior, prepare_prior_proposal,
                                    field_from_prior as corrected_field_from_prior)
import physical_prior_candidate.floor_profile as old
from reference_pmc import log_proposal
from run import ap_stats, tensor
from source_weight_mixture import field_from_prior
from transfer import History

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correction(theta, center):
    return lambda x: theta[0] + theta[1] * (x[:, 0] - center) / 30.


def reference_replay(profile, attenuation, obs, sx, stats, points, seed, proposals):
    """Call the UNCHANGED old backend, supplying exactly our frozen draws.

    Temporary process-local patches replace only sampling and record outputs;
    no shared source file or original probability calculation is edited.
    """
    components, used = [], []

    def fixed_draws(target, initial, center, root, seed, defensive_fraction):
        assert defensive_fraction == .1
        proposal = proposals[len(used)]
        torch.testing.assert_close(log_proposal(proposal['samples'], center, root,
                                                proposal['info']), proposal['log_proposal'])
        used.append(proposal)
        return proposal['samples'], proposal['reference_log_weights'], proposal['info']

    def record(*args, **kwargs):
        value = field_from_prior(*args, **kwargs)
        components.append(value)
        return value

    with patch('source_weight_mixture.sample_reference', fixed_draws), \
         patch.object(old, 'field_from_prior', record):
        result = old.field_from_profile(profile, attenuation, obs, sx, stats, points, seed)
    assert len(used) == len(proposals)
    result['component_log_evidence'] = tensor([v['log_evidence'] for v in components])
    return result


def compare_replay(profile, attenuation, obs, sx, stats, points, rss, seed):
    zero = lambda x: x.new_zeros(len(x))
    proposals = prepare_proposals(profile, attenuation, obs, sx, stats, zero, seed)
    actual = field_from_profile(profile, attenuation, obs, sx, stats, points, zero, proposals)
    reference = reference_replay(profile, attenuation, obs, sx, stats, points, seed, proposals)
    differences = {}
    for key in ['mean', 'variance', 'component_means', 'component_variances',
                'component_log_evidence', 'log_mass', 'log_evidence']:
        expected = torch.as_tensor(reference[key])
        torch.testing.assert_close(actual[key], expected, atol=1e-7, rtol=1e-9)
        differences[key] = float((actual[key] - expected).abs().max())
    left, right = scan_scores(rss, [actual], obs), old.scan_log_prob(rss, reference, obs)
    torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-9)
    differences['scan_log_prob'] = float((left - right).abs().max())
    return proposals, differences


def gradient_check(profile, attenuation, obs, sx, stats, points, rss, proposals):
    center = float(points[:, 0].mean())

    def evaluate(theta):
        return field_from_profile(profile, attenuation, obs, sx, stats, points,
                                  correction(theta, center), proposals)

    theta = tensor([.6, -.2]).requires_grad_()
    field = evaluate(theta)
    objective = -scan_scores(rss, [field], obs).mean()
    gradient, = torch.autograd.grad(objective, theta, retain_graph=True)
    wrong = dict(field, log_mass=torch.log_softmax(
        field['history_log_mass'] + field['component_log_evidence'].detach(), 0))
    detached, = torch.autograd.grad(-scan_scores(rss, [wrong], obs).mean(), theta, retain_graph=True)
    jacobian = torch.stack([torch.autograd.grad(z, theta, retain_graph=True)[0]
                            for z in field['component_log_evidence']])
    fd, evidence_fd = [], []
    with torch.no_grad():
        for column in range(2):
            delta = torch.zeros_like(theta); delta[column] = 1e-4
            plus, minus = evaluate(theta + delta), evaluate(theta - delta)
            fd.append(float((-scan_scores(rss, [plus], obs).mean()
                             + scan_scores(rss, [minus], obs).mean()) / (2e-4)))
            evidence_fd.append((plus['component_log_evidence']
                                - minus['component_log_evidence']) / (2e-4))
    fd = tensor(fd); evidence_fd = torch.stack(evidence_fd, 1)
    torch.testing.assert_close(gradient, fd, atol=1e-6, rtol=2e-4)
    torch.testing.assert_close(jacobian, evidence_fd, atol=1e-6, rtol=2e-4)
    assert not field['history_log_mass'].requires_grad
    assert all(not p['samples'].requires_grad and not p['log_proposal'].requires_grad for p in proposals)
    once = torch.log_softmax(field['history_log_mass'] + field['component_log_evidence'], 0)
    twice = torch.log_softmax(field['history_log_mass'] + 2 * field['component_log_evidence'], 0)
    torch.testing.assert_close(field['log_mass'], once, atol=0, rtol=0)
    return dict(parameters=theta.detach().tolist(), full_gradient=gradient.tolist(),
                finite_difference=fd.tolist(), gradient_max_error=float((gradient-fd).abs().max()),
                detached_logz_gradient=detached.tolist(),
                detached_gradient_gap=float((gradient-detached).norm()),
                component_logz_gradient=jacobian.tolist(),
                component_logz_fd_max_error=float((jacobian-evidence_fd).abs().max()),
                log_mass=field['log_mass'].detach().tolist(),
                twice_S_mass_gap=float((once.exp()-twice.exp()).detach().abs().max()),
                min_reweighted_ESS=min(float(e.detach()) for e in field['effective_samples']),
                fixed_proposals=True), field


def real_case():
    folder = ROOT / 'results/physical_prior_source_v1_20261026'
    meta = next(e for e in json.loads((folder / 'manifest.json').read_text())['events']
                if e['id'] == 'B0F1_s261026')
    adaptation_path = folder / meta['id'] / 'adaptation.json'
    fit = json.loads(adaptation_path.read_text())
    with np.load(folder / 'B0F1_geometry.npz') as geometry:
        origin, probes = geometry['origin'], tensor(geometry['probes'])
    raw = C.raw_floors()[meta['target']]
    lookup = {int(row): i for i, row in enumerate(raw['row_ids'])}
    rows = [lookup[row] for row in meta['support_rows']]
    support = group_scans(raw['rss'][rows], raw['xy'][rows] - origin)
    # AP chosen using S detections only; no Query row, location, or score read.
    ap = max(meta['APs'], key=lambda a: int((support['rss'][:, a] != 100).sum()))
    histories = [C.history(name, ap) for name in meta['histories']]
    assert all(h is not None for h in histories)
    base, obs = C.parameters()
    profile = old.FloorProfile(histories, fit['weights'], base, [0, 3], 1,
                               meta['candidate_floors'], probes)
    points = probes[torch.linspace(0, len(probes)-1, min(6, len(probes))).long()]
    sx, stats = tensor(support['positions']), ap_stats(support, ap)
    assert len(sx) == 5
    evidence = dict(event=meta['id'], AP=int(ap), support_rows=meta['support_rows'],
                    support_positions=sx.tolist(), source_names=meta['histories'],
                    weights=fit['weights'], attenuation=fit['attenuation']['attenuation'],
                    probes=points.tolist(), probe_origin='H-only shared coverage lattice',
                    probe_RSS=[-65., 100.], target_Q_used=False,
                    input_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                      [folder / 'manifest.json', adaptation_path, folder / 'B0F1_geometry.npz',
                       C.PARAMETERS, *[C.BANK / name / f'ap{ap}.npz' for name in meta['histories']]]})
    return profile, evidence['attenuation'], obs, sx, stats, points, tensor([[-65.], [100.]]), evidence


def toy_case():
    base = dict(mean=-90., amplitude=20., length=10., drift=4.)
    obs = dict(alpha=.9, beta=-85., detect_scale=3., noise_std=4.)
    hx = tensor([[-4., 0.], [0., 0.], [4., 0.]])
    histories = [History(hx, dict(mean=hx.new_full((3,), -70.), cov=.01*torch.eye(3)), base)
                 for _ in range(2)]
    profile = old.FloorProfile(histories, [.5, .5], base, [0, 2], 1, [1], hx)
    sx = hx[1:2]
    stats = dict(n=tensor([3.]), k=tensor([3.]), s1=tensor([-201.]), s2=tensor([13467.]))
    # Off-S positions are prescribed geometry, not selected by scores.
    points = tensor([[0., 0.], [15., 0.], [30., 0.]])
    return profile, 6., obs, sx, stats, points, tensor([[-63.], [100.]])


def boundary_check(problem, seed):
    profile, attenuation, obs, sx, stats, points, rss = problem
    zero = lambda x: x.new_zeros(len(x))
    proposals, replay = compare_replay(profile, 0., obs, sx, stats, points, rss, seed)
    collapsed = field_from_profile(profile, 0., obs, sx, stats, points, zero, proposals)
    assert collapsed['labels'] == ['identity']
    count = len(profile.labels)
    expanded = dict(collapsed,
        component_means=collapsed['component_means'].expand(count, -1),
        component_variances=collapsed['component_variances'].expand(count, -1),
        log_mass=profile.history_log_mass(0.))
    gap = float((scan_scores(rss, [collapsed], obs)-scan_scores(rss, [expanded], obs)).abs().max())
    assert gap < 1e-12
    # A nonzero g remains learnable on ordinary Bank; zero attenuation only
    # removes the floor structure, not the external correction or its gradient.
    bank_gradient, bank_g = gradient_check(profile, 0., obs, sx, stats, points, rss, proposals)
    assert max(abs(v) for v in bank_gradient['full_gradient']) > 1e-6
    bank_shift = float((bank_g['mean']-collapsed['mean']).detach().abs().max())
    assert bank_shift > 1e-4
    empty_stats = {key: value[:0] for key, value in stats.items()}
    theta = tensor([.6, -.2]).requires_grad_()
    g = correction(theta, float(points[:, 0].mean()))
    no_proposals = prepare_proposals(profile, attenuation, obs, sx[:0], empty_stats, g, seed)
    empty = field_from_profile(profile, attenuation, obs, sx[:0], empty_stats, points, g, no_proposals)
    expected = torch.stack([profile.component(attenuation, i).features(points)[0] + g(points)
                             for i in range(len(profile.labels))])
    torch.testing.assert_close(empty['component_means'], expected)
    torch.testing.assert_close(empty['log_mass'], profile.history_log_mass(attenuation))
    assert torch.equal(empty['component_log_evidence'], torch.zeros(len(profile.labels)))
    empty_gradient, = torch.autograd.grad(-scan_scores(rss, [empty], obs).mean(), theta)
    return dict(zero_attenuation_replay_max_error=max(replay.values()),
                merged_vs_expanded_probability_max_error=gap, empty_support_prior=True,
                ordinary_bank_nonzero_g_gradient=bank_gradient['full_gradient'],
                ordinary_bank_gradient_fd_max_error=bank_gradient['gradient_max_error'],
                ordinary_bank_mean_change=bank_shift,
                empty_support_gradient=empty_gradient.tolist())


def main():
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical checks run on ray-web only')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    output = HERE / 'check.json'
    if output.exists():
        raise FileExistsError(output)
    started = now()
    code = [HERE / 'core.py', Path(__file__), ROOT / 'physical_prior_candidate/floor_profile.py',
            *[ROOT / name for name in ['support_integral.py', 'reference_pmc.py', 'inference.py',
                                       'source_weight_mixture.py', 'history_bank.py', 'transfer.py']]]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in code}
    print(json.dumps(dict(started=started, code_sha256=hashes)), flush=True)
    real = real_case(); rows = []
    for name, problem, metadata in [('source_B0F1', real[:7], real[7]), ('two_components', toy_case(), {})]:
        profile, attenuation, obs, sx, stats, points, rss = problem
        proposals, replay = compare_replay(*problem, seed=261026)
        result, field = gradient_check(*problem, proposals)
        if name == 'two_components':
            assert result['detached_gradient_gap'] > 1e-3
            assert result['twice_S_mass_gap'] > 1e-3
            paired = torch.cat([rss, rss.flip(0)], 1)
            separate = scan_scores(paired, [field, field], obs)
            lp = [observation_log_prob(paired[:, a][None, :, None], field['component_means'][:, None],
                                       field['component_variances'][:, None], obs) for a in range(2)]
            wrong_shared = torch.logsumexp(field['log_mass'][:, None, None] + lp[0] + lp[1], 0)
            result['independent_vs_shared_floor_gap'] = float((separate-wrong_shared).detach().abs().max())
            assert result['independent_vs_shared_floor_gap'] > 1e-3
            try:
                field_from_profile(profile, attenuation, obs, sx, stats, points,
                                   lambda x: x.new_full((len(x),), 100.), proposals)
            except RuntimeError as error:
                assert 'Insufficient ESS' in str(error)
                result['stale_proposal_failure'] = str(error)
            else:
                raise AssertionError('Extreme correction did not expose insufficient ESS')
        row = dict(name=name, metadata=metadata, zero_g_replay_errors=replay, **result,
                   boundaries=boundary_check(problem, 261027))
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    assert {str(p.relative_to(ROOT)): digest(p) for p in code} == hashes
    result = dict(completed=True, started=started, finished=now(), code_sha256=hashes,
                  scope='Interface/gradient checks only; no training, target Q, full scan localization, T12 or T13',
                  threads=2, device='CPU', proposal_defensive_fraction=.1, rows=rows)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(completed=True, output=str(output))), flush=True)


def generic_main():
    """One real source event: generic Bank/intercept vs unchanged old backend."""
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical checks run on ray-web only')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    # Reuse the already implemented intercept kernel; no model implementation here.
    sys.path.insert(0, str(ROOT / 'physical_prior_candidate'))
    from physical_prior_candidate.source_trial import RandomInterceptPrior
    output = HERE / 'generic_check.json'
    if output.exists():
        raise FileExistsError(output)
    started = now()
    profile, _, obs, sx, stats, points, rss, metadata = real_case()
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in
              [HERE / 'core.py', Path(__file__), ROOT / 'physical_prior_candidate/source_trial.py',
               ROOT / 'support_integral.py', ROOT / 'reference_pmc.py', ROOT / 'inference.py',
               ROOT / 'source_weight_mixture.py']}
    rows = []
    for name, prior in [('bank', profile.bank), ('random_intercept', RandomInterceptPrior(
                         profile.histories, profile.weights, profile.base))]:
        theta = tensor([.6, -.2]).requires_grad_()
        center = float(points[:, 0].mean())
        g = correction(theta, center)
        proposal = prepare_prior_proposal(prior, obs, sx, stats, g, 261029)
        actual = corrected_field_from_prior(prior, obs, sx, stats, points, g, proposal)
        with patch('source_weight_mixture.sample_reference', return_value=(
                proposal['samples'], proposal['reference_log_weights'], proposal['info'])):
            expected = field_from_prior(MeanCorrectedPrior(prior, g), obs, sx, stats,
                                        points, 261029, defensive_fraction=.1)
        errors = {}
        for key in ['mean', 'variance', 'log_evidence']:
            delta = (actual[key]-torch.as_tensor(expected[key])).detach().abs().max()
            errors[key] = float(delta)
            torch.testing.assert_close(actual[key], torch.as_tensor(expected[key]), atol=1e-7, rtol=1e-9)
        score = observation_log_prob(rss, actual['mean'][None], actual['variance'][None], obs)
        old_score = observation_log_prob(rss, expected['mean'][None], expected['variance'][None], obs)
        torch.testing.assert_close(score, old_score, atol=1e-7, rtol=1e-9)
        errors['scan_log_prob'] = float((score-old_score).detach().abs().max())
        # Include logZ explicitly so this check verifies its gradient is exposed.
        loss = -score.mean() - .1*actual['log_evidence']
        gradient, = torch.autograd.grad(loss, theta)
        fd = []
        with torch.no_grad():
            for column in range(2):
                step = torch.zeros_like(theta); step[column] = 1e-4
                values = []
                for moved in [theta+step, theta-step]:
                    value = corrected_field_from_prior(prior, obs, sx, stats, points,
                                                      correction(moved, center), proposal)
                    values.append(-observation_log_prob(rss, value['mean'][None],
                                  value['variance'][None], obs).mean()-.1*value['log_evidence'])
                fd.append(float((values[0]-values[1])/(2e-4)))
        torch.testing.assert_close(gradient, tensor(fd), atol=1e-6, rtol=2e-4)
        assert gradient.abs().max() > 1e-6
        floor_gap = None
        if name == 'bank':
            delegated = field_from_profile(profile, 0., obs, sx, stats, points, g, [proposal])
            floor_gap = max(float((delegated[key]-actual[key]).detach().abs().max())
                            for key in ['mean', 'variance', 'log_evidence'])
            assert floor_gap < 1e-12
        rows.append(dict(prior=name, replay_errors=errors, gradient=gradient.tolist(),
                         finite_difference=fd, gradient_fd_max_error=float((gradient-tensor(fd)).abs().max()),
                         ESS=float(actual['effective_samples'].detach()), floor_wrapper_gap=floor_gap))
    assert all(digest(ROOT / path) == sha for path, sha in hashes.items())
    result = dict(completed=True, started=started, finished=now(), metadata=metadata,
                  scope='Generic API equivalence only; same source AP/S, no training or target performance',
                  code_sha256=hashes, rows=rows)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
