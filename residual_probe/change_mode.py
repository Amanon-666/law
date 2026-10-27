"""Zero-mean AP-specific random slopes along one normalized H-only direction.

f_a(x)=m_Bank,a(x)+c_a+delta_a(x)+u_a*q_a(x), u_a~N(0,tau^2).
u_a are independent across APs; tau is a shared S-only composite-likelihood
point choice, not an exact joint EB posterior. Existing w/sigma are fixed.

API for an evaluation driver (no data adapter or performance loop here):
  modes = build_modes(support_case, grid_arrays, heldout_paired_theta)
  grid_case = dict(support_case, **grid_arrays)
  changed = add_covariance(grid_case, modes[kind], tau)
  predictor = make_predictor(kind, event_id, tau, NEW_log_path)
  field, ess = zero_prediction(changed, obs, predictor)
select_tau(cases_by_AP, selected_modes_by_AP, obs) reads all AP S data only.

grid_arrays are the frozen H-union-S grid qm/qv/cross/phi_q, NOT prepared Q
fields. No Q RSS, Q coordinates or statistics enter normalization/selection.
The source theta only defines learned_mode's direction; prediction always
uses theta=0 and unchanged Bank means. Grid centering removes a globally
constant direction, but does not guarantee identifiability from five S.
Scale/sign invariance holds when rescaling does not cross the explicitly
fixed 1e-8 dB numerical-constant threshold. This is a mature random-slope /
low-rank GP mechanism, not a formula novelty claim.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from inference import observation_log_prob
from residual_probe.adaptive_proposal import AdaptivePrediction
from residual_probe.learning import prediction

TAU_GRID = (0., 5., 10., 15., 20.)
CONSTANT_RMS_DB = 1e-8
KINDS = ('history_mode', 'learned_mode')


@torch.no_grad()
def normalize_mode(raw_s, raw_g, has_history=True):
    """One center/RMS from the entire fixed grid, also used at S points."""
    center = raw_g.mean()
    rms = (raw_g - center).square().mean().sqrt()
    degenerate = not has_history or float(rms) <= CONSTANT_RMS_DB
    qs = torch.zeros_like(raw_s) if degenerate else (raw_s - center) / rms
    qg = torch.zeros_like(raw_g) if degenerate else (raw_g - center) / rms
    return dict(q_s=qs, q_g=qg, normalization=dict(center_dB=float(center), rms_dB=float(rms),
        degenerate=bool(degenerate), has_history=bool(has_history), grid_points=len(raw_g),
        constant_threshold_dB=CONSTANT_RMS_DB, measure='uniform over frozen H-union-S grid'))


@torch.no_grad()
def build_modes(support_case, grid_arrays, theta):
    """Return the two one-direction modes; theta is the fixed paired10 vector.

    Grid phi_q[:,5] is the H-present fraction; it also works with empty S.
    Fully absent H gives zero modes even if supplied values have roundoff.
    Only sm/phi_s and frozen GRID qm/phi_q are read; prepared Q is ignored.
    """
    active = bool((grid_arrays['phi_q'][:, 5] > 0).any())
    raw = dict(history_mode=(support_case['sm'], grid_arrays['qm']),
               learned_mode=(20 * (support_case['phi_s'] @ theta),
                             20 * (grid_arrays['phi_q'] @ theta)))
    return {kind: dict(kind=kind, **normalize_mode(*values, has_history=active))
            for kind, values in raw.items()}


def add_covariance(grid_case, mode, tau):
    """Add the complete rank-one block. Means, w/sigma and caches are untouched."""
    if tau == 0:
        return grid_case
    qs, qg = mode['q_s'], mode['q_g']
    variance = tau ** 2
    return dict(grid_case, sk=grid_case['sk'] + variance * qs[:, None] * qs[None],
                qv=grid_case['qv'] + variance * qg.square(),
                cross=grid_case['cross'] + variance * qg[:, None] * qs[None])


@torch.no_grad()
def select_tau(cases, modes, obs):
    """ALL eligible APs for one event/kind, each AP with its own latent u_a.

    cases/modes are mappings with the same AP keys. Only sm,diag(sk),n,k,s1
    and q_s enter this exact R1 observation composite likelihood. Correlations
    between S positions are ignored by this selector, not by final inference.
    A wholly S-invisible mode has a flat curve and deterministic tau=0.
    """
    first = next(iter(cases.values()))
    scores = first['sm'].new_zeros(len(TAU_GRID))
    for ap, case in cases.items():
        n, k = case['n'], case['k']
        if not bool((n == 1).all() and ((k == 0) | (k == 1)).all()):
            raise ValueError('Change-mode selection requires the frozen R1 single scans')
        rss = torch.where(k == 1, case['s1'], case['s1'].new_full(k.shape, 100.))
        variance = case['sk'].diag()
        qs = modes[ap]['q_s']
        for i, tau in enumerate(TAU_GRID):
            scores[i] += observation_log_prob(rss, case['sm'], variance + tau ** 2 * qs.square(), obs).sum()
    best = int(scores.argmax())
    return dict(tau=TAU_GRID[best], grid=list(TAU_GRID), composite_log_likelihood=scores.tolist(),
                APs=len(cases), boundary_hit=best in (0, len(TAU_GRID) - 1),
                scope='S-only exact R1 observation composite likelihood; not exact joint EB or calibrated posterior',
                means_unchanged=True, w_sigma_reselected=False, query_used=False)


def make_predictor(kind, event_id, tau, log_path):
    """Use a separate predictor per kind/event/tau; never write old proposals."""
    if kind not in KINDS:
        raise ValueError(kind)
    return AdaptivePrediction(f'change-mode-v1/{kind}/{event_id}/tau={tau:.17g}', log_path)


def zero_prediction(case, obs, predictor=prediction):
    """Source g is a covariance direction only: never pass source theta here."""
    width = case['phi_s'].shape[1]
    return predictor(case['sm'].new_zeros(width), case, obs, list(range(width)))


def _gaussian_condition(mean_s, mean_g, sk, gg, cross, y, noise2):
    a = torch.linalg.solve(sk + noise2 * torch.eye(len(sk)), cross.T).T
    return mean_g + a @ (y - mean_s), gg - a @ cross.T


def _synthetic_checks():
    # A nonconstant grid direction and a shared centering convention for S.
    rs = torch.tensor([1., -.5]); rg = torch.tensor([-1., 0., 1.])
    mode = normalize_mode(rs, rg)
    allq = torch.cat((mode['q_s'], mode['q_g']))
    kernel = allq[:, None] * allq[None]
    invariant_errors = []
    for scale, shift in [(1., 41.), (3., -8.), (-2., 13.)]:
        other = normalize_mode(scale * rs + shift, scale * rg + shift)
        q = torch.cat((other['q_s'], other['q_g']))
        error = float((q[:, None] * q[None] - kernel).abs().max())
        invariant_errors.append(error)
        torch.testing.assert_close(q[:, None] * q[None], kernel, atol=1e-12, rtol=1e-12)
    assert abs(float(mode['q_g'].mean())) < 1e-14
    torch.testing.assert_close(mode['q_g'].square().mean(), torch.tensor(1.))
    for item in [normalize_mode(torch.ones(2), torch.ones(3)),
                 normalize_mode(rs, rg, has_history=False),
                 normalize_mode(rs * 1e-10, rg * 1e-10)]:
        assert item['normalization']['degenerate'] and not bool(item['q_s'].any() or item['q_g'].any())

    # Explicit latent vector [base_f_S,base_f_G,u] versus marginal rank-one GP.
    # Grid points are far from S: only the old intercept couples base residuals.
    xs = torch.tensor([0., 1., 100., 200., 300.])
    base = torch.exp(-.5 * (xs[:, None] - xs[None]).square() / .6 ** 2) + .09
    sk, gg, cross = base[:2, :2], base[2:, 2:], base[2:, :2]
    tau, noise2, y = 2., .7, torch.tensor([2., -1.])
    case = dict(sm=torch.zeros(2), qm=torch.zeros(3), sk=sk, qv=gg.diag(), cross=cross)
    changed = add_covariance(case, mode, tau)
    newgg = gg + tau ** 2 * mode['q_g'][:, None] * mode['q_g'][None]
    mean, cov = _gaussian_condition(case['sm'], case['qm'], changed['sk'], newgg, changed['cross'], y, noise2)
    latent = torch.block_diag(base, torch.tensor([[tau ** 2]]))
    a = torch.cat((torch.eye(2), torch.zeros((2, 3)), mode['q_s'][:, None]), 1)
    b = torch.cat((torch.zeros((3, 2)), torch.eye(3), mode['q_g'][:, None]), 1)
    gain = latent @ a.T @ torch.linalg.inv(a @ latent @ a.T + noise2 * torch.eye(2))
    explicit_mean = b @ gain @ y
    explicit_cov = b @ (latent - gain @ a @ latent) @ b.T
    torch.testing.assert_close(mean, explicit_mean, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(cov, explicit_cov, atol=1e-12, rtol=1e-12)
    assert torch.linalg.eigvalsh(base + tau ** 2 * kernel).min() > 0
    assert add_covariance(case, mode, 0.) is case
    old_mean, old_cov = _gaussian_condition(case['sm'], case['qm'], sk, gg, cross, y, noise2)
    far_change = float((mean - old_mean).abs().max()); assert far_change > .1
    invisible = dict(mode, q_s=torch.zeros(2))
    unseen = add_covariance(case, invisible, tau)
    unseen_mean, unseen_cov = _gaussian_condition(case['sm'], case['qm'], unseen['sk'], newgg, unseen['cross'], y, noise2)
    torch.testing.assert_close(unseen_mean, old_mean, atol=0, rtol=0)
    torch.testing.assert_close(unseen_cov - old_cov, tau ** 2 * mode['q_g'][:, None] * mode['q_g'][None])
    # Exact R1 reconstruction; two APs, no S access to their mode amplitude.
    obs = dict(alpha=.9, beta=-85., detect_scale=4., noise_std=3.)
    support = dict(sm=torch.tensor([-70., -90.]), sk=torch.eye(2) * 4,
                   n=torch.ones(2), k=torch.tensor([1., 0.]), s1=torch.tensor([-69., 0.]))
    selection = select_tau({0: support, 1: support}, {0: invisible, 1: invisible}, obs)
    assert selection['tau'] == 0. and max(selection['composite_log_likelihood']) == min(selection['composite_log_likelihood'])
    direct = 2 * observation_log_prob(torch.tensor([-69., 100.]), support['sm'], support['sk'].diag(), obs).sum()
    torch.testing.assert_close(torch.tensor(selection['composite_log_likelihood'][0]), direct)
    return dict(shift_scale_sign_kernel_max_error=max(invariant_errors),
        invariance_threshold_limit='Nonzero scaling must not cross the fixed numerical-constant threshold',
        constant_and_empty_H_zero=True, tau0_identity=True,
        explicit_latent_mean_max_error=float((mean-explicit_mean).abs().max()),
        explicit_latent_covariance_max_error=float((cov-explicit_cov).abs().max()),
        complete_joint_block_PSD=True, far_point_min_S_distance=99., far_mean_change=far_change,
        qS_zero_cannot_update_amplitude=True, qS_zero_retains_prior_direction_variance=True,
        all_AP_S_selector_R1_exact=True, invisible_mode_selection=selection)


def _real_check(output):
    from unittest.mock import patch
    os.environ['LAW_ROOT'] = str(ROOT)
    from residual_probe.prepare import InterceptBankPrior, history_features, data_module
    data_module()
    from residual_probe.paired_features import features as pair_features
    from residual_probe.learning import load_case
    from residual_probe.localization import make_grid
    from support_integral import ImportanceSupport
    from source_weight_mixture import field_from_prior
    from run_multi_history import digest

    source = ROOT / 'results/source_residual_probe_v1_20261026'
    events = json.loads((source / 'prepared/manifest.json').read_text())['events']
    event = next(e for e in events if e['J'] == 2)
    model_path = source / 'learning_v2' / event['building'] / 'paired/model.json'
    model = json.loads(model_path.read_text())
    assert event['building'] not in model['training_buildings']
    assert model['feature_indices'] == list(range(10))
    theta = torch.tensor(model['theta'])
    # Choose using H presence only, not Q signal or localization performance.
    for ap, filename in zip(event['APs'], event['case_paths']):
        support = load_case(filename, source / 'paired_features')
        if bool((support['phi_s'][:, 5] > 0).any()):
            break
    prepared_path = Path(filename); before = digest(prepared_path)
    grid_path = source / 'grid_fields_v1' / event['id'] / f'ap{ap}.npz'
    grid_file = grid_path.parent / 'grid.npy'
    with np.load(grid_path, allow_pickle=False) as stored:
        assert str(stored['prepared_sha256']) == before
        assert str(stored['grid_sha256']) == digest(grid_file)
        arrays = {k: torch.from_numpy(stored[k]).double() for k in ['qm', 'qv', 'cross', 'phi_q']}
    grid = torch.from_numpy(np.load(grid_file, allow_pickle=False)).double()
    cases, meta = data_module().load_probe_episode(event['id'], source / 'preparation')
    raw = cases[ap]; obs = meta['obs']
    torch.testing.assert_close(make_grid(meta['grid']), grid, atol=0, rtol=0)
    prior = InterceptBankPrior(raw['histories'], event['weights'], meta['base'], event['intercept_std'])
    modes = build_modes(support, arrays, theta)
    poisoned = dict(support, rss=support['rss'] * 0 + 123., qx=raw['qx'] + 1e6)
    poison_modes = build_modes(poisoned, arrays, theta)
    for kind in KINDS:
        torch.testing.assert_close(modes[kind]['q_g'], poison_modes[kind]['q_g'], atol=0, rtol=0)
        torch.testing.assert_close(modes[kind]['q_s'], poison_modes[kind]['q_s'], atol=0, rtol=0)
    no_h = dict(support, phi_s=support['phi_s'] * 0)
    no_h_grid = dict(arrays, phi_q=arrays['phi_q'] * 0)
    assert all(not bool(m['q_s'].any() or m['q_g'].any()) for m in build_modes(no_h, no_h_grid, theta).values())
    grid_case = dict(support, **arrays)
    baseline, baseline_ess = zero_prediction(grid_case, obs)
    indices = torch.linspace(0, len(grid) - 1, min(8, len(grid))).long()
    rows = []
    for kind, mode in modes.items():
        zero_case = add_covariance(grid_case, mode, 0.)
        zero, _ = zero_prediction(zero_case, obs, make_predictor(kind, event['id'], 0., output / 'refreshes.jsonl'))
        for key in ['mean', 'variance']:
            torch.testing.assert_close(zero[key], baseline[key], atol=0, rtol=0)
        tau = 20.
        changed = add_covariance(grid_case, mode, tau)
        assert changed['sm'] is grid_case['sm'] and changed['qm'] is grid_case['qm']
        predictor = make_predictor(kind, event['id'], tau, output / 'refreshes.jsonl')
        value, ess = zero_prediction(changed, obs, predictor)
        effective = dict(changed, **predictor.proposals.get(changed['path'], {}))
        post = ImportanceSupport(effective['sm'], effective['sk'],
            {k: effective[k] for k in ['n', 'k', 's1', 's2']}, obs, effective['samples'], effective['log_proposal'])

        class RankOnePrior:
            def features(self, x):
                return prior.features(x)

            def direction(self, x):
                norm = mode['normalization']
                if norm['degenerate']:
                    return x.new_zeros(len(x))
                if kind == 'history_mode':
                    field = prior.features(x)[0]
                else:
                    phi = torch.cat((history_features(raw, x, meta['base']), pair_features(raw, x, meta['base'])), 1)
                    field = 20 * (phi @ theta)
                return (field - norm['center_dB']) / norm['rms_dB']

            def covariance(self, x, y, fx, fy):
                return prior.covariance(x, y, fx, fy) + tau ** 2 * self.direction(x)[:, None] * self.direction(y)[None]

            def diagonal(self, x, fx):
                return prior.diagonal(x, fx) + tau ** 2 * self.direction(x).square()

        explicit = RankOnePrior(); sx = raw['sx']; points = grid[indices]
        sm, sf = explicit.features(sx); qm, qf = explicit.features(points)
        sk = explicit.covariance(sx, sx, sf, sf)
        gg = explicit.covariance(points, points, qf, qf)
        cross = explicit.covariance(points, sx, qf, sf)
        torch.testing.assert_close(sm, changed['sm'], atol=1e-8, rtol=1e-9)
        torch.testing.assert_close(qm, changed['qm'][indices], atol=1e-8, rtol=1e-9)
        torch.testing.assert_close(sk, changed['sk'], atol=1e-7, rtol=1e-9)
        torch.testing.assert_close(cross, changed['cross'][indices], atol=1e-7, rtol=1e-9)
        torch.testing.assert_close(gg.diag(), changed['qv'][indices], atol=1e-7, rtol=1e-9)
        joint = torch.cat((torch.cat((sk, cross.T), 1), torch.cat((cross, gg), 1)), 0)
        minimum = float(torch.linalg.eigvalsh(joint).min()); assert minimum > -1e-7
        info = dict(log_detection_evidence=float(post.log_detection_evidence), effective_samples=float(ess))
        with patch('source_weight_mixture.sample_reference', return_value=(effective['samples'], post.log_weights, info)):
            reference = field_from_prior(explicit, obs, sx, raw['stats'], points, 0, defensive_fraction=.1)
        errors = {key: float((value[key][indices] - reference[key]).abs().max()) for key in ['mean', 'variance']}
        for key in errors:
            torch.testing.assert_close(value[key][indices], reference[key], atol=1e-6, rtol=1e-8)
        rows.append(dict(kind=kind, tau=tau, normalization=mode['normalization'], tau0_exact=True,
            means_unchanged=True, joint_block_min_eigenvalue=minimum,
            sk_max_error=float((sk-changed['sk']).abs().max()),
            cross_max_error=float((cross-changed['cross'][indices]).abs().max()),
            backend_prediction_max_error=errors, ESS=float(ess), S_only_refreshes=sum(predictor.counts.values()),
            refresh_namespace=predictor.namespace))
    assert digest(prepared_path) == before
    inputs = [prepared_path, grid_path, grid_file, model_path, source / 'paired_features' / event['id'] / f'ap{ap}.npz',
              source / 'prepared/parameters.json', source / 'preparation/manifest.json']
    return dict(event=event['id'], AP=ap, grid_points=len(grid), baseline_ESS=float(baseline_ess),
        rows=rows, normalization_ignores_Q_RSS_and_coordinates=True, no_H_modes_zero=True,
        w_sigma_unchanged=True, original_proposal_cache_unchanged=True,
        source_model_held_building_out=True, model_training_buildings=model['training_buildings'],
        inputs_sha256={str(p): digest(p) for p in inputs}, performance_scored=False, training=False)


def check():
    from run_multi_history import digest, write
    output = ROOT / 'results/source_change_mode_v1_20261027'
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write(output / 'check_context.json', dict(created=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
        mechanism=__doc__, code_sha256=digest(Path(__file__)), tau_grid=TAU_GRID,
        reverse_audit='Unchanged means/c/w/sigma. Normalization reads H-only frozen grid qm/Phi10 and S sm/Phi10, never Q RSS/coordinates. All-AP tau selection reads only S R1 observations and qS. Independent AP random amplitudes; not a cross-AP joint posterior.',
        scope='Specified synthetic and ONE real source AP checks only; no Q score, training or localization'))
    result = dict(synthetic=_synthetic_checks(), real=_real_check(output),
                  completed=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
                  seconds=time.perf_counter() - started, code_sha256=digest(Path(__file__)),
                  cpu_threads=torch.get_num_threads(), training=False, performance_scored=False)
    write(output / 'checks.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical checks stay on ray-web')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['check'])
    args = parser.parse_args(); globals()[args.stage]()
