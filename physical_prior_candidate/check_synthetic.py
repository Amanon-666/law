"""Three bounded mechanism cases, not a localization/performance experiment."""
import hashlib
import json
from pathlib import Path
import socket
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'operator_prior_v1'))

import numpy as np
import torch

from floor_profile import FloorProfile, fit_attenuation, field_from_profile, scan_log_prob
import operator_prior
from history_bank import BankPrior
from inference import gaussian_update, observation_log_prob
from source_weight_mixture import field_from_prior
from transfer import History


BASE = dict(mean=-90., amplitude=20., length=10., drift=.5)
OBS = dict(alpha=.95, beta=-110., detect_scale=3., noise_std=1.5)


def history(level, xy, base=BASE):
    return History(xy, dict(mean=xy.new_full((len(xy),), float(level)),
                           cov=.01 * torch.eye(len(xy))), base)


def stats(rss, scans=3):
    return dict(n=torch.full_like(rss, scans), k=torch.full_like(rss, scans),
                s1=scans * rss, s2=scans * rss.square())


def run():
    xy = torch.tensor([[-4., 0.], [0., 0.], [4., 0.]])
    h2 = [history(-66, xy), history(-72, xy)]
    profile = FloorProfile(h2, [.4, .6], BASE, [0, 3], 1, [0, 1, 2, 3], xy)
    two_a = profile.history_log_evidence(6.)[1]
    two_b = profile.history_log_evidence(2.)[0]
    torch.testing.assert_close(two_a, two_b, atol=1e-9, rtol=0)
    # Same H, same horizontal peak at all positions, different target truths.
    p6, p2 = profile.history_log_mass(6.).exp(), profile.history_log_mass(2.).exp()
    assert p6[1] > .999 and p2[0] > .999
    target_a = -60. - 6 * abs(1 - 1)
    target_b = -66. - 2 * abs(0 - 1)
    assert target_a == -60 and target_b == -68
    h3 = [h2[0], history(-66, xy), h2[1]]
    third = FloorProfile(h3, [.2, .4, .4], BASE, [0, 2, 3], 1, [0, 1, 2, 3], xy)
    gap = float(third.history_log_evidence(6.)[1] - third.history_log_evidence(2.)[0])
    assert gap > 15
    # The additional layer does not ALWAYS help: for two floors outside the
    # entire source/target interval, linear attenuation has identical contrasts.
    external = FloorProfile(h3, [.2, .4, .4], BASE, [0, 2, 3], 1, [-2, -1], xy)
    torch.testing.assert_close(external.expected[0], external.expected[1])
    # General J is exercised at J=4 with a nonuniform, permuted weight vector.
    floors4, weights4 = [0, 2, 3, 4], [.1, .2, .3, .4]
    hs4 = [history(-60 - 6 * abs(1 - z), xy) for z in floors4]
    four = FloorProfile(hs4, weights4, BASE, floors4, 1, list(range(5)), xy)
    order = [2, 0, 3, 1]
    four_perm = FloorProfile([hs4[i] for i in order], [weights4[i] for i in order], BASE,
                            [floors4[i] for i in order], 1, list(range(5)), xy)
    torch.testing.assert_close(four.history_log_mass(6), four_perm.history_log_mass(6),
                               atol=1e-8, rtol=1e-10)
    for left, right in zip(four.prior_moments(xy, 6), four_perm.prior_moments(xy, 6)):
        torch.testing.assert_close(left, right, atol=1e-8, rtol=1e-10)

    # Wrong-direction case: source-floor APs require weakening from their own
    # source, not max(H)+nonnegative kappa on every intervening target floor.
    asymmetric = []
    for target in [1, 2]:
        hs = [history(-50, xy), history(-68, xy)]
        model = FloorProfile(hs, [.5, .5], BASE, [0, 3], target, [0, 1, 2, 3], xy)
        mean, cov = model.prior_moments(xy, 6.)
        expected = -50 - 6 * target
        np.testing.assert_allclose(mean.numpy(), expected, atol=1e-5, rtol=0)
        assert torch.linalg.eigvalsh(cov).min() > 0
        reverse = FloorProfile(hs[::-1], [.5, .5], BASE, [3, 0], target, [0, 1, 2, 3], xy)
        torch.testing.assert_close(reverse.history_log_mass(6), model.history_log_mass(6))
        assert expected < max(-50, -68)
        asymmetric.append(dict(target=target, prediction=mean.tolist(), truth=expected,
                               hard_max_lower_bound=-50))

    # 8 synthetic, independent APs, four installation floors, ONE target anchor.
    sx = xy[1:2]
    profiles, statistics = [], []
    for i in range(8):
        floor, u = i % 4, -52. - i
        hs = [history(u - 6 * abs(floor - z), xy) for z in [0, 3]]
        profiles.append(FloorProfile(hs, [.4, .6], BASE, [0, 3], 1, [0, 1, 2, 3], xy))
        statistics.append(stats(sx.new_tensor([u - 6 * abs(floor - 1)])))
    fit = fit_attenuation(profiles, sx, statistics, OBS, list(range(13)), 0.)
    assert fit['attenuation'] == 6.
    empty = fit_attenuation(profiles, sx[:0], statistics, OBS, list(range(13)), 4.)
    assert empty['attenuation'] == 4 and empty['support_identified'] is False
    zero_n = [{k: torch.zeros_like(v) for k, v in st.items()} for st in statistics]
    no_evidence = fit_attenuation(profiles, sx, zero_n, OBS, [0., 6., 12.], 0.)
    assert no_evidence['score_range'] < 1e-12

    # Basic invariances and actual degeneracies, with no new test framework.
    p = profiles[1]
    m0, c0 = p.prior_moments(xy, 0)
    bm, bf = p.bank.features(xy)
    torch.testing.assert_close(m0, bm)
    torch.testing.assert_close(c0, p.bank.covariance(xy, xy, bf, bf))
    moved = xy + torch.tensor([200., -300.])
    moved_hs = [History(h.z + torch.tensor([200., -300.]),
                        dict(mean=h.m, cov=h.c), BASE) for h in p.histories]
    shifted = FloorProfile(moved_hs, [.4, .6], BASE, [0, 3], 1, [0, 1, 2, 3], moved)
    torch.testing.assert_close(shifted.history_log_mass(6), p.history_log_mass(6))
    torch.testing.assert_close(shifted.prior_moments(moved, 6)[0], p.prior_moments(xy, 6)[0])
    single = FloorProfile([h2[0]], [1.], BASE, [0], 1, [0, 1, 2, 3], xy)
    torch.testing.assert_close(single.history_log_mass(6).exp(), torch.full((5,), .2))
    no_probe = FloorProfile(h2, [.4, .6], BASE, [0, 3], 1, [0, 1, 2, 3], xy[:0])
    torch.testing.assert_close(no_probe.history_log_mass(6).exp(), torch.full((5,), .2))
    new = FloorProfile([None, None], [.4, .6], BASE, [0, 3], 1, [0, 1, 2, 3], xy)
    nm, nc = new.prior_moments(xy, 12.)
    old_empty = BankPrior([], [], BASE)
    om, of = old_empty.features(xy)
    torch.testing.assert_close(nm, om)
    torch.testing.assert_close(nc, old_empty.covariance(xy, xy, of, of))
    partial = FloorProfile([h2[0], None, h2[1]], [.2, .5, .3], BASE,
                          [0, 1, 3], 1, [0, 1, 2, 3], xy)
    assert float(partial.offset_units[1]) == .8
    diffuse_base = dict(BASE, drift=1e5)
    diffuse = FloorProfile(h2, [.4, .6], diffuse_base, [0, 3], 1, [0, 1, 2, 3], xy)
    np.testing.assert_allclose(diffuse.history_log_mass(6).exp().numpy(), .2, atol=1e-8)
    assert torch.isfinite(profile.history_log_mass(1e6)).all()

    # Existing PMC backend: one informative synthetic S, no target localization.
    # Compare to the analytic Gaussian evidence while detection is saturated.
    symmetric = FloorProfile([history(-70, xy), history(-70, xy)], [.5, .5], BASE,
                             [0, 2], 1, [0, 1, 2], xy)
    broad_obs = dict(OBS, noise_std=4.)
    st = stats(torch.tensor([-65.]))
    field = field_from_profile(symmetric, 6., broad_obs, sx, st, xy, 261026)
    logz = []
    for i in range(len(symmetric.labels)):
        prior = symmetric.component(6., i)
        sm, sf = prior.features(sx)
        sk = prior.covariance(sx, sx, sf, sf)
        logz.append(-gaussian_update(sm, sk, st, broad_obs)['loss'])
    logz = torch.stack(logz)
    once = torch.log_softmax(symmetric.history_log_mass(6.) + logz, 0)
    twice = torch.log_softmax(symmetric.history_log_mass(6.) + 2 * logz, 0)
    mass_error = float((field['log_mass'].exp() - once.exp()).abs().max())
    doubled_difference = float((field['log_mass'].exp() - twice.exp()).abs().max())
    assert mass_error < .01 and doubled_difference > .05
    assert all(v > 1000 for v in field['effective_samples'])
    observed = torch.tensor([[-65.], [100.]])
    lp = scan_log_prob(observed, field, broad_obs)
    manual = sum(float(w) * observation_log_prob(observed, m[None], v[None], broad_obs).exp()
                 for w, m, v in zip(field['log_mass'].exp(), field['component_means'],
                                    field['component_variances']))
    torch.testing.assert_close(lp.exp(), manual)
    zero = field_from_profile(symmetric, 0., broad_obs, sx, st, xy, 261026)
    baseline = field_from_prior(symmetric.bank, broad_obs, sx, st, xy, 261026,
                                defensive_fraction=.1)
    torch.testing.assert_close(zero['mean'], baseline['mean'], atol=0, rtol=0)
    torch.testing.assert_close(zero['variance'], baseline['variance'], atol=0, rtol=0)
    # Empty S: mixture uncertainty is the law-of-total-variance moment.
    no_s = field_from_profile(symmetric, 6., broad_obs, sx[:0], st, xy, 261026)
    pm, pc = symmetric.prior_moments(xy, 6.)
    torch.testing.assert_close(no_s['mean'], pm)
    torch.testing.assert_close(no_s['variance'], pc.diag())
    return dict(
        case1=dict(two_source_logevidence=[float(two_a), float(two_b)],
                   indistinguishable_history=[-66, -72], target_truths=[target_a, target_b],
                   third_floor_logevidence_gap=gap, external_floors_still_aliased=True),
        case2=dict(asymmetric_targets=asymmetric, source_permutation=True, prior_covariance_psd=True),
        case3=dict(shared_fit=fit, synthetic_APs=8, support_locations=1, scans_per_AP=3,
                   empty_support=empty, no_observations_score_range=no_evidence['score_range'],
                   coordinate_translation=True, partial_history=True, new_AP_fallback=True,
                   single_history_keeps_floor_uncertainty=True, no_overlap_keeps_floor_uncertainty=True,
                   noise_and_scale_limits=True, zero_attenuation_backend_exact=True,
                   empty_support_variance=True, mixture_scan_probability=True,
                   general_J_1_through_4=True, nonuniform_four_source_permutation=True,
                   floor_mass_error_vs_analytic=mass_error,
                   floor_mass_difference_vs_double_counted=doubled_difference,
                   floor_mass=field['log_mass'].exp().tolist(), effective_samples=field['effective_samples']))


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical checks are restricted to ray-web')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    started = datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
    result = run()
    result.update(scope='Synthetic structural checks only; no training or target localization',
                  started=started, completed=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
                  host=socket.gethostname(), torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
                  sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ['physical_prior_candidate/floor_profile.py',
                                       'physical_prior_candidate/check_synthetic.py',
                                       'history_bank.py', 'source_weight_mixture.py']})
    result['operator_prior'] = dict(path=operator_prior.__file__,
        sha256=hashlib.sha256(Path(operator_prior.__file__).read_bytes()).hexdigest())
    output = Path(__file__).with_name('synthetic_results.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
