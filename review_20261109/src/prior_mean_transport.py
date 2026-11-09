"""K1 deterministic transport with fixed H-only empirical scan residuals.

A detected historical value r_j becomes r_j-m_j^0+M_GP^+. This assumes the
empirical residual transfers; it is not an exact counterfactual posterior or
recovery of real target scans. No random worlds or posterior covariance needed.
"""
import numpy as np
import torch

from common_update_transport import common_update, _summary
from coupled_scan_transport import _finish, ORDER
from fields import Prior, Transfer
from observations import JITTER
from scalar_support import ScalarSupportIntegral
from uji import NO_DETECT


@torch.no_grad()
def build_transport(histories, support, task, selected):
    """Return mean_histories and diagnostics; leave real S and source roles intact.

The H-only source fields are evaluated before applying the S posterior mean.
For two sources selected['weight'] means [w, 1-w] in task['names'] order.
Source fields remain independent as in the existing GP. Partial absent sources
use Prior(None); an AP absent from all H has no detected carrier to transport.
"""
    names = list(task['names'])
    assert names == list(histories) and len(names) in (1, 2)
    assert len(support['positions']) == 1
    sx = next(iter(task['cases'].values()))['sx']
    assert sx.shape == (1, 2)
    assert np.array_equal(sx.detach().cpu().numpy(), support['positions'])
    weight, sigma = float(selected['weight']), float(selected['sigma'])
    assert 0 <= weight <= 1 and sigma >= 0
    if len(names) == 1:
        assert weight == 1.
    weights = sx.new_tensor([1.] if len(names) == 1 else [weight, 1-weight])
    base, obs = task['base'], task['obs']
    residual_variance = sx.new_tensor(base['drift']**2+sigma**2)
    positions, inverse = np.unique(np.concatenate(
        [histories[name]['positions'] for name in names]), axis=0, return_inverse=True)
    x = sx.new_tensor(positions)
    row_indices, offset = {}, 0
    for name in names:
        size = len(histories[name]['positions'])
        row_indices[name] = inverse[offset:offset+size][histories[name]['group']]
        offset += size
    arrays = {name:np.asarray(h['rss'], dtype=np.float64).copy() for name,h in histories.items()}
    ap_records, common_fields, applied_updates = [], [], []
    difference_fields, applied_differences = [], []

    for ap, case in sorted(task['cases'].items()):
        assert len(case['histories']) == len(names) and torch.equal(case['sx'], sx)
        present = sum(h is not None for h in case['histories'])
        if not present:
            ap_records.append(dict(AP=int(ap), present_sources=0, transported=False,
                                   reason='No detected historical carrier; AP unchanged.'))
            continue
        means, support_means, crosses, support_variances = [], [], [], []
        for history in case['histories']:
            prior = Prior(Transfer('identity'), history, dict(base, drift=0.))
            mean, fx = prior.features(x)
            support_mean, fs = prior.features(sx)
            means.append(mean)
            support_means.append(support_mean[0])
            crosses.append(prior.covariance(x, sx, fx, fs)[:, 0])
            support_variances.append(prior.covariance(sx, sx, fs, fs)[0, 0])
        means, support_means = torch.stack(means), torch.stack(support_means)
        crosses, support_variances = torch.stack(crosses), torch.stack(support_variances)
        m_s = (weights*support_means).sum()
        v_s = (weights.square()*support_variances).sum()+residual_variance
        posterior = ScalarSupportIntegral(m_s.reshape(1), v_s.reshape(1, 1),
                                          case['stats'], obs, ORDER)
        mass = posterior.log_weights.exp()
        mu = (mass*posterior.samples[:, 0]).sum()
        a = (weights.square()[:, None]*crosses).sum(0)
        kappa = common_update(a, v_s, mu, m_s)
        m_zero = (weights[:, None]*means).sum(0)
        delta = m_zero[None]-means+kappa[None]
        # Exact analytic old-common minus new-prior displacement under the
        # existing mixed convention: nominal contrast gain, jittered kappa.
        difference = (a[None]-weights[:, None]*crosses)/v_s*(mu-m_s)
        kappa_array = kappa.cpu().numpy()
        delta_array, difference_array = delta.cpu().numpy(), difference.cpu().numpy()
        common_fields.append(kappa_array)
        difference_fields.append(difference_array)
        source_records, ap_applied, ap_differences = {}, [], []
        for j,name in enumerate(names):
            detected = histories[name]['rss'][:, ap] != NO_DETECT
            indices = row_indices[name][detected]
            update = delta_array[j, indices]
            change_from_common = difference_array[j, indices]
            arrays[name][detected, ap] += update
            ap_applied.append(update)
            ap_differences.append(change_from_common)
            source_records[name] = dict(
                common_minus_prior_field=_summary([difference_array[j]]),
                common_minus_prior_applied=_summary([change_from_common]))
        applied_updates.extend(ap_applied)
        applied_differences.extend(ap_differences)
        ap_records.append(dict(AP=int(ap), present_sources=present, transported=True,
            support_prior_mean=float(m_s), support_prior_variance=float(v_s),
            support_posterior_mean=float(mu), posterior_nodes=len(mass),
            common_update=_summary([kappa_array]), applied_displacement=_summary(ap_applied),
            common_minus_prior_field=_summary([difference_array]),
            common_minus_prior_applied=_summary(ap_differences), sources=source_records))

    mean_histories, mean_range = _finish(histories, arrays)
    diagnostics = dict(sources=names, weights=weights.cpu().tolist(), sigma=sigma,
        support_positions=1, common_positions=len(positions), quadrature_order=ORDER,
        gain_jitter=JITTER, support_solver_jitter=JITTER,
        transported_difference='sum_l w_l m_l^0(X)-m_j^0(X)+kappa(X)',
        common_update_formula='sum_l w_l^2 C_l(X,S)/(v_S+JITTER)*(mu_S-m_S)',
        residual_interpretation='Fixed empirical residual r_j-m_j^0 from H; its transfer is assumed, not inferred.',
        source_means='Each m_j^0 depends on H only; selected mixture weights may depend on S.',
        target_mean='Existing retained-GP common mean; not a recovered real target scan.',
        common_minus_prior_formula='(sum_l w_l^2 C_l(X,S)-w_j C_j(X,S))/v_S*(mu_S-m_S)',
        difference_sign='Old common-update displacement minus this prior-mean displacement.',
        difference_interpretation='Nominal conditional contrast S increment; same jittered kappa cancels.',
        field_weighting='Each transported AP/source and unique historical position once; common update has no source axis.',
        applied_weighting='Each original detected historical scan/AP cell once.',
        common_update=_summary(common_fields), applied_displacement=_summary(applied_updates),
        common_minus_prior_field=_summary(difference_fields),
        common_minus_prior_applied=_summary(applied_differences),
        mean_range=mean_range, APs=ap_records,
        historical_mask_preserved=True, source_floor_features_unchanged=True,
        support_modified=False, support_inserted=False, RSS_clipping=False,
        random_worlds_generated=False, posterior_covariance_used=False,
        limitations=['K1 and one/two sources only; selected weights/sigma supplied by existing adaptation.',
            'No target detection reconstruction; all-H-absent APs have no transported rows.',
            'Fixed empirical residuals may include nontransferable historical acquisition effects.'])
    return dict(mean_histories=mean_histories, diagnostics=diagnostics)
