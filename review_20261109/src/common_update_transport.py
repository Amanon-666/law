"""Add the retained-GP common mean update to the frozen contrast transport.

The original conditional worlds are generated once and translated by the same
deterministic field. This preserves their contrasts and covariance; it does not
recover target detection patterns or transport acquisition-specific residuals.
"""
import numpy as np
import torch

from coupled_scan_transport import build_transport as contrast_transport, _finish
from fields import Prior, Transfer
from observations import JITTER
from uji import NO_DETECT


def common_update(cross, prior_variance, posterior_mean, prior_mean):
    """K1 mean correction, with predict_from_posterior's exact jitter convention.

cross is Cov(M(X), f_S), excluding the acquisition-specific residual. The
variance and posterior mean are the existing scalar solver's actual values.
"""
    return cross / (prior_variance + JITTER) * (posterior_mean - prior_mean)


def _summary(parts):
    count = sum(p.size for p in parts)
    if not count:
        return dict(count=0, minimum=None, maximum=None, mean=0., rms=0.,
                    maximum_absolute=0.)
    return dict(count=count, minimum=min(float(p.min()) for p in parts if p.size),
                maximum=max(float(p.max()) for p in parts if p.size),
                mean=sum(float(p.sum()) for p in parts)/count,
                rms=float(np.sqrt(sum(float(np.square(p).sum()) for p in parts)/count)),
                maximum_absolute=max(float(np.abs(p).max()) for p in parts if p.size))


@torch.no_grad()
def build_transport(histories, support, task, selected, worlds=8, seed=1150835):
    """Same API as coupled_scan_transport.build_transport; S is never modified.

Reuse its sampled worlds and actual scalar posterior moments, then add kappa
only to original detected cells. One value is shared by every source/world at
the same position and AP. Source labels, rows, coordinates and masks survive.
"""
    original = contrast_transport(histories, support, task, selected, worlds, seed)
    old_diagnostics = original['diagnostics']
    names = list(task['names'])
    sx = next(iter(task['cases'].values()))['sx']
    weights = sx.new_tensor(old_diagnostics['weights'])
    base = task['base']
    sizes = [len(histories[name]['positions']) for name in names]
    positions, inverse = np.unique(np.concatenate(
        [histories[name]['positions'] for name in names]), axis=0, return_inverse=True)
    x = sx.new_tensor(positions)
    row_indices, offset = {}, 0
    for name, size in zip(names, sizes):
        row_indices[name] = inverse[offset:offset+size][histories[name]['group']]
        offset += size

    mean_arrays = {name:h['rss'].copy() for name,h in original['mean_histories'].items()}
    world_arrays = [{name:h['rss'].copy() for name,h in world.items()}
                    for world in original['world_histories']]
    posterior_records = {r['AP']:r for r in old_diagnostics['APs']}
    field_updates, applied_updates = [], []
    for ap, case in sorted(task['cases'].items()):
        record = posterior_records[ap]
        if not record['transported']:
            continue
        cross = x.new_zeros(len(x))
        for weight, history in zip(weights, case['histories']):
            prior = Prior(Transfer('identity'), history, dict(base, drift=0.))
            _, fx = prior.features(x)
            _, fs = prior.features(sx)
            cross += weight.square()*prior.covariance(x, sx, fx, fs)[:, 0]
        kappa = common_update(cross, record['support_prior_variance'],
                              record['support_posterior_mean'],
                              record['support_prior_mean']).cpu().numpy()
        field_updates.append(kappa)
        for name in names:
            detected = histories[name]['rss'][:, ap] != NO_DETECT
            update = kappa[row_indices[name][detected]]
            applied_updates.append(update)
            mean_arrays[name][detected, ap] += update
            for arrays in world_arrays:
                arrays[name][detected, ap] += update

    mean_histories, mean_range = _finish(histories, mean_arrays)
    finished = [_finish(histories, arrays) for arrays in world_arrays]
    diagnostics = dict(old_diagnostics,
        mean_range=mean_range, world_ranges=[ranges for _,ranges in finished],
        transported_difference='sum_l w_l h_l_plus - h_j_plus + kappa',
        contrast_only_diagnostics=old_diagnostics,
        maximum_sampled_absolute_delta_semantics='Original contrast-only maximum; excludes kappa.',
        common_update=dict(
            formula='kappa=sum_j w_j^2 C_j(X,S)/(v_S+JITTER)*(mu_S-m_S)',
            gain_jitter=JITTER, posterior_moments_reused=True,
            field_summary=_summary(field_updates),
            field_summary_weighting='Each transported AP and unique historical position once.',
            applied_summary=_summary(applied_updates),
            applied_summary_weighting='Each original detected scan/AP cell once; worlds not counted again.',
            covariance='Same deterministic translation of every old world; no new sampling or variance.',
            all_absent_APs_transported=False,
            historical_mask_preserved=True, support_modified=False,
            source_floor_features_unchanged=True, RSS_clipping=False))
    return dict(mean_histories=mean_histories,
                world_histories=[h for h,_ in finished], diagnostics=diagnostics)
