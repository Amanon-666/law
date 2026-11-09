"""K1 source--target difference fields applied to real historical scans.

Sources are sampled jointly after conditioning on the same scalar S field.
Only M-h_j is transported: acquisition-specific residuals are not transported.
Worlds remain separate contexts; the consumer inserts the real S once per fit.
"""
import numpy as np
import torch

from fields import Prior, Transfer
from scalar_support import ScalarSupportIntegral
from uji import group_scans, NO_DETECT

JITTER = 1e-6
ORDER = 256


def condition_sources(means, covariances, weights, support_index,
                      residual_variance, posterior_samples, posterior_mass,
                      worlds, generator):
    """Return conditional draws [source, world, position] and exact node means.

The scalar posterior nodes/masses come from ScalarSupportIntegral. Full source
covariances use fixed 1e-6 Cholesky jitter; gains use the nominal covariance.
This is the stated numerical approximation, not a bitwise posterior identity.
The function is also usable with a small explicitly supplied Gaussian block.
"""
    s = support_index
    m_s = (weights*means[:, s]).sum()
    v_s = (weights.square()*covariances[:, s, s]).sum()+residual_variance
    gain = weights[:, None]*covariances[:, :, s]/v_s
    eye = torch.eye(means.shape[1], dtype=means.dtype, device=means.device)
    roots = torch.linalg.cholesky(covariances+JITTER*eye)
    normals = torch.randn((len(weights), worlds, means.shape[1]),
                          dtype=means.dtype, device=means.device, generator=generator)
    prior_draws = means[:, None, :]+normals@roots.transpose(-1, -2)
    residual = torch.randn(worlds, dtype=means.dtype, device=means.device,
                           generator=generator)*torch.sqrt(residual_variance)
    f_s_zero = (weights[:, None]*prior_draws[:, :, s]).sum(0)+residual
    # These draws are independent of the prior source/residual realizations.
    indices = torch.multinomial(posterior_mass, worlds, replacement=True,
                                generator=generator)
    f_s_plus = posterior_samples[indices]
    draws = prior_draws+gain[:, None, :]*(f_s_plus-f_s_zero)[None, :, None]
    posterior_mean = (posterior_mass*posterior_samples).sum()
    conditional_means = means+gain*(posterior_mean-m_s)
    return dict(draws=draws, means=conditional_means, support_prior_mean=m_s,
                support_prior_variance=v_s, gain=gain)


def _finish(histories, arrays):
    output = {}
    count, outside, minimum, maximum = 0, 0, None, None
    for name, original in histories.items():
        rss = arrays[name]
        detected = original['rss'] != NO_DETECT
        # The existing Tab interface reserves exactly 100 for non-detection.
        # Do not silently turn a transported detected value into that sentinel.
        assert np.array_equal(rss != NO_DETECT, detected)
        assert np.isfinite(rss).all()
        values = rss[detected]
        count += len(values)
        outside += int(((values < -105.) | (values > 0.)).sum())
        if len(values):
            lo, hi = float(values.min()), float(values.max())
            minimum = lo if minimum is None else min(minimum, lo)
            maximum = hi if maximum is None else max(maximum, hi)
        # Recompute sufficient statistics, retaining source name and metadata.
        output[name] = dict(original, **group_scans(rss, original['xy']))
    return output, dict(detected_cells=count, minimum=minimum, maximum=maximum,
                        outside_reference_interval=outside)


@torch.no_grad()
def build_transport(histories, support, task, selected, worlds=8, seed=1150835):
    """Return mean_histories, world_histories and concise provenance diagnostics.

Inputs share the existing registered coordinate origin and full AP indexing.
selected is the original strong-GP {weight, sigma} result, not refitted here.
For two sources weight means [weight, 1-weight] in task['names'] order. A
single-source call requires weight=1 and has an exactly zero difference field.
"""
    names = list(task['names'])
    assert names == list(histories) and len(names) in (1, 2)
    assert len(support['positions']) == 1 and worlds > 0
    first = next(iter(task['cases'].values()))
    sx = first['sx']
    assert sx.shape == (1, 2)
    assert np.array_equal(sx.detach().cpu().numpy(), support['positions'])
    weight = float(selected['weight'])
    assert 0 <= weight <= 1 and float(selected['sigma']) >= 0
    if len(names) == 1:
        assert weight == 1.
    weights = sx.new_tensor([1.] if len(names) == 1 else [weight, 1-weight])
    base, obs = task['base'], task['obs']
    residual_variance = sx.new_tensor(base['drift']**2+selected['sigma']**2)

    # Deduplicate X and S together, including when S is a historical position.
    # Repeated scans and equal positions therefore share exactly one field draw.
    sizes = [len(histories[name]['positions']) for name in names]
    positions, inverse = np.unique(np.concatenate(
        [histories[name]['positions'] for name in names]+[support['positions']]),
        axis=0, return_inverse=True)
    x = sx.new_tensor(positions)
    support_index = int(inverse[-1])
    row_indices, offset = {}, 0
    for name, size in zip(names, sizes):
        row_indices[name] = inverse[offset:offset+size][histories[name]['group']]
        offset += size
    generator = torch.Generator(device=sx.device).manual_seed(seed)
    mean_arrays = {name:np.asarray(h['rss'], dtype=np.float64).copy()
                   for name,h in histories.items()}
    world_arrays = [{name:rss.copy() for name,rss in mean_arrays.items()}
                    for _ in range(worlds)]
    ap_records = []
    maximum_delta = 0.

    for ap, case in sorted(task['cases'].items()):
        assert len(case['histories']) == len(names) and torch.equal(case['sx'], sx)
        present = any(h is not None for h in case['histories'])
        if not present:
            # Match InterceptBankPrior's all-absent branch: no drift addition.
            m_s = sx.new_tensor([base['mean']])
            v_s = sx.new_tensor([[base['amplitude']**2+selected['sigma']**2]])
            posterior = ScalarSupportIntegral(m_s, v_s, case['stats'], obs, ORDER)
            mass = posterior.log_weights.exp()
            post_mean = (mass*posterior.samples[:, 0]).sum()
            ap_records.append(dict(AP=int(ap), present_sources=0, transported=False,
                support_prior_mean=float(m_s[0]), support_prior_variance=float(v_s[0, 0]),
                support_posterior_mean=float(post_mean)))
            continue

        means, covariances = [], []
        for history in case['histories']:
            prior = Prior(Transfer('identity'), history, dict(base, drift=0.))
            mean, features = prior.features(x)
            covariance = prior.covariance(x, x, features, features)
            means.append(mean)
            covariances.append((covariance+covariance.T)/2)
        means, covariances = torch.stack(means), torch.stack(covariances)
        m_s = (weights*means[:, support_index]).sum().reshape(1)
        v_s = ((weights.square()*covariances[:, support_index, support_index]).sum()
               +residual_variance).reshape(1, 1)
        posterior = ScalarSupportIntegral(m_s, v_s, case['stats'], obs, ORDER)
        mass = posterior.log_weights.exp()
        samples = posterior.samples[:, 0]
        result = condition_sources(means, covariances, weights, support_index,
            residual_variance, samples, mass, worlds, generator)
        h_plus, h_mean = result['draws'], result['means']
        persistent = (weights[:, None, None]*h_plus).sum(0)
        persistent_mean = (weights[:, None]*h_mean).sum(0)
        delta = persistent[None]-h_plus
        mean_delta = persistent_mean[None]-h_mean
        if len(names) == 1:
            assert bool((delta == 0).all() and (mean_delta == 0).all())
        maximum_delta = max(maximum_delta, float(delta.abs().max()))
        for j, name in enumerate(names):
            detected = histories[name]['rss'][:, ap] != NO_DETECT
            indices = row_indices[name][detected]
            original_rss = histories[name]['rss'][detected, ap].astype(np.float64)
            change = mean_delta[j].detach().cpu().numpy()
            mean_arrays[name][detected, ap] = original_rss+change[indices]
            changes = delta[j].detach().cpu().numpy()
            for world in range(worlds):
                world_arrays[world][name][detected, ap] = original_rss+changes[world, indices]
        post_mean = (mass*samples).sum()
        post_variance = (mass*(samples-post_mean).square()).sum()
        ap_records.append(dict(AP=int(ap), present_sources=sum(h is not None for h in case['histories']),
            transported=True, support_prior_mean=float(m_s[0]), support_prior_variance=float(v_s[0, 0]),
            support_posterior_mean=float(post_mean), support_posterior_variance=float(post_variance)))

    mean_histories, mean_range = _finish(histories, mean_arrays)
    finished = [_finish(histories, arrays) for arrays in world_arrays]
    diagnostics = dict(worlds=worlds, seed=seed, sources=names, weights=weights.detach().cpu().tolist(),
        sigma=float(selected['sigma']), common_positions=len(positions), support_positions=1,
        quadrature_order=ORDER, sampling_cholesky_jitter=JITTER, support_solver_jitter=JITTER,
        numerical='Source sampling adds fixed diagonal jitter; nominal gains and existing scalar posterior are retained. No bitwise equivalence claim.',
        transported_difference='sum_l w_l h_l_plus - h_j_plus from the same joint realization',
        acquisition_residual_transported=False, source_floor_features_unchanged=True,
        support_inserted=False, RSS_clipping=False, rss_reference_interval=[-105., 0.],
        mean_range=mean_range, world_ranges=[ranges for _,ranges in finished],
        maximum_sampled_absolute_delta=maximum_delta, APs=ap_records,
        dependence='Sources and positions are jointly conditioned within each AP; GP AP fields remain independent. Real scan AP pairing is retained.')
    return dict(mean_histories=mean_histories,
                world_histories=[h for h,_ in finished], diagnostics=diagnostics)
