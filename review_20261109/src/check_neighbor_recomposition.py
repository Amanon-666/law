"""Bounded server-only checks of operator expressivity and posterior integration."""
import hashlib
import json
from pathlib import Path
import socket
import time
import torch
from fields import History, Prior, Transfer, rbf
from recomposition import Parameters, prepare, prior, features, condition_field, evaluate
from neighbor_recomposition import (NeighborParameters, prepare_neighbors, neighbor_prior,
    coefficients, operator_mean, neighbor_scores, refresh_case, build_field)
from recomposition_predict import locate
from posterior import SupportIntegral
from observations import _chol


def gap(a, b):
    return max(float((a[k]-b[k]).abs().max().detach()) if a[k].numel() else 0. for k in a)


def dense(model, b, weights):
    """Independent dense random-vector transform; never used for real grids."""
    points = torch.cat((b['sx'], b['grid']))
    all_points = torch.cat((points, b['anchors']))
    parts = []
    from neighbor_recomposition import support_summary
    for j, source in enumerate(b['sources']):
        p = source.at(points)
        p['missing'] = source.missing
        p['coverage'] = -torch.expm1(-rbf(points, torch.unique(b['positions'][j], dim=0), 1., b['base']['length']).sum(1))
        p['support_residual'], p['support_coverage'] = support_summary(
            points, b['sx'], b['test_stats'], b['ss'][j]['m'], b['base']['length'])
        parts.append(p)
    z = features(points, parts, b['positions'], b['delta'], b['base'])
    alpha = model.weights(z, b['za'], weights)
    _, _, lg, ls = model.values()
    mean = points.new_zeros(len(points))
    covariance = rbf(points, points, b['base']['drift'], b['base']['length'])+b['intercept']**2
    constant_gap, coefficient_mass = 0., 0.
    for j, (source, p, a, dz) in enumerate(zip(b['sources'], parts, b['aa'], b['delta'])):
        d, c = coefficients(model, p, a, dz, b['base']['length'])
        transform = torch.cat((torch.diag(d), c), 1)
        constant_gap = max(constant_gap, float((transform.sum(1)-1).abs().max().detach()))
        coefficient_mass = max(coefficient_mass, float(c.abs().sum(1).max().detach()))
        m, f = source.p.features(all_points)
        mean += alpha[j]*(transform@m)
        op = alpha[j, :, None]*transform
        covariance += ls.mul(2).exp()*op@source.p.covariance(all_points, all_points, f, f)@op.T
    mean = -75+lg.exp()*(mean+75)
    k = len(b['sx'])
    return dict(sm=mean[:k], gm=mean[k:], sk=covariance[:k, :k],
                cross=covariance[k:, :k], gv=covariance.diag()[k:]), covariance, constant_gap, coefficient_mass


def checks():
    started = time.monotonic()
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    base = dict(mean=-90., amplitude=7., length=3., drift=2.)
    obs = dict(alpha=.85, beta=-85., detect_scale=4., noise_std=5.)
    a = torch.tensor([[0., 0.], [2., 0.], [0., 3.]])
    histories = [History(a, dict(mean=torch.tensor([-70., -76., -88.]), cov=torch.eye(3)*8), base),
                 History(a+1, dict(mean=torch.tensor([-90., -80., -72.]), cov=torch.eye(3)*12), base)]
    positions, delta = [a, a+1], [1, -1]
    sx = torch.tensor([[.5, .2], [2., 2.]])
    grid = torch.tensor([[0., 1.], [1., 1.], [2., 1.], [3., 1.]])
    weights = torch.tensor([.3, .7])
    stats = dict(n=torch.tensor([2., 3.]), k=torch.tensor([1., 2.]),
                 s1=torch.tensor([-76., -162.]), s2=torch.tensor([76.**2, 2*81.**2]))
    pack = lambda hs=histories, ps=positions, ds=delta, s=sx, g=grid, st=stats, chunk=2: prepare_neighbors(hs, ps, ds, s, g, base, 3., st, chunk)
    b = pack()
    model = NeighborParameters(torch.zeros(9), torch.ones(9))
    old = Parameters(torch.zeros(9), torch.ones(9))
    with torch.no_grad():
        model.theta.copy_(torch.linspace(-.2, .2, 12)); old.theta.copy_(model.theta)
    legacy = prior(old, prepare(histories, positions, delta, sx, grid, base, 3.), weights)
    nested = gap(neighbor_prior(model, b, weights), legacy)
    assert nested < 1e-8
    with torch.no_grad(): model.neighbor.copy_(torch.linspace(-.12, .15, 10))
    out = neighbor_prior(model, b, weights)
    ref, covariance, row_gap, mass = dense(model, dict(b, test_stats=stats), weights)
    dense_gap = gap(out, ref)
    assert dense_gap < 1e-8 and row_gap < 1e-12 and mass <= .5
    assert float(torch.linalg.eigvalsh(covariance).min().detach()) > 0
    permutation = gap(out, neighbor_prior(model, pack(hs=histories[::-1], ps=positions[::-1], ds=delta[::-1]), weights.flip(0)))
    reordered_s = neighbor_prior(model, pack(s=sx.flip(0), st={k:v.flip(0) for k,v in stats.items()}), weights)
    undo = dict(reordered_s, sm=reordered_s['sm'].flip(0), sk=reordered_s['sk'].flip((0, 1)), cross=reordered_s['cross'].flip(1))
    support_permutation = gap(out, undo)
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    shift = torch.tensor([500., -700.])
    moved = [History(h.z@rotation+shift, dict(mean=h.m, cov=h.c), base) for h in histories]
    geometric = neighbor_prior(model, pack(hs=moved, ps=[p@rotation+shift for p in positions],
                               s=sx@rotation+shift, g=grid@rotation+shift), weights)
    transform_gap = gap(out, geometric)
    chunk_gap = gap(out, neighbor_prior(model, pack(chunk=1), weights))
    assert max(permutation, support_permutation, transform_gap, chunk_gap) < 1e-8
    shifted_stats = dict(stats, s1=stats['s1']+10*stats['k'],
                        s2=stats['s2']+20*stats['s1']+100*stats['k'])
    conditioned_change = gap(out, neighbor_prior(model, pack(st=shifted_stats), weights))
    assert conditioned_change > 1e-5

    integral = SupportIntegral(out['sm'].detach(), out['sk'].detach(), stats, obs, power=13, seed=115091)
    proposal = (integral.samples, torch.distributions.MultivariateNormal(integral.rss['mean'],
                scale_tril=_chol(integral.rss['cov'])).log_prob(integral.samples))
    case = dict(blocks=b, stats=stats, rss=torch.tensor([-75., 100.]), proposal=proposal)
    second = dict(case, rss=torch.tensor([100., -81.]))
    def objective():
        scores = sum(neighbor_scores(model, c, weights, obs)[0] for c in (case, second))
        return -scores.log_softmax(1)[[0, 1], [0, 3]].mean()
    grad, = torch.autograd.grad(objective(), model.neighbor)
    j = int(grad.abs().argmax()); original = float(model.neighbor[j].detach()); step = 1e-5
    with torch.no_grad(): model.neighbor[j] = original+step
    plus = float(objective().detach())
    with torch.no_grad(): model.neighbor[j] = original-step
    minus = float(objective().detach())
    with torch.no_grad(): model.neighbor[j] = original
    fd = (plus-minus)/(2*step); derivative_gap = abs(fd-float(grad[j]))
    assert derivative_gap < 2e-5 and float(grad.abs().max()) > 1e-6
    ep = dict(cases={0:case, 1:second}, grid=grid, truth=grid[[0, 3]],
              query_weights=torch.tensor([.5, .5]), nearest=torch.tensor([0, 3]), weights=weights, obs=obs)
    exact_grad, = torch.autograd.grad(objective()+model.penalty(), model.neighbor)
    model.zero_grad()
    evaluate(model, ep, backward=True, score_fn=neighbor_scores)
    adjoint_gap = float((exact_grad-model.neighbor.grad).abs().max())
    assert adjoint_gap < 1e-8

    for hs, count in ((histories, 0), (histories, 1), ([histories[0], None], 1), ([None, None], 1), ([], 1)):
        ps, ds, w = (positions, delta, weights) if hs else ([], [], weights[:0])
        st = {k:v[:count] for k,v in stats.items()}
        small = pack(hs=hs, ps=ps, ds=ds, s=sx[:count], st=st)
        p = neighbor_prior(model, small, w)
        f, _ = condition_field(p, dict(stats=st, proposal=None), obs) if count == 0 else (p, None)
        assert all(bool(torch.isfinite(v).all()) for v in f.values())
    absent = dict(n=stats['n'], k=stats['k']*0, s1=stats['s1']*0, s2=stats['s2']*0)
    missing = pack(st=absent)
    assert all(bool((s['support_residual'] == 0).all()) for s in missing['ss'])
    assert bool(torch.isfinite(neighbor_prior(model, missing, weights)['gv']).all())

    # Odd field about a symmetric center: scalar contrast cannot move the center.
    toybase = dict(mean=-75., amplitude=7., length=1., drift=2.)
    z = torch.tensor([[-1., 0.], [0., 0.], [1., 0.]])
    h = History(z, dict(mean=torch.tensor([-80., -75., -70.]), cov=torch.eye(3)), toybase)
    toy = prepare_neighbors([h], [z], [0], z[:0], z[1:2], toybase, 0., {k:v[:0] for k,v in stats.items()})
    express = NeighborParameters(torch.zeros(9), torch.ones(9))
    untouched = neighbor_prior(express, toy, weights.new_ones(1))['gm']
    with torch.no_grad(): express.theta[9] = .8
    scalar = neighbor_prior(express, toy, weights.new_ones(1))['gm']
    with torch.no_grad(): express.neighbor[0] = .8
    directional = neighbor_prior(express, toy, weights.new_ones(1))['gm']
    scalar_change = float((scalar-untouched).abs().max().detach())
    neighbor_change = float((directional-scalar).abs().max().detach())
    assert scalar_change < 1e-10 and neighbor_change > .01

    # Same seed, S-only proposals and readout in training and deployment.
    event = 'synthetic-interface'; ap = 0
    seed = int.from_bytes(hashlib.sha256(f'neighbor-refi/{event}/{ap}'.encode()).digest()[:4], 'little')
    adapted, info = refresh_case(model, dict(blocks=b, stats=stats), weights, obs, seed)
    direct, ess = neighbor_scores(model, dict(adapted, rss=case['rss']), weights, obs)
    cached = build_field(model, {ap:dict(histories=histories, sx=sx, stats=stats)}, positions, delta,
                         base, obs, dict(weights=weights.tolist(), sigma=dict(intercept_std=3.)), grid, event)
    deployed = locate(cached, case['rss'][:, None])['scores']
    deployment_gap = float((direct-deployed).abs().max().detach())
    assert deployment_gap < 1e-8
    return dict(complete=True, seconds=time.monotonic()-started, parameter_count=sum(p.numel() for p in model.parameters()),
        nonzero_legacy_nested_error=nested, dense_covariance_error=dense_gap, constant_row_error=row_gap,
        coefficient_abs_sum=mass, source_permutation_error=permutation, support_permutation_error=support_permutation,
        rigid_coordinate_transform_error=transform_gap, chunk_error=chunk_gap, posterior_gradient_error=derivative_gap,
        support_conditioning_effect=conditioned_change, two_AP_adjoint_gradient_error=adjoint_gap,
        posterior_gradient=float(grad[j]), symmetric_scalar_effect=scalar_change, symmetric_neighbor_effect=neighbor_change,
        empty_single_missing_history_checks=True, all_missing_support_descriptor=True,
        deployment_score_error=deployment_gap, deployment_ess=ess, deployment_sampler=info['sampler'],
        scope='Synthetic implementation/expressivity checks only; no training or real localization result.')


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    result = checks()
    result['code_sha256'] = {n:hashlib.sha256(Path(n).read_bytes()).hexdigest()
                            for n in ('neighbor_recomposition.py', 'check_neighbor_recomposition.py')}
    print(json.dumps(result, indent=2))
