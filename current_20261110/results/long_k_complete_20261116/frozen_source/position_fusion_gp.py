"""Existing arbitrary-source retained GP readout for position-score fusion.

No fitting, asset I/O, query-coordinate access or new integration policy lives
here. prepare_from_scans consumes absolute H/S coordinates; its load_history
callback returns existing fields already translated to the requested origin.
The caller performs the original support adaptation and may use move_task to
move the prepared task to CUDA afterwards. All GP arrays are float64 without
changing PyTorch's process-wide default dtype.

predict_retained preserves the original M-PMC support posterior and moment
readout. Only the grid/S cross covariance removes the acquisition-specific
residual kernel and intercept; support covariance and grid marginal variance
are unchanged. K1 keeps the frozen building-protocol seed exactly. General K
uses the same seed family and original proposal/ESS policy, with no new retry.
"""
import hashlib

import numpy as np
import torch

from building_information_gp_trial import cpu_proposal, move_task
from episodic_tasks import task_inputs
from fields import InterceptBankPrior, rbf
from observations import observation_log_prob, predict_from_posterior
from posterior import ImportanceSupport
from uji import group_scans


def prepare_from_scans(histories, support, base, obs, load_history, origin, bounds):
    """Build a CPU task from raw H/S, with all actual rows and source roles.

    histories maps source names to {rss,xy}; support has the same fields. Their
    xy arrays and bounds are ABSOLUTE metres in one registered frame. The
    callback load_history(source,ap) must return the fitted History in LOCAL
    coordinates (cache_z + cache_origin - origin), in float64. A detected AP
    with a missing asset raises; only a truly undetected source/AP becomes None
    through the original task_inputs rule. Supports any positive K and J.

    Bounds come exclusively from permitted H/S. Round them on the absolute
    2 m lattice before subtracting origin, preserving the shared fusion grid.
    No adaptation is performed here.
    """
    origin = np.asarray(origin, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    bounds = np.stack((np.floor(bounds[0]/2)*2, np.ceil(bounds[1]/2)*2))
    grouped = {name: group_scans(h['rss'], np.asarray(h['xy'], dtype=np.float64)-origin)
               for name, h in histories.items()}
    s = group_scans(support['rss'], np.asarray(support['xy'], dtype=np.float64)-origin)

    def load(source, ap):
        h = load_history(source, ap)
        assert h is not None, f'Missing detected historical AP: {source}/{ap}'
        assert h.z.dtype == h.m.dtype == h.c.dtype == torch.float64
        assert h.z.device.type == 'cpu'
        return h

    task = task_inputs(grouped, s, base, obs, load)
    axes = [np.arange(bounds[0, axis], bounds[1, axis]+1., 2.) for axis in range(2)]
    absolute_grid = np.stack(np.meshgrid(*axes, indexing='xy'), -1).reshape(-1, 2)
    task['grid'] = torch.as_tensor(absolute_grid-origin, dtype=torch.float64)
    task.update(origin=origin, bounds=bounds, absolute_grid=absolute_grid,
                aps=np.asarray(list(task['cases']), dtype=np.int64))
    return task


@torch.no_grad()
def _retained_field(task, state, ap, case, event_id, k, relation_collector):
    """Original single-AP posterior and field computation, without any Q input."""
    points = task['grid']
    base, obs = task['base'], task['obs']
    sigma = state['sigma']['intercept_std']
    sx = case['sx']
    assert len(sx) == k and sx.dtype == torch.float64
    prior = InterceptBankPrior(case['histories'], state['weights'], base, sigma)
    sm, sf = prior.features(sx)
    qm, qf = prior.features(points)
    sk = prior.covariance(sx, sx, sf, sf)
    qv = prior.diagonal(points, qf)
    cross = prior.covariance(points, sx, qf, sf)
    prepared = dict(sm=sm, sk=sk, phi_s=sm.new_zeros((len(sx), 10)), **case['stats'])
    seed = int.from_bytes(hashlib.sha256(
        f'multifloor/{event_id}/K{k}/{ap}'.encode()).digest()[:4], 'little')
    numerical, info = cpu_proposal(prepared, obs, sm.new_zeros(10), seed)
    posterior = ImportanceSupport(sm, sk, case['stats'], obs,
                                  numerical['samples'], numerical['log_proposal'])
    weights = posterior.log_weights.exp()
    ess = float(weights.square().sum().reciprocal())
    if ess < 1000:
        raise RuntimeError(f'AP {ap}: ESS {ess}')
    mean = weights@posterior.samples
    centered = posterior.samples-mean
    moments = dict(mean=mean, cov=(centered.T*weights)@centered)
    if relation_collector is not None:
        relation_collector(ap, prior, sm, sk, moments)
    amplitude = base['drift'] if any(h is not None for h in case['histories']) else base['amplitude']
    retained_cross = cross-rbf(points, sx, amplitude, base['length'])-sigma**2
    value = predict_from_posterior(qm, qv, retained_cross, sm, sk, moments)
    if (not torch.isfinite(value['mean']).all() or not torch.isfinite(value['variance']).all()
            or (value['variance'] < 0).any()):
        raise RuntimeError(f'AP {ap}: invalid retained propagated moments')
    diagnostic = dict(AP=int(ap), ESS=ess, sampler=info['sampler'], seed=seed, K=int(k))
    return value, diagnostic


@torch.no_grad()
def predict_retained(task, state, qrss, event_id, k, *, relation_collector=None):
    """Return full-scan scores [Nq,G], absolute means and diagnostics.

    task contains H/S-only inputs, possibly moved using the imported move_task;
    state is the original CPU-list adaptation output. qrss is a full raw AP
    matrix, including 100 for nondetection. This function never selects APs or
    geometry from Q, never changes adaptation state, and never reads Q labels.
    An optional collector receives the SAME already-computed per-AP S moments
    for a separate H-context consumer. It does not choose proposals or replace
    the original GP scores. The default path collects nothing.
    """
    points = task['grid']
    assert points.dtype == torch.float64 and k >= 1
    rss = torch.as_tensor(np.asarray(qrss)[:, task['aps']], dtype=torch.float64, device=points.device)
    scores = points.new_zeros((len(rss), len(points)))
    diagnostics = []
    for column, (ap, case) in enumerate(task['cases'].items()):
        value, diagnostic = _retained_field(task, state, ap, case, event_id, k, relation_collector)
        scores += observation_log_prob(rss[:, column, None], value['mean'][None],
                                       value['variance'][None], task['obs'])
        diagnostics.append(diagnostic)
    if not torch.isfinite(scores).all():
        raise RuntimeError('Invalid retained complete scan scores')
    raw = (scores.softmax(1)@points).cpu().numpy()+task['origin']
    return dict(scores=scores.cpu().numpy(), raw=raw,
                projected=np.clip(raw, task['bounds'][0], task['bounds'][1]), diagnostics=diagnostics)


@torch.no_grad()
def prepare_retained_fields(task, state, event_id, k, *, relation_collector=None):
    """Prepare fixed H/S-conditioned fields once, before any query is supplied.

    This uses the same per-AP seeds, posterior, ESS rule and retained cross as
    predict_retained. Storing fields costs two AP-by-grid float64 arrays; it
    changes neither adaptation nor the observation likelihood. Original
    predict_retained remains the streaming path for existing callers.
    """
    assert task['grid'].dtype == torch.float64 and k >= 1
    means, variances, diagnostics = [], [], []
    for ap, case in task['cases'].items():
        value, diagnostic = _retained_field(task, state, ap, case, event_id, k, relation_collector)
        means.append(value['mean'])
        variances.append(value['variance'])
        diagnostics.append(diagnostic)
    return dict(mean=torch.stack(means), variance=torch.stack(variances),
        grid=task['grid'], aps=np.asarray(list(task['cases']), dtype=np.int64),
        origin=np.asarray(task['origin']).copy(), bounds=np.asarray(task['bounds']).copy(),
        obs=dict(task['obs']), diagnostics=diagnostics)


@torch.no_grad()
def score_retained_fields(prepared, qrss):
    """Original complete-scan readout on an already fixed H/S preparation."""
    points = prepared['grid']
    rss = torch.as_tensor(np.asarray(qrss)[:, prepared['aps']], dtype=torch.float64, device=points.device)
    scores = points.new_zeros((len(rss), len(points)))
    for column in range(len(prepared['aps'])):
        scores += observation_log_prob(rss[:, column, None], prepared['mean'][column][None],
                                       prepared['variance'][column][None], prepared['obs'])
    if not torch.isfinite(scores).all():
        raise RuntimeError('Invalid retained complete scan scores')
    raw = (scores.softmax(1)@points).cpu().numpy()+prepared['origin']
    return dict(scores=scores.cpu().numpy(), raw=raw,
        projected=np.clip(raw, prepared['bounds'][0], prepared['bounds'][1]),
        diagnostics=prepared['diagnostics'])
