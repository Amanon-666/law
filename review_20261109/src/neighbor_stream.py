"""Exact all-AP score/gradient accumulation with one AP geometry resident.

Reconstruction trades computation for memory. Proposal samples and densities
are retained between the two passes; shared parameters must not change between
passes. This is the original two-pass ReFi adjoint, without all-AP grid caches.
"""
import hashlib
import copy
import resource
import time
import torch
from uji import tensor
from neighbor_recomposition import prepare_neighbors, refresh_case, neighbor_scores


def prepared_case(task, ap, delta, chunk=128, grid=None):
    raw = task['cases'][ap]
    grid = task['grid'] if grid is None else grid
    block = prepare_neighbors(raw['histories'], task['positions'], delta, raw['sx'], grid,
                              task['base'], task['state']['sigma']['intercept_std'], raw['stats'], chunk)
    return dict(blocks=block, stats=raw['stats'])


@torch.no_grad()
def prepare_cpu_proposals(model, task, delta, event_id):
    """M-PMC stays on CPU; the single candidate is irrelevant to the S prior."""
    assert task['grid'].device.type == 'cpu' and model.theta.device.type == 'cpu'
    weights = tensor(task['state']['weights'])
    proposals = {}
    for ap in task['cases']:
        case = prepared_case(task, ap, delta, grid=task['grid'][:1])
        seed = int.from_bytes(hashlib.sha256(f'episodic-refi/{event_id}/{ap}'.encode()).digest()[:4], 'little')
        case, _ = refresh_case(model, case, weights, task['obs'], seed)
        proposals[ap] = case['proposal']
    return proposals


def task_to_device(task, device):
    """Copy the fixed historical posterior buffers; do not recompute their fit."""
    def history_to_device(history):
        if history is None: return None
        moved = copy.copy(history)
        for name in ('z', 'm', 'c', 'chol', 'samples'):
            setattr(moved, name, getattr(history, name).to(device))
        return moved
    cases = {ap:dict(c, histories=[history_to_device(h) for h in c['histories']],
                     sx=c['sx'].to(device), stats={k:v.to(device) for k,v in c['stats'].items()})
             for ap,c in task['cases'].items()}
    return dict(task, cases=cases, grid=task['grid'].to(device),
                positions=[p.to(device) for p in task['positions']])


def coordinate_risk(scores, grid, truth, query_weights, distance_unit):
    """Exact deployed-mean MDE in source-fixed distance units.

    This is a point-decision risk, not a proper distribution score. The norm's
    zero subgradient is used at exact zero error; no smoothing is introduced.
    """
    errors = (scores.softmax(1) @ grid - truth).norm(dim=1)
    return (query_weights @ errors) / distance_unit


def evaluate_stream(model, task, delta, event_id, query, backward=False, proposals=None, progress=None, chunk=128, objective='nll'):
    """Q is isolated from field construction; adaptation state is already fixed.

    Linux ru_maxrss is in KiB. Called on the server, not on macOS. The optional
    progress hook receives timings/counts only, never input data or credentials.
    Only the source training adjoint changes with objective='mde'; deployed
    fields, scan likelihoods and coordinate expectation are unchanged.
    """
    if objective not in ('nll', 'mde'):
        raise ValueError(objective)
    device = task['grid'].device
    if device.type == 'cuda': torch.cuda.synchronize(device)
    started = time.monotonic()
    proposals = {} if proposals is None else proposals
    weights = tensor(task['state']['weights']).to(device)
    truth = tensor(query['xy']).to(device); grid = task['grid']
    qw = tensor(1/(len(query['positions'])*query['stats']['n'][query['group']])).to(device)
    # Preserve the CPU reference's tie breaking for quantized location labels.
    nearest = torch.cdist(truth.cpu(), grid.cpu()).argmin(1).to(device)
    total = grid.new_zeros((len(truth), len(grid)))
    esses = []

    def report(phase, count):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if peak > 4096:
            raise RuntimeError(f'process peak RSS {peak:.1f} MiB exceeds 4096 MiB at AP boundary')
        if progress is not None and (count == 1 or count % 25 == 0 or count == len(task['cases'])):
            progress(dict(phase=phase, APs_completed=count, seconds=time.monotonic()-started, peak_rss_MiB=peak))

    with torch.no_grad():
        for i, ap in enumerate(task['cases'], 1):
            case = prepared_case(task, ap, delta, chunk)
            if ap not in proposals:
                if device.type != 'cpu':
                    raise ValueError('GPU execution requires explicit CPU-generated proposals on the GPU')
                key = f'episodic-refi/{event_id}/{ap}'
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')
                case, _ = refresh_case(model, case, weights, task['obs'], seed)
                proposals[ap] = case['proposal']
            case['proposal'] = proposals[ap]
            case['rss'] = tensor(query['rss'][:, ap]).to(device)
            score, ess = neighbor_scores(model, case, weights, task['obs'])
            total += score
            if ess is not None: esses.append(ess)
            del case, score
            report('forward', i)
    if device.type == 'cuda': torch.cuda.synchronize(device)
    forward_seconds = time.monotonic()-started
    scores = total.requires_grad_(backward)
    logp = scores.log_softmax(1)
    nll = -(qw*logp[torch.arange(len(truth), device=device), nearest]).sum()
    errors = (logp.exp()@grid-truth).norm(dim=1)
    loss = nll if objective == 'nll' else coordinate_risk(scores, grid, truth, qw, task['base']['length'])
    if backward:
        adjoint, = torch.autograd.grad(loss, scores)
        for i, ap in enumerate(task['cases'], 1):
            case = prepared_case(task, ap, delta, chunk)
            case['proposal'] = proposals[ap]
            case['rss'] = tensor(query['rss'][:, ap]).to(device)
            score, _ = neighbor_scores(model, case, weights, task['obs'])
            if score.requires_grad:
                (score*adjoint).sum().backward()
            del case, score
            report('backward', i)
        model.penalty().backward()
    if device.type == 'cuda': torch.cuda.synchronize(device)
    metric = dict(nll=float(nll.detach()), mde=float(qw@errors.detach()), errors=errors.detach().tolist(),
                  objective=objective, data_loss=float(loss.detach()),
                  min_ess=min(esses) if esses else None, forward_seconds=forward_seconds,
                  seconds=time.monotonic()-started,
                  peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    return metric, proposals
