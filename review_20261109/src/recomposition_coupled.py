"""ReFi deployment and bounded-memory learning with current-prior adaptation.

Prepare H/S blocks once, then refit the task state after every shared update.
No query is accepted by adaptation. The original full support posterior and
scan likelihood remain the prediction path; inner VI only selects w/v.
"""
import hashlib
import torch
from observations import scan_log_prob
from posterior import proposal
from recomposition import prepare, condition_field
from recomposition_adaptation import fit, adapted_prior


@torch.no_grad()
def prepare_cases(task, delta):
    return {ap: dict(stats=c['stats'], blocks=prepare(c['histories'],
        task['positions'], delta, c['sx'], task['grid'], task['base'], 0.))
        for ap, c in task['cases'].items()}


def adapt_cases(model, cases, obs, event_id, differentiable=False):
    state = fit(model, cases, obs, differentiable=differentiable)
    prepared, diagnostics = {}, []
    with torch.no_grad():
        for ap, c in cases.items():
            p = adapted_prior(model, c['blocks'], state)
            k = len(p['sm'])
            if k:
                seed = int.from_bytes(hashlib.sha256(
                    f'coupled-refi/{event_id}/{ap}'.encode()).digest()[:4], 'little')
                numerical, info = proposal(dict(sm=p['sm'], sk=p['sk'],
                    phi_s=p['sm'].new_zeros((k, 0)), **c['stats']), obs,
                    p['sm'].new_empty(0), seed)
                sampling = (numerical['samples'], numerical['log_proposal'])
                diagnostics.append(dict(ap=ap, sampler=info['sampler']))
            else:
                sampling = None
            prepared[ap] = dict(c, proposal=sampling)
    return state, prepared, diagnostics


def scores_for(model, case, state, obs):
    field, ess = condition_field(adapted_prior(model, case['blocks'], state), case, obs)
    return scan_log_prob(case['rss'], field['mean'], field['variance'], obs), ess


def evaluate(model, ep, state, backward=False):
    """Two-pass complete-scan adjoint; consume the shared IFT graph only once.

    AP contributions use detached task-state leaves, accumulating direct
    model and state partials. A final state VJP supplies the indirect shared
    gradient. This avoids retaining every AP's field graph or backpropagating
    repeatedly through an already-freed common adaptation graph.
    """
    with torch.no_grad():
        total = ep['grid'].new_zeros((len(ep['truth']), len(ep['grid'])))
        esses = []
        for c in ep['cases'].values():
            score, ess = scores_for(model, c, state, ep['obs'])
            total += score
            if ess is not None: esses.append(ess)
    total.requires_grad_(backward)
    logp = total.log_softmax(1)
    loss = -(ep['query_weights']*logp[torch.arange(len(logp)), ep['nearest']]).sum()
    if backward:
        adjoint, = torch.autograd.grad(loss, total)
        keys = ('weights', 'intercept_variance')
        leaves = {key: state[key].detach().requires_grad_() for key in keys}
        for c in ep['cases'].values():
            score, _ = scores_for(model, c, leaves, ep['obs'])
            (score*adjoint).sum().backward()
        outputs = [state[key] for key in keys if state[key].requires_grad]
        gradients = [leaves[key].grad for key in keys if state[key].requires_grad]
        if outputs:
            torch.autograd.backward(outputs, gradients)
        model.penalty().backward()
    return dict(nll=float(loss.detach()), log_probability=logp.detach(),
                min_ess=min(esses) if esses else None)
