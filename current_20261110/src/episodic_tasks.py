"""Real source H/S/Q tasks for ReFi; no inherited target adaptation state.

Sampling adds support/source configurations, not independent environments.
The H/S-only adapter is also the deployment entry point. Query attachment is
separate so query rows cannot choose APs, geometry, weights or posterior state.
"""
import hashlib
import numpy as np
import torch
from predict import adapt
from uji import tensor, ap_stats
from support_recomposition import prepare_context, refresh_context_case
from episode_queries import set_queries


def sample_family(template, raw, seed, budgets=(1, 3, 5), queries=12, repeats=3):
    """Nested S, common disjoint Q, all single histories plus the full bank."""
    assert template['role'] == 'train'
    assert template['floor'] not in template['histories']
    assert tuple(sorted(set(budgets))) == tuple(budgets) and budgets[0] > 0
    rng = np.random.default_rng(seed)
    xy, groups, counts = np.unique(raw['xy'], axis=0, return_inverse=True, return_counts=True)
    eligible = np.flatnonzero(counts >= repeats)
    assert len(eligible) >= budgets[-1] + queries
    chosen = rng.choice(eligible, budgets[-1] + queries, replace=False)
    rows = [raw['row_ids'][rng.choice(np.flatnonzero(groups == g), repeats, replace=False)].tolist()
            for g in chosen]
    support = sum(rows[:budgets[-1]], [])
    query = sum(rows[budgets[-1]:], [])
    sources = [[f] for f in template['histories']]
    if len(template['histories']) > 1:
        sources.append(list(template['histories']))
    events = []
    for histories in sources:
        for k in budgets:
            events.append(dict(id=f"{template['floor']}_seed{seed}_H{'-'.join(histories)}_K{k}",
                role='train', floor=template['floor'], histories=histories,
                support_rows=support[:k*repeats], query_rows=query,
                history_rows={f: template['history_rows'][f] for f in histories},
                K=k, R=repeats, Q=queries, seed=seed,
                parent_event=template['id']))
    return events


def task_inputs(history_scans, support, base, obs, load_history):
    """Build deploy/train inputs using H/S only; coordinates share origin.

    history_scans preserves source identity and actual retained observations.
    load_history(source, ap) reads an existing fitted field. An AP with no
    detection in a source is absent there; a genuinely missing fitted asset
    must raise in the loader, never silently become an absent historical AP.
    """
    assert history_scans and len(support['positions'])
    names = list(history_scans)
    seen = (support['rss'] != 100).any(0)
    detected = {f: (h['rss'] != 100).any(0) for f, h in history_scans.items()}
    for mask in detected.values():
        seen |= mask
    geometry = np.concatenate([support['positions']] + [h['positions'] for h in history_scans.values()])
    low, high = np.floor(geometry.min(0)/2)*2, np.ceil(geometry.max(0)/2)*2
    axes = [np.arange(low[d], high[d]+1., 2.) for d in range(2)]
    grid = tensor(np.stack(np.meshgrid(*axes, indexing='xy'), -1).reshape(-1, 2))
    sx = tensor(support['positions'])
    cases = {int(ap): dict(histories=[load_history(f, int(ap)) if detected[f][ap] else None for f in names],
                          sx=sx, stats=ap_stats(support, int(ap))) for ap in np.flatnonzero(seen)}
    return dict(cases=cases, grid=grid,
                positions=[tensor(h['positions']) for h in history_scans.values()],
                names=names, base=base, obs=obs)


def adapt_task(history_scans, support, base, obs, load_history):
    """Fit this H/S task from scratch; no implicit reuse of another S fit."""
    task = task_inputs(history_scans, support, base, obs, load_history)
    task['state'] = adapt(task['cases'], base, obs)
    return task


@torch.no_grad()
def prepare_adapted(model, task, target_minus_source, event_id):
    """Build training-ready conditional fields from this task's fresh H/S fit.

    No query access and no shared parameter updates. The returned episode can
    be scored once Q is explicitly attached; proposal state is task-local.
    """
    weights = tensor(task['state']['weights'])
    ep = dict(id=event_id, role='train', grid=task['grid'], base=task['base'],
              obs=task['obs'], weights=weights, cases={})
    for ap, case in task['cases'].items():
        block = prepare_context(case['histories'], task['positions'], target_minus_source,
            case['sx'], task['grid'], task['base'], task['state']['sigma']['intercept_std'], case['stats'])
        key = f'{event_id}/{ap}'
        seed = int.from_bytes(hashlib.sha256(f'episodic-refi/{key}'.encode()).digest()[:4], 'little')
        prepared, _ = refresh_context_case(model, dict(blocks=block, stats=case['stats']), weights, task['obs'], seed)
        prepared['numerical_key'] = key
        ep['cases'][ap] = prepared
    return ep


def attach_queries(episode, query, support_positions):
    """Whole-coordinate support/query isolation, followed by original readout."""
    assert not (query['positions'][:, None] == support_positions[None]).all(-1).any()
    set_queries(episode, query)
    return episode
