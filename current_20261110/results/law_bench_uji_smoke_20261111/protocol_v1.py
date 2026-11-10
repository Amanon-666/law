"""Nested calibration budgets with Query coordinates kept outside Method Task.

Spatial bins only split trajectory receipts. Each selected bin contributes one
actual coordinate and at most R scans at that exact coordinate; no averaging,
coordinate snapping or synthetic repeat measurements. H remains all raw scans
from the other floors. This is a static resampling protocol, not a trajectory
tracking or official IPIN scoring protocol.
"""
import hashlib

import numpy as np

from .task import Task


def _partition(groups, acquisitions, n_query, max_k, rng):
    """Prefer two known acquisitions if both position budgets can be met."""
    all_groups = np.unique(groups)
    known = np.unique(acquisitions[acquisitions != -1])
    pairs = [(s, q) for s in known for q in known if s != q]
    for index in rng.permutation(len(pairs)):
        s_acq, q_acq = pairs[index]
        s_pool = set(np.unique(groups[acquisitions == s_acq]).tolist())
        q_pool = np.unique(groups[acquisitions == q_acq])
        if len(s_pool) < max_k or len(q_pool) < n_query:
            continue
        q_chosen = []
        remaining_s = set(s_pool)
        for group in rng.permutation(q_pool):
            if group in remaining_s and len(remaining_s) == max_k:
                continue
            q_chosen.append(int(group))
            remaining_s.discard(group)
            if len(q_chosen) == n_query:
                break
        if len(q_chosen) == n_query:
            s_chosen = rng.permutation(sorted(remaining_s))[:max_k]
            return s_chosen, q_chosen, s_acq, q_acq, True
    order = rng.permutation(all_groups)
    return order[n_query:n_query + max_k], order[:n_query], None, None, False


def _rows(data, groups, chosen, acquisition, repeats, rng):
    rows, labels, counts = [], [], []
    for label, group in enumerate(chosen):
        keep = groups == group
        if acquisition is not None:
            keep &= data['acq'] == acquisition
        candidates = np.flatnonzero(keep)
        positions, inverse = np.unique(data['xy'][candidates], axis=0, return_inverse=True)
        position = int(rng.integers(len(positions)))
        same_position = candidates[inverse.ravel() == position]
        selected = rng.permutation(same_position)[:repeats]
        rows.extend(selected.tolist())
        labels.extend([label] * len(selected))
        counts.append(len(selected))
    return np.asarray(rows, dtype=int), np.asarray(labels, dtype=int), counts


def make_tasks(dataset, building, floors, ap_ids, floor, *, budgets=(1, 3, 5),
               repeat=0, seed=20261111, R=3, query_positions=60):
    """Return [(Task, truth), ...] for one floor/repetition and nested K.

    truth contains xy/groups/row_indices and is passed only to the scorer.
    A floor lacking the requested position budget raises explicitly. Acquisition
    separation is reported, never assumed when acquisition identities are absent.
    """
    budgets = tuple(budgets)
    if not budgets or budgets != tuple(sorted(set(budgets))) or budgets[0] < 1:
        raise ValueError('Budgets must be increasing distinct positive integers')
    if R < 1 or query_positions < 1:
        raise ValueError('R and query_positions must be positive')
    if len(floors) < 2:
        raise ValueError('A deployment needs at least one other history floor')
    data = floors[floor]
    meta = data.get('meta', {})
    repeats = min(R, int(meta.get('max_supported_R', R)))
    grid = meta.get('split_grid_m')
    coordinates = np.floor(data['xy'] / grid).astype(np.int64) if grid else data['xy']
    unique, groups = np.unique(coordinates, axis=0, return_inverse=True)
    groups = groups.ravel()
    n_query = min(query_positions, len(unique) // 2)
    if n_query < 1 or len(unique) < n_query + budgets[-1]:
        raise ValueError(f'Insufficient positions: {len(unique)} for Q={n_query}, K={budgets[-1]}')
    key = f'{seed}/{dataset}/{building}/F{floor}/r{repeat}'
    split_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(split_seed)
    s_groups, q_groups, s_acq, q_acq, separated = _partition(
        groups, data['acq'], n_query, budgets[-1], rng)
    s_rows, _, counts = _rows(data, groups, s_groups, s_acq, repeats, rng)
    q_rows, q_labels, _ = _rows(data, groups, q_groups, q_acq, repeats, rng)
    history = {f: floors[f] for f in sorted(floors) if f != floor}
    truth = dict(xy=data['xy'][q_rows], groups=q_labels, row_indices=q_rows)
    tasks = []
    for k in budgets:
        selected = s_rows[:sum(counts[:k])]
        support = {name: data[name][selected] for name in ('rss', 'xy', 'acq')}
        task_meta = dict(protocol='static-resample-v1', seed=split_seed,
            R_requested=R, R_limit=repeats, support_scans_per_position=counts[:k],
            query_positions=n_query, query_scans=len(q_rows), split_grid_m=grid,
            acquisition_separated=separated,
            support_acquisition=None if s_acq is None else int(s_acq),
            query_acquisition=None if q_acq is None else int(q_acq),
            evaluation_status=meta.get('evaluation_status', 'not_declared'))
        tasks.append((Task(dataset, building, int(floor), k, repeat, list(ap_ids),
                           history, support, data['rss'][q_rows], task_meta), truth))
    return tasks
