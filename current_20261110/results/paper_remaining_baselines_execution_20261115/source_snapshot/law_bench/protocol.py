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
    """Sample positions first; acquisition preference cannot change coverage."""
    all_groups = np.unique(groups)
    order = rng.permutation(all_groups)
    s_chosen, q_chosen = order[n_query:n_query + max_k], order[:n_query]

    def common_acquisitions(chosen):
        return set.intersection(*(set(acquisitions[groups == g].tolist()) - {-1}
                                  for g in chosen))

    s_available = common_acquisitions(s_chosen)
    q_available = common_acquisitions(q_chosen)
    pairs = [(s, q) for s in sorted(s_available) for q in sorted(q_available) if s != q]
    if pairs:
        s_acq, q_acq = pairs[int(rng.integers(len(pairs)))]
        return s_chosen, q_chosen, s_acq, q_acq, True
    return s_chosen, q_chosen, None, None, False


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
    Uniformity is over recorded positions/bins, not over floor area. Sampled
    positions are never redrawn to improve geometric coverage or error scores.
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
    s_groups, q_groups, s_acq, q_acq, pair_available = _partition(
        groups, data['acq'], n_query, budgets[-1], rng)
    s_rows, _, counts = _rows(data, groups, s_groups, s_acq, repeats, rng)
    q_rows, q_labels, _ = _rows(data, groups, q_groups, q_acq, repeats, rng)
    history = {f: floors[f] for f in sorted(floors) if f != floor}
    truth = dict(xy=data['xy'][q_rows], groups=q_labels, row_indices=q_rows)
    tasks = []
    for k in budgets:
        selected = s_rows[:sum(counts[:k])]
        support = {name: data[name][selected] for name in ('rss', 'xy', 'acq')}
        s_ids, q_ids = set(support['acq'].tolist()), set(data['acq'][q_rows].tolist())
        separated = -1 not in s_ids | q_ids and not s_ids & q_ids
        task_meta = dict(protocol='static-resample-v2', seed=split_seed,
            R_requested=R, R_limit=repeats, support_scans_per_position=counts[:k],
            query_positions=n_query, query_scans=len(q_rows), split_grid_m=grid,
            acquisition_separated=separated,
            sampling='recorded_positions_first; acquisition_preference_preserves_positions',
            acquisition_pair_available=pair_available,
            support_acquisition_ids=sorted(s_ids), query_acquisition_ids=sorted(q_ids),
            support_acquisition=None if s_acq is None else int(s_acq),
            query_acquisition=None if q_acq is None else int(q_acq),
            evaluation_status=meta.get('evaluation_status', 'not_declared'))
        tasks.append((Task(dataset, building, int(floor), k, repeat, list(ap_ids),
                           history, support, data['rss'][q_rows], task_meta), truth))
    return tasks
