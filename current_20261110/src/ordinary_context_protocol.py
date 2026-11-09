"""Conventional random-context source fine-tuning on the full24 legal corpus.

This protocol changes only source context/query assignment. It creates no
model, wrapper, optimizer, file or prediction. Call original fit/backward once
per returned floor task, accumulate all three gradients and divide by three
before the optimizer step. The existing backward keeps both coordinate axes,
all eight official views and equal query-position weights.

The permitted pool is the union of catalog historical rows and the ACTUAL
support/fit-query rows of the original 24-step full24 schedule. This avoids
silently dropping authorized supervised examples if the older history asset
does not cover them, without expanding to every acquisition in the CSV.
Protected rows AND protected (floor,XY) positions are excluded from all roles.

Each draw partitions each floor's unique positions: ceil(20% of positions)
are query, all remaining positions are context. All allowed scans at a
position, across acquisitions, stay together. Each floor's query sees the
same pooled three-floor context, expressed as other-floor histories plus
its own-floor support so the original physical-floor feature remains valid.
AP selection, bounds and fitted transforms depend on context only.

The dense own-floor context is ordinary supervised-domain training, NOT a
K1/K3/K5 deployment episode or acquisition-held-out training. Deployment
continues to use original H plus the actual small S. Compared with full24,
this identifies the whole task-construction recipe, not separate effects of
cross-floor and cross-acquisition sampling. Equal update counts are not equal
compute or query exposure; callers must report actual cost and exposure.
"""
import hashlib
import json
import math
import random
from collections import defaultdict

import numpy as np

from check_tabicl_history_learning import bounds_and_aps
from source_episode_catalog import make_episode as catalog_episode
from tabicl_multitask_full_trial import events_for
from uji import group_scans


def allowed_rows(catalog):
    """Return the exact original training-corpus union using metadata only.

    `fit_extra_rows` explicitly records fit S/Q not already in the history
    union. Its contents are computed when the caller runs this on the server;
    no claim that it is empty is assumed from the catalog's construction.
    """
    floors = sorted(catalog['source_floors'])
    assert floors == ['B0F0', 'B0F1', 'B0F2']
    historical = {floor: set() for floor in floors}
    for bank in catalog['history_rows'].values():
        for floor, ids in bank.items():
            historical[floor].update(ids)
    fit_rows = {floor: set() for floor in floors}
    schedule, _ = events_for(catalog)
    assert len(schedule) == 24
    for batch in schedule:
        for spec in batch:
            assert spec['phase'] == 'fit'
            event = catalog_episode(catalog, **spec)
            fit_rows[event['floor']].update(event['support_rows'])
            fit_rows[event['floor']].update(event['query_rows'])
    protected = set(catalog['protected_rows'])
    pool = {floor: sorted(historical[floor] | fit_rows[floor]) for floor in floors}
    assert all(not protected.intersection(ids) for ids in pool.values())
    extra = {floor: sorted(fit_rows[floor]-historical[floor]) for floor in floors}
    return dict(by_floor=pool,
                history_only_rows={floor: sorted(ids) for floor, ids in historical.items()},
                fit_extra_rows=extra, history_union_covers_fit=not any(extra.values()))


def make_episode(rows, catalog, draw_index, seed):
    """Return three original-model packs and their complete row provenance.

    rows uses acquisition_tasks.load_acquisition_rows format. No protected RSS
    is accessed: only allowed rows are collected. Positions are absolute and
    use the original source coordinate frame (origin zero).

    For each p in result['tasks']:
        model.fit(p['histories'], p['support'], p['floor'],
                  p['aps'], p['bounds'], preprocessing_seed)
        model.backward(p['query']['rss'], p['query']['xy'], scaler)
    The caller averages these three episode gradients, keeping the shared
    model/optimizer across draws. `seed` controls only this metadata split;
    the original preprocessing seed is a separate, unchanged caller setting.
    """
    assert draw_index >= 0
    permissions = allowed_rows(catalog)
    floors = sorted(permissions['by_floor'])
    protected = {floor: {tuple(xy) for xy in catalog['protected_positions'][floor]}
                 for floor in floors}
    context_rows, query_rows, split = {}, {}, {}
    for floor in floors:
        groups = defaultdict(list)
        for row_id in permissions['by_floor'][floor]:
            row = rows[row_id]
            assert row['floor'] == floor
            xy = tuple(row['xy'])
            assert xy not in protected[floor]
            groups[xy].append(row_id)
        positions = sorted(groups)
        assert len(positions) >= 2
        identity = json.dumps(['ordinary_context', int(seed), int(draw_index), floor],
                              separators=(',', ':')).encode()
        split_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], 'little')
        rng = random.Random(split_seed)
        n_query = math.ceil(len(positions)*.2)
        selected = set(rng.sample(positions, n_query))
        # Canonical row order, not acquisition-driven order; every real row is
        # retained exactly once. Splitting never examines RSS or query scores.
        context_rows[floor] = sorted(i for xy, ids in groups.items() if xy not in selected for i in ids)
        query_rows[floor] = sorted(i for xy, ids in groups.items() if xy in selected for i in ids)
        split[floor] = dict(context_positions=len(positions)-n_query, query_positions=n_query,
                            context_scans=len(context_rows[floor]), query_scans=len(query_rows[floor]),
                            split_seed=split_seed)

    def collect(ids):
        return group_scans(np.asarray([rows[i]['rss'] for i in ids], dtype=np.int16),
                           np.asarray([rows[i]['xy'] for i in ids], dtype=np.float64))

    contexts = {floor: collect(context_rows[floor]) for floor in floors}
    tasks = []
    for floor in floors:
        histories = {name: contexts[name] for name in floors if name != floor}
        support = contexts[floor]
        query = collect(query_rows[floor])
        bounds, aps = bounds_and_aps(histories, support)
        eid = f'ordinary_context_seed{seed}_draw{draw_index}_{floor}'
        history_rows = {name: context_rows[name] for name in histories}
        event = dict(id=eid, role='ordinary_source_fit', floor=floor,
                     histories=list(histories), history_rows=history_rows,
                     support_rows=context_rows[floor], query_rows=query_rows[floor],
                     context_kind='pooled_dense_random_context_not_deployment_K')
        tasks.append(dict(floor=floor, event=event, histories=histories,
                          support=support, query=query, bounds=bounds, aps=aps,
                          history_rows=history_rows, support_rows=context_rows[floor],
                          query_rows=query_rows[floor]))
    return dict(tasks=tasks, draw_index=int(draw_index), seed=int(seed),
                context_rows=context_rows, query_rows=query_rows, split=split,
                permissions=permissions, floor_loss_weight=1/3)
