"""Use existing multi-building source assets for paired training recipes.

The old B0-only protocol and its frozen results stay unchanged. Each update
contains one task per source building. Episodic and ordinary recipes use the
same legal historical corpus; their support/query assignments intentionally
differ. A/B and augmented A+/C share the same schedule and augmentation key.
This module constructs tasks and a source-only empirical shift pool. It does
not load a model, train, select checkpoints, or open a held-out building.
"""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np

from acquisition_tasks import episode_scans
from check_tabicl_history_learning import bounds_and_aps
from source_episode_catalog import make_episode
from uji import group_scans


def building(floor):
    return floor.split('F')[0]


def generator(seed, *parts):
    value = json.dumps([seed, *parts], separators=(',', ':')).encode()
    return random.Random(int.from_bytes(hashlib.sha256(value).digest()[:8], 'little'))


def allowed_rows(catalog):
    """Exact union of the existing protected-position-safe history banks."""
    pool = {f: set() for f in catalog['source_floors']}
    for target, histories in catalog['history_rows'].items():
        for floor, ids in histories.items():
            assert floor != target and building(floor) == building(target)
            pool[floor].update(ids)
    protected = set(catalog['protected_rows'])
    assert all(ids and not ids.intersection(protected) for ids in pool.values())
    return {f: sorted(ids) for f, ids in pool.items()}


def training_schedule(catalog, updates, seed):
    """Balanced building visits; rotate floors and acquisition families.

    Floors are balanced to within one visit at a finite budget. Every family
    receives every K over a full three-family cycle, including when the family
    count is divisible by three. Actual counts are returned, not called equal
    FLOPs, equal query exposure, or independent environments.
    """
    families = defaultdict(list)
    floors = defaultdict(list)
    for family in catalog['families']:
        families[family['floor']].append(family['id'])
    assert set(families) == set(catalog['source_floors'])
    for floor in sorted(families):
        families[floor].sort()
        generator(seed, 'families', floor).shuffle(families[floor])
        floors[building(floor)].append(floor)
    buildings = sorted(floors)
    assert set(buildings) == set(catalog['contract']['source_buildings'])
    assert catalog['contract']['held_out_building'] not in buildings
    for b in buildings:
        generator(seed, 'floors', b).shuffle(floors[b])
    visits = Counter()
    schedule = []
    for step in range(updates):
        batch = []
        for b in buildings:
            floor = floors[b][step % len(floors[b])]
            visit = visits[floor]
            ids = families[floor]
            phase = visit // len(ids) if len(ids) % 3 == 0 else 0
            batch.append(dict(building=b, floor=floor,
                family_id=ids[visit % len(ids)], draw_index=step,
                k=(1, 3, 5)[(visit+phase) % 3], phase='fit',
                task_key=f'seed{seed}:step{step}:{b}:{floor}',
                loss_weight=1/len(buildings)))
            visits[floor] += 1
        schedule.append(batch)
    counts = {f: dict(visits=visits[f], by_K={str(k): sum(
        s['floor'] == f and s['k'] == k for batch in schedule for s in batch)
        for k in (1, 3, 5)}) for f in sorted(families)}
    return schedule, counts


def episodic_event(catalog, spec):
    event = make_episode(catalog, **{k: spec[k] for k in ('family_id', 'draw_index', 'k', 'phase')})
    assert event['floor'] == spec['floor']
    event['building'] = spec['building']
    event['task_key'] = spec['task_key']
    permitted = allowed_rows(catalog)
    assert set(event['support_rows']+event['query_rows']) <= set(permitted[event['floor']])
    return event


def episodic_pack(rows, catalog, spec):
    event = episodic_event(catalog, spec)
    p = episode_scans(rows, event, np.zeros(2))
    bounds, aps = bounds_and_aps(p['histories'], p['support'])
    return dict(p, event=event, floor=event['floor'], bounds=bounds, aps=aps,
                query_rows=event['query_rows'], loss_weight=spec['loss_weight'])


def ordinary_event(rows, catalog, spec, seed):
    """Original dense 80/20 whole-position recipe, within one building.

    All acquisition scans of a position remain together. Each source building
    supplies one selected target-floor loss per update, like the episodic arm;
    its dense source supervision and compute are measured separately.
    """
    permitted = allowed_rows(catalog)
    floors = sorted(f for f in permitted if building(f) == spec['building'])
    reserved = {f: {tuple(x) for x in catalog['protected_positions'][f]} for f in floors}
    contexts, queries = {}, {}
    for floor in floors:
        groups = defaultdict(list)
        for row_id in permitted[floor]:
            row = rows[row_id]
            assert row['floor'] == floor and row['xy'] not in reserved[floor]
            groups[row['xy']].append(row_id)
        positions = sorted(groups)
        chosen = set(generator(seed, 'ordinary', spec['task_key'], floor).sample(
            positions, math.ceil(.2*len(positions))))
        contexts[floor] = sorted(i for xy, ids in groups.items() if xy not in chosen for i in ids)
        queries[floor] = sorted(i for xy, ids in groups.items() if xy in chosen for i in ids)

    floor = spec['floor']
    histories = [f for f in floors if f != floor]
    event = dict(id='ordinary_'+spec['task_key'], role='ordinary_source_fit',
        building=spec['building'], floor=floor, task_key=spec['task_key'],
        histories=sorted(histories), history_rows={f: contexts[f] for f in histories},
        support_rows=contexts[floor], query_rows=queries[floor],
        context_kind='dense_random_context_not_deployment_K')
    return event


def ordinary_pack(rows, catalog, spec, seed):
    event = ordinary_event(rows, catalog, spec, seed)

    def collect(ids):
        return group_scans(np.asarray([rows[i]['rss'] for i in ids], dtype=np.int16),
                           np.asarray([rows[i]['xy'] for i in ids], dtype=np.float64))

    histories = {f: collect(ids) for f, ids in event['history_rows'].items()}
    support, query = collect(event['support_rows']), collect(event['query_rows'])
    bounds, aps = bounds_and_aps(histories, support)
    return dict(histories=histories, support=support, query=query, event=event,
        floor=event['floor'], bounds=bounds, aps=aps, query_rows=event['query_rows'],
        loss_weight=spec['loss_weight'])


def empirical_target_shifts(rows, catalog):
    """Finite source-only augmentation pool, not a future-change prior.

    For each unordered observed acquisition pair, use common nonprotected R3
    positions and APs detected in all six scans. Take the median AP mean
    difference at each position, then the median over positions. Five common
    usable positions are required, matching the maximum support budget; this
    is an explicit coverage rule, not an uncertainty guarantee. Signed copies
    and zero form the pool for that source building. The same target shift is
    later applied to S/Q; histories retain their real source differences.
    """
    permitted = {f: set(ids) for f, ids in allowed_rows(catalog).items()}
    pairs = sorted({tuple(sorted((f['support_group'], f['query_group'])))
                    for f in catalog['families']})
    records, excluded = [], []
    for left, right in pairs:
        a, b = catalog['groups'][left], catalog['groups'][right]
        floor = a['floor']
        assert floor == b['floor']
        maps = [{tuple(p['xy']): p['rows'] for p in g['train']} for g in (a, b)]
        shifts, ap_counts = [], []
        for xy in sorted(maps[0].keys() & maps[1].keys()):
            ids_a, ids_b = maps[0][xy], maps[1][xy]
            assert set(ids_a+ids_b) <= permitted[floor]
            x, y = (np.asarray([rows[i]['rss'] for i in ids], dtype=float) for ids in (ids_a, ids_b))
            detected = (x != 100).all(0) & (y != 100).all(0)
            if detected.any():
                shifts.append(float(np.median(y[:, detected].mean(0)-x[:, detected].mean(0))))
                ap_counts.append(int(detected.sum()))
        identity = dict(floor=floor, building=building(floor), groups=[left, right],
                        usable_positions=len(shifts))
        if len(shifts) < 5:
            excluded.append(dict(identity, reason='fewer than five common fully detected R3 positions'))
        else:
            records.append(dict(identity, delta_db=float(np.median(shifts)),
                APs_per_position_min=min(ap_counts), APs_per_position_median=float(np.median(ap_counts))))
    by_building = {}
    for b in catalog['contract']['source_buildings']:
        local = [r for r in records if r['building'] == b]
        assert local, f'No legal observed shift pool for source building {b}'
        pool = [0.] + [sign*r['delta_db'] for r in local for sign in (1., -1.)]
        by_building[b] = dict(pool_db=pool, acquisition_pairs=len(local),
            floors=sorted({r['floor'] for r in local}),
            max_abs_db=max(abs(v) for v in pool))
    return dict(by_building=by_building, records=records, excluded=excluded,
        scope='Only source-building nonprotected rows; observed paired differences, not identifiable device or persistent-environment effects',
        use='Finite symmetric target-only shift pool, conditional on the source training building; no held-out-building calibration',
        limitations='Pairs can share acquisitions; not independent environments. Detection masks stay fixed; no missing-value reconstruction or coverage simulation.')
