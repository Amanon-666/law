"""The frozen original 54 events, with original hierarchical error weights.

IMPORTANT: this hash-bound manifest contains ONE-based CSV row IDs, as does
data.load('uji').floor.row_ids. Both are converted to zero-based IDs before
lookup; truth exports zero-based CSV IDs, never floor-local row indices.
repeat 0/1 means AB/BA. No new sampling, averaging or history copying.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from .task import Task


MANIFEST_SHA256 = 'bb7db6eb8ce070a400f18057af46dc68f734f5db6f60235f4c60189e62d3c496'
CSV_SHA256 = '45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168'


def load_manifest(path):
    """Read the existing protocol only; reject another or edited manifest."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Expected the original building-information 54-event manifest')
    manifest = json.loads(raw)
    if manifest['csv_sha256'] != CSV_SHA256:
        raise ValueError('Legacy54 source CSV identity mismatch')
    return manifest


def make_tasks(manifest, building, floors, ap_ids, *, budgets, repeats, selected_floors=None):
    """Return original-order (Task, truth) pairs for a requested legal subset.

    H values reference the supplied full floor objects. Query coordinates and
    their groups are only in truth. Completeness of an aggregate refers to the
    records requested by the caller, not an assumption that all 54 were run.
    """
    budgets, repeats = set(budgets), set(repeats)
    if not budgets or not budgets <= {1, 3, 5} or not repeats or not repeats <= {0, 1}:
        raise ValueError('Legacy54 requires K in {1,3,5}, repeat in {0:AB,1:BA}')
    if building not in manifest['buildings']:
        raise ValueError('Legacy54 contains only B1 and B2')
    if list(ap_ids) != [f'WAP{i:03}' for i in range(1, 521)]:
        raise ValueError('Legacy54 requires the original ordered 520-AP axis')
    csv_path = Path(next(iter(floors.values()))['meta']['source'])
    with csv_path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != CSV_SHA256:
            raise ValueError('Loaded UJI CSV differs from the original legacy54 corpus')
    names = [f for f in manifest['floors'] if f.startswith(building+'F')]
    physical = {int(name.rsplit('F', 1)[1]): name for name in names}
    selected = set(physical) if selected_floors is None else set(selected_floors)
    if not selected <= set(physical):
        raise ValueError('Requested floor is absent from the original manifest')
    lookup = {}
    for floor, name in physical.items():
        ids = np.asarray(floors[floor]['row_ids'], dtype=np.int64)
        if ids.tolist() != manifest['history_rows'][name]:
            raise ValueError(f'Legacy54 needs the complete ordered raw floor: {name}')
        lookup[floor] = {int(row): i for i, row in enumerate(ids-1)}
    tasks = []
    for event in manifest['events']:
        floor = int(event['floor'].rsplit('F', 1)[1])
        repeat = {'AB': 0, 'BA': 1}[event['direction']]
        if (event['building'] != building or floor not in selected
                or event['K'] not in budgets or repeat not in repeats):
            continue
        data = floors[floor]
        csv_s = np.asarray(event['support_rows'], dtype=np.int64)-1
        csv_q = np.asarray(event['query_rows'], dtype=np.int64)-1
        si = np.asarray([lookup[floor][int(i)] for i in csv_s])
        qi = np.asarray([lookup[floor][int(i)] for i in csv_q])
        support = {key: data[key][si] for key in ('rss', 'xy', 'acq')}
        qxy = data['xy'][qi]
        positions, groups, counts = np.unique(qxy, axis=0, return_inverse=True, return_counts=True)
        s_positions, s_counts = np.unique(support['xy'], axis=0, return_counts=True)
        pair = manifest['floor_pairs'][event['floor']]
        if not (np.array_equal(positions, np.unique(pair['query_positions'], axis=0))
                and np.array_equal(s_positions, np.unique(pair['support_positions'][:event['K']], axis=0))
                and (counts == event['R']).all() and (s_counts == event['R']).all()):
            raise ValueError(f'Legacy54 row-to-position/repeat binding mismatch: {event["id"]}')
        for rows, expected in ((si, event['support_acquisition']), (qi, event['query_acquisition'])):
            ids = np.unique(data['acq'][rows])
            if (len(ids) != 1 or ids[0] < 0
                    or list(data['meta']['acquisition_identities'][int(ids[0])]) != expected):
                raise ValueError(f'Legacy54 acquisition binding mismatch: {event["id"]}')
        history = {int(name.rsplit('F', 1)[1]): floors[int(name.rsplit('F', 1)[1])]
                   for name in event['histories']}
        meta = dict(protocol='legacy54', legacy_event_id=event['id'], direction=event['direction'],
            seed=manifest['seed'], tabicl_seed=manifest['seed'], manifest_sha256=MANIFEST_SHA256,
            source_csv_path=manifest['csv_path'], source_csv_sha256=CSV_SHA256,
            manifest_row_index_kind='csv_one_based', query_row_index_kind='csv_zero_based',
            R_requested=event['R'], R_limit=event['R'], support_scans_per_position=s_counts.tolist(),
            query_positions=len(positions), query_scans=len(qi), acquisition_separated=True,
            support_acquisition=event['support_acquisition'], query_acquisition=event['query_acquisition'],
            acquisition_definition='USERID,PHONEID,UTC date; observed groups, not certified independent replicates',
            evaluation_status='exposed_development', legacy_event_weight=.5/len(names)/2/3,
            legacy_floors_in_building=len(names))
        truth = dict(xy=qxy, groups=groups.ravel(), row_indices=csv_q,
                     csv_query_rows=csv_q.copy(), csv_support_rows=csv_s,
                     query_row_index_kind='csv_zero_based', legacy_event_id=event['id'])
        tasks.append((Task('uji', building, floor, event['K'], repeat, list(ap_ids),
                           history, support, data['rss'][qi], meta), truth))
    return tasks


def overall(records):
    """Original equal-building/floor/direction/K/position metric on this subset.

    Each event keeps its original mass .5/n_floors/2/3, divided equally among
    its position-mean errors. Normalize once over completed records; failures
    stay in the denominator of completion coverage and are listed explicitly.
    p50/p90 are discrete weighted quantiles, with no linear interpolation.
    """
    done = [r for r in records if r['status'] == 'completed']
    result = dict(tasks=len(records), completed_tasks=len(done), failed_tasks=len(records)-len(done),
        complete=len(done) == len(records), failed_task_ids=[r['task_id'] for r in records if r['status'] != 'completed'],
        unfinished_optimization_tasks=[r['task_id'] for r in done
            if r.get('diagnostics', {}).get('status') == 'completed_unconverged'],
        mean=None, p50=None, p90=None, query_positions=0)
    if not done:
        return result
    values, weights = [], []
    for record in done:
        errors = np.asarray(record['scores']['position_errors'], dtype=float)
        values.append(errors)
        weights.append(np.full(len(errors), record['meta']['legacy_event_weight']/len(errors)))
    v, w = np.concatenate(values), np.concatenate(weights)
    w /= w.sum()
    order = np.argsort(v)
    cumulative = np.cumsum(w[order])
    result.update(mean=float(w @ v), query_positions=len(v),
        p50=float(v[order[np.searchsorted(cumulative, .5)]]),
        p90=float(v[order[np.searchsorted(cumulative, .9)]]))
    return result


def aggregate(records):
    """Same by-K result structure as score.aggregate, using legacy weights."""
    return {str(k): overall([r for r in records if r['K'] == k])
            for k in sorted({r['K'] for r in records})}
