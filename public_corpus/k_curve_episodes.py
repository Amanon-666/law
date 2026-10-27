"""Fixed-Q, nested-S source episodes for a K curve; no model execution."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import socket
from zoneinfo import ZoneInfo

import numpy as np

from data import group_scans
from public_corpus import probe_episodes as old


ROOT = old.ROOT
OUT = ROOT / 'results/k_curve_v1_20261027'
KS = (1, 3, 5, 10, 20)
SEEDS = (261026, 261027, 261028)
Q_COUNT = 12


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def history_rows(name, histories):
    return {floor: np.asarray(old.h_selection(name, floor)['rows'], dtype=int)
            for floor in histories}


def grid_definition(xy, boundary_source):
    low = np.floor(xy.min(0) / 2) * 2
    high = np.ceil(xy.max(0) / 2) * 2
    shape = np.rint((high - low) / 2).astype(int) + 1
    return dict(lower=low.tolist(), upper=high.tolist(), spacing=2.,
                shape_xy=shape.tolist(), points=int(np.prod(shape)),
                flatten_order='meshgrid(indexing=xy), C-order; y varies fastest',
                boundary_source=boundary_source, query_involved=False,
                coordinate_frame='raw public frame minus H origin')


def history_frame(building, histories, rows):
    xy = np.concatenate([building['floors'][h]['xy'][rows[h]] for h in histories])
    origin = xy.mean(0)
    shifted = xy - origin
    return origin, shifted, grid_definition(shifted, 'retained H positions only; outward 2-unit rounding')


def outside_grid(xy, grid):
    low, high = np.asarray(grid['lower']), np.asarray(grid['upper'])
    return ((xy < low) | (xy > high)).any(1)


def split_rows(target_floor, seed):
    """Reproduce old K5 row draws, then add fifteen nested S rows."""
    positions, group = np.unique(target_floor['xy'], axis=0, return_inverse=True)
    if len(positions) < 32:
        raise ValueError(f'Need 32 distinct target positions, found {len(positions)}')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(positions))
    first, query, extra = order[:5], order[5:17], order[17:32]

    def draw(indices):
        return np.asarray([rng.choice(np.flatnonzero(group == g)) for g in indices], dtype=int)

    s5 = draw(first)
    q12 = draw(query)
    s15 = draw(extra)
    return np.concatenate((first, extra)), query, np.concatenate((s5, s15)), q12


def selected_aps(building, target, histories, rows, support_rows):
    floors = building['floors']
    seen = (floors[target]['rss'][support_rows] != 100).any(0)
    for h in histories:
        seen |= (floors[h]['rss'][rows[h]] != 100).any(0)
    return np.flatnonzero(seen)


def make_manifest():
    old_manifest = json.loads((old.PREPARATION / 'manifest.json').read_text())
    anchors = {(e['building'], e['seed'], len(e['histories'])): e
               for e in old_manifest['events']}
    params = json.loads(old.PARAMETERS.read_text())
    events = []
    for name, (target, available) in old.CONFIGS.items():
        histories = sorted(available)
        building = old.building_data(name)
        floor = building['floors'][target]
        rows = history_rows(name, histories)
        origin, hxy, history_grid = history_frame(building, histories, rows)
        sources = {str(h): {k: old.h_selection(name, h)[k]
                            for k in ('cache', 'cache_origin', 'rows', 'scans', 'positions')}
                   for h in histories}
        for seed in SEEDS:
            s_positions, q_positions, s_rows, q_rows = split_rows(floor, seed)
            for k in KS:
                sr = s_rows[:k]
                ap_indices = selected_aps(building, target, histories, rows, sr)
                sxy = floor['xy'][sr] - origin
                grid = grid_definition(np.concatenate((hxy, sxy)),
                                       'retained H positions union current K Support only; outward 2-unit rounding')
                qxy = floor['xy'][q_rows] - origin
                outside_h = outside_grid(qxy, history_grid)
                outside = outside_grid(qxy, grid)
                entry = dict(id=f'{name}_F{target}_s{seed}_K{k}', building=name,
                             target=target, histories=histories, seed=seed, K=k, R=1,
                             Q_positions=Q_COUNT, support_position_indices=s_positions[:k].tolist(),
                             query_position_indices=q_positions.tolist(),
                             support_rows=sr.tolist(), query_rows=q_rows.tolist(),
                             support_scan_ids=floor['scan_ids'][sr].tolist(),
                             query_scan_ids=floor['scan_ids'][q_rows].tolist(),
                             AP_indices=ap_indices.tolist(), AP_ids=building['ap_ids'][ap_indices].tolist(),
                             origin=origin.tolist(), history_grid=history_grid, grid=grid,
                             query_outside_H_grid_positions=int(outside_h.sum()),
                             query_outside_H_grid_indices=np.flatnonzero(outside_h).tolist(),
                             query_outside_grid_positions=int(outside.sum()),
                             query_outside_grid_indices=np.flatnonzero(outside).tolist(),
                             history_metadata=sources,
                             coordinate_unit='UJI original coordinate distance' if name.startswith('UJI_') else 'metres',
                             position_kind=building['metadata']['position_kind'])
                if seed in old.SEEDS and k == 5:
                    anchor = anchors[name, seed, len(histories)]
                    if entry['support_scan_ids'] != anchor['support_scan_ids'] or \
                       entry['query_scan_ids'] != anchor['query_scan_ids'] or \
                       entry['AP_indices'] != anchor['AP_indices']:
                        raise AssertionError(f'K5 rows or APs changed for {name}, seed {seed}')
                events.append(entry)
    return dict(events=events, K=list(KS), seeds=list(SEEDS), R=1, Q_positions=Q_COUNT,
                base=params['base'], obs=params['obs'], parameters=str(old.PARAMETERS),
                source_preparation=str(old.PREPARATION / 'manifest.json'),
                design='All listed H floors sorted; old K5 S/Q rows; fixed Q; nested S; AP=H union current S; H-only origin and diagnostic history_grid; formal grid=H union current K S',
                query_RSS_used_for_split_AP_grid=False, source_development_only=True)


def load_manifest(path=OUT / 'manifest.json'):
    return json.loads(Path(path).read_text())


def build_task(meta):
    """Rebuild H/S/Q arrays from manifest rows; H cache paths are in metadata."""
    name, target, histories = meta['building'], meta['target'], meta['histories']
    building = old.building_data(name)
    floors = building['floors']
    rows = {h: np.asarray(meta['history_metadata'][str(h)]['rows'], dtype=int)
            for h in histories}
    s_rows = np.asarray(meta['support_rows'], dtype=int)
    q_rows = np.asarray(meta['query_rows'], dtype=int)
    ap_indices = selected_aps(building, target, histories, rows, s_rows)
    if ap_indices.tolist() != meta['AP_indices']:
        raise AssertionError(f'AP catalogue changed: {meta["id"]}')
    origin, hxy, history_grid = history_frame(building, histories, rows)
    grid = grid_definition(np.concatenate((hxy, floors[target]['xy'][s_rows] - origin)),
                           'retained H positions union current K Support only; outward 2-unit rounding')
    if history_grid != meta['history_grid'] or grid != meta['grid'] or \
       origin.tolist() != meta['origin']:
        raise AssertionError(f'H/S frame changed: {meta["id"]}')

    def scans(floor, selected):
        raw = floors[floor]
        out = group_scans(raw['rss'][selected][:, ap_indices], raw['xy'][selected] - origin)
        out.update(scan_ids=raw['scan_ids'][selected], raw_latlon=raw['raw_latlon'][selected],
                   ap_namespace=building['metadata']['ap_namespace'], row_indices=selected,
                   ap_ids=building['ap_ids'][ap_indices])
        return out

    support, query = scans(target, s_rows), scans(target, q_rows)
    query['scan_weights'] = 1 / (Q_COUNT * query['stats']['n'][query['group']])
    return dict(H=[scans(h, rows[h]) for h in histories], S=support, Q=query,
                target=target, histories=histories, dz=[h - target for h in histories],
                ap_indices=ap_indices, ap_ids=building['ap_ids'][ap_indices],
                metadata=dict(id=meta['id'], K=meta['K'], requested_R=1, actual_R=1,
                              origin=origin.tolist(), grid=grid,
                              ap_selection='retained H union current S; Q signal excluded'))


def check_and_write():
    old.verify_freeze(old.PREPARATION)
    manifest = make_manifest()
    by_split = {}
    for event in manifest['events']:
        by_split.setdefault((event['building'], event['seed']), []).append(event)
    if len(manifest['events']) != 60 or len(by_split) != 12:
        raise AssertionError('Expected 60 K cases from 12 building/seed splits')
    for group in by_split.values():
        group.sort(key=lambda e: e['K'])
        assert [e['K'] for e in group] == list(KS)
        for smaller, larger in zip(group, group[1:]):
            assert smaller['support_rows'] == larger['support_rows'][:smaller['K']]
            assert smaller['support_scan_ids'] == larger['support_scan_ids'][:smaller['K']]
            assert smaller['query_rows'] == larger['query_rows']
            assert smaller['query_scan_ids'] == larger['query_scan_ids']
            assert smaller['history_grid'] == larger['history_grid']
            assert all(a >= b for a, b in zip(smaller['grid']['lower'], larger['grid']['lower']))
            assert all(a <= b for a, b in zip(smaller['grid']['upper'], larger['grid']['upper']))
            assert set(smaller['AP_indices']) <= set(larger['AP_indices'])
        for event in group:
            assert not set(event['support_rows']) & set(event['query_rows'])
            assert not set(event['support_position_indices']) & set(event['query_position_indices'])
            assert len(event['AP_indices']) > 0
    for name, (target, available) in old.CONFIGS.items():
        for h in available:
            cache = Path(old.h_selection(name, h)['cache'])
            aps = {ap for event in manifest['events'] if event['building'] == name
                   for ap in event['AP_indices']}
            missing = [ap for ap in aps if not (cache / f'ap{ap}.npz').exists()]
            if missing:
                raise FileNotFoundError(f'{name} F{h} missing {len(missing)} AP caches')
    source_paths = [Path(__file__), Path(old.__file__), Path(old.source_task.__code__.co_filename),
                    ROOT / 'data.py', ROOT / 'history_bank.py', ROOT / 'transfer.py',
                    ROOT / 'inference.py', ROOT / 'reference_pmc.py', ROOT / 'run.py',
                    old.PARAMETERS, old.PREPARATION / 'manifest.json',
                    old.PREPARATION / 'freeze.json']
    source_paths += [old.DATA / 'trainingData.csv',
                     old.PUBLIC_BANK / 'freeze.json']
    for name, (target, available) in old.CONFIGS.items():
        if name.startswith('UJI_'):
            source_paths += [old.UJI_BANK / f'B{name[-1]}F{h}' / 'target.npz' for h in available]
        else:
            source_paths += [old.PUBLIC_DATA / name / f'F{f}.npz' for f in [target, *available]]
            source_paths += [old.PUBLIC_DATA / name / 'metadata.json']
            source_paths += [old.PUBLIC_BANK / name / f'F{h}' / 'selection.json' for h in available]
    previous_manifest = OUT / 'manifest.json'
    previous_check = OUT / 'data_check.json'
    if previous_manifest.exists():
        old_events = json.loads(previous_manifest.read_text())['events']
        for before, after in zip(old_events, manifest['events']):
            for key in ('id', 'histories', 'support_rows', 'query_rows', 'AP_indices',
                        'origin', 'query_outside_H_grid_positions',
                        'query_outside_H_grid_indices'):
                assert before[key] == after[key], (after['id'], key)
            assert before['grid'] == after['history_grid'], after['id']
    previous_sha = digest(previous_manifest) if previous_manifest.exists() else None
    previous_time = (datetime.fromtimestamp(previous_manifest.stat().st_mtime,
                                            ZoneInfo('Asia/Singapore')).isoformat()
                     if previous_manifest.exists() else None)
    prior_check = json.loads(previous_check.read_text()) if previous_check.exists() else {}
    check = dict(events=60, splits=12, K=list(KS), nested_S=True, fixed_Q=True,
                 disjoint_S_Q=True, old_K5_scan_IDs_equal=True,
                 history_grid_same_across_K=True, current_H_S_grid_only_expands=True,
                 complete_H_union_current_S_AP=True, Q_RSS_used_for_selection=False,
                 query_outside_H_grid_by_event={e['id']: e['query_outside_H_grid_positions']
                                                for e in manifest['events']},
                 query_outside_grid_by_event={e['id']: e['query_outside_grid_positions']
                                              for e in manifest['events']},
                 previous_manifest_sha256=previous_sha,
                 previous_manifest_modified_at=previous_time,
                 revised_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
                 revision_reason='Before any model scoring, Pisa Q positions outside the H-only grid exposed an S-coverage restriction. Formal grid now uses retained H plus current K Support positions; H-only origin and diagnostic history_grid remain fixed.',
                 source_sha256={str(p): digest(p) for p in source_paths},
                 history_cache_source='Existing UJI/public H caches selected by probe_episodes.h_selection; no H refits',
                 model_source='Existing history_bank/transfer/inference/reference_pmc; no model run',
                 data_rights='Raw target training-floor positions and current S RSS select rows/AP; H plus current S coordinates set grid; Q positions/IDs fixed before K and used only for evaluation-domain diagnosis; Q RSS never selects AP/origin/grid; R=1 walking',
                 training=False, localization=False, model_scoring=False)
    if 'transfer' in prior_check:
        check['transfer'] = prior_check['transfer']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'data_check.json').write_text(json.dumps(check, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(events=60, splits=12,
                          AP_range=[min(len(e['AP_indices']) for e in manifest['events']),
                                    max(len(e['AP_indices']) for e in manifest['events'])],
                          query_outside_H_cases=sum(e['query_outside_H_grid_positions'] > 0
                                                    for e in manifest['events']),
                          query_outside_formal_cases=sum(e['query_outside_grid_positions'] > 0
                                                         for e in manifest['events']))), flush=True)


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Source data check stays on ray-web')
    check_and_write()
