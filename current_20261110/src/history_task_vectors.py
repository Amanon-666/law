"""Source-supported target AP-vector augmentation; CPU-only profiling CLI.

Whole, AP-identified same-floor acquisition contrasts replace the scalar
offset. Each retained AP has >=5 same-XY positions with all six R3 scans
detected. Unsupported APs receive no artificial change, which is not evidence
that their real shift is zero. Draw one vector measured on the same target floor and
one whole-vector sign; never resample or permute AP components. Apply it to
detected S/Q entries, retain H RSS, and reuse the frozen common rigid transform.

Repeated locations cancel a stable common spatial map only under that
assumption. Contrasts still include acquisition/device effects and detection
selection; they do not identify a persistent floor effect shared by independent
S/Q acquisitions. Added perturbations can widen existing domain differences.
This is a source-supported regularizer, not a calibrated future-shift law,
proof of S necessity, or a replacement for the original consumer/inference.

The CLI reads only the existing nonprotected B0/B2 row whitelist, profiles the
frozen schedule, and checks two actual packs on CPU. No model, backward,
optimizer, localization prediction, protected data or B1 data is used.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import time

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[_name] = '2'

import numpy as np

from history_task_augmentation import apply_transform, sample_transform
from history_task_protocol import allowed_rows, building, episodic_event
from uji import group_scans

ROOT = Path(__file__).resolve().parent
PRE = ROOT/'results/history_task_preflight_20261110'
OUT = ROOT/'results/history_task_vector_profile_20261110_floor_matched'
KIND = 'whole_source_acquisition_AP_vector_v2_floor_matched'


def build_vector_profile(rows, catalog):
    """Return sparse whole-vector records on the original 520-WAP identity axis."""
    permitted = {f: set(ids) for f, ids in allowed_rows(catalog).items()}
    pairs = sorted({tuple(sorted((f['support_group'], f['query_group'])))
                    for f in catalog['families']})
    records, excluded = [], []
    for left, right in pairs:
        a, b = catalog['groups'][left], catalog['groups'][right]
        floor = a['floor']
        assert floor == b['floor']
        maps = [{tuple(p['xy']): p['rows'] for p in g['train']} for g in (a, b)]
        differences = [[] for _ in range(520)]
        for xy in sorted(maps[0].keys() & maps[1].keys()):
            ids_a, ids_b = maps[0][xy], maps[1][xy]
            assert len(ids_a) == len(ids_b) == 3
            assert set(ids_a+ids_b) <= permitted[floor]
            x, y = (np.asarray([rows[i]['rss'] for i in ids], dtype=float) for ids in (ids_a, ids_b))
            detected = (x != 100).all(0) & (y != 100).all(0)
            delta = y.mean(0)-x.mean(0)
            for ap in np.flatnonzero(detected):
                differences[ap].append(float(delta[ap]))
        aps = [a for a, values in enumerate(differences) if len(values) >= 5]
        identity = dict(floor=floor, building=building(floor), groups=[left, right],
                        common_R3_positions=len(maps[0].keys() & maps[1].keys()))
        if not aps:
            excluded.append(dict(identity, reason='No AP with five fully detected common R3 positions'))
            continue
        delta = np.asarray([np.median(differences[a]) for a in aps])
        scalar = float(np.median(delta))
        residual = delta-scalar
        records.append(dict(identity, id=left+'__'+right, ap_indices=aps,
            delta_db=delta.tolist(), positions_per_ap=[len(differences[a]) for a in aps],
            APs=len(aps), scalar_median_db=scalar,
            residual_RMS_db=float(np.sqrt(np.mean(residual**2))),
            residual_MAD_db=float(np.median(np.abs(residual))),
            AP_std_db=float(np.std(delta))))
    by_building = {}
    for b in catalog['contract']['source_buildings']:
        ids = [i for i, r in enumerate(records) if r['building'] == b]
        assert ids, f'No supported AP vectors in source building {b}'
        by_building[b] = dict(record_indices=ids, vectors=len(ids))
    return dict(kind=KIND, AP_axis='zero-based WAP001..WAP520', records=records,
        excluded=excluded, by_building=by_building,
        sampling='Uniform whole supported vector measured on the selected target floor, then one equiprobable whole-vector sign; no scalar addition or amplitude sampling',
        limitations=['AP-wise fully detected subset; not latent RSS or complete coverage',
            'Pairs and positions are dependent, not future population estimates',
            'Missing evidence means no added perturbation, not zero real bias',
            'S/Q shared increment is an augmentation assumption despite distinct real acquisition groups',
            'Adding any increment can widen existing differences; not a distribution-preserving replacement'])


def sample_vector_transform(history_names, target_floor, profile, *, seed, task_key, coordinates=True):
    """Choose only from frozen source metadata/pool; no S/Q values choose draws."""
    coordinate = sample_transform(history_names, target_floor, seed=seed, task_key=task_key,
        range_evidence='Frozen source same-XY acquisition vector profile; scalar disabled',
        target_shift_pool_db=[0.], coordinates=coordinates)
    payload = json.dumps([KIND, int(seed), task_key, building(target_floor)], separators=(',', ':')).encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little'))
    # AP identities can be floor-specific: a different floor's sparse vector
    # can change no target entries. Match by the prespecified floor metadata,
    # never by query RSS, labels, overlap or scores; Aplus/C keep the same draw.
    indices = [i for i in profile['by_building'][building(target_floor)]['record_indices']
               if profile['records'][i]['floor'] == target_floor]
    assert indices, f'No supported vector for source target floor {target_floor}'
    index = int(rng.choice(indices))
    sign = int(rng.choice([-1, 1]))
    vector = profile['records'][index]
    return dict(kind=KIND, coordinate=coordinate, vector_id=vector['id'],
        source_building=vector['building'], source_floor=vector['floor'], sign=sign, ap_indices=list(vector['ap_indices']),
        target_delta_db=(sign*np.asarray(vector['delta_db'])).tolist(),
        unsupported_AP_rule='No artificial increment', scalar_added_db=0., clipping=False, rounding=False)


def apply_vector_transform(p, record):
    """Fresh fit/backward pack; replay preserves all row and acquisition metadata."""
    assert record['kind'] == KIND and record['source_building'] == building(p['event']['floor'])
    assert record['coordinate']['target_shift_db'] == 0.
    aps = np.asarray(record['ap_indices'], dtype=int)
    delta = np.asarray(record['target_delta_db'], dtype=float)
    assert len(aps) == len(delta) and len(set(aps.tolist())) == len(aps)
    assert np.isfinite(delta).all() and ((aps >= 0) & (aps < 520)).all()
    out = apply_transform(p, record['coordinate'])
    trace = deepcopy(record)
    trace['coordinate'] = out.pop('augmentation')
    if np.any(delta != 0):
        full_delta = np.zeros(520)
        full_delta[aps] = delta
        for role in ('support', 'query'):
            scan = out[role]
            original = np.asarray(scan['rss'])
            detected = original != 100
            rss = np.where(detected, original.astype(float)+full_delta, 100.)
            if not np.isfinite(rss[detected]).all() or np.any(rss[detected] == 100):
                raise ValueError('AP increment collides with missing sentinel or is nonfinite')
            scan.update(group_scans(rss, scan['xy']))
    # RSS masks are unchanged, so original coordinate-transform H/S bounds/APs
    # are also exactly the bounds_and_aps output after the vector increment.
    out['augmentation'] = trace
    return out


def schedule_coverage(rows, catalog, schedule):
    """Count observed Q detections absent from S and absent from H separately."""
    by_k = {str(k): dict(events=0, Q_detections=0, Q_detected_S_unseen=0,
                        Q_detected_H_unseen=0) for k in (1, 3, 5)}
    h_seen = {}
    for batch in schedule:
        for spec in batch:
            event = episodic_event(catalog, spec)
            if event['floor'] not in h_seen:
                ids = [i for ids in event['history_rows'].values() for i in ids]
                h_seen[event['floor']] = (np.asarray([rows[i]['rss'] for i in ids]) != 100).any(0)
            s = (np.asarray([rows[i]['rss'] for i in event['support_rows']]) != 100).any(0)
            q = np.asarray([rows[i]['rss'] for i in event['query_rows']]) != 100
            counts = by_k[str(spec['k'])]
            counts['events'] += 1
            counts['Q_detections'] += int(q.sum())
            counts['Q_detected_S_unseen'] += int((q & ~s).sum())
            counts['Q_detected_H_unseen'] += int((q & ~h_seen[event['floor']]).sum())
    for counts in by_k.values():
        n = counts['Q_detections']
        counts['S_unseen_fraction'] = counts['Q_detected_S_unseen']/n
        counts['H_unseen_fraction'] = counts['Q_detected_H_unseen']/n
    return dict(by_K=by_k, weighting='Observed query scan-AP detections pooled within K; repeated schedule occurrences retained',
        scope='Original source episodic scans only; no augmented values, protected rows or held-out building',
        limitation='Not independent samples or future population estimates. H coverage does not imply S coverage; S detection alone does not identify a persistent bias.')


def check_pack(p, profile, spec, seed):
    from history_task_preflight import assert_same
    from check_tabicl_history_learning import bounds_and_aps
    before = deepcopy(p)
    record = sample_vector_transform(p['histories'], p['event']['floor'], profile,
                                    seed=seed, task_key=spec['task_key'])
    changed = apply_vector_transform(p, record)
    replay = apply_vector_transform(p, json.loads(json.dumps(changed['augmentation'])))
    assert_same(changed, replay)
    assert_same(p, before)
    delta = np.zeros(520)
    delta[record['ap_indices']] = record['target_delta_db']
    rotation = np.asarray(record['coordinate']['rotation'])
    center = np.asarray(changed['augmentation']['coordinate']['coordinate_center'])
    parts = [(h, changed['histories'][name], False) for name, h in p['histories'].items()]
    parts += [(p[role], changed[role], True) for role in ('support', 'query')]
    for raw, aug, target in parts:
        mask = raw['rss'] != 100
        assert np.array_equal(mask, aug['rss'] != 100)
        assert np.array_equal(aug['xy'], (raw['xy']-center)@rotation.T+center)
        if target:
            expected = np.broadcast_to(delta, raw['rss'].shape)[mask]
            assert np.allclose(aug['rss'][mask]-raw['rss'][mask], expected, rtol=0, atol=1e-12)
        else:
            assert_same(raw['rss'], aug['rss'])
    for key in ('event', 'query_rows'):
        assert_same(p[key], changed[key])
    bounds, aps = bounds_and_aps(changed['histories'], changed['support'])
    assert_same(bounds, changed['bounds'])
    assert_same(aps, changed['aps'])
    identity = sample_vector_transform(p['histories'], p['event']['floor'], profile,
        seed=seed, task_key=spec['task_key'], coordinates=False)
    identity['target_delta_db'] = [0.]*len(identity['ap_indices'])
    unchanged = apply_vector_transform(p, identity)
    unchanged.pop('augmentation')
    assert_same(p, unchanged)
    return dict(event=p['event']['id'], task_key=spec['task_key'],
        context_scans=sum(len(h['rss']) for h in p['histories'].values())+len(p['support']['rss']),
        query_scans=len(p['query']['rss']), transform=changed['augmentation'],
        changed_detected_entries={role: int(((p[role]['rss'] != 100) &
            (p[role]['rss'] != changed[role]['rss'])).sum()) for role in ('support', 'query')},
        mask_and_rows_preserved=True, H_RSS_unchanged=True, coordinate_rule=True,
        replay_exact=True, identity_exact=True, original_unchanged=True, bounds_APs_recomputed_match=True)


def run(args):
    assert socket.gethostname() == 'ray-web', 'No local numerical execution'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    from history_task_preflight import (CSV, SOURCE, digest, load_allowed, read, write, now)
    from history_task_protocol import episodic_pack, ordinary_pack
    import torch
    torch.set_num_threads(2)
    started = time.monotonic()
    old_complete, old_freeze = read(PRE/'complete.json'), read(PRE/'freeze.json')
    assert old_complete['complete'] and old_complete['weights_unchanged']
    for name in ('freeze', 'source_profile', 'schedule'):
        assert digest(PRE/f'{name}.json') == old_complete[f'{name}_sha256']
    old = old_freeze['contract']
    for name, value in old['code'].items():
        assert digest(ROOT/name) == value
    for path in (CSV, SOURCE/'catalog.json', SOURCE/'freeze.json', SOURCE/'complete.json'):
        assert digest(path) == old['inputs'][str(path)]
    catalog, scheduled = read(SOURCE/'catalog.json'), read(PRE/'schedule.json')
    assert catalog['contract']['source_buildings'] == ['B0', 'B2']
    assert catalog['contract']['held_out_building'] == 'B1'
    permitted = allowed_rows(catalog)
    inputs = [CSV, SOURCE/'catalog.json', SOURCE/'freeze.json', SOURCE/'complete.json']
    inputs += [PRE/f'{n}.json' for n in ('freeze', 'source_profile', 'schedule', 'complete')]
    args.output.mkdir(exist_ok=False)
    write(args.output/'freeze.json', dict(created=now(), contract=dict(
        code={**old['code'], Path(__file__).name: digest(Path(__file__))},
        inputs={str(p): digest(p) for p in inputs}, schedule_seed=old['seed'],
        selected=old['selected'], construction='Same-floor same-XY paired R3; six detections; AP median over >=5 eligible positions',
        transform='Whole AP-identified vector and whole sign, replacing scalar; frozen common rotation; H RSS unchanged; no new AP noise/clip/round/dropout',
        scope='CPU source evidence/identifiability check, not performance or vector GPU/backward validation',
        resources=dict(gpu=0, cpu_threads=2, model_calls=0, backward_calls=0, optimizer_steps=0))))
    rows = load_allowed(CSV, permitted, rss=True)
    reserved = {f: {tuple(x) for x in catalog['protected_positions'][f]} for f in permitted}
    assert all(rows[i]['floor'] == f and rows[i]['xy'] not in reserved[f]
               for f, ids in permitted.items() for i in ids)
    profile = build_vector_profile(rows, catalog)
    write(args.output/'profile.json', profile)
    coverage = schedule_coverage(rows, catalog, scheduled['schedule'])
    checks = {}
    for recipe in ('ordinary', 'episodic'):
        spec = old['selected'][recipe]['spec']
        p = (ordinary_pack(rows, catalog, spec, old['seed']) if recipe == 'ordinary'
             else episodic_pack(rows, catalog, spec))
        checks[recipe] = check_pack(p, profile, spec, old['seed'])
    write(args.output/'complete.json', dict(complete=True, created=now(), seconds=time.monotonic()-started,
        freeze_sha256=digest(args.output/'freeze.json'), profile_sha256=digest(args.output/'profile.json'),
        preflight_sha256=digest(PRE/'complete.json'), code_sha256=digest(Path(__file__)), source_only=True,
        source_profile_sha256=digest(PRE/'source_profile.json'), schedule_sha256=digest(PRE/'schedule.json'),
        coverage=coverage, recipes=checks, vectors_by_building=profile['by_building'],
        protected_RSS_parsed=0, held_out_RSS_parsed=0, model_calls=0, backward_calls=0,
        limits=profile['limitations']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args())
