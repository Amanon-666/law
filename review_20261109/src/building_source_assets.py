"""Two-building ReFi H/S assets; no learning, query parsing or scoring.

One lexical acquisition pair per building, with all same-building histories.
Reuse exact B0 source calibration and exact matching historical posteriors.
This is a development preparation, not a localization result.
"""
import json
import socket
import time
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch

import history_fit
from building_source_tasks import building
from contrast_source_operator import contrast_history_context
from episodic_tasks import task_inputs
from fields import History, Prior, Transfer
from paired_acquisition_tasks import make_quartet, verify_quartet
from paired_assets import read_allowed
from predict import adapt
from recomposition_trial import ROOT, CSV, digest, write, now, subset
from source_episode_catalog import read_metadata, verify_catalog
from uji import ap_stats, tensor

CATALOG = ROOT/'results/building_source_tasks_20261106'
PRIOR = ROOT/'results/paired_assets_20261106'
OUT = ROOT/'results/building_source_assets_20261106'


def load_field(record, origin, base):
    """A stationary GP can change origin without changing its posterior."""
    path = ROOT/record['path']
    assert digest(path) == record['sha256']
    with np.load(path) as a:
        return History(tensor(a['z'] + (a['origin']-origin)),
                       dict(mean=tensor(a['mean']), cov=tensor(a['cov'])), base)


def run():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    OUT.mkdir(exist_ok=False)
    catalog = json.loads((CATALOG/'catalog.json').read_text())
    pairs = json.loads((CATALOG/'paired_catalog.json').read_text())['pairs']
    parent = json.loads((CATALOG/'freeze.json').read_text())
    old = json.loads((PRIOR/'freeze.json').read_text())
    old_done = json.loads((PRIOR/'complete.json').read_text())
    old_check = json.loads((PRIOR/'verification.json').read_text())
    old_histories = json.loads((PRIOR/'histories.json').read_text())
    parameters = json.loads((PRIOR/'parameters.json').read_text())
    assert old_done['complete'] and old_check['passed']
    assert digest(PRIOR/'freeze.json') == old_done['freeze_sha256']
    assert digest(PRIOR/'histories.json') == old_done['histories_sha256']
    assert digest(PRIOR/'parameters.json') == old_done['parameters_sha256'] == old_histories['parameters_sha256']
    assert digest(CSV) == parent['CSV_sha256'] == old['inputs'][str(CSV)]
    assert all(digest(ROOT/n) == sha for n, sha in parent['code'].items())
    assert all(digest(path) == sha for path, sha in old['code'].items())
    metadata = read_metadata(CSV, set(catalog['source_floors']))
    verify_catalog(metadata, catalog)
    protected = set(catalog['protected_rows'])
    assert set(old['allowed_rows']) <= set(metadata)-protected
    selected = [next(p for p in pairs if building(p['floor']) == b) for b in ('B0', 'B2')]
    tasks_meta, needed = [], set()
    for pair in selected:
        events = {phase: make_quartet(catalog, pair, 0, phase=phase)
                  for phase in ('fit', 'validation')}
        for phase, quartet in events.items():
            verify_quartet(metadata, catalog, quartet, phase)
        hrows = events['fit'][0]['history_rows']
        origin = np.mean([metadata[i]['xy'] for ids in hrows.values() for i in ids], axis=0)
        for ids in hrows.values(): needed.update(ids)
        for e in events['fit']: needed.update(e['support_rows'])
        tasks_meta.append(dict(pair=pair, events=events, origin=origin.tolist()))
    assert not needed & protected
    assert {building(metadata[i]['floor']) for i in needed} == {'B0', 'B2'}
    source_files = list(ROOT.glob('*.py'))
    inputs = [CSV] + [CATALOG/n for n in ('catalog.json', 'paired_catalog.json', 'freeze.json')]
    inputs += [PRIOR/n for n in ('freeze.json', 'complete.json', 'verification.json', 'parameters.json', 'histories.json')]
    frozen = dict(created=now(), tasks=tasks_meta, inputs={str(p): digest(p) for p in inputs},
        code={p.name: digest(p) for p in source_files}, parsed_rows=sorted(needed),
        calibration_rows=old['allowed_rows'], protected_rows=sorted(protected),
        held_out_building='B1', selected_pair_weights=[.5, .5],
        calibration='Fixed clean B0F0/F1/F2 base and observation calibration; no B2 refit',
        reuse='Exact RSS row sequence and same parameters; stationary GP coordinate-origin translation only',
        scope='Two lexical pairs; four K5 R3 H/S tasks with full H3/H4, fresh S adaptation and H-only neural context',
        limits=['No learning updates, query RSS parsing, query score or performance claim.',
                'B0-only nuisance calibration may mismatch B2; not joint calibration.',
                'Only one pair and one S layout per building, not the full 17-pair curriculum.',
                'B1 has been used in past development; excluded here, not a fresh blind test.',
                'No cross-building coordinate or AP alignment; each H bank stays within its building.',
                'Larger H banks change geometry and AP availability; previous H2 scores are not matched controls.'],
        inner_maxiter=640, inner_max_projected_gradient=1e-3,
        resources=dict(cpu=2, gpu=0, seconds=600, downloads=0))
    write(OUT/'freeze.json', frozen)
    (OUT/'code').mkdir()
    for p in source_files: (OUT/'code'/p.name).write_bytes(p.read_bytes())
    # Only H and the chosen S scans; even nonprotected source Q is not parsed.
    raw = read_allowed(CSV, needed)
    base, obs = parameters['base'], parameters['obs']
    context, solver_rows, records, task_records, reuse_checks = {}, [], [], [], []
    original_minimize = history_fit.minimize

    def monitored(*args, **kwargs):
        result = original_minimize(*args, **kwargs)
        grad = np.asarray(result.jac).copy()
        for j, (low, high) in enumerate(kwargs.get('bounds') or [(None, None)]*len(grad)):
            if low is not None and result.x[j] <= low+1e-10 and grad[j] > 0: grad[j] = 0
            if high is not None and result.x[j] >= high-1e-10 and grad[j] < 0: grad[j] = 0
        row = dict(context, success=bool(result.success), iterations=int(result.nit),
                   projected_gradient=float(np.abs(grad).max()), message=str(result.message))
        solver_rows.append(row)
        with (OUT/'solver.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        if not (np.isfinite(result.fun) and np.isfinite(result.x).all() and np.isfinite(grad).all()
                and row['projected_gradient'] <= frozen['inner_max_projected_gradient']):
            raise RuntimeError(f'Unfinished history VI: {row}')
        return result

    for spec in tasks_meta:
        floor, origin = spec['pair']['floor'], np.array(spec['origin'])
        event = spec['events']['fit'][0]
        histories = {h: subset(raw[h], ids, origin) for h, ids in event['history_rows'].items()}
        loaded = {}
        for h, scans in histories.items():
            reuse = event['history_rows'][h] == old_histories['history_rows'].get(h)
            old_records = {r['ap']: r for r in old_histories['fields'] if r['floor'] == h}
            aps = [int(ap) for ap in np.flatnonzero((scans['stats']['k'] > 0).any(0))]
            if reuse: assert set(old_records) == set(aps)
            for ap in aps:
                if reuse:
                    record = dict(old_records[ap], path=str((PRIOR/old_records[ap]['path']).relative_to(ROOT)), reused=True)
                else:
                    context.clear(); context.update(floor=h, ap=ap)
                    with patch.object(history_fit, 'minimize', monitored):
                        field = history_fit.single_history(dict(scans=scans), ap, base, obs, steps=640)
                    path = OUT/'histories'/h/f'ap{ap}.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, z=field.z.numpy(), mean=field.m.numpy(),
                                        cov=field.c.numpy(), origin=origin)
                    record = dict(floor=h, ap=ap, path=str(path.relative_to(ROOT)), sha256=digest(path), reused=False)
                field = load_field(record, origin, base)
                np.testing.assert_allclose(field.z.numpy(), scans['positions'], rtol=0, atol=1e-8)
                loaded[h, ap] = field
                records.append(record)
            if reuse:
                # Original and translated full mean/covariance at identical physical sites.
                record = next(r for r in records if r['floor'] == h)
                original = load_field(record, np.array(old['origin']), base)
                x = original.z[:5]
                shifted = x + tensor(np.array(old['origin'])-origin)
                a = Prior(Transfer('identity'), original, dict(base, drift=0.))
                b = Prior(Transfer('identity'), loaded[h, record['ap']], dict(base, drift=0.))
                am, af = a.features(x); bm, bf = b.features(shifted)
                ac, bc = a.covariance(x,x,af,af), b.covariance(shifted,shifted,bf,bf)
                torch.testing.assert_close(am, bm, rtol=1e-8, atol=1e-6)
                torch.testing.assert_close(ac, bc, rtol=1e-8, atol=1e-6)
                reuse_checks.append(dict(floor=h, ap=record['ap'], mean_gap=float((am-bm).abs().max()),
                                         covariance_gap=float((ac-bc).abs().max())))
            print(json.dumps(dict(stage='history', floor=h, fields=len(aps), reused=reuse,
                                  seconds=time.monotonic()-started)), flush=True)
        for side, e in zip(('A', 'B'), (spec['events']['fit'][0], spec['events']['fit'][2])):
            support = subset(raw[floor], e['support_rows'], origin)
            task = task_inputs(histories, support, base, obs, lambda h, ap: loaded[h, ap])
            assert task['names'] == e['histories'] and len(task['names']) == (3 if building(floor)=='B0' else 4)
            task['state'] = adapt(task['cases'], base, obs)
            task['context'] = contrast_history_context(histories, task['state']['weights'], base, obs)
            path = OUT/f'{floor}_{side}.pt'
            torch.save(task, path)
            # Reload the future learning input; verify its permission boundary and S statistics.
            check = torch.load(path, weights_only=False)
            assert set(check) == {'cases','grid','positions','names','base','obs','state','context'}
            assert len(check['context']['histories']) == len(e['histories'])
            expected = (support['rss'] != 100).any(0)
            for h in histories.values(): expected |= (h['rss'] != 100).any(0)
            assert list(check['cases']) == list(np.flatnonzero(expected))
            for ap, case in check['cases'].items():
                for key, value in ap_stats(support, ap).items():
                    assert torch.equal(value, case['stats'][key])
                assert torch.equal(case['sx'], tensor(support['positions']))
                assert len(case['histories']) == len(e['histories'])
            record = dict(floor=floor, side=side, pair=spec['pair']['id'], path=path.name,
                sha256=digest(path), support_rows=e['support_rows'], histories=e['histories'],
                APs=list(check['cases']), grid_points=len(check['grid']), state=check['state'])
            task_records.append(record)
            write(OUT/'tasks.json', task_records)
            print(json.dumps(dict(stage='adapted', floor=floor, side=side, APs=len(check['cases']),
                                  weights=check['state']['weights'], seconds=time.monotonic()-started)), flush=True)
    write(OUT/'histories.json', dict(parameters_path=str(PRIOR/'parameters.json'),
        parameters_sha256=digest(PRIOR/'parameters.json'), fields=records,
        history_rows={h: ids for s in tasks_meta for h, ids in s['events']['fit'][0]['history_rows'].items()}))
    write(OUT/'verification.json', dict(passed=True, reused_origin_checks=reuse_checks,
        all_field_coordinates_match_exact_H_rows=True, reloaded_task_S_stats_and_APs_match=True,
        source_code_unchanged=all(digest(ROOT/n)==sha for n,sha in frozen['code'].items()),
        excluded_protected_rows=len(protected), B1_RSS_parsed=0, query_RSS_parsed=0))
    assert all(digest(ROOT/n)==sha for n,sha in frozen['code'].items())
    write(OUT/'complete.json', dict(complete=True, created=now(), elapsed=time.monotonic()-started,
        freeze_sha256=digest(OUT/'freeze.json'), tasks_sha256=digest(OUT/'tasks.json'),
        histories_sha256=digest(OUT/'histories.json'), verification_sha256=digest(OUT/'verification.json'),
        tasks=len(task_records), selected_pairs=[p['id'] for p in selected],
        history_fields_by_floor=dict(Counter(r['floor'] for r in records)),
        reused_fields=sum(r['reused'] for r in records), fitted_fields=sum(not r['reused'] for r in records),
        parsed_RSS_rows=len(needed), protected_RSS_rows_parsed=0, B1_RSS_rows_parsed=0,
        learning_updates=0, localization_scores=0, calibration_updates=0,
        max_VI_projected_gradient=max(r['projected_gradient'] for r in solver_rows)))
    print((OUT/'complete.json').read_text(), flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    run()
