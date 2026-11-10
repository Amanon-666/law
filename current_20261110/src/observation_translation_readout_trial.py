"""Post-result readout comparison for the frozen quantized observation model.

Only source fit labels define interval representatives. Both the expected map
and Query use those same representatives, discarding within-bin RSS precision.
S conditions the original features only: no real S rows are appended to 3NN.
No training, probability-temperature choice, GP fit or new fusion is performed.
Use external timeout 900s; the internal 840s limit stops at event boundaries.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import time
import socket
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from observation_translation_model import ObservationTranslationModel, observation_classes
from observation_translation_tasks import ObservationTranslationContext
from observation_translation_trial import (ROOT, RESULTS, CSV, CATALOG, ASSETS, CODE,
    read, budget, source_pack, load_acquisition_rows, context, read_scans, read_truth,
    digest, now, write)

METHODS = ('quantized_map_3NN', 'observation_translation', 'GP',
           'ordinary_HS', 'GP_ordinary_HS_half')


def representatives(model, source_freeze, start):
    """Recreate only the original 72 fit tables and their exact sample weights."""
    catalog = read(CATALOG)
    specs = source_freeze['contract']['schedule']
    assert len(specs) == 72 and all(s['phase'] == 'fit' for s in specs)
    rows = load_acquisition_rows(CSV, set(catalog['source_floors']))
    mass = np.zeros(len(model.cuts) + 2)
    total = np.zeros_like(mass)
    for spec in specs:
        budget(start, 840)
        event, pack = source_pack(rows, catalog, spec)
        table = ObservationTranslationContext(pack['histories'], pack['support'], event['floor']).source_scan_table(pack['query'], event)
        assert table['role'] == 'train'
        labels = observation_classes(table['y'], model.cuts)
        weights = table['sample_weight']
        mass += np.bincount(labels, weights=weights, minlength=len(mass))
        total += np.bincount(labels, weights=weights * table['y'], minlength=len(mass))
    assert np.all(mass > 0)
    values = total / mass
    values[0] = -105.
    return values, mass


def map_readout(prepared, query_rss, values, bounds):
    """Original RMS/stable-3NN rule, with matched quantization and no S append."""
    map_rss = np.exp(prepared.log_mass) @ values
    query = values[observation_classes(query_rss[:, prepared.aps], prepared.cuts)]
    distance = np.sqrt(((query[:, None] - map_rss[None]) ** 2).mean(-1))
    idx = np.argsort(distance, axis=1, kind='stable')[:, :3]
    weights = 1 / np.maximum(np.take_along_axis(distance, idx, axis=1), 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.clip((weights[:, :, None] * prepared.cand_xy[idx]).sum(1),
                   bounds[0], bounds[1])


def target_summary(manifest, records):
    """Same position/building/floor/direction/K weights as the frozen trial."""
    samples = []; events = {e['id']: e for e in manifest['events']}
    for r in records:
        e = events[r['event']]
        with np.load(r['scored_path']) as a:
            _, g = np.unique(a['truth'], axis=0, return_inverse=True)
            n = int(g.max()) + 1
            errors = {m: np.array([a[m][g == i].mean() for i in range(n)]) for m in METHODS}
        nf = sum(f.startswith(e['building']) for f in manifest['floors'])
        samples.append(dict(event=e, errors=errors, weight=.5 / nf / 2 / n))
    def agg(ss, method):
        v = np.concatenate([s['errors'][method] for s in ss])
        w = np.concatenate([np.full(len(s['errors'][method]), s['weight']) for s in ss])
        w /= w.sum(); order = np.argsort(v)
        return dict(mean=float(w @ v), position_mean_p90=float(v[order][np.searchsorted(np.cumsum(w[order]), .9)]))
    return {m: dict(agg(samples, m), **{'by_' + key: {
        str(c): agg([s for s in samples if s['event'][key] == c], m) for c in choices}
        for key, choices in (('K', (1, 3, 5)), ('floor', manifest['floors']), ('building', manifest['buildings']))}) for m in METHODS}


def main(args):
    start = time.monotonic()
    source, target = read(args.source/'complete.json'), read(args.target/'complete.json')
    assert source['complete'] and target['complete'] and source['training_tables'] == 72
    sf = read(args.source/'freeze.json'); tf = read(args.target/'freeze.json')
    assert digest(args.source/'freeze.json') == source['freeze_sha256']
    assert digest(args.target/'freeze.json') == target['freeze_sha256']
    model_path = Path(source['model_path'])
    assert digest(model_path) == source['model_sha256'] == target['model_sha256']
    for name in CODE:
        assert digest(ROOT/name) == sf['code'][name] == tf['code'][name]
    assert sf['inputs'][str(CSV)] == digest(CSV) and sf['inputs'][str(CATALOG)] == digest(CATALOG)
    manifest = read(ASSETS/'manifest.json')
    assert tf['contract']['events'] == manifest['events'] and len(manifest['events']) == 54
    assert tf['inputs'][str(ASSETS/'manifest.json')] == digest(ASSETS/'manifest.json')
    assert manifest['csv_sha256'] == digest(CSV)
    cached = {r['event']: r for r in target['records']}
    assert set(cached) == {e['id'] for e in manifest['events']}
    paths = [CSV, CATALOG, model_path, args.source/'freeze.json', args.source/'complete.json',
             args.target/'freeze.json', args.target/'complete.json', ASSETS/'manifest.json']
    for r in cached.values():
        path = Path(r['prediction_path'])
        assert digest(path) == r['prediction_sha256']; paths.append(path)
    args.output.mkdir(exist_ok=False)
    write(args.output/'freeze.json', dict(created=now(),
        code={name: digest(ROOT/name) for name in (*CODE, Path(__file__).name)},
        inputs={str(p): digest(p) for p in paths},
        contract=dict(events=manifest['events'], methods=METHODS, source_schedule=sf['contract']['schedule'],
            source_representatives='Original 72 fit tables, original sample_weight; missing=-105; no validation labels',
            readout='Expected interval-representative map versus identically quantized Query, fixed3NN, no appended S',
            comparison='Post-result readout comparison; same quantized model/information, no new fusion or target selection',
            resources=dict(CPU_threads=2, GPU=0, downloads=0, seconds=840, external_timeout=900),
            scope='Same exposed UJI54; correlated K/directions, not independent validation')))
    model = ObservationTranslationModel.load(model_path)
    values, mass = representatives(model, sf, start)
    rep_path = args.output/'source_representatives.json'
    write(rep_path, dict(cuts=model.cuts.tolist(), representatives=values.tolist(), class_weight=mass.tolist(),
                        source_model_sha256=source['model_sha256'], training_tables=72))
    records = []
    for event in manifest['events']:
        budget(start, 840)
        histories, support, _, _, aps, bounds = context(CSV, manifest, event, 'HS')
        prepared = model.prepare(histories, support, event['floor'])
        assert np.array_equal(aps, prepared.aps)
        qrss = read_scans(CSV, event['query_rows'], False)['rss']
        points = {'quantized_map_3NN': map_readout(prepared, qrss, values, bounds)}
        with np.load(cached[event['id']]['prediction_path']) as a:
            assert np.array_equal(a['query_rows'], event['query_rows'])
            assert np.array_equal(a['cand_xy'], prepared.cand_xy)
            points.update({m: a[m].copy() for m in METHODS[1:]})
        path = args.output/f"{event['id']}_unscored.npz"
        np.savez_compressed(path, **points, query_rows=event['query_rows'],
                            cand_xy=prepared.cand_xy, aps=prepared.aps, bounds=bounds)
        truth = read_truth(CSV, event['query_rows'])
        _, group, counts = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
        weights = 1 / (len(counts) * counts[group])
        errors = {m: np.linalg.norm(p - truth, axis=1) for m, p in points.items()}
        scored = args.output/f"{event['id']}_scored.npz"
        np.savez_compressed(scored, truth=truth, weights=weights, **errors)
        records.append(dict(event=event['id'], floor=event['floor'], K=event['K'],
            prediction_path=str(path), prediction_sha256=digest(path), scored_path=str(scored),
            scored_sha256=digest(scored), mde={m: float(weights @ errors[m]) for m in METHODS}))
        write(args.output/'progress.json', dict(records=records))
    assert digest(model_path) == source['model_sha256']
    summary = target_summary(manifest, records)
    for m in METHODS[1:]:
        assert summary[m] == target['summary'][m]
    write(args.output/'complete.json', dict(complete=True, created=now(), seconds=time.monotonic()-start,
        freeze_sha256=digest(args.output/'freeze.json'), model_sha256=source['model_sha256'],
        representatives_sha256=digest(rep_path), records=records, summary=summary,
        training_updates=0, GP_fits=0, parameters_selected=False, post_result_readout_comparison=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=RESULTS/'observation_translation_source_20261110_nanfix')
    parser.add_argument('--target', type=Path, default=RESULTS/'observation_translation_target_20261110')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web'
    with threadpool_limits(limits=2):
        main(args)
