"""Explicit source learning and deployment for the continuous scan translator.

This entrypoint has not been run. It adopts Claude's successful hurdle reader
and fits it under the project's protected, three-floor source permissions.
The supervision scope differs from both Claude's four-floor prototype and the
previous 72 sparse query tables. Existing results are references, not results
of this implementation. Source learning and target evaluation are separate.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import socket
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from acquisition_tasks import load_acquisition_rows
from continuous_observation_tasks import (
    ContinuousObservationTranslationContext, full_source_episode)
from continuous_observation_translation import ContinuousObservationTranslationModel
from observation_translation_trial import (
    ROOT, RESULTS, OLD, CATALOG, CSV, ASSETS, REFERENCE, TRANSLATED,
    read, budget, context, read_scans, read_truth, digest, now, write)

METHOD = 'continuous_observation_translation'
METHODS = (METHOD, 'GP', 'ordinary_HS', 'GP_ordinary_HS_half',
           'GP_translated_3NN_half')
CODE = ('continuous_observation_trial.py', 'continuous_observation_tasks.py',
        'continuous_observation_translation.py', 'observation_translation_tasks.py',
        'observation_translation_trial.py', 'source_episode_catalog.py',
        'acquisition_tasks.py', 'building_information_tab_trial.py', 'uji.py',
        'vertical_translator/translator.py')


def freeze(out, inputs, contract):
    out.mkdir(exist_ok=False)
    write(out/'freeze.json', dict(created=now(),
        code={name: digest(ROOT/name) for name in CODE},
        inputs={str(path): digest(path) for path in inputs}, contract=contract,
        resources=dict(CPU_threads=2, GPU=0, downloads=0,
                       external_timeout_required=True),
        runtime=dict(numpy=np.__version__, scipy=scipy.__version__,
                     sklearn=sklearn.__version__)))


def source_pack(rows, catalog, spec):
    return full_source_episode(rows, catalog, spec)


def source_summary(records):
    floors = sorted({record['floor'] for record in records})
    def mean(selected):
        return float(np.mean([np.mean([r['mde'] for r in selected if r['floor'] == f])
                              for f in floors]))
    return dict(mean=mean(records),
        by_floor={f: float(np.mean([r['mde'] for r in records if r['floor'] == f]))
                  for f in floors},
        by_K={str(k): mean([r for r in records if r['K'] == k]) for k in (1, 3, 5)})


def train(args):
    start = time.monotonic()
    old = read(OLD/'complete.json')
    original = read(OLD/'freeze.json')
    catalog = read(CATALOG)
    assert old['complete'] and old['freeze_sha256'] == digest(OLD/'freeze.json')
    assert catalog['source_floors'] == ['B0F0', 'B0F1', 'B0F2']
    assert original['inputs'][str(CSV)] == digest(CSV)
    assert original['inputs'][str(CATALOG)] == digest(CATALOG)
    specs = [spec for batch in original['schedule'] for spec in batch]
    validation = original['validation']
    assert len(specs) == 72 and len(validation) == 27
    assert all(s['phase'] == 'fit' for s in specs)
    assert all(s['phase'] == 'validation' for s in validation)
    freeze(args.output, [CSV, CATALOG, OLD/'freeze.json', OLD/'complete.json'], dict(
        phase='source_train', seed=args.seed, max_iter=400, seconds=args.seconds,
        schedule=specs, validation=validation, source_floors=catalog['source_floors'],
        supervision='All legal scans in each original query acquisition, excluding all protected rows and maximal K5 support positions',
        weighting='Equal task, equal source position, equal scan within position, equal H/S-selected AP',
        scale='Two groups of source target floors; cross-fitted mean residual plus within-position detected variance',
        reader=dict(beta=128, eps=.02, offsets=[-8, -4, 0, 4, 8],
                    origin='Fixed collaborator development configuration; no selection in this run'),
        source_comparison='Shared B0F0-F2 legal corpus; larger supervised query coverage than the existing 72-table training budget',
        development_scope='Reader configuration previously developed with B1/B2-trained fields; exposed development evidence, not fully B0-only model selection',
        target_opened=False, additional_GP_fit=False, Tab_training=False))
    rows = load_acquisition_rows(CSV, set(catalog['source_floors']))
    tables, training_events = [], []
    for spec in specs:
        budget(start, args.seconds)
        event, pack = source_pack(rows, catalog, spec)
        table = ContinuousObservationTranslationContext(
            pack['histories'], pack['support'], event['floor']).source_cell_table(
                pack['query'], event)
        tables.append(table)
        training_events.append(dict(event=event['id'], floor=event['floor'], K=event['K'],
                                    query_rows=event['query_rows'], cells=len(table['n'])))
    write(args.output/'training_events.json', training_events)
    budget(start, args.seconds)
    model = ContinuousObservationTranslationModel().fit(tables, seed=args.seed, max_iter=400)
    model_path = args.output/'model.joblib'
    model.save(model_path)
    model_sha = digest(model_path)
    write(args.output/'model_saved.json', dict(created=now(), model_sha256=model_sha,
        training_tables=len(tables), training_events_sha256=digest(args.output/'training_events.json')))
    del tables
    records = []
    for spec in validation:
        budget(start, args.seconds)
        event, pack = source_pack(rows, catalog, spec)
        prediction = model.prepare(pack['histories'], pack['support'], event['floor']).predict(
            pack['query']['rss'])
        path = args.output/f"{event['id']}_unscored.npz"
        np.savez_compressed(path, **prediction, query_rows=event['query_rows'])
        _, group, counts = np.unique(pack['query']['xy'], axis=0,
                                      return_inverse=True, return_counts=True)
        weight = 1 / (len(counts) * counts[group.ravel()])
        error = np.linalg.norm(prediction['projected'] - pack['query']['xy'], axis=1)
        records.append(dict(event=event['id'], floor=event['floor'], K=event['K'],
            mde=float(weight @ error), prediction_path=str(path), prediction_sha256=digest(path)))
        write(args.output/'progress.json', dict(records=records, model_sha256=model_sha))
    write(args.output/'complete.json', dict(complete=True, created=now(), phase='source_train',
        seconds=time.monotonic()-start, freeze_sha256=digest(args.output/'freeze.json'),
        model_path=str(model_path), model_sha256=model_sha, training_tables=len(specs),
        training_events_sha256=digest(args.output/'training_events.json'),
        validation_events=len(records), summary=source_summary(records), records=records,
        target_opened=False, parameters_selected=False))


def target_summary(manifest, records):
    events = {event['id']: event for event in manifest['events']}
    samples = []
    for record in records:
        event = events[record['event']]
        with np.load(record['scored_path']) as saved:
            _, group = np.unique(saved['truth'], axis=0, return_inverse=True)
            group = group.ravel()
            count = int(group.max()) + 1
            error = {name: np.array([saved[name][group == i].mean() for i in range(count)])
                     for name in METHODS}
        floors = sum(f.startswith(event['building']) for f in manifest['floors'])
        samples.append(dict(event=event, error=error, weight=.5/floors/2/count))
    def aggregate(selected, method):
        values = np.concatenate([sample['error'][method] for sample in selected])
        weights = np.concatenate([np.full(len(sample['error'][method]), sample['weight'])
                                  for sample in selected])
        weights /= weights.sum()
        order = np.argsort(values)
        return dict(mean=float(weights @ values), position_mean_p90=float(
            values[order][np.searchsorted(np.cumsum(weights[order]), .9)]))
    return {method: dict(aggregate(samples, method), **{
        'by_'+key: {str(value): aggregate(
            [sample for sample in samples if sample['event'][key] == value], method)
            for value in choices}
        for key, choices in (('K', (1, 3, 5)), ('floor', manifest['floors']),
                             ('building', manifest['buildings']))}) for method in METHODS}


def evaluate(args):
    start = time.monotonic()
    source = read(args.source/'complete.json')
    source_freeze = read(args.source/'freeze.json')
    assert source['complete'] and source['phase'] == 'source_train' and not source['target_opened']
    assert source['freeze_sha256'] == digest(args.source/'freeze.json')
    model_path = Path(source['model_path'])
    assert digest(model_path) == source['model_sha256']
    assert all(digest(ROOT/name) == source_freeze['code'][name] for name in CODE)
    manifest = read(ASSETS/'manifest.json')
    assert read(ASSETS/'complete.json')['manifest_sha256'] == digest(ASSETS/'manifest.json')
    assert manifest['csv_sha256'] == digest(CSV) and len(manifest['events']) == 54
    reference = read(REFERENCE/'complete.json')
    translated = read(TRANSLATED/'complete.json')
    assert reference['complete'] and translated['complete']
    assert reference['freeze_sha256'] == digest(REFERENCE/'freeze.json')
    assert translated['freeze_sha256'] == digest(TRANSLATED/'freeze.json')
    assert read(TRANSLATED/'freeze.json')['contract']['events'] == manifest['events']
    refs = {record['event']: record for record in reference['records']}
    maps = {record['event']: record for record in translated['records']}
    assert set(refs) == set(maps) == {event['id'] for event in manifest['events']}
    inputs = [CSV, model_path, args.source/'complete.json', args.source/'freeze.json',
              ASSETS/'manifest.json', ASSETS/'complete.json', REFERENCE/'complete.json',
              REFERENCE/'freeze.json', TRANSLATED/'complete.json', TRANSLATED/'freeze.json']
    for record in [*refs.values(), *maps.values()]:
        path = Path(record['prediction_path'])
        assert digest(path) == record['prediction_sha256']
        inputs.append(path)
    freeze(args.output, inputs, dict(phase='target_evaluate', seconds=args.seconds,
        events=manifest['events'], methods=METHODS, model_sha256=source['model_sha256'],
        source_contract=source_freeze['contract'], training_updates=0,
        information='H/S prepare, Query RSS prediction, persist prediction and position scores, then open Query truth',
        reference='Reuse existing GP/ordinary/translated predictions; no new fits or combinations',
        scope='Exposed B1/B2 development comparison; original four-floor translated system is a system reference, not matched source training'))
    model = ContinuousObservationTranslationModel.load(model_path)
    records = []
    for event in manifest['events']:
        budget(start, args.seconds)
        histories, support, *_ = context(CSV, manifest, event, 'HS')
        prepared = model.prepare(histories, support, event['floor'])
        prediction = prepared.predict(read_scans(CSV, event['query_rows'], False)['rss'])
        points = {METHOD: prediction['projected']}
        for record, names in ((refs[event['id']], METHODS[1:4]),
                              (maps[event['id']], METHODS[4:])):
            with np.load(record['prediction_path']) as saved:
                assert np.array_equal(saved['query_rows'], event['query_rows'])
                points.update({name: saved[name].copy() for name in names})
        path = args.output/f"{event['id']}_unscored.npz"
        np.savez_compressed(path, **points, query_rows=event['query_rows'],
            cand_xy=prediction['cand_xy'], log_position_mass=prediction['log_position_mass'])
        truth = read_truth(CSV, event['query_rows'])
        _, group, counts = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
        weights = 1 / (len(counts) * counts[group.ravel()])
        errors = {name: np.linalg.norm(value - truth, axis=1) for name, value in points.items()}
        scored = args.output/f"{event['id']}_scored.npz"
        np.savez_compressed(scored, truth=truth, **errors)
        records.append(dict(event=event['id'], floor=event['floor'], K=event['K'],
            prediction_path=str(path), prediction_sha256=digest(path), scored_path=str(scored),
            scored_sha256=digest(scored), mde={name: float(weights @ error) for name, error in errors.items()}))
        write(args.output/'progress.json', dict(records=records, model_sha256=source['model_sha256']))
    write(args.output/'complete.json', dict(complete=True, created=now(), phase='target_evaluate',
        seconds=time.monotonic()-start, freeze_sha256=digest(args.output/'freeze.json'),
        model_sha256=source['model_sha256'], records=records, summary=target_summary(manifest, records),
        training_updates=0, GP_fits=0, target_selection=False,
        tail_metric='Pooled weighted quantile of per-position mean scan error; not a mean of event quantiles'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('train', 'evaluate'), required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--seed', type=int, default=11509159)
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web' and args.seconds > 0
    assert args.phase != 'evaluate' or args.source is not None
    with threadpool_limits(limits=2):
        (train if args.phase == 'train' else evaluate)(args)
