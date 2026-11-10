"""Fixed-model K1 stability over all five previously frozen support positions.

Pre-execution rule: retain all 18 x 5 events, reuse the original 18 K1 outputs,
and make exactly 72 new forwards. No query-based anchor selection or new fits.
For each context all four new predictions are saved before query truth is read.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import socket
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from continuous_observation_translation import ContinuousObservationTranslationModel
from continuous_observation_trial import METHOD
from observation_translation_trial import ROOT, ASSETS, CSV, read, budget
from building_information_tab_trial import digest, now, read_scans, read_truth, write


def anchor_events(manifest):
    """Construct all K1 substitutions from frozen K5 row blocks, without labels."""
    events = manifest['events']
    result = []
    for original in (e for e in events if e['K'] == 1):
        full = next(e for e in events if e['floor'] == original['floor']
                    and e['direction'] == original['direction'] and e['K'] == 5)
        pair = manifest['floor_pairs'][original['floor']]
        positions = pair['support_positions']
        assert len(positions) == 5 and len(full['support_rows']) == 15
        assert len(set(full['support_rows'])) == 15
        assert len(set(map(tuple, positions))) == 5
        assert original['support_rows'] == full['support_rows'][:3]
        assert original['query_rows'] == full['query_rows']
        assert original['histories'] == full['histories']
        assert original['support_acquisition'] == full['support_acquisition']
        assert original['query_acquisition'] == full['query_acquisition']
        assert not set(full['support_rows']).intersection(full['query_rows'])
        assert not set(map(tuple, positions)).intersection(map(tuple, pair['query_positions']))
        for index, xy in enumerate(positions, 1):
            result.append(dict(original, id=f"{original['id']}_anchor{index}",
                original_event=original['id'], anchor_index=index, anchor_xy=xy,
                support_rows=full['support_rows'][3*(index-1):3*index]))
    assert len(result) == 90 and len({e['original_event'] for e in result}) == 18
    return result


def summarize(records):
    """Pool position-mean errors with original hierarchy and equal anchor weight."""
    samples = []
    buildings = sorted({r['building'] for r in records})
    for record in records:
        with np.load(record['scored_path']) as saved:
            _, group = np.unique(saved['truth'], axis=0, return_inverse=True)
            group = group.ravel()
            values = np.bincount(group, weights=saved[METHOD]) / np.bincount(group)
        floors = {r['floor'] for r in records if r['building'] == record['building']}
        directions = {r['direction'] for r in records if r['floor'] == record['floor']}
        samples.append(dict(record=record, values=values,
            weight=1. / (len(buildings)*len(floors)*len(directions)*5*len(values))))

    def aggregate(selected):
        values = np.concatenate([s['values'] for s in selected])
        weights = np.concatenate([np.full(len(s['values']), s['weight']) for s in selected])
        weights /= weights.sum()
        order = np.argsort(values)
        return dict(mean=float(weights @ values), position_mean_p90=float(
            values[order[np.searchsorted(np.cumsum(weights[order]), .9)]]))

    def anchor_summary(selected):
        overall = aggregate(selected)
        anchors = {str(i): aggregate([s for s in selected if s['record']['anchor_index'] == i])
                   for i in range(1, 6)}
        means = [a['mean'] for a in anchors.values()]
        return dict(all_anchors=overall, by_anchor=anchors,
            anchor_mean_range=dict(min=min(means), max=max(means), span=max(means)-min(means)),
            anchor1_relative=dict(mean_delta=anchors['1']['mean']-overall['mean'],
                p90_delta=anchors['1']['position_mean_p90']-overall['position_mean_p90'],
                mean_rank_ascending=1+sum(value < means[0] for value in means)))

    return dict(anchor_summary(samples), by_floor={floor: anchor_summary(
        [s for s in samples if s['record']['floor'] == floor])
        for floor in sorted({r['floor'] for r in records})})


def run(args):
    os.chdir(ROOT)
    start = time.monotonic()
    source, source_freeze = read(args.source/'complete.json'), read(args.source/'freeze.json')
    target, target_freeze = read(args.target/'complete.json'), read(args.target/'freeze.json')
    manifest = read(ASSETS/'manifest.json')
    assert source['complete'] and target['complete']
    assert source['phase'] == 'source_train' and target['phase'] == 'target_evaluate'
    assert source['freeze_sha256'] == digest(args.source/'freeze.json')
    assert target['freeze_sha256'] == digest(args.target/'freeze.json')
    assert source['model_sha256'] == target['model_sha256']
    model_path = Path(source['model_path'])
    assert digest(model_path) == source['model_sha256']
    assert digest(ASSETS/'manifest.json') == read(ASSETS/'complete.json')['manifest_sha256']
    assert manifest['csv_sha256'] == digest(CSV)
    assert target_freeze['contract']['events'] == manifest['events']
    assert len(manifest['floors']) == 9 and manifest['buildings'] == ['B1', 'B2']
    for name, sha in source_freeze['code'].items():
        assert digest(ROOT/name) == sha
    for name, sha in target_freeze['code'].items():
        assert digest(ROOT/name) == sha
    plan = anchor_events(manifest)
    originals = {r['event']: r for r in target['records'] if r['K'] == 1}
    assert set(originals) == {e['original_event'] for e in plan}
    paths = [args.source/'complete.json', args.source/'freeze.json',
             args.target/'complete.json', args.target/'freeze.json', model_path, CSV,
             ASSETS/'manifest.json', ASSETS/'complete.json']
    for record in originals.values():
        for key in ('prediction', 'scored'):
            path = Path(record[key+'_path'])
            assert digest(path) == record[key+'_sha256']
            paths.append(path)
    args.output.mkdir(exist_ok=False)
    write(args.output/'freeze.json', dict(created=now(),
        code={**source_freeze['code'], Path(__file__).name: digest(Path(__file__))},
        inputs={str(p): digest(p) for p in paths}, contract=dict(
            plan=plan, source_model_sha256=source['model_sha256'], seconds=args.seconds,
            resources=dict(CPU_threads=2, GPU=0, downloads=0, external_timeout_seconds=900),
            forwards=72, cached_events=18, total_events=90, K=1, R=3,
            rule='All five original K5 locations, original H and Query, frozen model and reader; all anchor outcomes retained',
            ordering='Save all four new anchor predictions for a context before opening its Query truth',
            weighting='Equal building, floor within building, direction, anchor index and Query position; equal scans within position',
            p90='Weighted quantile of pooled position-mean errors, including all anchors for the overall statistic',
            comparison='Original anchor1 compared only to this same model across all five anchors; original GP/Tab outputs are not comparisons for substituted anchors',
            scope='Exposed target support-location sensitivity; correlated anchors and Query reuse, not independent-building validation or oracle anchor selection',
            original_source_contract=source_freeze['contract'])))
    model = ContinuousObservationTranslationModel.load(model_path)
    records, forwards = [], 0
    for original_id, old in originals.items():
        budget(start, args.seconds)
        group = [e for e in plan if e['original_event'] == original_id]
        histories = {name: read_scans(CSV, manifest['history_rows'][name], True)
                     for name in group[0]['histories']}
        support_rows = [i for event in group for i in event['support_rows']]
        full_support = read_scans(CSV, support_rows, True)
        expected_xy = np.repeat(np.asarray([e['anchor_xy'] for e in group]), 3, axis=0)
        assert np.array_equal(full_support['xy'], expected_xy)
        query_rows = group[0]['query_rows']
        query_rss = read_scans(CSV, query_rows, False)['rss']
        with np.load(old['prediction_path']) as saved:
            assert np.array_equal(saved['query_rows'], query_rows)
        pending = []
        for event in group:
            budget(start, args.seconds)
            if event['anchor_index'] == 1:
                path = Path(old['prediction_path'])
            else:
                offset = 3*(event['anchor_index']-1)
                support = {key: full_support[key][offset:offset+3].copy() for key in ('rss', 'xy')}
                prediction = model.prepare(histories, support, event['floor']).predict(query_rss)
                forwards += 1
                path = args.output/f"{event['id']}_unscored.npz"
                np.savez_compressed(path, **{METHOD: prediction['projected']},
                    query_rows=query_rows, support_rows=event['support_rows'],
                    cand_xy=prediction['cand_xy'], log_position_mass=prediction['log_position_mass'])
            pending.append((event, path, digest(path)))

        truth = read_truth(CSV, query_rows)
        assert not set(map(tuple, truth)).intersection(map(tuple, expected_xy))
        _, inverse, counts = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
        weights = 1. / (len(counts)*counts[inverse.ravel()])
        for event, path, prediction_sha in pending:
            if event['anchor_index'] == 1:
                scored = Path(old['scored_path'])
                with np.load(scored) as saved:
                    assert np.array_equal(saved['truth'], truth)
                    errors = saved[METHOD]
            else:
                with np.load(path) as saved:
                    errors = np.linalg.norm(saved[METHOD]-truth, axis=1)
                scored = args.output/f"{event['id']}_scored.npz"
                np.savez_compressed(scored, truth=truth, **{METHOD: errors}, query_rows=query_rows)
            records.append(dict(event=event['id'], original_event=original_id,
                floor=event['floor'], building=event['building'], direction=event['direction'],
                anchor_index=event['anchor_index'], anchor_xy=event['anchor_xy'],
                support_rows=event['support_rows'], K=1, R=3, reused=event['anchor_index'] == 1,
                prediction_path=str(path), prediction_sha256=prediction_sha,
                scored_path=str(scored), scored_sha256=digest(scored), mde=float(weights @ errors)))
        write(args.output/'progress.json', dict(records=records, forwards=forwards,
            model_sha256=source['model_sha256']))
        print(f'Completed {len(records)}/90 events; {forwards}/72 new forwards', flush=True)

    budget(start, args.seconds)
    assert len(records) == 90 and forwards == 72
    assert digest(model_path) == source['model_sha256']
    summary = summarize(records)
    original = target['summary'][METHOD]['by_K']['1']
    assert np.isclose(summary['by_anchor']['1']['mean'], original['mean'])
    assert np.isclose(summary['by_anchor']['1']['position_mean_p90'], original['position_mean_p90'])
    write(args.output/'complete.json', dict(complete=True, created=now(),
        seconds=time.monotonic()-start, freeze_sha256=digest(args.output/'freeze.json'),
        model_sha256=source['model_sha256'], records=records, summary=summary,
        forwards=forwards, reused_events=18, training_updates=0, GP_fits=0,
        anchor_selection=False, independent_building_validation=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/continuous_observation_source_20261110')
    parser.add_argument('--target', type=Path, default=ROOT/'results/continuous_observation_target_20261110')
    parser.add_argument('--output', type=Path, default=ROOT/'results/continuous_observation_support_stability_20261110')
    parser.add_argument('--seconds', type=float, default=840.)
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 840
    with threadpool_limits(limits=2):
        run(args)
