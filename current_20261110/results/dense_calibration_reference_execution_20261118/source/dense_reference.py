"""Target-only sufficient-calibration WKNN on the original 39 long-K queries."""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
from threadpoolctl import threadpool_limits

PROJECTS = Path('/home/panyushuo/projects/panyushuo')
BASE = PROJECTS / 'paper_bench_wt/current_20261110'
CORE = PROJECTS / 'ReFi-UJI-Core-20261031'
CLAUDE = PROJECTS / 'claude_fp_wt/current_20261110/src/results'
sys.path[:0] = [str(BASE / 'src'), str(CORE)]

from law_bench import data, methods_simple, protocol, run, score, task as task_module
from law_bench.methods_simple import WKNN_S
from law_bench.task import Task

BUDGETS = (1, 3, 5, 10, 20)
BUILDINGS = ('B0', 'B1', 'B2')


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return dict(path=str(path), sha256=run._sha(path))


def array_sha(value):
    value = np.ascontiguousarray(value)
    result = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    result.update(value.tobytes())
    return result.hexdigest()


def matching(record, task, truth):
    assert record['status'] == 'completed' and record['task_id'] == task.task_id
    assert record['query_row_indices'] == truth['row_indices'].tolist()
    assert record['query_groups'] == truth['groups'].tolist()
    assert record['meta'] == task.meta


def prepare(output):
    assert not (output / 'freeze.json').exists()
    buildings, axes = data.load('uji')
    units, sparse = [], []
    references = {}
    for building in BUILDINGS:
        paper = BASE / 'results' / f'paper_uji_{building}_kcurve_20261111'
        old = CLAUDE / f'bench_uji_{building}_kcurve_20261111'
        paper_freeze, old_freeze = read(paper / 'freeze.json'), read(old / 'freeze.json')
        cfg = paper_freeze['configuration']
        assert cfg['budgets'] == list(BUDGETS) and cfg['repeats'] == [0, 1, 2]
        assert cfg['seed'] == 20261111 and cfg['R'] == 3 and cfg['query_positions'] == 60
        assert old_freeze['code']['methods_simple.py'] == run._sha(methods_simple.__file__)
        assert paper_freeze['target_data'] == run._data_identity(
            {building: buildings[building]}, {building: axes[building]})
        references[building] = dict(paper_freeze=binding(paper / 'freeze.json'),
                                    sparse_freeze=binding(old / 'freeze.json'))
        for floor in sorted(buildings[building]):
            values = buildings[building][floor]
            _, position_ids = np.unique(values['xy'], axis=0, return_inverse=True)
            position_ids = position_ids.ravel()
            for repeat in (0, 1, 2):
                episodes = protocol.make_tasks('uji', building, buildings[building], axes[building],
                    floor, budgets=BUDGETS, repeat=repeat, seed=20261111, R=3, query_positions=60)
                reference, truth = episodes[-1]
                reference_events, reference_meta = {}, {}
                for task, current_truth in episodes:
                    assert np.array_equal(current_truth['row_indices'], truth['row_indices'])
                    assert np.array_equal(current_truth['groups'], truth['groups'])
                    assert np.array_equal(task.Q_rss, reference.Q_rss)
                    paper_path = paper / 'events/refi' / (task.task_id + '.json')
                    sparse_path = old / 'events/wknn_S' / (task.task_id + '.json')
                    matching(read(paper_path), task, current_truth)
                    old_record = read(sparse_path)
                    matching(old_record, task, current_truth)
                    assert old_record['diagnostics']['information'] == 'S'
                    assert old_record['diagnostics']['context_scans'] == len(task.S['rss'])
                    reference_events[str(task.K)] = binding(paper_path)
                    reference_meta[str(task.K)] = task.meta
                    sparse.append(dict(**binding(sparse_path), task_id=task.task_id,
                                       building=building, floor=floor, repeat=repeat, K=task.K))
                    if task.task_id == 'uji/B0/F0/K20/r0':
                        check_task, check_record = task, old_record
                query_positions = np.unique(position_ids[truth['row_indices']])
                excluded = np.isin(position_ids, query_positions)
                allowed = ~excluded
                acquisition = reference.meta['support_acquisition']
                if acquisition is not None:
                    allowed &= values['acq'] == acquisition
                rows = np.flatnonzero(allowed)
                positions, counts = np.unique(position_ids[rows], return_counts=True)
                assert len(positions) >= 20 and not np.intersect1d(positions, query_positions).size
                assert len(rows) and (values['rss'][rows] != 100).any()
                support = {name: values[name][rows] for name in ('rss', 'xy', 'acq')}
                units.append(dict(query_unit_id=f'uji/{building}/F{floor}/r{repeat}',
                    building=building, floor=floor, repeat=repeat,
                    representative_task_id=reference.task_id, reference_events=reference_events,
                    reference_meta_by_K=reference_meta, meta=reference.meta,
                    query_row_indices=truth['row_indices'].tolist(), query_groups=truth['groups'].tolist(),
                    calibration_row_indices=rows.tolist(), excluded_query_position_rows=np.flatnonzero(excluded).tolist(),
                    support_acquisition=acquisition, raw_floor_positions=len(np.unique(position_ids)),
                    raw_floor_scans=len(values['rss']), excluded_query_positions=len(query_positions),
                    excluded_query_position_scans=int(excluded.sum()),
                    calibration_positions=len(positions), calibration_scans=len(rows),
                    calibration_scans_per_position=counts.tolist(),
                    model_inputs_sha256={name: array_sha(value) for name, value in
                        [('S_rss', support['rss']), ('S_xy', support['xy']), ('S_acq', support['acq']),
                         ('Q_rss', reference.Q_rss)]}))
    assert len(units) == 39 and len(sparse) == 195
    tick = time.perf_counter()
    check_prediction = WKNN_S().predict(check_task)
    difference = float(np.max(np.abs(check_prediction - np.asarray(check_record['prediction']))))
    assert difference <= 1e-9
    import building_information_tab_trial as nearest_module
    csv_path = Path(next(iter(buildings['B0'].values()))['meta']['source'])
    code = [Path(__file__), Path(data.__file__), Path(protocol.__file__), Path(score.__file__),
            Path(methods_simple.__file__), Path(task_module.__file__), Path(run.__file__),
            Path(nearest_module.__file__)]
    frozen = dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        identity='Empirical target-only sufficient-calibration WKNN reference',
        query_units=39, sparse_events_reused=195, reference_budgets=list(BUDGETS),
        repeats=[0, 1, 2], seed=20261111, R=3, query_positions_requested=60,
        source_csv=binding(csv_path), target_data=run._data_identity(buildings, axes),
        code_sha256={str(path): run._sha(path) for path in code},
        nearest3_function_sha256=hashlib.sha256(inspect.getsource(nearest_module.nearest3).encode()).hexdigest(),
        source_consistency=dict(task_id=check_task.task_id, prediction_max_abs_difference=difference,
                                seconds=time.perf_counter() - tick),
        review=dict(
            given_information='Target-floor labelled scans at all legal non-Query positions and original Query RSS.',
            learning_object='Existing deterministic 3-WKNN with its original missing-value and weighting rule.',
            partition='Exclude every scan at each Query coordinate before applying the original support-acquisition restriction.',
            deployment='WKNN receives an empty H, legal labelled S, the unchanged AP axis and Query RSS. Query coordinates remain in partitioning and scoring.',
            parameters='Missing RSS -110, RMS over S-observed APs, stable nearest3, inverse distance floor1e-8 and S-derived outward2m bounds.',
            acquisition='Use the original support_acquisition when defined. Use the original unseparated acquisition pool otherwise.',
            counterexamples='Zero RSS distance uses the existing floor. Dataset scan order preserves stable ties. Empty context and absent APs follow the existing error behavior.',
            interpretation='This measured reference varies both calibration positions and scans per position. Its cost is reported directly.',
            falsifiable_comparison='Compare the same Query units and position-balanced errors. More calibration can still worsen this fixed WKNN rule.',
            uncertainty='Repeated draws share recorded positions. This empirical reference carries no theoretical optimum claim.'),
        resources=dict(CPU_threads=2, GPU_devices=[]), references=references,
        units=units, sparse_references=sparse)
    run._write(output / 'freeze.json', frozen)
    print(json.dumps(dict(stage='prepared', units=39, sparse_references=195,
        acquisition_restricted_units=sum(unit['support_acquisition'] is not None for unit in units),
        positions_range=[min(unit['calibration_positions'] for unit in units), max(unit['calibration_positions'] for unit in units)],
        scans_range=[min(unit['calibration_scans'] for unit in units), max(unit['calibration_scans'] for unit in units)],
        source_prediction_difference=difference, freeze_sha256=run._sha(output / 'freeze.json'))), flush=True)


def statistics(records):
    errors = [error for record in records for error in record['scores']['position_errors']]
    positions = [record['calibration_positions'] for record in records]
    scans = [record['calibration_scans'] for record in records]
    return dict(query_units=len(records), query_positions=len(errors), mean=float(np.mean(errors)),
        p90=float(np.quantile(errors, .9)),
        calibration_positions_mean=float(np.mean(positions)),
        calibration_positions_min=min(positions), calibration_positions_max=max(positions),
        calibration_scans_mean=float(np.mean(scans)),
        calibration_scans_min=min(scans), calibration_scans_max=max(scans))


def summarize(records):
    buildings = sorted({record['building'] for record in records})
    return dict(overall=statistics(records),
        by_building={building: statistics([record for record in records if record['building'] == building])
                     for building in buildings},
        by_floor={f'{building}F{floor}': statistics([record for record in records
                    if record['building'] == building and record['floor'] == floor])
                  for building in buildings for floor in sorted({record['floor'] for record in records
                      if record['building'] == building})},
        by_draw={str(repeat): statistics([record for record in records if record['repeat'] == repeat])
                 for repeat in (0, 1, 2)})


def evaluate(output):
    started = time.perf_counter()
    frozen = read(output / 'freeze.json')
    freeze_sha = run._sha(output / 'freeze.json')
    for path, expected in frozen['code_sha256'].items():
        assert run._sha(path) == expected
    assert run._sha(frozen['source_csv']['path']) == frozen['source_csv']['sha256']
    buildings, axes = data.load('uji')
    assert run._data_identity(buildings, axes) == frozen['target_data']
    records = []
    for unit in frozen['units']:
        path = output / 'events/dense_wknn' / (unit['query_unit_id'] + '.json')
        if path.exists():
            event = read(path)
            assert event['freeze_sha256'] == freeze_sha and event['status'] == 'completed'
        else:
            values = buildings[unit['building']][unit['floor']]
            rows = unit['calibration_row_indices']
            query_rows = unit['query_row_indices']
            support = {name: values[name][rows] for name in ('rss', 'xy', 'acq')}
            query_rss = values['rss'][query_rows]
            actual = {name: array_sha(value) for name, value in
                [('S_rss', support['rss']), ('S_xy', support['xy']), ('S_acq', support['acq']), ('Q_rss', query_rss)]}
            assert actual == unit['model_inputs_sha256']
            task = Task('uji', unit['building'], unit['floor'], unit['calibration_positions'],
                        unit['repeat'], axes[unit['building']], {}, support, query_rss, unit['meta'])
            model = WKNN_S()
            tick = time.perf_counter()
            prediction = model.predict(task)
            prediction_seconds = time.perf_counter() - tick
            truth = dict(xy=values['xy'][query_rows], groups=np.asarray(unit['query_groups']))
            event = dict(query_unit_id=unit['query_unit_id'], task_id=unit['query_unit_id'],
                method='dense_wknn_target_only', building=unit['building'], floor=unit['floor'],
                repeat=unit['repeat'], status='completed', meta=unit['meta'],
                query_row_indices=query_rows, query_groups=unit['query_groups'],
                prediction=prediction.tolist(), scores=score.score(prediction, truth),
                calibration_positions=unit['calibration_positions'], calibration_scans=unit['calibration_scans'],
                calibration_scans_per_position=unit['calibration_scans_per_position'],
                calibration_source_row_ids=values['row_ids'][rows].tolist(),
                calibration_row_indices=rows, excluded_query_position_rows=unit['excluded_query_position_rows'],
                excluded_query_position_scans=unit['excluded_query_position_scans'],
                support_acquisition=unit['support_acquisition'],
                calibration_acquisition_ids=sorted(set(support['acq'].tolist())),
                query_acquisition_ids=sorted(set(values['acq'][query_rows].tolist())),
                diagnostics=model.diagnostics, prediction_seconds=prediction_seconds,
                reference_events=unit['reference_events'], reference_meta_by_K=unit['reference_meta_by_K'],
                model_inputs_sha256=actual, freeze_sha256=freeze_sha)
            run._write(path, event)
        records.append(event)
        print(json.dumps(dict(stage='query_unit_completed', query_unit=unit['query_unit_id'],
            completed_units=len(records), total_units=39, positions=event['calibration_positions'],
            scans=event['calibration_scans'], prediction_seconds=event['prediction_seconds'])), flush=True)

    sparse = []
    for reference in frozen['sparse_references']:
        source = Path(reference['path'])
        assert run._sha(source) == reference['sha256']
        path = output / 'events/reused_sparse_wknn_S' / (reference['task_id'] + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            shutil.copyfile(source, path)
        assert run._sha(path) == reference['sha256']
        event = read(path)
        sparse.append(dict(event, building=reference['building'],
            query_unit_id=f'uji/{reference["building"]}/F{event["floor"]}/r{event["repeat"]}',
            calibration_positions=event['K'], calibration_scans=event['diagnostics']['context_scans']))
    assert len(records) == 39 and len(sparse) == 195
    summary = dict(complete=True, method_identity=frozen['identity'], unique_query_units=39,
        original_sparse_events_reused=195, freeze_sha256=freeze_sha,
        dense=summarize(records), sparse_by_K={str(K): summarize([record for record in sparse if record['K'] == K])
                                             for K in BUDGETS},
        per_query_unit=[dict(query_unit_id=record['query_unit_id'], building=record['building'],
            floor=record['floor'], repeat=record['repeat'], calibration_positions=record['calibration_positions'],
            calibration_scans=record['calibration_scans'], prediction_seconds=record['prediction_seconds'],
            mean=record['scores']['mean'], p90=record['scores']['p90']) for record in records],
        cost=dict(evaluation_process_seconds=time.perf_counter() - started,
                  dense_prediction_seconds=sum(record['prediction_seconds'] for record in records),
                  CPU_threads=2, GPU_devices=[]),
        interpretation=dict(metric='Pooled position-balanced scan errors on the same 39 Query units.',
            calibration='Dense calibration includes every legal non-Query-position scan within the original Support acquisition rule.',
            repeated_draws='The 39 draws reuse recorded floor data. The five sparse budgets are repeated evaluations of these same Query units.',
            comparison='Both position counts and per-position scan counts vary. Report their measured costs separately.'),
        records=[dict(path=str((output / 'events/dense_wknn' / (record['query_unit_id'] + '.json')).relative_to(output)),
                      sha256=run._sha(output / 'events/dense_wknn' / (record['query_unit_id'] + '.json')))
                 for record in records])
    run._write(output / 'summary.json', summary)
    csv_rows = []
    for label, selected in [('dense', records)] + [(str(K), [record for record in sparse if record['K'] == K])
                                                   for K in BUDGETS]:
        groups = [('all', 'all', 'all', selected)]
        groups.extend((building, 'all', 'all', [record for record in selected if record['building'] == building])
                      for building in BUILDINGS)
        groups.extend((record['building'], record['floor'], record['repeat'], [record]) for record in selected)
        for building, floor, repeat, group in groups:
            csv_rows.append(dict(reference='sufficient_calibration' if label == 'dense' else 'reused_sparse',
                nominal_K=label, building=building, floor=floor, draw=repeat, **statistics(group)))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    run._write(output / 'complete.json', dict(complete=True, dense_events=39, sparse_events_reused=195,
        summary_sha256=run._sha(output / 'summary.json'), summary_csv_sha256=run._sha(output / 'summary.csv'),
        freeze_sha256=freeze_sha))
    print(json.dumps(dict(stage='completed', dense_events=39, sparse_events_reused=195,
                         dense=summary['dense']['overall'], cost=summary['cost'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'run'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        (prepare if args.phase == 'prepare' else evaluate)(args.output)
