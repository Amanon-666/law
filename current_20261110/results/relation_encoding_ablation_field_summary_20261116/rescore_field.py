"""Summarize saved field predictions with their original Query coordinates."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(records):
    errors = [value for record in records for value in record['scores']['position_errors']]
    return dict(
        unique_task_scenarios=len({record['task_id'] for record in records}),
        training_seed_runs=len({record['training_seed'] for record in records}),
        task_seed_observations=len(records),
        query_position_budget_seed_observations=len(errors),
        mean=float(np.mean(errors)),
        p90=float(np.quantile(errors, .9)),
    )


def grouped(records):
    floors = sorted({record['floor'] for record in records})
    return dict(
        overall=statistics(records),
        by_K={str(K): statistics([record for record in records if record['K'] == K])
              for K in (1, 3, 5)},
        by_floor={str(floor): statistics([record for record in records if record['floor'] == floor])
                  for floor in floors},
        by_floor_K={
            str(floor): {
                str(K): statistics([record for record in records
                                    if record['floor'] == floor and record['K'] == K])
                for K in (1, 3, 5)
            }
            for floor in floors
        },
    )


def main(args):
    raw = args.raw / 'results'
    frozen = read(raw / 'freeze.json')
    truth_document = read(args.truth)
    assert truth_document['source_csv_sha256'] == next(iter(frozen['original_inputs'].values()))
    assert digest(args.score_module) == 'c4abfa7fbe008ac994a6ba4790c7ef3f25da11d07c5391367043936d95af6f3e'
    spec = importlib.util.spec_from_file_location('original_position_score', args.score_module)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)

    records, field_records = [], []
    max_original_error_difference = 0.
    for path in sorted((raw / 'events').rglob('*.json')):
        event = read(path)
        scene = truth_document['truths'][f'F{event["floor"]}/r{event["repeat"]}']
        assert event['query_row_indices'] == scene['query_row_indices']
        assert event['query_groups'] == scene['query_groups']
        truth = dict(xy=np.asarray(scene['xy']), groups=np.asarray(scene['query_groups']))
        prediction = event['diagnostics']['field_projected']
        scores = scorer.score(prediction, truth)
        seed, arm = event['training_seed'], event['arm']
        record = dict(task_id=event['task_id'], method=arm + '_field', training_seed=seed,
                      member_count=1, floor=event['floor'], K=event['K'], repeat=event['repeat'],
                      scores=scores, source_event=str(path), source_event_sha256=digest(path))
        field_records.append(record)
        records.append(record)
        records.append(dict(record, method=arm + '_complete', scores=event['scores']))
        if arm == 'original_single_member':
            original_path = (args.reference / 'events/A_field' / f'seed_{seed}' /
                             (event['task_id'] + '.json'))
            original = read(original_path)
            assert original['prediction'] == prediction
            difference = np.max(np.abs(np.asarray(scores['position_errors']) -
                                       np.asarray(original['scores']['position_errors'])))
            max_original_error_difference = max(max_original_error_difference, float(difference))
    assert len(field_records) == 216 and max_original_error_difference <= 1e-9

    for method in ('adopted_5seed_field', 'adopted_5seed_complete'):
        paths = sorted((args.reference / 'events' / method / 'reference').rglob('*.json'))
        assert len(paths) == 36
        for path in paths:
            event = read(path)
            scene = truth_document['truths'][f'F{event["floor"]}/r{event["repeat"]}']
            assert event['query_row_indices'] == scene['query_row_indices']
            assert event['query_groups'] == scene['query_groups']
            records.append(dict(task_id=event['task_id'], method=method, training_seed=None,
                                member_count=5, floor=event['floor'], K=event['K'],
                                repeat=event['repeat'], scores=event['scores'],
                                source_event=str(path), source_event_sha256=digest(path)))

    methods = {}
    for method in sorted({record['method'] for record in records}):
        selected = [record for record in records if record['method'] == method]
        seeds = sorted({record['training_seed'] for record in selected},
                       key=lambda seed: -1 if seed is None else seed)
        by_seed = {str(seed): grouped([record for record in selected
                                      if record['training_seed'] == seed]) for seed in seeds}
        methods[method] = dict(member_count=selected[0]['member_count'],
                               pooled_over_training_seeds=grouped(selected),
                               by_training_seed=by_seed)
        if seeds != [None]:
            means = [by_seed[str(seed)]['overall']['mean'] for seed in seeds]
            methods[method]['training_seed_mean_error'] = dict(
                values=means, mean=float(np.mean(means)), sample_std=float(np.std(means, ddof=1)))

    provenance = dict(
        raw_directory=str(args.raw.resolve()),
        raw_light_manifest_sha256=digest(args.raw / 'file_manifest_light.json'),
        original_freeze_sha256=digest(raw / 'freeze.json'),
        query_truth_path=str(args.truth.resolve()), query_truth_sha256=digest(args.truth),
        source_csv_sha256=truth_document['source_csv_sha256'],
        data_module_sha256=truth_document['data_module_sha256'],
        score_module_path=str(args.score_module.resolve()), score_module_sha256=digest(args.score_module),
        derivation_script_path=str(Path(__file__).resolve()), derivation_script_sha256=digest(Path(__file__)),
        original_field_reference_directory=str(args.reference.resolve()),
        original_field_position_error_max_abs_difference=max_original_error_difference,
        model_inference_performed=False, training_performed=False,
    )
    args.field_scores.write_text(json.dumps(dict(complete=True, events=216, provenance=provenance,
                                                records=field_records), indent=2, allow_nan=False) + '\n')
    output = dict(
        complete=True, unique_task_scenarios=36, training_seeds=[1, 2, 3],
        derived_field_events=216, methods=methods, provenance=provenance,
        field_scores_path=str(args.field_scores.resolve()), field_scores_sha256=digest(args.field_scores),
        metric='Original position-balanced score. P90 is computed from pooled position errors.',
        aggregation='Each training seed is retained. Seed repetitions describe training variability on the same 36 scenarios.',
    )
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    rows = []
    for method, values in methods.items():
        sections = [('pooled_training_seeds', values['pooled_over_training_seeds'])]
        sections.extend(values['by_training_seed'].items())
        for seed, section in sections:
            groups = [('all', 'all', section['overall'])]
            groups.extend(('all', K, value) for K, value in section['by_K'].items())
            groups.extend((floor, 'all', value) for floor, value in section['by_floor'].items())
            groups.extend((floor, K, value) for floor, ks in section['by_floor_K'].items()
                          for K, value in ks.items())
            for floor, K, value in groups:
                rows.append(dict(method=method, member_count=values['member_count'],
                                 training_seed=seed, floor=floor, K=K, **value))
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(output=str(args.output.resolve()),
                          original_field_position_error_max_abs_difference=max_original_error_difference,
                          pooled={method: values['pooled_over_training_seeds']['overall']
                                  for method, values in methods.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'truth', 'reference', 'score-module', 'field-scores', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
