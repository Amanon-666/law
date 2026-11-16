"""Paired single-member field comparison using the shared original GP cache."""
import argparse
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from source_candidate import BASE, CORE, GP_CACHE, KEEP, OUTPUT, SEEDS, bindings, digest, read, write, array_digest

sys.path.insert(0, str(BASE/'src'))
from law_bench import run
from law_bench.data import load
from law_bench.protocol import make_tasks
from law_bench.score import score, aggregate
sys.path.insert(0, str(CORE))
from frozen_hurdle_translator import FrozenHurdleTranslator
from hurdle_gp_pool import fuse_hurdle_gp

REFERENCE = BASE/'results/paper_uji_B1_ext_20261111'


class SingleMemberTranslator(FrozenHurdleTranslator):
    def __init__(self, member, source_freeze):
        # Independent member identity, using the unchanged deployment arithmetic.
        self.models = [member]
        self.freeze = source_freeze


def pooled(records):
    errors = [e for record in records for e in record['scores']['position_errors']]
    return dict(tasks=len(records), query_position_budget_count=len(errors),
        mean=float(np.mean(errors)), p90=float(np.quantile(errors, .9)))


def evaluate(output, gp_cache):
    frozen, trained = read(output/'freeze.json'), read(output/'trained.json')
    freeze_sha = digest(output/'freeze.json')
    assert trained['complete'] and trained['freeze_sha256'] == freeze_sha
    assert frozen['code_sha256'] == {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    for path, expected in frozen['deployment_code_sha256'].items():
        assert digest(path) == expected
    original, _, original_members, _ = bindings()
    cache_complete = read(gp_cache/'complete.json')
    assert cache_complete['complete'] and len(cache_complete['tasks']) == 36
    assert not cache_complete['training_performed'] and cache_complete['new_history_fields'] == 0
    cache_freeze = read(gp_cache/'freeze.json')
    assert cache_complete['freeze_sha256'] == digest(gp_cache/'freeze.json')
    assert cache_freeze['reference_freeze_sha256'] == digest(REFERENCE/'freeze.json')
    reference_freeze = read(REFERENCE/'freeze.json')
    buildings, axes = load('uji')
    assert run._data_identity({'B1': buildings['B1']}, {'B1': axes['B1']}) == reference_freeze['target_data']
    assert run._data_identity({b: buildings[b] for b in ['B0', 'B2']},
        {b: axes[b] for b in ['B0', 'B2']}) == reference_freeze['source_data']
    translators, model_bindings = {}, {}
    for seed in SEEDS:
        a = next(r for r in original_members if r['seed'] == seed)
        b = next(r for r in trained['models'] if r['seed'] == seed)
        for arm, record, source_freeze in [('original_single_member', a, original),
                                         ('drop_relation_single_member', b, frozen)]:
            assert digest(record['path']) == record['sha256']
            member = joblib.load(record['path'])
            assert member['seed'] == seed and member['source_buildings'] == [0, 2]
            if arm == 'drop_relation_single_member':
                assert member['freeze_sha256'] == freeze_sha and np.array_equal(member['kept_features'], KEEP)
            translators[(arm, seed)] = SingleMemberTranslator(member, source_freeze)
            model_bindings[(arm, seed)] = record
    records = {key: [] for key in translators}
    adopted = []
    completed_tasks = 0
    for floor in sorted(buildings['B1']):
        for repeat in (0, 1, 2):
            tasks = make_tasks('uji', 'B1', buildings['B1'], axes['B1'], floor,
                budgets=(1, 3, 5), repeat=repeat, seed=20261111, R=3, query_positions=60)
            for task, truth in tasks:
                original_path = REFERENCE/'events/refi_GP'/(task.task_id+'.json')
                old = read(original_path)
                assert old['status'] == 'completed'
                assert old['query_row_indices'] == truth['row_indices'].tolist()
                assert old['query_groups'] == truth['groups'].tolist() and old['meta'] == task.meta
                adopted.append(old)
                cache_path = gp_cache/(task.task_id+'.npz')
                metadata = read(cache_path.with_suffix('.json'))
                binding = metadata['binding']
                assert binding['task_id'] == task.task_id and binding['reference_event_sha256'] == digest(original_path)
                assert binding['query_rows'] == old['query_row_indices'] and binding['query_groups'] == old['query_groups']
                assert binding['meta'] == task.meta and digest(cache_path) == metadata['sha256']
                assert metadata['new_history_fields'] == 0 and metadata['original_projection_max_abs_difference'] <= 1e-6
                with np.load(cache_path, allow_pickle=False) as saved:
                    gp = {key: saved[key] for key in ['grid', 'scores', 'raw', 'projected', 'aps', 'bounds']}
                candidate_hash = None
                for (arm, seed), translator in translators.items():
                    record_path = output/'events'/f'{arm}_seed{seed}'/(task.task_id+'.json')
                    model_record = model_bindings[(arm, seed)]
                    if record_path.exists():
                        event = read(record_path)
                        assert event['status'] == 'completed' and event['meta'] == task.meta
                        assert event['query_row_indices'] == old['query_row_indices'] and event['query_groups'] == old['query_groups']
                        assert event['diagnostics']['model_sha256'] == model_record['sha256']
                        assert event['diagnostics']['GP_surface_sha256'] == metadata['sha256']
                    else:
                        prepared = translator.prepare({f'B1F{f}': h for f, h in task.H.items()}, task.S, f'B1F{floor}')
                        field = prepared.predict(task.Q_rss)
                        assert np.array_equal(field['aps'], gp['aps']) and np.array_equal(prepared.bounds, gp['bounds'])
                        fused = fuse_hurdle_gp(prepared.cand_xy, field['log_position_mass'],
                            gp['grid'], gp['scores'], gp['raw'], prepared.bounds)
                        prediction = fused['predictions']['probability_half']
                        event = dict(task_id=task.task_id, floor=floor, K=task.K, repeat=repeat,
                            arm=arm, training_seed=seed, member_count=1, status='completed', meta=task.meta,
                            prediction=prediction.tolist(), scores=score(prediction, truth),
                            query_row_indices=old['query_row_indices'], query_groups=old['query_groups'],
                            diagnostics=dict(model_path=model_record['path'], model_sha256=model_record['sha256'],
                                candidate_sha256=array_digest(prepared.cand_xy), GP_surface_path=str(cache_path),
                                GP_surface_sha256=metadata['sha256'], original_reference_event_sha256=digest(original_path),
                                field_projected=field['projected'].tolist(), GP_projected=gp['projected'].tolist(),
                                pool='probability_half', freeze_sha256=freeze_sha))
                        record_path.parent.mkdir(parents=True, exist_ok=True)
                        write(record_path, event)
                    if candidate_hash is None:
                        candidate_hash = event['diagnostics']['candidate_sha256']
                    assert candidate_hash == event['diagnostics']['candidate_sha256']
                    records[(arm, seed)].append(event)
                completed_tasks += 1
                print(json.dumps(dict(stage='paired_task_complete', scenarios=completed_tasks,
                    unique_scenarios=36, task_id=task.task_id)), flush=True)
    assert completed_tasks == 36 and all(len(values) == 36 for values in records.values())
    per_seed = {str(seed): {arm: dict(overall=pooled(records[(arm, seed)]),
        by_K=aggregate(records[(arm, seed)])) for arm in ['original_single_member', 'drop_relation_single_member']}
        for seed in SEEDS}
    variability = {}
    for arm in ['original_single_member', 'drop_relation_single_member']:
        variability[arm] = {metric: dict(mean=float(np.mean([per_seed[str(s)][arm]['overall'][metric] for s in SEEDS])),
            sample_std=float(np.std([per_seed[str(s)][arm]['overall'][metric] for s in SEEDS], ddof=1)))
            for metric in ['mean', 'p90']}
    paired = [dict(seed=seed,
        delta_mean=per_seed[str(seed)]['drop_relation_single_member']['overall']['mean']-
            per_seed[str(seed)]['original_single_member']['overall']['mean'],
        delta_p90=per_seed[str(seed)]['drop_relation_single_member']['overall']['p90']-
            per_seed[str(seed)]['original_single_member']['overall']['p90']) for seed in SEEDS]
    summary = dict(complete=True, unique_task_scenarios=36, training_seeds=list(SEEDS),
        paired_member_count=1, scored_events=216, freeze_sha256=freeze_sha,
        GP_cache_complete_sha256=digest(gp_cache/'complete.json'),
        by_training_seed=per_seed, training_seed_variability=variability, paired_seed_deltas=paired,
        adopted_five_member_reference=dict(member_count=5, overall=pooled(adopted), by_K=aggregate(adopted),
            source=str(REFERENCE), predictions_recomputed=False),
        aggregation='Pool per-position errors within each seed. Report variability across three paired training runs on the same 36 scenarios.',
        model_bindings=[dict(record, arm=arm) for (arm, seed), record in model_bindings.items()])
    write(output/'summary.json', summary)
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['arm', 'training_seed', 'K', 'tasks', 'mean', 'p90'])
        for seed in SEEDS:
            for arm, values in per_seed[str(seed)].items():
                for K, item in [('pooled', values['overall']), *values['by_K'].items()]:
                    writer.writerow([arm, seed, K, item['tasks'], item['mean'], item['p90']])
        reference = summary['adopted_five_member_reference']
        for K, item in [('pooled', reference['overall']), *reference['by_K'].items()]:
            writer.writerow(['adopted_five_member_reference', '', K, item['tasks'], item['mean'], item['p90']])
    print(json.dumps(dict(stage='relation_ablation_complete', unique_scenarios=36, training_seeds=list(SEEDS))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--gp-cache', type=Path, default=GP_CACHE)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        evaluate(args.output, args.gp_cache)
