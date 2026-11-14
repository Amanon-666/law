"""Read-only geometry export for completed static-resample-v2 tasks.

Run from the original server src directory with the existing DKT-UJI Python.
Only official raw arrays and completed result JSON are read. JSON goes to stdout.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from law_bench.data import load
from law_bench.protocol import make_tasks
from law_bench.run import _data_identity


BASE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110')
RESULTS = BASE / 'results'
SRC = BASE / 'src'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path):
    return {'path': str(path), 'sha256': digest(path)}


def query_binding(rows, groups):
    payload = {'query_row_indices': rows, 'query_groups': groups}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main(extraction_code_sha256):
    translator_path = SRC / 'vertical_translator/translator.py'
    assignments = ast.parse(translator_path.read_text()).body
    constants = {}
    for node in assignments:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == 'SLOTS':
                constants['SLOTS'] = ast.literal_eval(node.value)
            if isinstance(target, ast.Tuple) and [n.id for n in target.elts] == ['R_SAME', 'R_SMOOTH', 'R_LABEL']:
                constants.update(zip(['R_SAME', 'R_SMOOTH', 'R_LABEL'], ast.literal_eval(node.value)))
    translator_sha = digest(translator_path)
    assert digest(SRC / 'claude_hurdle_20261110/translator.py') == translator_sha

    configurations = [
        (f'paper_uji_B{building}_20261111', f'paper_uji_B{building}_20261111',
         f'paper_uji_B{building}_ext_20261111/events/refi_GP/uji/B{building}/F0/K1/r0.json')
        for building in range(3)
    ]
    configurations.append((
        'paper_tampere_TIE1_F0_ext_cuda_subset_20261114',
        'paper_tampere_TIE1_20261111',
        'paper_tampere_TIE1_F0_ext_cuda_subset_20261114/events/refi_GP/tampere/TIE1/F0/K1/r0.json',
    ))
    loaded = {}
    raw_files = {}
    provenance = []
    families = []
    tasks = []
    for freeze_run, reference_run, model_binding in configurations:
        freeze_path = RESULTS / freeze_run / 'freeze.json'
        freeze = json.loads(freeze_path.read_text())
        config = freeze['configuration']
        assert config['protocol'] == 'static-resample-v2'
        assert config['budgets'] == [1, 3, 5]
        for filename in ['data.py', 'protocol.py', 'run.py', 'task.py']:
            assert digest(SRC / 'law_bench' / filename) == freeze['code'][filename]
        binding_path = RESULTS / model_binding
        bound_event = json.loads(binding_path.read_text())
        bound_code = bound_event['diagnostics']['provenance']['refi']['code']
        assert bound_code['claude_hurdle_20261110/translator.py'] == translator_sha
        assert digest(SRC / 'frozen_hurdle_translator.py') == bound_code['frozen_hurdle_translator.py']

        dataset = config['dataset']
        building = config['building']
        if dataset not in loaded:
            loaded[dataset] = load(dataset, root=config['root'])
        buildings, axes = loaded[dataset]
        floors = buildings[building]
        identity = _data_identity({building: floors}, {building: axes[building]})
        assert identity == freeze['target_data']
        data_sources = next(iter(floors.values()))['meta']['source']
        for filename in [data_sources] if isinstance(data_sources, str) else data_sources:
            path = Path(filename)
            raw_files[str(path)] = record(path)
        provenance.append({
            'dataset': dataset, 'building': building, 'configuration': config,
            'freeze': record(freeze_path), 'target_data': identity,
            'translator_binding_event': record(binding_path),
        })

        for floor in config['floors'] or sorted(floors):
            for repeat in config['repeats']:
                rebuilt = make_tasks(
                    dataset, building, floors, axes[building], floor,
                    budgets=tuple(config['budgets']), repeat=repeat, seed=config['seed'],
                    R=config['R'], query_positions=config['query_positions'],
                )
                first_task, first_truth = rebuilt[0]
                groups = np.asarray(first_truth['groups'])
                group_ids = np.unique(groups)
                query_xy = np.stack([first_truth['xy'][groups == group][0] for group in group_ids])
                for group, xy in zip(group_ids, query_xy):
                    assert np.array_equal(first_truth['xy'][groups == group], np.broadcast_to(xy, first_truth['xy'][groups == group].shape))
                all_history_xy = np.unique(np.concatenate([h['xy'] for h in first_task.H.values()]), axis=0)
                slot_floors = [f for f in first_task.H if f - floor in constants['SLOTS']]
                slot_history_xy = np.concatenate([first_task.H[f]['xy'] for f in slot_floors])
                slot_history_rss = np.concatenate([first_task.H[f]['rss'] for f in slot_floors])
                all_distance = cKDTree(all_history_xy).query(query_xy)[0]
                slot_tree = cKDTree(slot_history_xy)
                slot_distance = slot_tree.query(query_xy)[0]
                local_rows = slot_tree.query_ball_point(query_xy, r=constants['R_SMOOTH'])
                history_active = (slot_history_rss != 100).any(0)
                active_count = int(history_active.sum())
                assert active_count > 0
                local_ap_count = [int((slot_history_rss[indices] != 100).any(0).sum()) for indices in local_rows]
                local_fraction = np.asarray(local_ap_count, dtype=float) / active_count
                q_rows = first_truth['row_indices'].tolist()
                q_groups = groups.tolist()
                family = {
                    'dataset': dataset, 'building': building, 'floor': floor, 'repeat': repeat,
                    'query_binding_sha256': query_binding(q_rows, q_groups),
                    'query_group_ids': group_ids.tolist(), 'query_xy': query_xy.tolist(),
                    'query_row_indices': q_rows, 'query_groups': q_groups,
                    'history_floors': sorted(first_task.H), 'slot_history_floors': slot_floors,
                    'nearest_all_H_distance_m': all_distance.tolist(),
                    'nearest_slot_H_distance_m': slot_distance.tolist(),
                    'slot_H_active_AP_count': active_count,
                    'local_slot_H_AP_count_within_R_SMOOTH': local_ap_count,
                    'local_slot_H_AP_available_fraction': local_fraction.tolist(),
                    'tasks': [],
                }
                previous_support = None
                for task, truth in rebuilt:
                    event_path = RESULTS / reference_run / 'events/refi' / (task.task_id + '.json')
                    event = json.loads(event_path.read_text())
                    assert event['status'] == 'completed'
                    assert event['query_row_indices'] == truth['row_indices'].tolist() == q_rows
                    assert event['query_groups'] == truth['groups'].tolist() == q_groups
                    assert event['meta'] == task.meta
                    support_xy = np.unique(task.S['xy'], axis=0)
                    assert len(support_xy) == task.K
                    if previous_support is not None:
                        assert set(map(tuple, previous_support)) <= set(map(tuple, support_xy))
                    previous_support = support_xy
                    support_distance = cKDTree(support_xy).query(query_xy)[0]
                    task_geometry = {
                        'task_id': task.task_id, 'K': task.K, 'repeat': repeat,
                        'reference': record(event_path), 'meta': task.meta,
                        'support_xy': support_xy.tolist(),
                        'nearest_S_distance_m': support_distance.tolist(),
                    }
                    family['tasks'].append(task_geometry)
                    tasks.append(task.task_id)
                families.append(family)

    assert len(tasks) == 120 and len(families) == 40
    output = {
        'status': 'completed', 'prepared_on': '2026-11-14',
        'extraction_code_sha256': extraction_code_sha256,
        'server_base': str(BASE), 'protocol': 'static-resample-v2',
        'operation': 'Read-only raw data reconstruction and post-hoc geometry; no method prediction or fitting.',
        'constants': constants,
        'translator_source': record(translator_path),
        'frozen_adapter_source': record(SRC / 'frozen_hurdle_translator.py'),
        'protocol_sources': [record(SRC / 'law_bench' / name) for name in ['data.py', 'protocol.py', 'run.py', 'task.py']],
        'raw_sources': list(raw_files.values()),
        'source_runs': provenance,
        'validation': {'target_data_identity_verified': True, 'frozen_loader_protocol_hashes_verified': True,
                       'query_rows_groups_and_meta_equal': True, 'nested_support_verified': True,
                       'tasks': len(tasks), 'families': len(families)},
        'distance_definition': '2D Euclidean metres in the frozen common XY frame. All H is every legal other target-building floor. Slot H uses the frozen translator SLOTS only. Each Query group has one verified actual XY.',
        'local_AP_definition': 'At the actual Query XY, number of AP columns detected (!=100) in at least one slot-H scan within R_SMOOTH, divided by AP columns detected anywhere in slot H. This is a post-hoc H-only availability fraction, with no Query RSS or labels entering inference.',
        'families': families,
    }
    print(json.dumps(output, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main(digest(Path(__file__)))
