"""Bind unchanged tower H arrays to completed legacy single-source GP fields."""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from law_bench import run
from law_bench.data import load
from law_bench.method_retained_gp import history_key
from law_bench.protocol import make_tasks

BASE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110')
CORE = Path('/home/panyushuo/projects/panyushuo/ReFi-UJI-Core-20261031')
WORK = Path(__file__).resolve().parent
OLD = CORE/'results/uts_official_GP_20261111'
SOURCE = BASE/'results/paper_uts_New_IT_20261111'
CACHE = BASE/'results/paper_external_fullH_cache_20261113'
PARAMETERS = CORE/'results/paired_assets_20261106/parameters.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    freeze, complete = read(OLD/'freeze.json'), read(OLD/'histories_complete.json')
    parameter_sha = digest(PARAMETERS)
    assert complete['complete'] and complete['freeze_sha256'] == digest(OLD/'freeze.json')
    assert parameter_sha == freeze['parameters_sha256']
    old_metadata = {r['path']: r['sha256'] for r in complete['records']}
    buildings, axes = load('uts')
    floors, aps = buildings['New IT'], axes['New IT']
    original = read(SOURCE/'freeze.json')
    target_identity = run._data_identity({'New IT': floors}, {'New IT': aps})
    assert target_identity == original['target_data']
    identity = SimpleNamespace(dataset='uts', building='New IT', ap_ids=aps, H=floors)
    legacy_identity = SimpleNamespace(dataset='uts', building='FEIT',
        ap_ids=[ap.rsplit('/', 1)[-1] for ap in aps], H=floors)
    records = []
    for task_floor in (4, 8):
        for task, truth in make_tasks('uts', 'New IT', floors, aps, task_floor,
                budgets=(1, 3, 5), repeat=0, seed=20261111, R=3, query_positions=60):
            path = SOURCE/'events/refi'/(task.task_id+'.json')
            event = read(path)
            assert event['status'] == 'completed'
            assert event['query_row_indices'] == truth['row_indices'].tolist()
            assert event['query_groups'] == truth['groups'].tolist()
            assert event['meta'] == task.meta
            assert sorted(task.H) == [f for f in range(4, 13) if f != task_floor]
            records.append(dict(task_id=task.task_id, source_event=str(path),
                source_event_sha256=digest(path), H_floors=sorted(task.H),
                query_scans=len(task.Q_rss), query_positions=task.meta['query_positions'],
                Q_rows_groups_meta_match=True, original_refi_seconds=event['seconds']))
    imported = []
    for history in freeze['histories']:
        floor = history['floor']
        if floor not in floors:
            continue
        data = floors[floor]
        old_key = history_key(legacy_identity, floor, parameter_sha)
        key = history_key(identity, floor, parameter_sha)
        assert old_key == history['key'], (floor, 'Full H arrays/AP axis/parameters mismatch')
        detected = np.flatnonzero((data['rss'] != 100).any(0)).tolist()
        assert detected == history['APs']
        origin = data['xy'].mean(0)
        expected_z = np.unique(data['xy']-origin, axis=0)
        destination = CACHE/key
        destination.mkdir(parents=True, exist_ok=True)
        binding = dict(kind='uts_official_GP_20261111_verified_single_source_import',
            source=str(OLD), source_freeze_sha256=digest(OLD/'freeze.json'),
            source_complete_sha256=digest(OLD/'histories_complete.json'),
            parameters_sha256=parameter_sha, floor=floor, legacy_history_key=old_key,
            history_key=key, rows=len(data['rss']), positions=len(expected_z),
            origin=origin.tolist(), fields=len(detected),
            verified='Full H rss/xy/acq and raw AP axis reproduce original history key',
            transformation='Only building and AP namespace labels changed; npz bytes unchanged',
            reused_legacy_system_predictions=False)
        write(destination/'legacy_binding.json', binding)
        old_seconds = 0.0
        for ap in detected:
            source = OLD/'cache'/old_key/f'{ap}.npz'
            metadata = source.with_suffix('.json')
            assert digest(metadata) == old_metadata[str(metadata)]
            saved = read(metadata)
            assert saved['history_key'] == old_key
            assert saved['ap_id'] == legacy_identity.ap_ids[ap]
            assert digest(source) == saved['sha256']
            with np.load(source, allow_pickle=False) as arrays:
                assert np.array_equal(arrays['origin'], origin)
                assert np.array_equal(arrays['z'], expected_z)
            target = destination/source.name
            target_metadata = target.with_suffix('.json')
            record = dict(history_key=key, ap_id=aps[ap], sha256=saved['sha256'],
                solver=saved['solver'], imported_from=dict(path=str(source),
                    sha256=saved['sha256'], metadata_sha256=digest(metadata),
                    binding_path=str(destination/'legacy_binding.json')))
            if target.exists():
                assert digest(target) == saved['sha256']
                assert read(target_metadata) == record
            else:
                shutil.copyfile(source, target)
                write(target_metadata, record)
            old_seconds += saved['seconds']
        imported.append(dict(binding, original_field_fit_seconds=old_seconds))
        print(json.dumps(dict(floor=floor, fields=len(detected),
            positions=len(expected_z), original_fit_seconds=old_seconds)), flush=True)
    assert len(imported) == 9
    write(WORK/'preparation.json', dict(status='prepared_before_formal_run',
        task_count=len(records), tasks=records, target_data=target_identity,
        source_data=original['source_data'], source_run=str(SOURCE),
        source_model_sha256=read(SOURCE/'source_models/refi.json')['sha256'],
        parameters_sha256=parameter_sha, imported_histories=imported,
        reused_fields=sum(h['fields'] for h in imported), expected_new_GP_field_fits=0,
        scope='UTS tower F4 edge/F8 middle only; K1/3/5 r0, all other tower H floors',
        data_policy='Frozen three-UJI five-model; unchanged H/S, no Query labels in adaptation',
        comparison_boundary='Legacy 16-floor/B0B2 system scores are not reused'))


if __name__ == '__main__':
    main()
