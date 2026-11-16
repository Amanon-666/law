"""Frozen source-only deletion of seven relationship-feature columns."""
import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

PROJECTS = Path('/home/panyushuo/projects/panyushuo')
CORE = PROJECTS/'ReFi-UJI-Core-20261031'
BASE = PROJECTS/'paper_bench_wt/current_20261110'
ORIGINAL = CORE/'results/claude_hurdle_sources02_ens5_20261111'
FRAMEWORK = BASE/'results/tab_field_framework_formal_20261115/source'
GP_CACHE = BASE/'results/tab_field_framework_formal_20261115/gp_surfaces'
OUTPUT = BASE/'results/relation_encoding_ablation_20261115'
DROP = np.array([43, 46, 47, 48, 49, 51, 54], dtype=int)
KEEP = np.array([*range(43), 44, 45, 50, 52, 53], dtype=int)
SEEDS = (1, 2, 3)
sys.path.insert(0, str(CORE/'claude_hurdle_20261110'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def crop_table(table):
    return (table[0][:, KEEP], *table[1:])


def bindings():
    original, complete = read(ORIGINAL/'freeze.json'), read(ORIGINAL/'complete.json')
    assert complete['complete'] and complete['freeze_sha256'] == digest(ORIGINAL/'freeze.json')
    assert original['source_buildings'] == [0, 2] and original['seeds'] == [1, 2, 3, 4, 5]
    assert original['draws'] == 4
    for name in original['upstream_git_blobs']:
        path = CORE/'claude_hurdle_20261110'/name
        expected = next(v for p, v in original['code'].items() if p.endswith('/'+name))
        assert digest(path) == expected
    for path, expected in original['inputs'].items():
        assert digest(path) == expected
    versions = {name: version(name) for name in original['versions']}
    assert versions == original['versions']
    members = [r for r in complete['models'] if r['seed'] in SEEDS]
    for record in members:
        assert digest(record['path']) == record['sha256']
        model = joblib.load(record['path'])
        assert model['seed'] == record['seed'] and model['source_buildings'] == [0, 2]
        assert model['draws'] == 4 and model['freeze_sha256'] == complete['freeze_sha256']
        assert model['source_rows_by_floor'] == original['source_rows_by_floor']
    return original, complete, members, versions


def mapping_check():
    # The appended OOF mean is relative to the selected outer columns.
    x = np.arange(110, dtype=np.float32).reshape(2, 55)
    local = np.array([0, 7, 40, 47])
    outer = KEEP[local]
    reduced = x[:, KEEP][:, local]
    assert np.array_equal(reduced, x[:, outer])
    mean = np.array([1.5, 2.5])
    assert np.array_equal(np.column_stack((reduced, mean)), np.column_stack((x[:, outer], mean)))
    assert len(outer) == reduced.shape[1] and not np.intersect1d(outer, DROP).size


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    original, complete, members, versions = bindings()
    trace_path = FRAMEWORK/'source_cells.py'
    spec = importlib.util.spec_from_file_location('relation_source_unit_trace', trace_path)
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
    import hurdle as H
    mapping_check()
    frozen = dict(status='source_frozen_before_training', source_buildings=[0, 2],
        training_seeds=list(SEEDS), original_input_features=55,
        kept_feature_columns=KEEP.tolist(), dropped_feature_columns=DROP.tolist(), features=48,
        dropped_semantics=['target-minus-history AP delta', 'pooled delta', 'home-slot group delta',
            'history-strength group delta', 'nearest delta', 'nearest historical reference', 'seen fraction'],
        retained_support_semantics=['target ymax', 'target ymean', 'nearest target y', 'distance', 'K'],
        draws=4, cell_table_max_rows=6000, count_weights='Original n/nd weights and within-position variance',
        own_crossfit='Original two alternating source-floor halves, fitted separately for each deleted-column member',
        source_permission='Same full source corpus, cell selection and RNG calls as each original A member',
        original_freeze_path=str(ORIGINAL/'freeze.json'), original_freeze_sha256=complete['freeze_sha256'],
        original_complete_sha256=digest(ORIGINAL/'complete.json'), original_members=members,
        original_source_floors=original['source_floors'], original_source_rows_by_floor=original['source_rows_by_floor'],
        original_code=original['code'], original_inputs=original['inputs'], versions=versions,
        source_trace_path=str(trace_path), source_trace_sha256=digest(trace_path),
        deployment_code_sha256={str(p): digest(p) for p in [CORE/'frozen_hurdle_translator.py',
            CORE/'hurdle_gp_pool.py', CORE/'vertical_translator/translator.py']},
        code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
        evaluation=dict(dataset='uji', building='B1', floors=[0, 1, 2, 3], budgets=[1, 3, 5],
            repeats=[0, 1, 2], R=3, query_positions=60, protocol='static-resample-v2', seed=20261111,
            source_runs=3, unique_task_scenarios=36, arms=['original_single_member', 'drop_relation_single_member'],
            adopted_reference='Original five-member ReFi+GP, shown separately',
            reader=dict(beta=128., eps=.02, offsets=True), pool='probability_half',
            GP_cache=str(GP_CACHE), GP_rule='Read the original fixed adaptation/surface cache',
            summary='Paired per-training-seed statistics over the same 36 tasks; three seed runs describe training variability'),
        execution=dict(CPU_threads=2, GPU=0, state='prepared_pending_unified_Tab_completion'))
    freeze_path = output/'freeze.json'
    if freeze_path.exists():
        assert read(freeze_path) == frozen
    else:
        write(freeze_path, frozen)
    pairs = [(b, floor) for b in (0, 2) for floor in H.V.FLOORS[b]]
    source_rows = {f'B{b}F{floor}': int(((H.C.BLD == b)&(H.C.FLR == floor)).sum()) for b, floor in pairs}
    assert source_rows == original['source_rows_by_floor']
    audit = []
    seed1_reference = BASE/'results/tab_field_framework_probe_20261115/probe_v2/source_manifest.json'
    old_parity = read(seed1_reference)['original_cell_table_parity']
    started = time.monotonic()
    for seed in SEEDS:
        for half, group in enumerate((pairs[0::2], pairs[1::2])):
            rng, traced_rng = np.random.default_rng([seed, half]), np.random.default_rng([seed, half])
            raw = H.cell_table(group, rng, draws=4)
            traced = trace.cell_table_with_units(H, group, traced_rng, draws=4)
            assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(raw, traced[:5]))
            assert rng.bit_generator.state == traced_rng.bit_generator.state
            reduced = crop_table(raw)
            assert reduced[0].shape == (len(raw[0]), 48)
            assert all(a is b for a, b in zip(raw[1:], reduced[1:]))
            hashes = [array_digest(a) for a in raw]
            if seed == 1:
                assert hashes == old_parity[half]['array_sha256']
            audit.append(dict(seed=seed, half=half, source_floor_pairs=group, rows=len(raw[0]),
                original_array_names=['X55', 'n', 'nd', 'mu', 'variance'], original_array_sha256=hashes,
                reduced_X48_sha256=array_digest(reduced[0]), source_unit_list_sha256=array_digest(traced[5]),
                all_sample_rows_and_weights_identical=True, traced_array_and_rng_parity=True,
                seed1_existing_manifest_match=True if seed == 1 else None,
                sampling_rng_after_sha256=hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()))
            del raw, traced, reduced
    write(output/'source_audit.json', dict(passed=True, source_only=True,
        seconds=time.monotonic()-started, freeze_sha256=digest(freeze_path), halves=audit,
        existing_seed1_manifest=str(seed1_reference), existing_seed1_manifest_sha256=digest(seed1_reference),
        historical_array_binding='Seed1 also matches its stored manifest. Seeds2/3 are regenerated from each original member\'s frozen code/CSV/RNG recipe and verified against its identical unit-traced path.',
        mapping=dict(outer='KEEP[field.cols]', inner='Original det/mean/scale cols remain relative',
            scale_appended_mean='Index len(field.cols), after the selected 48-dimensional outer columns'),
        HGB_training_performed=False, target_localization_evaluated=False))
    print(json.dumps(dict(stage='source_preparation_complete', seconds=time.monotonic()-started,
        halves=len(audit), features=48, seeds=list(SEEDS))), flush=True)


def train(output):
    import hurdle as H
    original, _, _, _ = bindings()
    frozen = read(output/'freeze.json')
    assert read(output/'source_audit.json')['passed']
    assert read(output/'source_audit.json')['freeze_sha256'] == digest(output/'freeze.json')
    assert frozen['code_sha256'] == {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    records = []
    for seed in SEEDS:
        path, metadata = output/f'seed_{seed}.joblib', output/f'seed_{seed}.json'
        if metadata.exists():
            record = read(metadata)
            assert record['freeze_sha256'] == digest(output/'freeze.json') and digest(path) == record['sha256']
        else:
            started = time.monotonic()
            cell_table = H.cell_table
            with patch.object(H, 'cell_table', lambda *a, **kw: crop_table(cell_table(*a, **kw))):
                field = H.HurdleField([0, 2], seed=seed, draws=4)
            outer = KEEP[field.cols]
            assert not np.intersect1d(outer, DROP).size
            assert len(field.cols) in field.scale.cols
            payload = dict(kind='claude_hurdle_v1', variant='drop_relation_encoding', seed=seed,
                source_buildings=[0, 2], draws=4, source_floors=original['source_floors'],
                source_rows_by_floor=original['source_rows_by_floor'], cols=outer,
                freeze_sha256=digest(output/'freeze.json'), original_features=55,
                kept_features=KEEP, dropped_features=DROP,
                **{name: dict(estimator=getattr(field, name).est, cols=getattr(field, name).cols)
                   for name in ('det', 'mean', 'scale')})
            temporary = path.with_suffix('.tmp')
            joblib.dump(payload, temporary)
            temporary.replace(path)
            record = dict(seed=seed, path=str(path), sha256=digest(path), bytes=path.stat().st_size,
                seconds=time.monotonic()-started, freeze_sha256=digest(output/'freeze.json'),
                outer_cols=outer.tolist(), scale_appended_mean_index=len(field.cols), own_two_half_OOF=True)
            write(metadata, record)
            print(json.dumps(dict(stage='deleted_relation_member_trained', **record)), flush=True)
        records.append(record)
    write(output/'trained.json', dict(complete=True, models=records,
        freeze_sha256=digest(output/'freeze.json'), source_only=True, target_localization_evaluated=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'train'], required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        (prepare if args.phase == 'prepare' else train)(args.output)
