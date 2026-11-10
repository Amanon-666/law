"""Run one explicit heldout building with shared tasks and visible failures.

Example: python -m law_bench.run --dataset uji --building B1 \
  --source-buildings B0 --methods wknn_S wknn_HS --output results/bench_B1
No dataset/model downloads. Source fits and completed tasks are reusable with
--resume only when input arrays, configuration and benchmark code still match.
This runner does not launch a multi-dataset or complete leave-one-building run.
Use --protocol legacy54 --manifest <original manifest> to retain the old 54
events' rows and hierarchical metric, one building at a time. Sources are B0;
repeat 0/1 selects AB/BA rather than new random draws. Static defaults are unchanged.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time

import joblib
import numpy as np

from .protocol import make_tasks
from .score import aggregate, score


METHODS = {
    'wknn_S': ('methods_simple', 'WKNN_S', {}),
    'wknn_HS': ('methods_simple', 'WKNN_HS', {}),
    'tabicl': ('method_tabicl', 'TabICL', {}),
    'tabicl_ordinary': ('method_tabicl', 'TabICL', {'variant': 'ordinary'}),
    'tabicl_full24': ('method_tabicl', 'TabICL', {'variant': 'full24'}),
    'scaml': ('method_scaml', 'ScaMLMethod', {}),
    'tcms': ('method_tcms', 'TCMSMethod', {}),
    'retained_gp': ('method_retained_gp', 'RetainedGPMethod', {}),
    'adopted_A': ('method_adopted', 'AdoptedMethod', {'variant': 'A'}),
    'adopted_A_GP': ('method_adopted', 'AdoptedMethod', {'variant': 'A_GP'}),
    'original_A': ('method_adopted', 'AdoptedMethod', {'variant': 'A'}),
    'original_A_GP': ('method_adopted', 'AdoptedMethod', {'variant': 'A_GP'}),
    'frozen_v1': ('method_frozen_field', 'FrozenFieldMethod', {'variant': 'v1'}),
    'frozen_v1_GP': ('method_frozen_field', 'FrozenFieldMethod', {'variant': 'v1_GP'}),
    'gp_S': ('methods_claude', 'GpS', {}),
    'gp_histmean': ('methods_claude', 'GpHS', {}),
    'ours': ('methods_claude', 'OursMethod', {}),
    'ours_A': ('methods_claude', 'OursMethod', {'variant': 'A'}),
    'trained_A_GP': ('method_trained_field_gp', 'TrainedFieldGP', {}),
    'ours_noM2': ('methods_claude', 'OursMethod', {'variant': 'noM2'}),
    'ours_C1': ('methods_claude', 'OursMethod', {'variant': 'C1'}),
    'ours_v1': ('methods_claude', 'OursMethod', {'variant': 'v1'}),
    'ours_naive': ('methods_claude', 'OursMethod', {'variant': 'naive'}),
    # No-M2 predictions do not consume theta; skip the unused source search.
    'ours_naive_noM2': ('methods_claude', 'OursMethod',
                       {'variant': 'naive_noM2', 'fixed_theta': (16.0, 0.0, 1.0, 1.0)}),
}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=_json_default, allow_nan=False) + '\n')
    temporary.replace(path)


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _data_identity(buildings, ap_ids):
    identity = {}
    for building, floors in sorted(buildings.items()):
        h = hashlib.sha256(json.dumps(ap_ids[building], default=_json_default).encode())
        for floor, values in sorted(floors.items()):
            h.update(str(floor).encode())
            h.update(json.dumps(values.get('meta', {}), sort_keys=True, default=_json_default).encode())
            for name in ('rss', 'xy', 'acq'):
                a = np.ascontiguousarray(values[name])
                h.update(str((name, a.shape, a.dtype.str)).encode())
                h.update(a.tobytes())
        identity[building] = h.hexdigest()
    return identity


def _method(name, options, output):
    module, class_name, defaults = METHODS[name]
    kwargs = dict(defaults, **options.get(name, {}))
    if name == 'tcms':
        kwargs.setdefault('diagnostics_dir', str(output / 'tcms_restarts'))
    constructor = getattr(importlib.import_module('.' + module, __package__), class_name)
    return constructor(**kwargs)


def evaluate(args):
    from .data import load

    legacy = manifest = None
    aggregate_results = aggregate
    if args.protocol == 'legacy54':
        from . import legacy54 as legacy
        if args.manifest is None:
            raise ValueError('legacy54 requires the original --manifest')
        if (args.dataset != 'uji' or args.source_dataset not in (None, 'uji')
                or args.source_root is not None
                or args.source_buildings not in (None, ['B0'])):
            raise ValueError('legacy54 uses the original UJI corpus and B0 sources')
        manifest = legacy.load_manifest(args.manifest)
        if (args.R not in (None, manifest['R'])
                or args.seed not in (None, manifest['seed'])
                or args.query_positions is not None):
            raise ValueError('legacy54 preserves the manifest R, seed and all Query rows')
        args.R, args.seed = manifest['R'], manifest['seed']
        args.repeats = [0, 1] if args.repeats is None else args.repeats
        args.source_buildings = ['B0']
        aggregate_results = legacy.aggregate
    else:
        if args.manifest is not None:
            raise ValueError('--manifest is only used by --protocol legacy54')
        args.R = 3 if args.R is None else args.R
        args.seed = 20261111 if args.seed is None else args.seed
        args.repeats = list(range(5)) if args.repeats is None else args.repeats
        args.query_positions = 60 if args.query_positions is None else args.query_positions

    output = args.output.resolve()
    buildings, ap_ids = load(args.dataset, root=args.root)
    target = buildings[args.building]
    source_dataset = args.source_dataset or args.dataset
    if source_dataset == args.dataset and args.source_root is None:
        source_data, source_aps = buildings, ap_ids
    else:
        source_data, source_aps = load(source_dataset, root=args.source_root)
    source_names = args.source_buildings if args.source_buildings is not None else [
        b for b in source_data if (source_dataset, b) != (args.dataset, args.building)]
    if source_dataset == args.dataset and args.building in source_names:
        raise ValueError('The heldout building cannot be a source building')
    sources = [dict(dataset=source_dataset, building=b, floors=source_data[b],
                    ap_ids=source_aps[b]) for b in source_names]
    options = json.loads(args.method_options.read_text()) if args.method_options else {}
    configuration = {k: str(v) if isinstance(v, Path) else v
                     for k, v in vars(args).items() if k not in ('resume', 'output', 'method_options')}
    configuration['method_options'] = options
    package = Path(__file__).resolve().parent
    frozen = dict(protocol=args.protocol, configuration=configuration,
        target_data=_data_identity({args.building: target}, {args.building: ap_ids[args.building]}),
        source_data=_data_identity({b: source_data[b] for b in source_names}, source_aps),
        code={p.name: _sha(p) for p in sorted(package.glob('*.py'))})
    if manifest is not None:
        frozen['manifest'] = dict(path=str(args.manifest.resolve()), sha256=_sha(args.manifest),
                                  metric='legacy equal building/floor/direction/K/position; discrete quantiles')
    freeze_path = output / 'freeze.json'
    if args.resume:
        if json.loads(freeze_path.read_text()) != frozen:
            raise ValueError('Resume inputs/configuration/code differ; use a new output directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write(freeze_path, frozen)

    tasks, unavailable = [], []
    if legacy is not None:
        tasks = legacy.make_tasks(manifest, args.building, target, ap_ids[args.building],
                                  budgets=args.budgets, repeats=args.repeats, selected_floors=args.floors)
    else:
        for floor in args.floors if args.floors is not None else sorted(target):
            for repeat in args.repeats:
                try:
                    tasks.extend(make_tasks(args.dataset, args.building, target, ap_ids[args.building],
                        floor, budgets=tuple(args.budgets), repeat=repeat, seed=args.seed,
                        R=args.R, query_positions=args.query_positions))
                except ValueError as error:
                    for k in args.budgets:
                        unavailable.append(dict(task_id=f'{args.dataset}/{args.building}/F{floor}/K{k}/r{repeat}',
                            floor=floor, K=k, repeat=repeat, status='protocol_unavailable', error=str(error)))
    _write(output / 'task_inventory.json', dict(
        ready=[dict(task_id=t.task_id, **t.meta) for t, _ in tasks], unavailable=unavailable))

    summaries = {}
    for name in args.methods:
        paths = {t.task_id: output / 'events' / name / (t.task_id + '.json') for t, _ in tasks}
        records = {key: json.loads(path.read_text()) for key, path in paths.items() if path.exists()}
        pending = [(t, truth) for t, truth in tasks
                   if records.get(t.task_id, {}).get('status') != 'completed']
        fit_path = output / 'source_models' / (name + '.json')
        model_path = fit_path.with_suffix('.joblib')
        model, fit_error = None, None
        fit_info = json.loads(fit_path.read_text()) if fit_path.exists() else {}
        if pending:
            tick = time.monotonic()
            try:
                if fit_info.get('status') == 'completed':
                    if _sha(model_path) != fit_info['sha256']:
                        raise ValueError('Source model checksum mismatch')
                    model = joblib.load(model_path)
                else:
                    model = _method(name, options, output)
                    model.fit_source(sources)
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = model_path.with_suffix('.tmp')
                    joblib.dump(model, temporary, compress=3)
                    temporary.replace(model_path)
                    fit_info = dict(status='completed', seconds=time.monotonic() - tick,
                        sha256=_sha(model_path), source_buildings=source_names)
                    _write(fit_path, fit_info)
            except Exception as error:
                fit_error = repr(error)
                fit_info = dict(status='failed', seconds=time.monotonic() - tick, error=fit_error)
                _write(fit_path, fit_info)
        for task, truth in pending:
            record = dict(task_id=task.task_id, floor=task.floor, K=task.K, repeat=task.repeat,
                          meta=task.meta, status='failed')
            tick = time.monotonic()
            if fit_error is not None:
                record.update(stage='source_fit', error=fit_error)
            else:
                try:
                    prediction = model.predict(task)
                    record.update(status='completed', scores=score(prediction, truth),
                        prediction=np.asarray(prediction).tolist(),
                        query_row_indices=truth['row_indices'].tolist(),
                        query_groups=truth['groups'].tolist())
                except Exception as error:
                    record.update(stage='prediction_or_score', error=repr(error))
                record['diagnostics'] = getattr(model, 'diagnostics', {})
            record['seconds'] = time.monotonic() - tick
            previous = records.get(task.task_id)
            if previous:
                record['previous_attempts'] = previous.get('previous_attempts', []) + [
                    {k: v for k, v in previous.items() if k != 'previous_attempts'}]
            _write(paths[task.task_id], record)
            records[task.task_id] = record
            print(json.dumps(dict(method=name, task=task.task_id, status=record['status'])), flush=True)
        all_records = list(records.values()) + unavailable
        summary = dict(method=name, source_fit=fit_info, by_K=aggregate_results(all_records),
            by_floor={str(f): aggregate_results([r for r in all_records if r['floor'] == f])
                      for f in sorted({r['floor'] for r in all_records})},
            task_records=[str(paths[t.task_id].relative_to(output)) for t, _ in tasks],
            unavailable=unavailable)
        if legacy is not None:
            summary['overall'] = legacy.overall(all_records)
            summary['metric'] = frozen['manifest']['metric']
        _write(output / (name + '.json'), summary)
        summaries[name] = summary
    _write(output / 'summary.json', summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--building', required=True, help='Exactly one heldout building')
    parser.add_argument('--protocol', choices=['static-resample-v2', 'legacy54'], default='static-resample-v2')
    parser.add_argument('--manifest', type=Path, help='Original frozen 54-event manifest; legacy54 only')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--source-dataset')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--source-buildings', nargs='+')
    parser.add_argument('--floors', type=int, nargs='+')
    parser.add_argument('--budgets', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--repeats', type=int, nargs='+', help='Static: default 0..4; legacy54: 0=AB, 1=BA')
    parser.add_argument('--seed', type=int, help='Static default 20261111; legacy54 keeps the manifest seed')
    parser.add_argument('--R', type=int, help='Static default 3; legacy54 keeps the manifest R')
    parser.add_argument('--query-positions', type=int, help='Static only, default 60; legacy54 keeps every Query row')
    parser.add_argument('--methods', choices=list(METHODS), nargs='+', default=['wknn_S', 'wknn_HS'])
    parser.add_argument('--method-options', type=Path, help='JSON mapping method names to constructor arguments')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if os.uname().sysname != 'Linux':
        parser.error('Run benchmark computation on the lab server, not the local Mac')
    os.nice(10)
    evaluate(args)


if __name__ == '__main__':
    main()
