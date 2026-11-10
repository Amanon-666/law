"""Existing retained GP on the shared Task, with explicit parameter provenance.

parameters_path is a caller-supplied base/obs JSON. provenance is a dict with
parameters_sha256, training_sources=[{dataset, building}, ...], and evidence
(a citation/path identifying its calibration freeze). fit_source only checks
that these identities belong to the permitted source buildings; it does not
calibrate replacement parameters. No legal parameters means a non-runnable
fold, never borrowing target-trained parameters.

cache_dir holds <history_key>/<AP-index>.npz plus .json: z/mean/cov/origin, a file
checksum and solver diagnostics. history_key binds the full actual H arrays,
AP axis, source-floor XY mean origin and parameter bytes, independently of
target/S/K/repeat. Cached fields stay in that source frame and are translated
only when loaded into a task. uji_assets_dir can point to the existing
results/building_information_assets_20261109: import_uji_assets verifies its
full-H row/AP/parameter bindings and copies fields without fitting. Other old
asset formats are not imported; FPS128 fields cannot stand in for full H.
Missing fields raise unless fit_missing_history=True, which
runs the original CPU full-covariance VI (640 iterations maximum). Its cost
grows steeply with unique H positions; no history subsampling is performed.

Run serially or in isolated processes: history solver monitoring temporarily
patches the original module. Statically reviewed only; no numerical run yet.
"""
from pathlib import Path
from unittest.mock import patch
import csv
import hashlib
import json

import numpy as np


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history_key(task, floor, parameters_sha256):
    """Public identity helper for verified conversion of existing full-H assets."""
    identity = dict(dataset=task.dataset, building=task.building, floor=int(floor),
                    ap_ids=list(task.ap_ids), parameters_sha256=parameters_sha256,
                    origin=np.asarray(task.H[floor]['xy'], dtype=np.float64).mean(0).tolist(),
                    frame='source_xy_mean_v1')
    h = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for name, dtype in [('rss', np.float64), ('xy', np.float64), ('acq', np.int64)]:
        value = np.ascontiguousarray(task.H[floor][name], dtype=dtype)
        h.update(str((name, value.dtype.str, value.shape)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


class RetainedGPMethod:
    name = 'retained_gp'

    def __init__(self, *, parameters_path, provenance, cache_dir,
                 fit_missing_history=False, uji_assets_dir=None):
        self.parameters_path = Path(parameters_path)
        self.parameters_sha256 = _digest(self.parameters_path)
        self.parameters = json.loads(self.parameters_path.read_text())
        self.provenance = json.loads(json.dumps(provenance))
        if self.provenance['parameters_sha256'] != self.parameters_sha256:
            raise ValueError('Non-runnable GP: parameter provenance checksum mismatch')
        self.training_sources = {(s['dataset'], s['building'])
                                 for s in self.provenance['training_sources']}
        if not self.training_sources or not self.provenance['evidence']:
            raise ValueError('Non-runnable GP: explicit calibration provenance required')
        # This audited asset was calibrated on protected-legal UJI B0 F0-F2.
        if (self.parameters_sha256 ==
                'f14dbededc1fb40630ec1c11a4ae49abd0f1098d01d31987141c99529ef369de'
                and self.training_sources != {('uji', 'B0')}):
            raise ValueError('Known paired-assets parameters must be bound to uji/B0')
        self.base, self.obs = self.parameters['base'], self.parameters['obs']
        self.cache_dir = Path(cache_dir)
        self.fit_missing_history = fit_missing_history
        self.uji_assets_dir = None if uji_assets_dir is None else Path(uji_assets_dir)
        self._imported_history_keys = set()
        self.allowed_sources = None
        self._fields = {}
        self.diagnostics = {}
        self.state = None

    def fit_source(self, sources):
        """Validate existing calibration; no new training or extra-building H fit."""
        self.allowed_sources = None
        allowed = {(s['dataset'], s['building']) for s in sources}
        if not self.training_sources <= allowed:
            raise ValueError('Non-runnable GP fold: parameters use non-source buildings')
        self.allowed_sources = allowed
        return self

    def import_uji_assets(self, task, assets_dir):
        """Import only this Task's H from the verified nine-floor UJI assets.

        No GP/model imports or inference. The cache key identifies data and
        parameters; actual solver iterations and the old maxiter are recorded
        separately. Existing entries are kept, not overwritten.
        """
        if (task.dataset != 'uji' or task.building not in ('B1', 'B2')
                or self.training_sources != {('uji', 'B0')}
                or self.allowed_sources is None
                or ('uji', task.building) in self.allowed_sources):
            raise ValueError('UJI asset import requires a permitted B0-source/B1-or-B2-heldout fold')
        assets = Path(assets_dir).resolve()
        read = lambda name: json.loads((assets/name).read_text())
        complete, manifest, index, freeze = (read(name) for name in
            ('complete.json', 'manifest.json', 'histories.json', 'freeze.json'))
        if not (complete['complete']
                and _digest(assets/'manifest.json') == complete['manifest_sha256'] == index['manifest_sha256'] == freeze['manifest_sha256']
                and _digest(assets/'histories.json') == complete['histories_sha256']
                and _digest(assets/'freeze.json') == complete['freeze_sha256']
                and _digest(index['parameters_path']) == self.parameters_sha256 == index['parameters_sha256'] == complete['parameters_sha256']
                and freeze['source_calibration']['floors'] == ['B0F0', 'B0F1', 'B0F2']
                and index['history_rows'] == manifest['history_rows']):
            raise ValueError('UJI legacy asset manifest/parameter binding mismatch')
        csv_path = Path(manifest['csv_path'])
        if _digest(csv_path) != manifest['csv_sha256']:
            raise ValueError('UJI legacy CSV checksum mismatch')
        aps = [f'WAP{i:03}' for i in range(1, 521)]
        if list(task.ap_ids) != aps:
            raise ValueError('UJI legacy asset AP axis must be the original WAP001..WAP520')
        names = {f'{task.building}F{floor}': floor for floor in task.H}
        raw = {name: [] for name in names}
        with csv_path.open(newline='') as stream:
            for row_id, row in enumerate(csv.DictReader(stream), 1):
                name = f"B{row['BUILDINGID']}F{row['FLOOR']}"
                if name in raw:
                    raw[name].append((row_id, [int(row[ap]) for ap in aps],
                                      [float(row['LONGITUDE']), float(row['LATITUDE'])]))
        imported = []
        for name, floor in names.items():
            h = task.H[floor]
            rows = [r[0] for r in raw[name]]
            rss, xy = np.asarray([r[1] for r in raw[name]]), np.asarray([r[2] for r in raw[name]])
            if not (rows == manifest['history_rows'][name] == np.asarray(h['row_ids']).tolist()
                    and np.array_equal(rss, h['rss']) and np.array_equal(xy, h['xy'])):
                raise ValueError(f'UJI legacy field does not match full raw H rows/RSS/XY: {name}')
            records = [r for r in index['fields'] if r['floor'] == name]
            if sorted(r['ap'] for r in records) != np.flatnonzero((rss != 100).any(0)).tolist():
                raise ValueError(f'UJI legacy field AP coverage mismatch: {name}')
            origin = np.asarray(h['xy'], dtype=np.float64).mean(0)
            key = history_key(task, floor, self.parameters_sha256)
            destination = self.cache_dir/key
            destination.mkdir(parents=True, exist_ok=True)
            # One small binding per history, shared by all its AP field records.
            binding = dict(kind='building_information_assets_20261109', history_key=key,
                source_assets=str(assets), complete_sha256=_digest(assets/'complete.json'),
                manifest_sha256=complete['manifest_sha256'], histories_sha256=complete['histories_sha256'],
                freeze_sha256=complete['freeze_sha256'], csv_path=str(csv_path), csv_sha256=manifest['csv_sha256'],
                parameters_path=index['parameters_path'], parameters_sha256=self.parameters_sha256,
                floor=name, row_ids=rows, ap_ids=aps, full_raw_rows_RSS_XY_verified=True,
                acquisition_note='Legacy fit uses all raw rows, without acquisition grouping',
                legacy_max_iterations=freeze['inner_maxiter'], source_origin=origin.tolist())
            binding_path = destination/'legacy_binding.json'
            binding_path.write_text(json.dumps(binding, indent=2)+'\n')
            for record in records:
                old_path = assets.parent.parent/record['path']
                if _digest(old_path) != record['sha256']:
                    raise ValueError(f'UJI legacy field checksum mismatch: {old_path}')
                ap = record['ap']
                path = destination/f'{ap}.npz'
                metadata = path.with_suffix('.json')
                if metadata.exists():
                    saved = json.loads(metadata.read_text())
                    if (saved['history_key'] != key or saved['ap_id'] != aps[ap]
                            or saved['sha256'] != _digest(path)):
                        raise ValueError(f'Existing GP cache binding mismatch: {path}')
                    continue
                with np.load(old_path, allow_pickle=False) as old:
                    expected_z = np.unique(xy-old['origin'], axis=0)
                    if not np.array_equal(old['z'], expected_z):
                        raise ValueError(f'UJI legacy field coordinate grid mismatch: {old_path}')
                    z = old['z']+old['origin']-origin
                    np.savez_compressed(path, z=z, mean=old['mean'], cov=old['cov'], origin=origin)
                    old_origin = old['origin'].tolist()
                metadata.write_text(json.dumps(dict(history_key=key, ap_id=aps[ap],
                    sha256=_digest(path), solver=record.get('solver', {'status': 'unknown'}),
                    imported_from=dict(path=str(old_path), sha256=record['sha256'], origin=old_origin,
                        binding_path=str(binding_path), binding_sha256=_digest(binding_path),
                        transformation='z + old_origin - source_origin; mean/cov unchanged')),
                    indent=2)+'\n')
                imported.append(str(path))
            self._imported_history_keys.add(key)
        return imported

    def predict(self, task):
        self.state = None
        self.diagnostics = dict(task_id=task.task_id, completed=False,
            parameters_sha256=self.parameters_sha256, provenance=self.provenance,
            history_fields=[])
        try:
            if self.allowed_sources is None:
                raise ValueError('Non-runnable GP: call fit_source with this fold first')
            identity = (task.dataset, task.building)
            if identity in self.training_sources or identity in self.allowed_sources:
                raise ValueError('Non-runnable GP fold: heldout building appears in sources')

            import history_fit
            from fields import History
            from uji import group_scans, tensor
            from predict import adapt
            from position_fusion_gp import prepare_from_scans, prepare_retained_fields
            from cached_retained_readout import score_cached_fields

            histories = {str(f): task.H[f] for f in sorted(task.H)}
            origin = np.concatenate([h['xy'] for h in histories.values()]).mean(0)
            geometry = np.concatenate([h['xy'] for h in histories.values()] + [task.S['xy']])
            bounds = np.stack([geometry.min(0), geometry.max(0)])
            source_origins = {f: np.asarray(h['xy'], dtype=np.float64).mean(0)
                              for f, h in histories.items()}
            keys = {str(f): history_key(task, f, self.parameters_sha256)
                    for f in sorted(task.H)}
            if self.uji_assets_dir is not None and not set(keys.values()) <= self._imported_history_keys:
                self.diagnostics['imported_fields'] = self.import_uji_assets(task, self.uji_assets_dir)
            grouped = {}
            self.diagnostics.update(history_keys=keys, origin=origin.tolist(),
                source_origins={f: o.tolist() for f, o in source_origins.items()},
                H_rows={f: len(h['rss']) for f, h in histories.items()},
                S_rows=len(task.S['rss']), output='projected')

            def load_cached(source, ap):
                key = keys[source]
                cache_origin = source_origins[source]
                slot = (key, ap)
                diagnostic = dict(source=source, AP=ap, ap_id=task.ap_ids[ap],
                                  history_key=key)
                self.diagnostics['history_fields'].append(diagnostic)
                if slot in self._fields:
                    field, cache_origin, solver = self._fields[slot]
                    diagnostic.update(cache='memory', solver=solver)
                    return field, cache_origin
                path = self.cache_dir/key/f'{ap}.npz'
                metadata = path.with_suffix('.json')
                if metadata.exists():
                    record = json.loads(metadata.read_text())
                    if (record['history_key'] != key or record['ap_id'] != task.ap_ids[ap]
                            or record['sha256'] != _digest(path)):
                        raise ValueError(f'GP history cache identity/checksum mismatch: {path}')
                    solver = record['solver']
                    with np.load(path, allow_pickle=False) as arrays:
                        if not np.array_equal(arrays['origin'], cache_origin):
                            raise ValueError(f'GP history cache origin mismatch: {path}')
                        field = History(tensor(arrays['z']),
                            dict(mean=tensor(arrays['mean']), cov=tensor(arrays['cov'])), self.base)
                    diagnostic.update(cache='disk', solver=solver)
                else:
                    if not self.fit_missing_history:
                        raise RuntimeError(f'Non-runnable GP: full-H field missing: {source}/{ap}; '
                                           'explicit full-H fitting or verified cache conversion required')
                    if source not in grouped:
                        h = histories[source]
                        grouped[source] = group_scans(h['rss'], np.asarray(h['xy'])-cache_origin)
                    solver = {}
                    diagnostic.update(cache='fitted', solver=solver)
                    original = history_fit.minimize

                    def monitored(*args, **kwargs):
                        result = original(*args, **kwargs)
                        gradient = np.asarray(result.jac).copy()
                        for j, (lo, hi) in enumerate(kwargs['bounds']):
                            if lo is not None and result.x[j] <= lo+1e-10 and gradient[j] > 0:
                                gradient[j] = 0
                            if hi is not None and result.x[j] >= hi-1e-10 and gradient[j] < 0:
                                gradient[j] = 0
                        solver.update(success=bool(result.success), iterations=int(result.nit),
                            projected_gradient=float(np.abs(gradient).max()), message=str(result.message))
                        # Same acceptance rule as the original IPIN history-asset runner.
                        if not (np.isfinite(result.fun) and np.isfinite(result.x).all()
                                and np.isfinite(gradient).all() and solver['projected_gradient'] <= 1e-3):
                            raise RuntimeError(f'Unfinished GP history fit: {diagnostic}')
                        return result

                    with patch.object(history_fit, 'minimize', monitored):
                        field = history_fit.single_history(dict(scans=grouped[source]), ap,
                                                           self.base, self.obs, steps=640)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, z=field.z.numpy(), mean=field.m.numpy(),
                                        cov=field.c.numpy(), origin=cache_origin)
                    metadata.write_text(json.dumps(dict(history_key=key, ap_id=task.ap_ids[ap],
                        sha256=_digest(path), solver=solver), indent=2)+'\n')
                self._fields[slot] = field, cache_origin, solver
                return field, cache_origin

            def load(source, ap):
                field, cache_origin = load_cached(source, ap)
                return History(field.z + tensor(cache_origin) - tensor(origin),
                               dict(mean=field.m, cov=field.c), self.base)

            context = prepare_from_scans(histories, task.S, self.base, self.obs, load,
                                         origin, bounds)
            if not context['cases']:
                raise ValueError('Non-runnable GP: no AP detected in H/current S')
            if len(next(iter(context['cases'].values()))['sx']) != task.K:
                raise ValueError('GP Task K disagrees with actual unique support positions')
            self.diagnostics.update(APs=context['aps'].tolist(), bounds=context['bounds'].tolist())
            self.state = adapt(context['cases'], self.base, self.obs)
            self.diagnostics['state'] = self.state
            prepared = prepare_retained_fields(context, self.state, task.task_id, task.K)
            self.diagnostics['retained_fields'] = prepared['diagnostics']
            result = score_cached_fields(prepared, task.Q_rss)
            self.diagnostics['completed'] = True
            return result['projected']
        except Exception as error:
            self.diagnostics['error'] = repr(error)
            raise
