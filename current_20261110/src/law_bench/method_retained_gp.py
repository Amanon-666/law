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
only when loaded into a task. Old asset manifests require
an explicit, verified conversion into this format; FPS128 fields cannot stand
in for full H. Missing fields raise unless fit_missing_history=True, which
runs the original CPU full-covariance VI (640 iterations maximum). Its cost
grows steeply with unique H positions; no history subsampling is performed.

Run serially or in isolated processes: history solver monitoring temporarily
patches the original module. Statically reviewed only; no numerical run yet.
"""
from pathlib import Path
from unittest.mock import patch
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
                    frame='source_xy_mean_v1', history_steps=640)
    h = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for name, dtype in [('rss', np.float64), ('xy', np.float64), ('acq', np.int64)]:
        value = np.ascontiguousarray(task.H[floor][name], dtype=dtype)
        h.update(str((name, value.dtype.str, value.shape)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


class RetainedGPMethod:
    name = 'retained_gp'

    def __init__(self, *, parameters_path, provenance, cache_dir,
                 fit_missing_history=False):
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
