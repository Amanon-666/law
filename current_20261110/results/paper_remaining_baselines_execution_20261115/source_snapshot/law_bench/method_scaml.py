"""Official ScaML core with the existing corrected restart-selection adapter.

Duck-types law_bench.task.Method; no second Task definition. fit_source is a
no-op: this baseline fits source GPs to this deployment's H, then the target
GP to current S. It does not consume the benchmark's extra training buildings.
All retained H scans and repeats are used. Query enters only after fitting.

Dependencies are imported on predict, from the existing server snapshot and
its locked deps (BoTorch .7.3 / GPyTorch1.9). No installation or old batch-runner
execution. Original correction patches process-global BoTorch fitting; run
methods serially in this locked environment or in isolated worker processes,
never concurrent threads. Other preloaded BoTorch/GPyTorch versions are not
compatible with this legacy optimizer trace API.
This wrapper has been statically reviewed only, not numerically exercised.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np


RESUME = Path('/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0')


class ScaMLMethod:
    name = 'scaml'

    def __init__(self, *, legacy_root=RESUME/'Law-Probabilistic-Transfer-20261023',
                 reference_root=RESUME/'scamlgp-reference-6717858'):
        self.legacy_root, self.reference_root = Path(legacy_root), Path(reference_root)
        self._sources = {}
        self.diagnostics = {}

    def fit_source(self, sources):
        """No extra-building fitting; original source GPs use task.H only."""
        return None

    def _dependencies(self):
        previous = sys.path[:]
        sys.path[:0] = [str(self.reference_root), str(self.reference_root/'deps'), str(self.legacy_root)]
        try:
            import torch
            import botorch
            import gpytorch
            if botorch.__version__ != '0.7.3' or gpytorch.__version__ != '1.9.0':
                raise RuntimeError('ScaML requires the existing locked deps in an isolated process')
            from external_baselines.scaml_adapter import fit_sources, ScaMLCoordinates
            from external_baselines.scaml_training_selection import TrainingSelectionTrace
            from external_baselines.scaml_run import exact_context
        finally:
            sys.path[:] = previous
        return torch, fit_sources, ScaMLCoordinates, TrainingSelectionTrace, exact_context

    def predict(self, task):
        """Return absolute XY [Nq,2], un-clipped; diagnostics are per last call.

        Cache source models only under exact H/AP/frame identity. Source seeds
        use its SHA256 prefix; target restarts keep the old fixed seed261026.
        restarts=5 means one initial fit PLUS five sampled restarts in ScaML.
        """
        torch, fit_sources, Coordinates, Trace, exact_context = self._dependencies()
        names = sorted(task.H)
        seen = (np.asarray(task.S['rss']) != 100).any(0)
        for name in names:
            seen |= (np.asarray(task.H[name]['rss']) != 100).any(0)
        aps = np.flatnonzero(seen)
        assert len(aps) and names and len(task.S['xy'])
        origin = np.concatenate([np.asarray(task.H[n]['xy'], dtype=np.float64) for n in names]).mean(0)
        histories = [dict(rss=np.asarray(task.H[n]['rss'])[:, aps],
                          xy=np.asarray(task.H[n]['xy'], dtype=np.float64)-origin) for n in names]
        support = dict(rss=np.asarray(task.S['rss'])[:, aps],
                       xy=np.asarray(task.S['xy'], dtype=np.float64)-origin)
        identity = dict(dataset=task.dataset, building=task.building,
                        histories=[str(n) for n in names],
                        ap_ids=[str(task.ap_ids[i]) for i in aps], origin=origin.tolist())
        h = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
        for history in histories:
            for key in ('rss', 'xy'):
                a = np.ascontiguousarray(history[key])
                h.update(str((a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
        key = h.hexdigest()
        self.diagnostics = dict(task_id=task.task_id, completed=False, source_key=key,
            cached_source=key in self._sources, H_rows=[len(v['rss']) for v in histories],
            S_rows=len(support['rss']), APs=len(aps), origin=origin.tolist(),
            restarts=5, maxiter=500, source_seed=int(key[:8], 16), target_seed=261026,
            source_information='Task H only; benchmark source buildings unused')
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            with exact_context():
                if key not in self._sources:
                    trace = Trace(maxiter=500)
                    source_diag = self.diagnostics['source_fit'] = {}
                    try:
                        with trace.observe():
                            models = fit_sources(histories, seed=int(key[:8], 16), restarts=5)
                        selected = [[trace.selected(m) for m in axis.values()] for axis in models]
                        source_diag['selected'] = selected
                        assert all(s['optimizer_success'] for axis in selected for s in axis)
                    finally:
                        source_diag['attempts'] = trace.attempts
                    self._sources[key] = (models, source_diag)
                models, source_diag = self._sources[key]
                self.diagnostics['source_fit'] = source_diag
                trace = Trace(maxiter=500)
                target_diag = self.diagnostics['target_fit'] = {}
                try:
                    with trace.observe():
                        model = Coordinates(models, support, seed=261026, restarts=5)
                    selected = [trace.selected(m) for m in model.models]
                    target_diag['selected'] = selected
                    assert all(s['optimizer_success'] for s in selected)
                    prediction, variance = model.predict(np.asarray(task.Q_rss)[:, aps])
                finally:
                    target_diag['attempts'] = trace.attempts
            if (not np.isfinite(prediction).all() or not np.isfinite(variance).all()
                    or (variance < 0).any()):
                raise FloatingPointError('Invalid ScaML prediction/variance')
            self.diagnostics.update(completed=True, optimizer_success=True,
                status='completed; optimizer success is not a global-optimum guarantee')
            return prediction + origin
        except Exception as error:
            self.diagnostics['error'] = repr(error)
            raise
        finally:
            torch.set_default_dtype(previous_dtype)
