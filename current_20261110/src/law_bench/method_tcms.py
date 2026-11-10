"""Existing TCMS terminal-consistent continuation on the shared bench Task.

No new Task, source subsampling, output clipping, jitter fix or objective.
The project SourcePair implements TC-MSStack equations; it is not author code.
fit_source is a no-op; H/S alone fit each deployment. K1's whole-position LOO
prediction reduces to the H-conditioned mean, so its objective cannot identify
target noise; do not present K1 as validated multi-anchor adaptation.

Continuation writes actual restart traces under the caller's diagnostics_dir.
Its globals and PyTorch dtype are scoped/restored; use serial calls or isolated
processes, not concurrent threads. All dependencies load lazily on the existing
server environment. No fitting or numerical checks have run for this wrapper.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np


LEGACY = Path('/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/Law-Probabilistic-Transfer-20261023')
SETTINGS = dict(restarts=5, maxiter=300, maxfun=1000, projected_gradient_tolerance=1e-3)


class TCMSMethod:
    name = 'tcms'

    def __init__(self, diagnostics_dir, *, legacy_root=LEGACY):
        self.output = Path(diagnostics_dir).resolve()
        self.legacy_root = Path(legacy_root)
        self.diagnostics = {}

    def fit_source(self, sources):
        """No additional building training; original fitting uses task H/S."""
        return None

    def _dependencies(self):
        previous = sys.path[:]
        sys.path.insert(0, str(self.legacy_root))
        try:
            import torch
            from external_baselines import tcms, tcms_continuation
        finally:
            sys.path[:] = previous
        return torch, tcms, tcms_continuation

    def predict(self, task):
        """Return un-clipped XY; preserve incomplete convergence in diagnostics.

        diagnostics_dir must be a bench-owned directory, not a legacy result
        folder. Restart reuse is bound to the exact H/S arrays, APs, settings,
        solver code and Task identity; no Query data enter that binding.
        """
        torch, tcms, continuation = self._dependencies()
        names = sorted(task.H)
        seen = (np.asarray(task.S['rss']) != 100).any(0)
        for n in names:
            seen |= (np.asarray(task.H[n]['rss']) != 100).any(0)
        aps = np.flatnonzero(seen)
        assert len(aps) and names and len(task.S['xy'])
        histories = [dict(rss=np.asarray(task.H[n]['rss'])[:, aps],
                          xy=np.asarray(task.H[n]['xy'], dtype=np.float64)) for n in names]
        sxy = np.asarray(task.S['xy'], dtype=np.float64)
        _, group = np.unique(sxy, axis=0, return_inverse=True)
        support = dict(rss=np.asarray(task.S['rss'])[:, aps], xy=sxy, group=group.ravel())
        h = hashlib.sha256(json.dumps(dict(task_id=task.task_id, settings=SETTINGS,
            histories=[str(n) for n in names], ap_ids=[str(task.ap_ids[i]) for i in aps]),
            sort_keys=True).encode())
        for block in [*histories, support]:
            for key in ('rss', 'xy'):
                a = np.ascontiguousarray(block[key])
                h.update(str((a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
        for path in (Path(__file__), Path(tcms.__file__), Path(continuation.__file__),
                     self.legacy_root/'tcms_stack_bridge.py', tcms.ORIGINAL, tcms.SOLVE):
            h.update(Path(path).read_bytes())
        event = h.hexdigest()
        self.diagnostics = dict(task_id=task.task_id, completed=False, settings=dict(SETTINGS),
            restart_directory=str(self.output/event), H_rows=[len(v['rss']) for v in histories],
            S_rows=len(sxy), S_positions=len(np.unique(group)), APs=len(aps), axes=[],
            source_information='Task H/S only; benchmark source buildings unused',
            K1_boundary='One S position removes all its scans in group LOO; target noise is unidentifiable')
        previous_dtype = torch.get_default_dtype()
        previous_out, previous_event = continuation.OUT, continuation.CURRENT_EVENT
        torch.set_default_dtype(torch.float64)
        continuation.OUT, continuation.CURRENT_EVENT = self.output, event
        try:
            fitted = []
            for axis in range(2):
                pairs, center, scale = tcms.pairs_for(dict(H=histories, S=support), axis)
                parameters, diagnostic = continuation.fit(pairs, 261026+axis, SETTINGS)
                self.diagnostics['axes'].append(dict(axis=axis, **diagnostic))
                fitted.append((pairs, parameters))
            # Both coordinate fits are fixed before any Query RSS is consumed.
            xq = tcms.features(np.asarray(task.Q_rss)[:, aps])
            with torch.no_grad():
                prediction = np.column_stack([
                    (tcms.weights(ps) @ torch.stack([pair.predict(p, xq)
                        for pair, p in zip(pairs, ps)])).numpy() for pairs, ps in fitted])
            prediction = prediction*scale + center
            if not np.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite TCMS coordinates')
            converged = all(r['selected_converged'] for r in self.diagnostics['axes'])
            self.diagnostics.update(completed=True, both_axes_converged=converged,
                status='completed_converged' if converged else 'completed_unconverged')
            return prediction
        except Exception as error:
            self.diagnostics['error'] = repr(error)
            raise
        finally:
            continuation.OUT, continuation.CURRENT_EVENT = previous_out, previous_event
            torch.set_default_dtype(previous_dtype)
