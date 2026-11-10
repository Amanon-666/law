"""Reuse a completed benchmark ours_A source model with fixed retained-GP pooling.

source_run contains freeze.json and source_models/ours_A.{json,joblib}.
Current source arrays/APs/metadata must exactly match that training freeze;
the installed methods_claude.py/ours.py must match its recorded source code.
Run from the matching source snapshot if these files have since changed.
No A training, alternate reader, candidate deduplication or GP history fitting.
GP parameters need their own legal source permissions in the supplied fold;
UJI-source parameters may serve non-UJI targets when UJI is an allowed source.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from .method_retained_gp import RetainedGPMethod


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class TrainedFieldGP:
    name = 'trained_A_GP'

    def __init__(self, *, source_run, gp_options):
        self.source_run = Path(source_run).resolve()
        self.model_path = self.source_run/'source_models/ours_A.joblib'
        self.fit_path = self.model_path.with_suffix('.json')
        self.freeze_path = self.source_run/'freeze.json'
        self.bindings = {str(p): _sha(p) for p in (self.model_path, self.fit_path, self.freeze_path)}
        if gp_options.get('fit_missing_history', False):
            raise ValueError('Trained A+GP requires existing complete H fields; fitting missing history is disabled')
        self.gp = RetainedGPMethod(**gp_options)
        self.model = None
        self.allowed_sources = None
        self.provenance, self.diagnostics = {}, {}

    def _verify(self):
        for path, digest in self.bindings.items():
            if _sha(path) != digest:
                raise ValueError(f'Saved A training asset changed: {path}')
        freeze = json.loads(self.freeze_path.read_text())
        package = Path(__file__).resolve().parent
        for name in ('methods_claude.py', 'ours.py'):
            if _sha(package/name) != freeze['code'][name]:
                raise ValueError(f'A source code differs from its training freeze: {name}; use matching snapshot')
        return freeze

    def fit_source(self, sources):
        """Verify the completed fit and exact current source pool; never call A.fit_source."""
        from .run import _data_identity

        self.allowed_sources = None
        freeze = self._verify()
        fit = json.loads(self.fit_path.read_text())
        config = freeze['configuration']
        dataset = config['source_dataset'] or config['dataset']
        names = fit['source_buildings']
        expected = {(dataset, b) for b in names}
        allowed = {(s['dataset'], s['building']) for s in sources}
        if (fit['status'] != 'completed' or fit['sha256'] != self.bindings[str(self.model_path)]
                or 'ours_A' not in config['methods'] or set(names) != set(freeze['source_data'])
                or allowed != expected or len(sources) != len(expected)):
            raise ValueError('Saved ours_A fit/source identities do not match this fold')
        actual = _data_identity({s['building']: s['floors'] for s in sources},
                                {s['building']: s['ap_ids'] for s in sources})
        if actual != freeze['source_data']:
            raise ValueError('Saved A source rows/APs/metadata differ from the current permitted source pool')
        self.gp.fit_source(sources)
        self.allowed_sources = allowed
        self.provenance = dict(source_run=str(self.source_run), bindings=self.bindings,
            A_training_sources=[dict(dataset=dataset, building=b) for b in names],
            source_data=freeze['source_data'],
            A_source_code={n: freeze['code'][n] for n in ('methods_claude.py', 'ours.py')},
            original_method_options=config['method_options'].get('ours_A', {}),
            GP_parameter_provenance=self.gp.provenance, training_performed=False)
        return self

    def predict(self, task):
        self.diagnostics = dict(task_id=task.task_id, completed=False, provenance=self.provenance)
        try:
            if self.allowed_sources is None:
                raise ValueError('Call fit_source to validate the saved A/GP assets for this fold')
            identity = (task.dataset, task.building)
            if identity in self.allowed_sources or identity in self.gp.training_sources:
                raise ValueError('Non-runnable A+GP fold: target building was used by A or GP source fitting')
            self._verify()
            from . import methods_claude as C
            # Reject modules imported from a different source tree/version.
            for module in (C, C.O):
                if _sha(module.__file__) != self.provenance['A_source_code'][Path(module.__file__).name]:
                    raise ValueError('Loaded A module does not match the saved source fit')
            if self.model is None:
                import joblib
                model = joblib.load(self.model_path)
                if (not isinstance(model, C.OursMethod) or model.variant != 'A'
                        or model.model.variant != 'A' or model.inf):
                    raise ValueError('Expected saved OursMethod A with its original probability reader')
                self.model = model
            projected, candidates, log_mass, bounds = C.predict_mass(self.model, task)
            self.diagnostics.update(A_projected=np.asarray(projected).tolist(),
                candidate_rows=len(candidates), bounds=np.asarray(bounds).tolist(),
                candidate_policy='Unchanged A candidates, including empirical S duplicate atoms')
            try:
                gp = self.gp.predict_scores(task)
            finally:
                self.diagnostics['GP'] = self.gp.diagnostics
            self.diagnostics['GP_projected'] = np.asarray(gp['projected']).tolist()
            aps = C._hs_aps(task)
            if not np.array_equal(aps, gp['aps']) or not np.array_equal(bounds, gp['bounds']):
                raise ValueError('A/GP H-current-S AP axes or bounds differ')
            from hurdle_gp_pool import fuse_hurdle_gp
            fused = fuse_hurdle_gp(candidates, log_mass, gp['grid'], gp['scores'], gp['raw'], bounds)
            self.diagnostics.update(completed=True, pool='probability_half', APs=aps.tolist())
            return fused['predictions']['probability_half']
        except Exception as error:
            self.diagnostics['error'] = repr(error)
            raise
