"""Frozen five-seed v1_newcode candidate from the completed nine_v2 trial.

source_dir is the immutable claude_field_posterior_20261111 source snapshot,
with sibling claude_hurdle_20261110/, vertical_translator/ and
history_reliability.py. Never point at Claude's changing working source.
Only the saved B0-four-floor models are loaded; fit_source verifies permission
and provenance without training. The original ensemble_field and old
condition(16,.3,1,0) feed PreparedFrozenHurdle, preserving all H/S atoms and
the original probability reader. Optional GP uses the existing Task-specific
retained fields and unchanged probability_half pool, not old predictions.

Pickles require the original law_bench_ours module name. Incompatible modules
already loaded in the process are rejected; run in a separate process then.
"""
from pathlib import Path
import hashlib
import importlib
import json
import os
import sys

import numpy as np

from .method_adopted import GP_PARAMETERS_SHA256


MODEL_DIR = Path('/home/panyushuo/projects/panyushuo/claude_fp_wt/current_20261110/src/'
                 'claude_field_posterior_20261111/results/nine_v2_20261111/models')
CODE_SHA256 = {
    'law_bench_ours.py': '1d3761feaf964497f3408df01cc639ad3b6619743ae0592bcbd1d16165990705',
    'nine_floor_v2.py': '5f0aefb4a4142901ff84671cb88a09fc738a1bf0e2a4075dcfd641dcc8527038',
}
MODEL_SHA256 = (
    'a5c8523c61325d4bfd664dc0fdf0d3da591c7b0112a9c7c137e45e5f962e83e3',
    '2e106093f80305a77c6c15c1051a3b17c69077bcc158de3cadc5848a447d0676',
    'f9e7ca05c4ddd96867b3a328ec6cf478d4712b4faea1c035d958021599818e74',
    '8b9c220d3665187e61a5b42c1d6e07935bbaf5b3273fb0aa5c108a3e1d94dfb9',
    'fe3cc4cc59a5f60a9ca79e766e627b74a3d68fb27263501a1f31e6e7398f5262',
)
CSV_SHA256 = '45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168'
TRANSLATOR_SHA256 = '50b51aa05330ec1b7335b8be7f8c72f7200de198a933c950e19a9acfd582eacd'
RELIABILITY_SHA256 = 'd567b69f87f0a15c807f4fa494827754be34ba217786e95f8c160231dbf3fc56'


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class FrozenFieldMethod:
    def __init__(self, *, source_dir, model_dir=MODEL_DIR, variant='v1', gp_options=None):
        if variant not in ('v1', 'v1_GP'):
            raise ValueError('Frozen field variant must be v1 or v1_GP')
        self.source_dir, self.model_dir = Path(source_dir).resolve(), Path(model_dir).resolve()
        self.variant, self.name = variant, 'frozen_' + variant
        self.ops = self.module = self.allowed_sources = None
        self.provenance, self.diagnostics = {}, {}
        self.gp = None
        if variant == 'v1_GP':
            if gp_options is None:
                raise ValueError('v1_GP requires explicit retained-GP parameter provenance and cache options')
            from .method_retained_gp import RetainedGPMethod
            self.gp = RetainedGPMethod(**gp_options)
            if self.gp.parameters_sha256 != GP_PARAMETERS_SHA256:
                raise ValueError('Frozen v1_GP requires the original B0 F0-F2 GP parameters')

    def fit_source(self, sources):
        self.allowed_sources = None
        allowed = {(s['dataset'], s['building']) for s in sources}
        if ('uji', 'B0') not in allowed:
            raise ValueError('Non-runnable frozen-v1 fold: UJI B0 must be an allowed source')
        selection = json.loads((self.model_dir/'theta_selection.json').read_text())
        if selection['code_sha256'] != CODE_SHA256:
            raise ValueError('Model directory is not the frozen five-seed nine_v2 candidate')
        for name, digest in CODE_SHA256.items():
            if _sha(self.source_dir/name) != digest:
                raise ValueError(f'Frozen source version mismatch: {name}')
        models = []
        for seed, digest in enumerate(MODEL_SHA256, 1):
            path = self.model_dir/f'v1_seed{seed}.joblib'
            if _sha(path) != digest:
                raise ValueError(f'Frozen v1 model checksum mismatch: {path}')
            models.append(dict(seed=seed, path=str(path), sha256=digest))
        self.provenance = dict(source_dir=str(self.source_dir), code_sha256=CODE_SHA256,
            models=models, training_sources=[dict(dataset='uji', building='B0')],
            source_floors=[0, 1, 2, 3], source_csv_sha256=CSV_SHA256,
            source_permission='Original complete B0 four-floor corpus',
            conditioning=[16.0, 0.3, 1.0, 0.0], model_training=False)
        if self.gp is not None:
            self.gp.fit_source(sources)
        self.allowed_sources = allowed
        return self

    def _load(self):
        import joblib

        sibling = self.source_dir.parent
        modules = {
            'law_bench_ours': (self.source_dir/'law_bench_ours.py', CODE_SHA256['law_bench_ours.py']),
            'translator': (sibling/'claude_hurdle_20261110/translator.py', TRANSLATOR_SHA256),
            'history_reliability': (sibling/'history_reliability.py', RELIABILITY_SHA256),
            'vertical_translator.translator': (sibling/'vertical_translator/translator.py', TRANSLATOR_SHA256),
        }
        for name, (path, digest) in modules.items():
            if _sha(path) != digest:
                raise ValueError(f'Frozen source dependency mismatch: {path}')
            if name in sys.modules and _sha(sys.modules[name].__file__) != digest:
                raise RuntimeError(f'Incompatible {name} already loaded; use an isolated process')
        previous_path, previous_vt = sys.path[:], os.environ.get('VT_DIR')
        sys.path[:0] = [str(self.source_dir), str(sibling/'claude_hurdle_20261110'), str(sibling)]
        os.environ['VT_DIR'] = str(sibling/'claude_hurdle_20261110')
        try:
            module = importlib.import_module('law_bench_ours')
            ops, csv_paths = [], set()
            for seed, digest in enumerate(MODEL_SHA256, 1):
                path = self.model_dir/f'v1_seed{seed}.joblib'
                if _sha(path) != digest:
                    raise ValueError(f'Frozen v1 model changed before loading: {path}')
                saved = joblib.load(path)
                if (saved['seed'] != seed or saved['code_sha256'] != CODE_SHA256
                        or saved['source'] != 'UJI B0' or saved['model'].rel):
                    raise ValueError(f'Frozen v1 source/model identity mismatch: {path}')
                csv_paths.add(saved['csv'])
                ops.append(saved['model'])
            for path in csv_paths:
                if _sha(path) != CSV_SHA256:
                    raise ValueError('Frozen v1 source CSV binding mismatch')
            self.module, self.ops = module, ops
        finally:
            sys.path[:] = previous_path
            if previous_vt is None:
                os.environ.pop('VT_DIR', None)
            else:
                os.environ['VT_DIR'] = previous_vt

    def predict(self, task):
        self.diagnostics = dict(task_id=task.task_id, completed=False, provenance=self.provenance)
        try:
            if self.allowed_sources is None:
                raise ValueError('Call fit_source to validate the frozen-v1 fold first')
            identity = (task.dataset, task.building)
            if identity == ('uji', 'B0') or identity in self.allowed_sources:
                raise ValueError('Non-runnable frozen-v1 fold: heldout building appears in source permissions')
            if self.ops is None:
                self._load()
            from frozen_hurdle_translator import PreparedFrozenHurdle, _support_fields
            O = self.module
            seen = (task.S['rss'] != 100).any(0)
            for h in task.H.values():
                seen |= (h['rss'] != 100).any(0)
            aps = np.flatnonzero(seen)
            maps = O.make_maps(task.H)
            X, field = O.ensemble_field(self.ops, maps, task.floor, aps)
            sr, sx = task.S['rss'][:, aps], task.S['xy']
            p, mean, variance = O.condition(X, field['p'], field['m'], field['s2'],
                                           sr, sx, (16.0, 0.3, 1.0, 0.0))
            Ps, sp, sm, sv = _support_fields(sr, sx)
            candidates = np.vstack([X, Ps])
            bounds = np.stack([np.floor(candidates.min(0)/2)*2, np.ceil(candidates.max(0)/2)*2])
            prepared = PreparedFrozenHurdle(candidates, aps,
                np.clip(np.vstack([p, sp]), 1e-4, 1-1e-4), np.vstack([mean, sm]),
                np.vstack([variance, sv]), bounds, task.S['rss'].shape[1])
            hurdle = prepared.predict(task.Q_rss)
            prediction = hurdle['projected']
            self.diagnostics.update(APs=aps.tolist(), candidate_rows=len(candidates), bounds=bounds.tolist())
            if self.gp is not None:
                from hurdle_gp_pool import fuse_hurdle_gp
                try:
                    gp = self.gp.predict_scores(task)
                finally:
                    self.diagnostics['GP'] = self.gp.diagnostics
                if not np.array_equal(aps, gp['aps']):
                    raise ValueError('Frozen v1 and GP detected AP axes differ')
                fused = fuse_hurdle_gp(candidates, hurdle['log_position_mass'],
                    gp['grid'], gp['scores'], gp['raw'], bounds)
                prediction = fused['predictions']['probability_half']
                self.diagnostics['pool'] = 'probability_half'
            self.diagnostics['completed'] = True
            return prediction
        except Exception as error:
            self.diagnostics['error'] = repr(error)
            raise
