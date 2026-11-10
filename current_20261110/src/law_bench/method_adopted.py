"""Adopted original five-model A, optionally with the unchanged fixed GP pool.

AdoptedMethod(variant='A') uses FrozenHurdleTranslator directly. variant='A_GP'
also requires gp_options: the existing RetainedGPMethod constructor arguments,
including its full-H cache directory and original B0-three-floor parameters.
Neither source fitting nor loading rebuilds A. GP computes this Task's real
S-conditioned fields; no old 54-event prediction is reused.

A was trained on full UJI B0 F0-F3 (5249 raw rows); GP hyperparameters used
the protected-legal B0 F0-F2 subset. Source permissions must include UJI B0;
all B0 target tasks are rejected. These assets cannot supply three-fold LOBO.
Keep the original full AP input axis for A's pooled support features, then
let its existing reader select H/current-S detected APs. Do not prefilter it.
"""
from pathlib import Path
import hashlib
import json

import numpy as np


RESULTS = Path('/home/panyushuo/projects/panyushuo/ReFi-UJI-Core-20261031/results')
A_FREEZE_SHA256 = '371f5815a44f7c4ddac67cbb1ad7932ba90e3189958d5723f4984f86cf1d08a7'
GP_PARAMETERS_SHA256 = 'f14dbededc1fb40630ec1c11a4ae49abd0f1098d01d31987141c99529ef369de'


class AdoptedMethod:
    def __init__(self, *, variant='A', generation_dir=RESULTS/'claude_hurdle_b0_ens5_20261110',
                 gp_options=None):
        if variant not in ('A', 'A_GP'):
            raise ValueError('Adopted variant must be A or A_GP')
        self.variant, self.name = variant, 'adopted_' + variant
        self.generation_dir = Path(generation_dir)
        self.translator = None
        self.allowed_sources = None
        self.provenance = {}
        self.diagnostics = {}
        self.gp = None
        if variant == 'A_GP':
            if gp_options is None:
                raise ValueError('A_GP requires explicit retained-GP parameter provenance and cache options')
            from .method_retained_gp import RetainedGPMethod
            self.gp = RetainedGPMethod(**gp_options)
            if self.gp.parameters_sha256 != GP_PARAMETERS_SHA256:
                raise ValueError('Adopted A_GP requires the original paired B0 F0-F2 GP parameters')

    def fit_source(self, sources):
        """Check the existing assets and fold permission; no training occurs."""
        self.allowed_sources = None
        allowed = {(s['dataset'], s['building']) for s in sources}
        if ('uji', 'B0') not in allowed:
            raise ValueError('Non-runnable adopted fold: UJI B0 must be an allowed source')
        freeze_path = self.generation_dir/'freeze.json'
        freeze = json.loads(freeze_path.read_text())
        complete = json.loads((self.generation_dir/'complete.json').read_text())
        if not (complete['complete'] and complete['freeze_sha256'] == A_FREEZE_SHA256
                and hashlib.sha256(freeze_path.read_bytes()).hexdigest() == A_FREEZE_SHA256
                and freeze['source_buildings'] == [0] and freeze['source_floors'] == [0, 1, 2, 3]):
            raise ValueError('Original adopted five-model A asset binding mismatch')
        self.provenance = dict(generation_dir=str(self.generation_dir), freeze_sha256=A_FREEZE_SHA256,
            training_sources=[dict(dataset='uji', building='B0')], source_floors=freeze['source_floors'],
            source_rows_by_floor=complete['source_rows_by_floor'], models=complete['models'],
            source_permission=freeze['source_permission'])
        if self.gp is not None:
            self.gp.fit_source(sources)
        self.allowed_sources = allowed
        return self

    def predict(self, task):
        self.diagnostics = dict(task_id=task.task_id, variant=self.variant, completed=False,
                                A_provenance=self.provenance)
        try:
            if self.allowed_sources is None:
                raise ValueError('Call fit_source to validate the adopted assets for this fold first')
            identity = (task.dataset, task.building)
            if identity == ('uji', 'B0') or identity in self.allowed_sources:
                raise ValueError('Non-runnable adopted fold: target building appears in source permissions')
            if self.translator is None:
                from frozen_hurdle_translator import FrozenHurdleTranslator
                # Original loader verifies all five saved models and feature-code identity.
                self.translator = FrozenHurdleTranslator(self.generation_dir)
            histories = {f'{task.building}F{f}': h for f, h in sorted(task.H.items())}
            prepared = self.translator.prepare(histories, task.S, f'{task.building}F{task.floor}')
            hurdle = prepared.predict(task.Q_rss)
            self.diagnostics.update(APs=hurdle['aps'].tolist(), candidate_rows=len(hurdle['cand_xy']),
                bounds=prepared.bounds.tolist(), history_scans=sum(len(h['rss']) for h in task.H.values()),
                support_scans=len(task.S['rss']))
            prediction = hurdle['projected']
            if self.gp is not None:
                from hurdle_gp_pool import fuse_hurdle_gp
                try:
                    gp = self.gp.predict_scores(task)
                finally:
                    self.diagnostics['GP'] = self.gp.diagnostics
                if not np.array_equal(hurdle['aps'], gp['aps']):
                    raise ValueError('A and GP must use the same H/current-S detected AP axis')
                fused = fuse_hurdle_gp(hurdle['cand_xy'], hurdle['log_position_mass'],
                    gp['grid'], gp['scores'], gp['raw'], prepared.bounds)
                prediction = fused['predictions']['probability_half']
                self.diagnostics.update(pool='probability_half', GP_source_floors=['B0F0', 'B0F1', 'B0F2'],
                    interpolation='original probability-space bilinear 2m-grid interpolation',
                    candidate_policy='original A entries including coincident H/S atoms')
            self.diagnostics['completed'] = True
            return prediction
        except Exception as error:
            self.diagnostics['error'] = repr(error)
            raise
