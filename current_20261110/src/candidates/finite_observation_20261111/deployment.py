"""Portable five-seed finite-observation field, followed by the original reader.

No source CSV imports, training, count conditioning or S-prior correction.
Input H/S and the returned PreparedFrozenHurdle follow the existing interface.
"""
import json
from pathlib import Path

import joblib
import numpy as np

from frozen_hurdle_translator import PreparedFrozenHurdle, _sha256, _support_fields
from .features import SCHEMA, SCHEMA_ID, WIDTH, build_context


KIND = 'claude_hurdle_finite_observation_v1'
ROOT = Path(__file__).resolve().parents[2]


class FiniteObservationTranslator:
    def __init__(self, generation_dir):
        root = Path(generation_dir)
        complete = json.loads((root/'complete.json').read_text())
        self.freeze = json.loads((root/'freeze.json').read_text())
        freeze_sha = _sha256(root/'freeze.json')
        assert complete['complete'] and complete['freeze_sha256'] == freeze_sha
        assert complete['kind'] == self.freeze['kind'] == KIND
        assert self.freeze['schema_id'] == SCHEMA_ID
        assert self.freeze['feature_schema'] == SCHEMA and self.freeze['feature_width'] == WIDTH
        assert self.freeze['source_buildings'] == complete['source_buildings'] == [0]
        assert self.freeze['source_floors'] == complete['source_floors'] == {'0': [0, 1, 2, 3]}
        assert self.freeze['source_rows_by_floor'] == complete['source_rows_by_floor']
        assert self.freeze['seeds'] == [1, 2, 3, 4, 5] and self.freeze['draws'] == 4
        for name, sha in self.freeze['deployment_code'].items():
            assert _sha256(ROOT/name) == sha, name
        records = sorted(complete['models'], key=lambda r: r['seed'])
        assert [r['seed'] for r in records] == self.freeze['seeds']
        self.models = []
        for record in records:
            path = root/record['path']
            assert _sha256(path) == record['sha256']
            model = joblib.load(path)
            assert model['kind'] == KIND and model['seed'] == record['seed']
            assert model['schema_id'] == SCHEMA_ID and model['feature_schema'] == SCHEMA
            assert model['feature_width'] == WIDTH and model['freeze_sha256'] == freeze_sha
            assert model['source_buildings'] == [0] and model['draws'] == 4
            assert model['source_rows_by_floor'] == self.freeze['source_rows_by_floor']
            assert _sha256(root/model['source_tasks']['path']) == model['source_tasks']['sha256']
            self.models.append(model)

    def prepare(self, histories, support, target_floor):
        building, floor = target_floor.rsplit('F', 1)
        floor = int(floor)
        maps = {}
        for name, scans in histories.items():
            hb, hf = name.rsplit('F', 1)
            assert hb == building and int(hf) != floor
            maps[int(hf)] = scans
        context = build_context(maps, support, floor)
        xy, aps = context['cand_xy'], context['aps']
        features = context['features'].reshape(-1, WIDTH)
        fields = []
        for model in self.models:
            X = features[:, model['cols']]
            det, mean, scale = (model[key] for key in ('det', 'mean', 'scale'))
            p = det['estimator'].predict_proba(X[:, det['cols']])[:, 1]
            m = mean['estimator'].predict(X[:, mean['cols']])
            # Preserve the original float64 appended deployment mean.
            scale_X = np.hstack([X, m[:, None]])
            s2 = np.exp(scale['estimator'].predict(scale_X[:, scale['cols']]))
            fields.append(tuple(value.reshape(len(xy), len(aps)) for value in (p, m, s2)))
        ps, ms, vs = (np.stack([field[k] for field in fields]) for k in range(3))
        p, m, s2 = ps.mean(0), ms.mean(0), vs.mean(0) + ms.var(0)
        Ps, sp, sm, sv = _support_fields(np.asarray(support['rss'])[:, aps],
                                        np.asarray(support['xy'], dtype=np.float64))
        candidates = np.vstack([xy, Ps])
        bounds = np.stack([np.floor(candidates.min(0)/2)*2, np.ceil(candidates.max(0)/2)*2])
        return PreparedFrozenHurdle(candidates, aps,
            np.clip(np.vstack([p, sp]), 1e-4, 1-1e-4),
            np.vstack([m, sm]), np.vstack([s2, sv]), bounds, context['ap_count'])
