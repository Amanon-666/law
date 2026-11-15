"""Native field parameters with the adopted candidate construction and reader."""
import gc
import json
from pathlib import Path

import numpy as np
import torch

from frozen_hurdle_translator import PreparedFrozenHurdle, _support_fields, T
from native_tab import NativeFieldRole
from source_probe import digest


def predict_role(model, X, chunk=512):
    result = []
    for start in range(0, len(X), chunk):
        q = X[start:start+chunk]
        value = model.estimator.predict_proba(q)[:, 1] if model.role == 'det' else model.estimator.predict(q)
        assert np.isfinite(value).all()
        result.append(value)
    return np.concatenate(result)


class NativeObservationField:
    def __init__(self, models):
        self.models = models

    @classmethod
    def from_assets(cls, path):
        assets = json.loads(Path(path).read_text())
        models = {}
        for kind, record in assets['contexts'].items():
            checkpoint = Path(record['checkpoint'])
            context_path = Path(record['path'])
            assert digest(checkpoint) == record['checkpoint_sha256']
            assert digest(context_path) == record['sha256']
            model = NativeFieldRole(kind, checkpoint, views=1)
            with np.load(context_path) as context:
                model.fit_context(context['X'], context['y'], seed=assets['seed'])
            model.model.eval()
            model.estimator.kv_cache = True
            model.estimator.cache_mode_ = 'kv'
            model.estimator._build_kv_cache()
            models[kind] = model
        return cls(models)

    def parameters_at(self, X):
        p = predict_role(self.models['det'], X)
        mean = predict_role(self.models['mean'], X)
        scale_X = np.column_stack((X, mean.astype(np.float32)))
        log_variance = predict_role(self.models['scale'], scale_X)
        return dict(det=p, mean=mean, scale=log_variance)

    def prepare(self, task):
        maps = {}
        seen = (task.S['rss'] != 100).any(0)
        for floor, history in task.H.items():
            maps[int(floor)] = T.position_map(history['rss'], history['xy'])
            seen |= (history['rss'] != 100).any(0)
        aps = np.flatnonzero(seen)
        candidates = np.unique(np.concatenate([m[0] for m in maps.values()]), axis=0)
        F, reference = T.history_features(candidates, task.floor, maps)
        F = F.astype(np.float32)
        support_reference = T.naive_ref_at(task.S['xy'], task.floor, maps)
        S = T.support_features(candidates, reference, task.S['xy'], task.S['rss'],
                               support_reference, F).astype(np.float32)
        cells = len(candidates)*len(aps)
        X = np.hstack((F[:, aps].reshape(cells, -1), S[:, aps].reshape(cells, -1)))
        assert X.shape[1] == 55
        parameters = self.parameters_at(X)
        p = parameters['det'].reshape(len(candidates), len(aps))
        mean = parameters['mean'].reshape(len(candidates), len(aps))
        variance = np.exp(parameters['scale'].astype(np.float64)).reshape(len(candidates), len(aps))
        assert np.isfinite(variance).all() and (variance > 0).all()
        Ps, sp, sm, sv = _support_fields(task.S['rss'][:, aps], task.S['xy'])
        xy = np.vstack((candidates, Ps))
        bounds = np.stack((np.floor(xy.min(0)/2)*2, np.ceil(xy.max(0)/2)*2))
        return PreparedFrozenHurdle(xy, aps,
            np.clip(np.vstack((p, sp)), 1e-4, 1-1e-4),
            np.vstack((mean, sm)), np.vstack((variance, sv)), bounds, task.S['rss'].shape[1])

    def close(self):
        self.models.clear()
        gc.collect()
        torch.cuda.empty_cache()
