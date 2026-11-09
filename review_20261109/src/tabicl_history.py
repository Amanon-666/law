"""Released TabICLv2 as a history+support conditional coordinate predictor.

This comparator reuses the old LAW bank recipe, not ReFi's learned operator.
It requires registered XY/AP identities and an observed physical floor number.
No target gradient updates, automatic downloads, or query-fitted preprocessing.
"""
import hashlib
from pathlib import Path
import sys

import numpy as np

SERVER = Path('/home/panyushuo/projects/panyushuo')
PACKAGE = SERVER/'RNP-UJI/.deps_tabicl'
CHECKPOINT = SERVER/'RNP-UJI/tabicl_weights/tabicl-regressor-v2-20260212.ckpt'
sys.path.insert(0, str(PACKAGE))
from tabicl import TabICLRegressor


def features(rss, floor, aps):
    x = np.where(rss[:, aps] == 100, 0., (rss[:, aps].astype(float)+105.)/105.)
    return np.column_stack((x, np.full(len(x), float(floor))))


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FrozenHistoryRegressor:
    def __init__(self, checkpoint=CHECKPOINT, device='cpu'):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.loaded = TabICLRegressor(model_path=self.checkpoint, allow_auto_download=False, device=device)
        self.loaded._load_model()
        self.loaded.model_.to(device).eval()

    def fit(self, histories, support, target_floor, aps, bounds, seed):
        """Fit context transforms only; scans contain raw RSS and known XY.

        Target query data are intentionally absent from this interface. Floor
        keys use UJI BnFm semantics; this is not arbitrary-building registration.
        """
        self.aps = np.asarray(aps)
        self.floor = int(target_floor.split('F')[1])
        self.bounds = np.asarray(bounds)
        contexts = [(int(name.split('F')[1]), h) for name, h in histories.items()]
        contexts.append((self.floor, support))
        x = np.concatenate([features(h['rss'], f, self.aps) for f, h in contexts])
        y = np.concatenate([h['xy'] for _, h in contexts])
        self.origin = y.mean(0)
        self.estimators = []
        for axis in range(2):
            inner = TabICLRegressor(model_path=self.checkpoint, allow_auto_download=False,
                device=self.device, n_jobs=2, n_estimators=8, batch_size=1, random_state=seed)
            # Same official model-reuse pattern as the existing bank runner.
            inner.model_ = self.loaded.model_
            inner.model_config_ = self.loaded.model_config_
            inner.model_path_ = self.checkpoint
            inner._load_model = lambda: None
            inner.fit(x, y[:, axis]-self.origin[axis])
            self.estimators.append(inner)
        return self

    def predict(self, query_rss):
        x = features(query_rss, self.floor, self.aps)
        raw = np.column_stack([m.predict(x) for m in self.estimators])+self.origin
        assert np.isfinite(raw).all()
        return dict(raw=raw, projected=np.clip(raw, self.bounds[0], self.bounds[1]))
