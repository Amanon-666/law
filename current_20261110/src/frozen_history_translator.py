"""Deploy the saved five-seed HS translator using only supplied real H/S.

No CSV, global floor maps, Query, fitting or model selection is used here.
The current contract is a shared 520-AP axis and registered XY/physical-floor
names within one building. It does not infer cross-building registration.
Output is mean encoded RSS (missing=-105), not a physical GP latent field.
S conditions the translated map; this module neither appends S nor duplicates
its rows. The caller passes real S once to the existing Tab context consumer.
Empty S uses the original12 NaN features, without a claim of trained S0 risk.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from vertical_translator import translator as T


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class FrozenHistoryTranslator:
    def __init__(self, generation_dir):
        directory = Path(generation_dir)
        complete = json.loads((directory/'complete.json').read_text())
        freeze_path = directory/'freeze.json'
        freeze = json.loads(freeze_path.read_text())
        assert complete['complete'] and digest(freeze_path) == complete['freeze_sha256']
        assert freeze['train'] == 'b0only' and freeze['seeds'] == [1, 2, 3, 4, 5]
        assert digest(T.__file__) == freeze['code'][str(Path(T.__file__))]
        assert [record['seed'] for record in complete['seeds']] == freeze['seeds']
        self.models = []
        for record in complete['seeds']:
            model = next(m for m in record['models'] if m['kind'] == 'HS')
            assert digest(model['path']) == model['sha256']
            saved = joblib.load(model['path'])
            assert saved['kind'] == 'HS' and saved['seed'] == record['seed']
            assert saved['freeze_sha256'] == complete['freeze_sha256']
            assert saved['feature_count'] == 55 and saved['source_buildings'] == [0]
            assert saved['versions'] == freeze['versions']
            assert saved['estimator'].n_features_in_ == len(saved['cols'])
            self.models.append((saved['estimator'], saved['cols']))

    def translate(self, histories, support, target_floor, aps):
        """Return cand_xy and mean_encoded_rss[:,aps-order], both float64.

        H and S are {rss,xy}; all AP columns must remain present until the
        pooled/home/strength-bin S features are formed. No input is modified.
        Per-seed clipping precedes averaging, exactly as the original runner.
        """
        building, floor = target_floor.rsplit('F', 1)
        target = int(floor)
        assert histories and target_floor not in histories
        maps = {}
        for name, scans in histories.items():
            source_building, source_floor = name.rsplit('F', 1)
            assert source_building == building and int(source_floor) != target
            rss, xy = np.asarray(scans['rss'], dtype=np.float64), np.asarray(scans['xy'], dtype=np.float64)
            assert rss.ndim == 2 and rss.shape[1] == 520 and xy.shape == (len(rss), 2)
            maps[int(source_floor)] = T.position_map(rss, xy)
        assert len(maps) == len(histories)
        sx, srss = np.asarray(support['xy'], dtype=np.float64), np.asarray(support['rss'], dtype=np.float64)
        assert srss.shape == (len(sx), 520) and sx.shape == (len(sx), 2)
        aps = np.asarray(aps, dtype=np.int64)
        assert aps.ndim == 1 and len(aps) and np.all((aps >= 0) & (aps < 520))
        xy = np.unique(np.concatenate([m[0] for m in maps.values()]), axis=0)
        history, reference = T.history_features(xy, target, maps)
        history = history.astype(np.float32)
        if len(sx):
            sref = T.naive_ref_at(sx, target, maps)
            features = T.support_features(xy, reference, sx, srss, sref, history).astype(np.float32)
        else:
            features = np.full((len(xy), 520, 12), np.nan, dtype=np.float32)
        X = np.hstack([history[:, aps].reshape(-1, 43), features[:, aps].reshape(-1, 12)])
        predictions = [np.clip(model.predict(X[:, columns]), T.ND, 0).reshape(len(xy), len(aps))
                       for model, columns in self.models]
        return dict(cand_xy=xy, mean_encoded_rss=np.mean(predictions, axis=0))
