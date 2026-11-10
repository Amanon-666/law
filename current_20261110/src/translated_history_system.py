"""Frozen H/S translation, original 3NN readout, and fixed GP half-average.

This packages the evaluated algorithm; it performs no training, GP fitting,
weight selection or Query-label access. Maps are mean encoded RSS, not GP
observations. The independent GP branch must receive its original real H/S.
XY/AP/floor registration follows FrozenHistoryTranslator's existing contract.
"""
import numpy as np

from frozen_history_translator import FrozenHistoryTranslator
from vertical_translator import translator as T


class FrozenTranslatedHistorySystem:
    def __init__(self, generation_dir):
        self.translator = FrozenHistoryTranslator(generation_dir)

    def fit(self, histories, support, target_floor, aps, bounds):
        """Build the map from H/S, then append real S exactly once for 3NN.

        This is deployment context preparation, not estimator fitting. Inputs
        retain full raw AP columns, with missing scans encoded as 100.
        """
        self.aps = np.asarray(aps, dtype=np.int64).copy()
        self.bounds = np.asarray(bounds, dtype=np.float64).copy()
        assert self.bounds.shape == (2, 2)
        self.map = self.translator.translate(histories, support, target_floor, self.aps)
        self.reference_rss = np.vstack([
            self.map['mean_encoded_rss'],
            T.encode(np.asarray(support['rss'], dtype=np.float64)[:, self.aps]),
        ])
        self.reference_xy = np.vstack([
            self.map['cand_xy'], np.asarray(support['xy'], dtype=np.float64),
        ])
        return self

    def predict(self, query_rss, gp_prediction):
        """Return both readouts in the supplied Query order and absolute XY.

        gp_prediction is the original GP branch's [Nquery,2] position array;
        its row correspondence is the caller's responsibility. No GP update
        or additional projection of the fixed half-average is performed.
        """
        query = np.asarray(query_rss, dtype=np.float64)
        gp = np.asarray(gp_prediction, dtype=np.float64)
        assert query.ndim == 2 and query.shape[1] == 520 and gp.shape == (len(query), 2)
        distance = np.sqrt(((T.encode(query[:, self.aps])[:, None]
                             - self.reference_rss[None]) ** 2).mean(-1))
        indices = np.argsort(distance, axis=1, kind='stable')[:, :3]
        weights = 1 / np.maximum(np.take_along_axis(distance, indices, axis=1), 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        translated = np.clip((weights[:, :, None] * self.reference_xy[indices]).sum(1),
                             self.bounds[0], self.bounds[1])
        return dict(translated_3NN=translated,
                    GP_translated_3NN_half=0.5 * (gp + translated))
