"""H/S-conditioned features and real single-scan source supervision.

The existing source catalog/pack owns protected-row and acquisition provenance
checks. This module does not read files, fit models or choose a likelihood.
at(xy) uses only H/S at supplied candidate coordinates; absolute XY is not a
feature. source_scan_table uses real source Q coordinates and RSS as supervised
examples, never position-mean pseudo-labels. Validation tables remain validation
data: accepting that role does not authorize fitting on their labels.
The registered 520-AP axis and physical-floor/XY correspondence are required.
"""
import numpy as np

from vertical_translator import translator as T


class ObservationTranslationContext:
    def __init__(self, histories, support, target_floor):
        building, floor = target_floor.rsplit('F', 1)
        self.target_floor, self.target = target_floor, int(floor)
        assert histories and target_floor not in histories
        self.maps = {}
        seen = np.zeros(520, dtype=bool)
        for name, scans in histories.items():
            source_building, source_floor = name.rsplit('F', 1)
            assert source_building == building and int(source_floor) != self.target
            rss = np.asarray(scans['rss'], dtype=np.float64)
            xy = np.asarray(scans['xy'], dtype=np.float64)
            assert rss.ndim == 2 and rss.shape[1] == 520 and xy.shape == (len(rss), 2)
            self.maps[int(source_floor)] = T.position_map(rss, xy)
            seen |= (rss != 100).any(axis=0)
        assert len(self.maps) == len(histories)
        self.srss = np.asarray(support['rss'], dtype=np.float64).copy()
        self.sxy = np.asarray(support['xy'], dtype=np.float64).copy()
        assert self.srss.shape == (len(self.sxy), 520) and self.sxy.shape == (len(self.sxy), 2)
        self.aps = np.flatnonzero(seen | (self.srss != 100).any(axis=0))
        assert len(self.aps), 'No AP detected in H or S'
        self.sref = T.naive_ref_at(self.sxy, self.target, self.maps) if len(self.sxy) else None

    def at(self, xy):
        """Return [Nposition, NselectedAP, 55] float32 in self.aps order.

        Calculate all 520 APs before selection so pooled/home/strength-bin S
        statistics retain the original translator meaning and dtype sequence.
        """
        xy = np.asarray(xy, dtype=np.float64)
        assert xy.ndim == 2 and xy.shape[1] == 2
        history, reference = T.history_features(xy, self.target, self.maps)
        history = history.astype(np.float32)
        if len(self.sxy):
            features = T.support_features(xy, reference, self.sxy, self.srss,
                                          self.sref, history).astype(np.float32)
        else:
            features = np.full((len(xy), 520, 12), np.nan, dtype=np.float32)
        return np.concatenate([history[:, self.aps], features[:, self.aps]], axis=-1)

    def source_scan_table(self, query, event):
        """Flatten scan-major/AP-minor supervision; y keeps missing atom 100.

        Each unique Q position has equal total weight, divided equally among
        its scans and the H/S-selected APs. No detection/class rebalancing is
        performed. Supplied rows must follow the existing catalog/pack order;
        Q-only APs are excluded, never used to expand the input AP directory.
        """
        assert event['role'] in ('train', 'source_validation')
        assert event['floor'] == self.target_floor
        assert tuple(event['support_acquisition'][:2]) != tuple(event['query_acquisition'][:2])
        assert not set(event['support_rows']).intersection(event['query_rows'])
        assert tuple(query['acquisition_group']) == tuple(event['query_acquisition'])
        rss = np.asarray(query['rss'])
        xy = np.asarray(query['xy'], dtype=np.float64)
        assert rss.shape == (len(event['query_rows']), 520) and xy.shape == (len(rss), 2)
        assert len(self.sxy) == len(event['support_rows']) and len(rss)
        assert not set(map(tuple, self.sxy)).intersection(map(tuple, xy))
        positions, inverse, counts = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        X = self.at(positions)[inverse].reshape(-1, 55)
        y = rss[:, self.aps].reshape(-1).copy()
        weight = 1.0 / (len(positions) * counts[inverse] * len(self.aps))
        return dict(X=X, y=y, sample_weight=np.repeat(weight, len(self.aps)),
                    aps=self.aps.copy(), query_rows=list(event['query_rows']), role=event['role'])
