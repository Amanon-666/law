"""Source-only full-acquisition supervision and continuous observation cells.

H/S retain the catalog's original rows. Fit Q expands only within its original
acquisition, excluding every protected row and the complete draw's K5 positions.
Validation keeps the catalog Q. Cell statistics preserve position/scan/AP
weighting for Bernoulli detection and conditional detected squared loss.
"""
import numpy as np

from acquisition_tasks import episode_scans
from observation_translation_tasks import ObservationTranslationContext
from source_episode_catalog import make_episode
from uji import group_scans


def full_source_episode(rows, catalog, spec):
    """Return (event, pack) with unchanged H/S and explicitly scoped source Q."""
    source_floors = {'B0F0', 'B0F1', 'B0F2'}
    assert set(catalog['source_floors']) == source_floors
    event = make_episode(catalog, **spec)
    protected = set(catalog['protected_rows'])
    assert event['floor'] in source_floors
    assert set(event['history_rows']) <= source_floors
    assert not protected.intersection(event['support_rows'])
    assert all(not protected.intersection(ids) for ids in event['history_rows'].values())
    assert event['support_acquisition'][:2] != event['query_acquisition'][:2]

    if event['role'] == 'train':
        maximum = make_episode(catalog, **dict(spec, k=5))
        support_positions = {tuple(rows[i]['xy']) for i in maximum['support_rows']}
        query_acquisition = tuple(event['query_acquisition'])
        event['query_rows'] = sorted(
            i for i, row in rows.items()
            if row['floor'] == event['floor']
            and tuple(row['acquisition']) == query_acquisition
            and i not in protected
            and tuple(row['xy']) not in support_positions
        )
        assert event['query_rows'] and not protected.intersection(event['query_rows'])
        assert not support_positions.intersection(tuple(rows[i]['xy']) for i in event['query_rows'])
        event['id'] += '_full_query_acquisition'
        event['supervision_scope'] = 'all_query_acquisition_rows_excluding_protected_and_full_K5_positions'
    else:
        assert event['role'] == 'source_validation'
        assert set(event['query_rows']) <= protected
        event['supervision_scope'] = 'original_catalog_protected_query_rows'

    pack = episode_scans(rows, event, np.zeros(2))
    assert not set(map(tuple, pack['support']['xy'])).intersection(map(tuple, pack['query']['xy']))
    return event, pack


class ContinuousObservationTranslationContext(ObservationTranslationContext):
    def source_cell_table(self, query, event):
        """Return position-major/AP-minor cells, including zero detections.

        For a cell, detection probability is nd/n and the scan-averaged detected
        squared loss is (nd/n) * ((prediction - mean_detected)**2 +
        variance_detected). Multiplying these terms by weight recovers equal
        position, within-position scan, and H/S-selected AP weighting.
        """
        assert event['role'] in ('train', 'source_validation')
        assert event['floor'] == self.target_floor
        assert event['floor'] in ('B0F0', 'B0F1', 'B0F2')
        assert tuple(event['support_acquisition'][:2]) != tuple(event['query_acquisition'][:2])
        assert not set(event['support_rows']).intersection(event['query_rows'])
        assert tuple(query['acquisition_group']) == tuple(event['query_acquisition'])
        rss = np.asarray(query['rss'])
        xy = np.asarray(query['xy'], dtype=np.float64)
        assert rss.shape == (len(event['query_rows']), 520) and xy.shape == (len(rss), 2)
        assert len(self.sxy) == len(event['support_rows']) and len(rss)
        assert not set(map(tuple, self.sxy)).intersection(map(tuple, xy))

        grouped = group_scans(rss[:, self.aps], xy)
        positions, stats = grouped['positions'], grouped['stats']
        nd = stats['k']
        mean = np.divide(stats['s1'], nd, out=np.zeros_like(nd), where=nd > 0)
        second_moment = np.divide(stats['s2'], nd, out=np.zeros_like(nd), where=nd > 0)
        variance = np.maximum(second_moment - mean ** 2, 0.0)
        n = np.broadcast_to(stats['n'][:, None], nd.shape).reshape(-1).copy()
        cells = nd.size
        return dict(
            X=self.at(positions).reshape(cells, 55),
            n=n, nd=nd.reshape(-1), mean_detected=mean.reshape(-1),
            variance_detected=variance.reshape(-1), weight=np.full(cells, 1.0 / cells),
            floor=event['floor'], role=event['role'], event_id=event['id'],
            representation='continuous_scan_cells_v1',
        )
