"""Real source adaptation examples for a fixed tabular observation predictor.

The present interface is K1 with two registered history sources. Target S is
conditioning information, never a second set of self-prediction labels.
"""
import numpy as np

from acquisition_tasks import episode_scans
from source_episode_catalog import make_episode
from tabicl_observation_features import HistoricalObservationTable, SOURCE_FEATURES


class AdaptationFeatures:
    def __init__(self, histories, support, target_floor, base):
        assert len(histories) == 2 and len(support['positions']) == 1
        self.history = HistoricalObservationTable(histories, support, target_floor, base)
        self.aps = self.history.aps
        self.support_position = support['positions'][0]
        self.length = self.history.length
        self.at_support = self.history.at(support['positions'])[0]
        st = support['stats']
        count = st['k'][0, self.aps]
        total = float(st['n'][0])
        valid = count > 0
        mean = np.divide(st['s1'][0, self.aps], count,
                         out=np.zeros_like(count), where=valid)
        second = np.divide(st['s2'][0, self.aps], count,
                           out=np.zeros_like(count), where=valid)
        self.local_support = np.column_stack((count/total,
            np.where(valid, (mean+75)/20, 0),
            np.where(valid, np.sqrt(np.maximum(second-mean*mean, 0))/20, 0),
            np.full(len(count), np.log1p(total)), valid.astype(float)))
        self.common_support, common_names = self._common_state(support)
        source_names = tuple(f'source_{j}:{key}' for j in range(2) for key in SOURCE_FEATURES)
        self.feature_names = tuple(f'H_query:{n}' for n in source_names)
        self.feature_names += tuple(f'H_support:{n}' for n in source_names)
        self.feature_names += ('S_detection_fraction', 'S_detected_mean', 'S_detected_std',
                               'S_log_repeats', 'S_detected_valid', 'dx_from_S', 'dy_from_S')
        self.feature_names += common_names
        assert len(self.feature_names) == 112

    def _common_state(self, support):
        h = self.at_support
        m1, m2 = (h[:, i] for i in (2, 10))
        v1, v2 = (h[:, i] for i in (6, 14))
        z = np.column_stack((np.ones(len(h)), v1, v2,
            v1*(m1+75)/20, v2*(m2+75)/20,
            v1*np.maximum(m1+70, 0)/20, v2*np.maximum(m2+70, 0)/20))
        rss = support['rss'][:, self.aps].astype(float)
        detected = rss != 100
        value = np.where(detected, (rss+75)/20, 0)
        n = rss.size
        detected_count = detected.sum(0)
        upper = np.triu_indices(z.shape[1])
        all_second = (z.T@z)/len(z)
        detected_second = (z.T@(z*detected_count[:, None]))/n
        detection_score = z.T@detected_count/n
        rss_score = z.T@value.sum(0)/n
        common = np.concatenate((all_second[upper], detection_score,
            [detected.mean()], detected_second[upper], rss_score,
            [np.square(value).sum()/n, np.log1p(n)]))
        names = tuple(f'S_all_z{i}_z{j}' for i,j in zip(*upper))
        names += tuple(f'S_z{i}_detection' for i in range(7))
        names += ('S_all_detection_fraction',)
        names += tuple(f'S_detected_z{i}_z{j}' for i,j in zip(*upper))
        names += tuple(f'S_z{i}_RSS' for i in range(7))
        names += ('S_RSS_second_moment', 'S_log_observations')
        assert common.shape == (73,)
        return common, names

    def at(self, xy):
        xy = np.asarray(xy, dtype=float)
        shape = (len(xy), len(self.aps))
        broadcast = lambda value:np.broadcast_to(value, (*shape, value.shape[-1]))
        delta = (xy-self.support_position)/self.length
        return np.concatenate((self.history.at(xy), broadcast(self.at_support),
            broadcast(self.local_support), broadcast(delta[:, None, :]),
            broadcast(self.common_support)), axis=-1)


def build_examples(rows, catalog, base):
    """Eighteen real AB tasks; no protected labels or target S self-labels."""
    chunks, labels, records = [], [], []
    protected = set(catalog['protected_rows'])
    feature_names = None
    for family in catalog['families']:
        for draw in (0, 1):
            event = make_episode(catalog, family['id'], draw, k=1, phase='fit')
            ids = event['query_rows'][:18:3]
            assert len(ids) == len({rows[i]['xy'] for i in ids}) == 6
            input_ids = set(event['support_rows']) | {
                i for values in event['history_rows'].values() for i in values}
            assert not input_ids.intersection(ids)
            assert not protected.intersection(input_ids | set(ids))
            assert event['support_acquisition'][:2] != event['query_acquisition'][:2]
            selected = dict(event, query_rows=ids)
            scans = episode_scans(rows, selected, np.zeros(2))
            descriptor = AdaptationFeatures(scans['histories'], scans['support'], event['floor'], base)
            if feature_names is None:
                feature_names = descriptor.feature_names
            assert feature_names == descriptor.feature_names
            # Query coordinates are supervised input locations. Its observed
            # RSS enters labels below, never AdaptationFeatures.
            x = descriptor.at(scans['query']['positions'])[scans['query']['group']]
            y = scans['query']['rss'][:, descriptor.aps].astype(float)
            chunks.append(x.reshape(-1, len(feature_names)))
            labels.append(y.ravel())
            records.append(dict(event=event['id'], floor=event['floor'], family=family['id'],
                draw=draw, support_rows=event['support_rows'], query_label_rows=ids,
                APs=descriptor.aps.tolist(), observations=y.size,
                detections=int((y != 100).sum()),
                support_state=descriptor.common_support.tolist()))
    x = np.concatenate(chunks)
    y = np.concatenate(labels)
    detected = y != 100
    assert np.isfinite(x).all() and np.isfinite(y).all()
    context = dict(x_D=x, y_D=detected.astype(float), x_R=x[detected], y_R=y[detected],
        n_D=len(y), n_R=int(detected.sum()), n_scans=sum(len(r['query_label_rows']) for r in records),
        n_APs=[len(r['APs']) for r in records])
    return context, dict(events=records, feature_names=feature_names,
                         source_floors=catalog['source_floors'], source_floor_count=3)


class AdaptationContext:
    """Adapter for the unchanged two-head frozen observation regressor."""
    def __init__(self, descriptor, examples):
        self.descriptor = descriptor
        self.aps = descriptor.aps
        self.examples = examples

    def support_context(self):
        # Interface name inherited from the original target-only model. These
        # labels are real source Query observations, not this target's S.
        return self.examples

    def at(self, xy):
        return self.descriptor.at(xy)
