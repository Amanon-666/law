"""History-only features and real support labels for AP observation tables.

Inputs use uji.group_scans: each source has actual scan rows and position-level
n/k/s1/s2 statistics. Features describe historical observations, not latent GP
truth or independent target evidence. No query RSS or query labels enter here.
"""
import numpy as np

from uji import NO_DETECT


SOURCE_FEATURES = (
    'log_total_scan_weight',
    'detection_fraction',
    'detected_mean',
    'detected_std',
    'mean_l_minus_mean_2l',
    'log_detected_effective_scan_count',
    'detected_valid',
    'relative_floor',
)


class HistoricalObservationTable:
    """Retain one feature block per source and one row per position/AP.

    Sources are sorted by their UJI names; AP indices are the ascending union
    of detections in H/S. Relative floor means source floor minus target floor.
    Kernel weights use exp(-distance^2/(2*length^2)), with no truncation radius.
    Thus coverage is a continuous weighted scan count; far-away observations
    have small positive coverage until numerical underflow, not a hard boundary.
    """
    def __init__(self, histories, support, target_floor, base):
        self.source_names = tuple(sorted(histories))
        self.length = float(base['length'])
        self.support = support
        target = int(str(target_floor).rsplit('F', 1)[-1])
        seen = (support['rss'] != NO_DETECT).any(axis=0)
        for name in self.source_names:
            seen |= (histories[name]['rss'] != NO_DETECT).any(axis=0)
        self.aps = np.flatnonzero(seen)
        self.feature_names = tuple(
            f'{name}:{feature}' for name in self.source_names for feature in SOURCE_FEATURES)
        self.sources = []
        for name in self.source_names:
            h = histories[name]
            stats = h['stats']
            self.sources.append(dict(
                positions=np.asarray(h['positions'], dtype=np.float64),
                n=np.asarray(stats['n'], dtype=np.float64),
                k=np.asarray(stats['k'], dtype=np.float64)[:, self.aps],
                s1=np.asarray(stats['s1'], dtype=np.float64)[:, self.aps],
                s2=np.asarray(stats['s2'], dtype=np.float64)[:, self.aps],
                relative_floor=int(str(name).rsplit('F', 1)[-1])-target,
            ))

    def at(self, xy):
        """Return float64 [position, AP, feature] using H and coordinates only.

        With zero coverage all observation summaries are zero. With coverage
        but no detections, log_total_scan_weight remains positive while the
        detection fraction, detected summaries and detected_valid are zero.
        The zero RSS summaries are placeholders, never observed RSS labels.
        """
        xy = np.asarray(xy, dtype=np.float64)
        blocks = []
        for h in self.sources:
            delta = (xy[:, None, :] - h['positions'][None, :, :]) / self.length
            distance2 = np.sum(delta * delta, axis=-1)
            w = np.exp(-.5 * distance2)
            w2 = np.exp(-.125 * distance2)  # Same kernel with length 2*l.
            total = w @ h['n']
            count = w @ h['k']
            count2 = w2 @ h['k']
            valid = count > 0
            valid2 = count2 > 0
            fraction = np.divide(count, total[:, None], out=np.zeros_like(count),
                                 where=total[:, None] > 0)
            mean = np.divide(w @ h['s1'], count, out=np.zeros_like(count), where=valid)
            mean2 = np.divide(w2 @ h['s1'], count2, out=np.zeros_like(count2), where=valid2)
            second = np.divide(w @ h['s2'], count, out=np.zeros_like(count), where=valid)
            # A weighted population variance; max removes cancellation below 0.
            std = np.sqrt(np.maximum(second-mean*mean, 0.))
            contrast = np.where(valid & valid2, mean-mean2, 0.)
            # Every detected scan has its position's kernel weight. Repeated
            # scans therefore contribute k*w^2, not (k*w)^2, in the denominator.
            # This is a weight-concentration count, not independent AP evidence
            # or GP posterior precision; correlated acquisitions remain so.
            squared_weight = (w*w) @ h['k']
            effective = np.divide(count*count, squared_weight, out=np.zeros_like(count),
                                  where=squared_weight > 0)
            blocks.append(np.stack((
                np.broadcast_to(np.log1p(total)[:, None], count.shape),
                fraction, mean, std, contrast, np.log1p(effective),
                valid.astype(np.float64), np.full(count.shape, h['relative_floor'], dtype=np.float64),
            ), axis=-1))
        if not blocks:
            return np.empty((len(xy), len(self.aps), 0), dtype=np.float64)
        return np.concatenate(blocks, axis=-1)

    def support_context(self):
        """Real S scan-major/AP-minor rows, retaining repeated acquisitions.

        x_D/y_D include all S scans times selected APs, including non-detections.
        x_R/y_R preserve that order after selecting detected rows only. No
        imputation or averaging creates signal labels. Empty contexts remain
        empty for the consuming model to reject if it requires observations.
        """
        support = self.support
        values = np.asarray(support['rss'][:, self.aps], dtype=np.float64)
        detected = values != NO_DETECT
        per_position = self.at(support['positions'])
        per_scan = per_position[support['group']]
        count = values.size
        x_d = per_scan.reshape(count, len(self.feature_names))
        y_d = detected.reshape(count).astype(np.float64)
        keep = detected.reshape(count)
        return dict(x_D=x_d, y_D=y_d, x_R=x_d[keep], y_R=values.reshape(count)[keep],
                    n_D=count, n_R=int(keep.sum()), n_scans=len(values), n_APs=len(self.aps),
                    flatten_order='scan-major, AP-minor (ascending table.aps)')
