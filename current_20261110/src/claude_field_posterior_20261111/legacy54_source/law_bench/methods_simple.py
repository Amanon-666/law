"""Shared-Task wrappers around the existing 54-event 3-WKNN rule.

building_information_tab_trial.nearest3 is the existing truth-independent
extraction of claude_hurdle_20261110.core.nearest3: missing=-110, RMS RSS
distance, stable three-neighbour selection, inverse-distance weights with
1e-8 floor, and clipping to the permitted geometry's outward 2 m bounds.
Importing core itself would eagerly read UJI and construct its manifests.
"""
import numpy as np


class WKNN_S:
    name = 'wknn_S'
    use_history = False

    def fit_source(self, sources):
        return None

    def predict(self, task):
        from building_information_tab_trial import nearest3

        parts = [task.H[f] for f in sorted(task.H)] if self.use_history else []
        parts.append(task.S)
        rss = np.concatenate([p['rss'] for p in parts])
        xy = np.concatenate([p['xy'] for p in parts])
        aps = np.flatnonzero((rss != 100).any(0))
        if not len(rss) or not len(aps):
            raise ValueError('WKNN needs context scans with at least one detected AP')
        bounds = np.stack([np.floor(xy.min(0)/2)*2, np.ceil(xy.max(0)/2)*2])
        result = nearest3(rss, xy, task.Q_rss, aps, bounds)
        self.diagnostics = dict(task_id=task.task_id, context_scans=len(rss),
            information='HS' if self.use_history else 'S', APs=aps.tolist(),
            neighbours=min(3, len(rss)), bounds=bounds.tolist(), output='projected')
        return result['projected']


class WKNN_HS(WKNN_S):
    name = 'wknn_HS'
    use_history = True
