"""Released TabICLv2 history-context recipe on the shared law_bench Task.

Reuse the installed public checkpoint and existing eight-estimator recipe.
No source fine-tuning, downloads, history subsampling or Query-fitted input
transform. Output bounds use H/current S only, as in the existing recipe.
"""
import numpy as np


class TabICL:
    name = 'tabicl'

    def __init__(self, checkpoint=None, seed=261026):
        self.checkpoint, self.seed = checkpoint, seed
        self.regressor = None

    def fit_source(self, sources):
        return self

    def predict(self, task):
        from tabicl_history import CHECKPOINT, FrozenHistoryRegressor

        if self.regressor is None:
            path = CHECKPOINT if self.checkpoint is None else self.checkpoint
            self.regressor = FrozenHistoryRegressor(checkpoint=path, device='cpu')
        histories = {f'B0F{f}': d for f, d in sorted(task.H.items())}
        seen = (task.S['rss'] != 100).any(0)
        for history in histories.values():
            seen |= (history['rss'] != 100).any(0)
        aps = np.flatnonzero(seen)
        xy = np.concatenate([d['xy'] for d in histories.values()] + [task.S['xy']])
        bounds = np.stack([np.floor(xy.min(0) / 2) * 2, np.ceil(xy.max(0) / 2) * 2])
        seed = self.seed + task.repeat
        self.regressor.fit(histories, task.S, f'B0F{task.floor}', aps, bounds, seed)
        result = self.regressor.predict(task.Q_rss)
        self.diagnostics = dict(task_id=task.task_id, checkpoint=str(self.regressor.checkpoint),
            source_fit=False, history_scans=sum(len(d['rss']) for d in histories.values()),
            support_scans=len(task.S['rss']), ap_count=len(aps), seed=seed,
            output='projected', bounds=bounds.tolist())
        return result['projected']
