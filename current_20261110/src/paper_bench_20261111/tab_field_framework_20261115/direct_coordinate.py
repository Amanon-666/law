"""Single-view source learning for the original H+S coordinate predictor.

The caller owns source train/development splits, seeds and optimizer steps.
source_task_loss accumulates the original position-balanced pinball gradient.
fit_task and predict_task expose the same frozen deployment path.
"""
import hashlib

import numpy as np
import torch

from tabicl_history import CHECKPOINT, TabICLRegressor, features
from tabicl_history_learning import EpisodicHistoryRegressor


PUBLIC_REGRESSOR_SHA256 = '0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a'


class DirectCoordinateRegressor(EpisodicHistoryRegressor):
    def __init__(self, checkpoint=CHECKPOINT, *, seed, device='cuda:0', amp=True):
        super().__init__(checkpoint=checkpoint, device=device)
        self.seed = int(seed)
        self.amp = bool(amp)
        self.view_count = 1
        self.norm_methods = ['none']
        self.enable_recompute()
        self.model = self.loaded.model_
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.scaler = torch.amp.GradScaler('cuda', init_scale=1024.) if self.amp else None
        with self.checkpoint.open('rb') as stream:
            self.checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()

    def fit(self, histories, support, target_floor, aps, bounds, seed):
        """Original context construction with one official unnormalized view."""
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
                device=self.device, n_jobs=2, n_estimators=self.view_count,
                norm_methods=self.norm_methods, batch_size=1, random_state=seed,
                use_amp=self.amp)
            inner.model_ = self.loaded.model_
            inner.model_config_ = self.loaded.model_config_
            inner.model_path_ = self.checkpoint
            inner._load_model = lambda: None
            inner.fit(x, y[:, axis]-self.origin[axis])
            self.estimators.append(inner)
        return self

    def fit_task(self, task):
        """Fit transformations from every original H/S row and observed AP."""
        # These labels encode physical floors, as in the original bench adapter.
        histories = {f'B0F{floor}': data for floor, data in sorted(task.H.items())}
        seen = (task.S['rss'] != 100).any(0)
        for history in histories.values():
            seen |= (history['rss'] != 100).any(0)
        aps = np.flatnonzero(seen)
        xy = np.concatenate([h['xy'] for h in histories.values()]+[task.S['xy']])
        bounds = np.stack([np.floor(xy.min(0)/2)*2, np.ceil(xy.max(0)/2)*2])
        self.fit(histories, task.S, f'B0F{task.floor}', aps, bounds,
                 self.seed+task.repeat)
        self.diagnostics = dict(task_id=task.task_id, H_rows={f: len(h['rss'])
            for f, h in task.H.items()}, S_rows=len(task.S['rss']), APs=aps.tolist(),
            context_rows=sum(len(h['rss']) for h in task.H.values())+len(task.S['rss']),
            bounds=bounds.tolist(), view_count=self.view_count,
            norm_methods=list(self.norm_methods), amp=self.amp)
        return self

    def source_task_loss(self, task, truth):
        """Accumulate source gradients, returning the original scalar loss.

        The caller zeroes gradients, then unscales/steps/updates self.scaler.
        Formal B0 training and B2 development are selected by the caller.
        """
        if task.dataset != 'uji' or task.building not in ('B0', 'B2'):
            raise ValueError('Source coordinate gradients require UJI B0 or B2')
        if task.meta['protocol'] != 'static-resample-v2':
            raise ValueError('Source coordinate episodes require static-resample-v2')
        support_positions = set(map(tuple, np.asarray(task.S['xy'])))
        query_positions = set(map(tuple, np.asarray(truth['xy'])))
        if support_positions & query_positions:
            raise ValueError('Source Support and Query positions must be disjoint')
        self.fit_task(task)
        return self.backward(task.Q_rss, truth['xy'], scaler=self.scaler)

    def predict_task(self, task):
        self.fit_task(task)
        return self.predict(task.Q_rss)['projected']

    def save(self, path, *, source_updates, provenance):
        information = dict(provenance, direct_coordinate=dict(
            view_count=self.view_count, norm_methods=list(self.norm_methods),
            seed=self.seed, amp=self.amp, recompute=True,
            base_checkpoint_sha256=PUBLIC_REGRESSOR_SHA256,
            loaded_checkpoint_sha256=self.checkpoint_sha256))
        return super().save(path, source_updates=source_updates, provenance=information)
