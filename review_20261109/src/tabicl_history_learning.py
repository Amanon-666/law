"""Episodic source learning for the existing H+S TabICL predictor.

Learn the shared conditional coordinate distributions, not target-floor weights.
fit(H,S) is inherited unchanged; only labeled SOURCE queries enter backward().
All raw history rows and source floor features remain available at deployment.
This changes the learning object relative to ReFi's field operator and must not
be described as an implementation of that operator or an established improvement.

The official 999-quantile pinball objective is averaged across axes and the same
eight preprocessing views used by inference. Query positions balance the loss;
context H/S alone fits transforms, AP filtering, origins and output scales.
An optimizer must persist across episodes. Target deployment uses fit/predict
with frozen shared weights, with no target query gradients or batch adaptation.
"""
from contextlib import nullcontext

import numpy as np
import torch

from tabicl_history import FrozenHistoryRegressor, features


class EpisodicHistoryRegressor(FrozenHistoryRegressor):
    def enable_recompute(self):
        """Enable the released model's activation checkpointing, keeping weights."""
        names = []
        for name, module in self.loaded.model_.named_modules():
            if hasattr(module, 'recompute'):
                module.recompute = True
                names.append(name)
        self.loaded.model_config_ = dict(self.loaded.model_config_, recompute=True)
        return names

    def views(self, query_rss, axis):
        """Use the fitted deployment transforms; query labels are absent."""
        estimator = self.estimators[axis]
        query = features(query_rss, self.floor, self.aps)
        query = estimator.X_encoder_.transform(query)
        transformed = estimator.ensemble_generator_.transform(query, mode='both')
        x = np.concatenate([pair[0] for pair in transformed.values()])
        y = np.concatenate([pair[1] for pair in transformed.values()])
        return x, y

    def backward(self, query_rss, query_xy, scaler=None):
        """Accumulate one equal-axis/equal-position source episode gradient.

        No optimizer creation, stepping or clearing here: callers can average
        episodes explicitly. Views run serially on the model's device. An
        explicit CUDA GradScaler enables fp16 autocast; no retry is performed.
        """
        query_xy = np.asarray(query_xy, dtype=np.float64)
        _, group, count = np.unique(query_xy, axis=0, return_inverse=True, return_counts=True)
        model = self.loaded.model_
        device = next(model.parameters()).device
        weights = torch.as_tensor(1. / (len(count) * count[group]), dtype=torch.float32, device=device)
        model.train()
        total = 0.
        for axis, estimator in enumerate(self.estimators):
            x, y = self.views(query_rss, axis)
            centered = np.asarray(query_xy[:, axis] - self.origin[axis], dtype=np.float32)
            target = estimator.y_scaler_.transform(centered[:, None]).ravel()
            target = torch.as_tensor(target, dtype=torch.float32, device=device)
            for view in range(len(x)):
                context = torch.autocast('cuda',dtype=torch.float16) if scaler is not None else nullcontext()
                with context:
                    prediction = model(torch.as_tensor(x[view:view+1], dtype=torch.float32, device=device),
                                       torch.as_tensor(y[view:view+1], dtype=torch.float32, device=device),
                                       embed_with_test=False)
                    alpha = torch.linspace(0, 1, prediction.shape[-1]+2,
                                           dtype=prediction.dtype, device=device)[1:-1]
                    diff = target[None, :, None] - prediction
                    per_query = torch.maximum(alpha*diff, (alpha-1)*diff).mean(dim=(0, 2))
                    loss = (per_query*weights).sum() / (2*len(x))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite source episode pinball loss')
                (scaler.scale(loss) if scaler is not None else loss).backward()
                total += float(loss.detach())
        model.eval()
        return total

    def save(self, path, *, source_updates, provenance):
        """Official-loader-compatible shared model; no fitted target context."""
        torch.save(dict(config=self.loaded.model_config_,
            state_dict={k:v.detach().cpu() for k,v in self.loaded.model_.state_dict().items()},
            source_updates=source_updates, provenance=provenance), path)
