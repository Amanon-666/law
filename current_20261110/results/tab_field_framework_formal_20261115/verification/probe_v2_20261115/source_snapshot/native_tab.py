"""Native full-parameter TabICLv2 classifiers/regressors for field roles."""
from pathlib import Path

import numpy as np
import torch
from tabicl import TabICLClassifier, TabICLRegressor


class NativeFieldRole:
    def __init__(self, role, checkpoint, device='cuda:0', views=1):
        self.role = role
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.views = views
        constructor = TabICLClassifier if role == 'det' else TabICLRegressor
        options = dict(model_path=self.checkpoint, allow_auto_download=False, device=device,
                       n_estimators=views, norm_methods=['none'], n_jobs=2, batch_size=1, use_amp=False)
        if role == 'det':
            options['softmax_temperature'] = 1.
        self.options = options
        self.constructor = constructor
        loaded = constructor(**options)
        loaded._load_model()
        self.model = loaded.model_.to(device)
        self.model_config = dict(loaded.model_config_, recompute=True)
        assert self.model_config['dropout'] == 0.
        for module in self.model.modules():
            if hasattr(module, 'recompute'):
                module.recompute = True
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

    def fit_context(self, X, y, seed=1):
        estimator = self.constructor(**self.options, random_state=seed)
        estimator.model_ = self.model
        estimator.model_config_ = self.model_config
        estimator.model_path_ = self.checkpoint
        estimator._load_model = lambda: None
        estimator.fit(X, y)
        if self.role == 'det':
            assert estimator.classes_.tolist() == [0, 1]
        self.estimator = estimator
        return self

    def transformed_views(self, X):
        estimator = self.estimator
        numerical = estimator.X_encoder_.transform(X)
        transformed = estimator.ensemble_generator_.transform(numerical, mode='both')
        x = np.concatenate([p[0] for p in transformed.values()])
        y = np.concatenate([p[1] for p in transformed.values()])
        shuffles = []
        if self.role == 'det':
            for values in estimator.ensemble_generator_.class_shuffles_.values():
                shuffles.extend(values)
            assert len(shuffles) == len(x)
        return x, y, shuffles

    def forward_views(self, transformed):
        self.model.train()
        x, y, shuffles = transformed
        predictions = []
        for view in range(len(x)):
            output = self.model(torch.as_tensor(x[view:view + 1], dtype=torch.float32, device=self.device),
                                torch.as_tensor(y[view:view + 1], dtype=torch.float32, device=self.device),
                                embed_with_test=False)[0].float()
            if self.role == 'det':
                # Match the official inverse class-shuffle decode before CE.
                output = output[:, :2][:, shuffles[view]]
            else:
                output = output.mean(-1)
                output = output * float(self.estimator.y_scaler_.scale_[0]) + float(self.estimator.y_scaler_.mean_[0])
            predictions.append(output)
        return predictions

    def loss(self, predictions, target, weight, training_mean_weight):
        device = predictions[0].device
        target = torch.as_tensor(target, dtype=torch.float32, device=device)
        weight = torch.as_tensor(weight, dtype=torch.float32, device=device)
        losses = []
        denominator = len(weight) * training_mean_weight
        for prediction in predictions:
            if self.role == 'det':
                log_probability = prediction.log_softmax(-1)
                loss = -(target * log_probability[:, 1] + (weight - target) * log_probability[:, 0]).sum() / denominator
            else:
                loss = (weight * (prediction - target).square()).sum() / denominator
            losses.append(loss)
        return torch.stack(losses).mean()


def context_labels(data, role, indices, scale_X, scale_target):
    if role == 'det':
        positive = data['nd'][indices].astype(int)
        negative = (data['n'][indices] - data['nd'][indices]).astype(int)
        X = np.concatenate((np.repeat(data['X'][indices], positive, axis=0),
                            np.repeat(data['X'][indices], negative, axis=0)))
        y = np.r_[np.ones(positive.sum(), dtype=int), np.zeros(negative.sum(), dtype=int)]
    else:
        count = data['nd'][indices].astype(int)
        features = data['X'] if role == 'mean' else scale_X
        values = data['mu'] if role == 'mean' else scale_target
        X = np.repeat(features[indices], count, axis=0)
        y = np.repeat(values[indices], count)
    return X, y
