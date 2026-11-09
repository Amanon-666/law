"""H-only TabICLv2 means and decoder-input features, with a fixed AP schema.

This wrapper changes the context and preprocessing used by the earlier H+S
model; it does not inherit that model's measured accuracy. It supplies neither
a joint covariance nor a target adaptation rule. No S/Q labels or statistics
are used here, and no synthetic history rows are added.

Package contract: the installed RNP-UJI/.deps_tabicl implementation exposes
EnsembleGenerator._generate_ensemble/transform, PreprocessingPipeline,
UniqueFeatureFilter(threshold=0), and model.predict_stats. Its predict_stats
mean is the average of crossing-corrected quantiles, NOT the separate
QuantileDistribution.mean() tail integral. Pin the package hash in the runner.
"""
import random

import numpy as np
import torch

from tabicl_history import features
from tabicl_history_learning import EpisodicHistoryRegressor
from tabicl._sklearn.preprocessing import (
    EnsembleGenerator, PreprocessingPipeline, UniqueFeatureFilter,
)


class _HistoryPreprocessor:
    """Original preprocessing on varying H columns; physical units otherwise."""

    def __init__(self, x, *, normalization_method, outlier_threshold, random_state):
        self.varying = np.any(x != x[0], axis=0)
        self.center = x[0].copy()
        self.original = None
        if self.varying.any():
            self.original = PreprocessingPipeline(
                normalization_method=normalization_method,
                outlier_threshold=outlier_threshold,
                random_state=random_state,
            ).fit(x[:, self.varying])
        self.X_transformed_ = np.asarray(x, dtype=np.float64) - self.center
        if self.original is not None:
            self.X_transformed_[:, self.varying] = self.original.X_transformed_

    def transform(self, x):
        # features() already encodes RSS in units of 105 dB and floor in floors.
        # Constant columns bypass both variance-based scaling and power/outlier
        # transforms: an AP unseen in H retains its distinct target RSS values.
        out = np.asarray(x, dtype=np.float64) - self.center
        if self.original is not None:
            out[:, self.varying] = self.original.transform(x[:, self.varying])
        return out


class _HistoryEnsemble(EnsembleGenerator):
    """Instance-local fit override; retain the official view assembly code."""

    def fit(self, x, y):
        # Official fit hard-creates threshold=1; changing a constructor option
        # or a previously fitted mask cannot preserve the downstream dimensions.
        assert not self.classification
        self.norm_methods_ = list(self.norm_methods or ["none", "power"])
        self.unique_filter_ = UniqueFeatureFilter(threshold=0).fit(x)
        self.X_ = self.unique_filter_.transform(x)
        self.y_ = np.asarray(y)
        self.n_features_in_ = self.X_.shape[1]
        self.rng_ = random.Random(self.random_state)
        self.ensemble_configs_, self.feature_shuffles_, _ = self._generate_ensemble()
        self.preprocessors_ = {
            method: _HistoryPreprocessor(
                self.X_, normalization_method=method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
            )
            for method in self.ensemble_configs_
        }
        return self


class HistoryFeatures(EpisodicHistoryRegressor):
    """Frozen, H-conditioned features for unlabeled target RSS rows.

    The caller supplies the complete 520-column AP schema in fixed order, not
    an H/S/Q detection-selected union. H is nonempty. Output means are raw XY,
    without bounds projection; phi is post-ICL-LayerNorm, before the decoder,
    with no additional feature normalization. Axis/view identities are retained.
    """

    def fit_history(self, histories, target_floor, aps, bounds, seed):
        aps = np.asarray(aps)
        if aps.shape != (520,) or not np.array_equal(np.sort(aps), np.arange(520)):
            raise ValueError('aps must contain the complete fixed 520-column schema')
        first = next(iter(histories.values()))
        empty = dict(rss=np.empty((0, first['rss'].shape[1]), dtype=first['rss'].dtype),
                     xy=np.empty((0, 2), dtype=np.float64))
        super().fit(histories, empty, target_floor, aps, bounds, seed)
        x = np.concatenate([
            features(h['rss'], int(name.split('F')[1]), self.aps)
            for name, h in histories.items()
        ])
        for estimator in self.estimators:
            old = estimator.ensemble_generator_
            replacement = _HistoryEnsemble(**old.get_params(deep=False))
            replacement.fit(estimator.X_encoder_.transform(x), old.y_)
            estimator.ensemble_generator_ = replacement
            # fit() uses the official default kv_cache=False; encoding below
            # deliberately follows the same uncached full-context views path.
            assert estimator.model_kv_cache_ is None
        model = self.loaded.model_
        model.requires_grad_(False)
        model.eval()
        self.feature_dim = model.icl_predictor.decoder[0].in_features
        self.history_size = len(x)
        return self

    def encode(self, rss):
        """Return mean[N,2] and phi[N,2,8,D], without reading target labels."""
        n = len(rss)
        if n == 0:
            return dict(mean=np.empty((0, 2), dtype=np.float64),
                        phi=np.empty((0, 2, 8, self.feature_dim), dtype=np.float32))
        model = self.loaded.model_
        model.eval()
        parameter = next(model.parameters())
        # Match the existing official inference path: float32 model/input on
        # the explicitly selected loader device, with each fitted estimator's
        # fixed official inference configuration. Do not inherit train/autocast.
        assert parameter.dtype == torch.float32
        device = parameter.device
        means = np.empty((n, 2), dtype=np.float64)
        phi = np.empty((n, 2, 8, self.feature_dim), dtype=np.float32)
        for axis, estimator in enumerate(self.estimators):
            x, y = self.views(rss, axis)
            assert x.shape[0] == y.shape[0] == 8
            assert y.shape[1] == self.history_size
            view_means = []
            for view in range(8):
                captured = []

                def capture_decoder_input(module, args):
                    captured.append(args[0].detach().float().cpu().clone())

                handle = model.icl_predictor.decoder.register_forward_pre_hook(capture_decoder_input)
                try:
                    with torch.no_grad(), torch.autocast(device.type, enabled=False):
                        mean = model.predict_stats(
                            torch.as_tensor(x[view:view+1], dtype=torch.float32, device=device),
                            torch.as_tensor(y[view:view+1], dtype=torch.float32, device=device),
                            output_type='mean', embed_with_test=False,
                            inference_config=estimator.inference_config_,
                        )
                finally:
                    handle.remove()
                # One table per call: the installed ICL manager passes the full
                # H+query row sequence through decoder once. Stop on API drift.
                assert len(captured) == 1
                assert captured[0].shape == (1, self.history_size+n, self.feature_dim)
                phi[:, axis, view] = captured[0][0, self.history_size:].numpy()
                assert mean.shape == (1, n)
                restored = estimator.y_scaler_.inverse_transform(
                    mean.detach().float().cpu().numpy().reshape(-1, 1)
                ).ravel()
                view_means.append(restored)
            means[:, axis] = np.mean(view_means, axis=0) + self.origin[axis]
        if not (np.isfinite(means).all() and np.isfinite(phi).all()):
            raise FloatingPointError('Nonfinite H-only TabICL mean/features')
        return dict(mean=means, phi=phi)
