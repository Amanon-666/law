"""AP-bound GP relations for the existing trained Tab history consumer.

Uses the same raw H/S table, preprocessing, coordinate targets, quantile
readout, ensemble averaging and projection as the current consumer. Only a
separately owned relation encoder is trainable; the supplied trained Tab
backbone remains frozen. No training schedule, optimizer step, GP fit,
automatic download or experiment launch is defined in this module.

Build a RetainedInfluenceCollector from exactly the task/H used by the GP,
pass it as relation_collector to predict_retained, and give collector.finish()
to fit(..., relations=...). predict() then uses the graph without Q labels;
backward() and backward_system() may use labeled SOURCE queries only. The
former retains the existing single-Tab pinball comparator. The latter, called
by InfluencePositionSystem.backward_source(), accumulates the actual fixed-
half system's position-balanced distance gradient. Neither method starts an
optimizer or an approved experiment; no improvement is claimed by its presence.

The first row-pooled prototype could not bind a relation to its AP. This one
keeps that binding through constant-column removal, the actual per-axis/view
shuffle, and the native ordered feature groups. It neither changes RSS nor
assumes learned AP identifiers transfer between buildings. GP and original
Tab remain usable independently. enabled=False delegates to original predict.
No old representation/KV cache is accepted by the enabled path.

Required per-source d/u/b analytic moments bypass learned anchor averaging
before the existing AP-bound column injection. They preserve an already
computed additive response and uncertainty terms, not a new physical
calibration model or a performance guarantee. v2 sidecars explicitly record
their moment names; older v1 sidecars are not implicitly upgraded.
"""
from contextlib import nullcontext

import numpy as np
import torch

from context_influence_adapter import ColumnInfluenceEncoder, column_influence
from context_influence_features import FEATURE_NAMES, MOMENT_NAMES
from tabicl_history import features, state_hash
from tabicl_history_learning import EpisodicHistoryRegressor
from uji import group_scans


class InfluenceHistoryRegressor(EpisodicHistoryRegressor):
    def __init__(self, checkpoint, device='cpu', hidden_dim=32):
        # Explicit checkpoint: reuse a chosen existing asset, never silently
        # substitute public weights for the project's trained backbone.
        super().__init__(checkpoint, device)
        network = self.loaded.model_
        assert network.max_classes == 0
        col = network.col_embedder
        assert col.feature_group in (True, 'same') and col.feature_group_size == 3
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self.relation_encoder = ColumnInfluenceEncoder(len(FEATURE_NAMES), col.embed_dim,
            col.feature_group_size, hidden_dim).to(device)

    def fit(self, histories, support, target_floor, aps, bounds, seed, *, relations):
        """Bind a complete H/S-only graph to exactly the original context rows."""
        aps = np.asarray(aps, dtype=np.int64)
        assert np.array_equal(aps, relations['aps'])
        assert list(histories) == relations['source_names']
        assert list(FEATURE_NAMES) == relations['feature_names']
        assert list(MOMENT_NAMES) == relations['moment_names']
        lengths = [len(h['rss']) for h in histories.values()]
        assert np.array_equal(np.r_[0, np.cumsum(lengths)], relations['source_offsets'])
        raw = np.concatenate([h['rss'][:, aps] for h in histories.values()])
        xy = np.concatenate([h['xy'] for h in histories.values()])
        assert np.array_equal(raw, relations['raw_history_rss'])
        assert np.array_equal(xy, relations['row_xy_absolute'])
        s = group_scans(support['rss'], support['xy'])
        # GP local-to-absolute conversion may incur rounding, not a changed
        # position; statistics and raw H rows must still agree exactly.
        assert np.allclose(s['positions'], relations['support_xy_absolute'], rtol=0, atol=1e-8)
        for key in ('n', 'k', 's1', 's2'):
            actual = (np.broadcast_to(s['stats']['n'], (len(aps), len(s['positions'])))
                      if key == 'n' else s['stats'][key][:, aps].T)
            assert np.array_equal(actual, relations['support_stats'][key])
        super().fit(histories, support, target_floor, aps, bounds, seed)
        for estimator in self.estimators:
            assert estimator.model_kv_cache_ is None
            assert set(estimator.ensemble_generator_.norm_methods_) <= {'none', 'power'}
        device = next(self.relation_encoder.parameters()).device
        self.relation_features = relations['features'].detach().to(device)
        self.relation_covariance = relations['normalized_covariance'].detach().to(device)
        self.relation_moments = relations['moments'].detach().to(device)
        assert self.relation_features.shape[:3] == (len(raw), len(aps), len(s['positions']))
        assert self.relation_covariance.shape == (len(aps), len(s['positions']), len(s['positions']))
        assert self.relation_moments.shape == (len(raw), len(aps), 3)
        return self

    def _bound_views(self, query_rss, axis):
        """Return actual view arrays with matching original-AP slot maps."""
        estimator = self.estimators[axis]
        assert estimator.model_kv_cache_ is None
        query = features(query_rss, self.floor, self.aps)
        query = estimator.X_encoder_.transform(query)
        generator = estimator.ensemble_generator_
        data = generator.transform(query, mode='both')
        # Numeric input encoder and current normalizers do not mix columns.
        # Last input column is the physical floor, not another AP.
        slots = np.r_[np.arange(len(self.aps), dtype=np.int64), -1]
        keep = generator.unique_filter_.features_to_keep_
        assert keep.shape == slots.shape
        slots = slots[keep]
        for method, (x, y) in data.items():
            assert method in {'none', 'power'}
            shuffles = generator.feature_shuffles_[method]
            maps = np.stack([slots[np.asarray(order, dtype=np.int64)] for order in shuffles])
            assert maps.shape == (len(x), x.shape[-1]) and len(x) == len(y)
            yield x, y, maps

    def _scope(self, maps):
        return column_influence(self.loaded.model_, self.relation_encoder,
            self.relation_features, self.relation_covariance, torch.as_tensor(maps), self.relation_moments)

    def predict(self, query_rss, *, enabled=True):
        """Original official quantile-mean readout, with batch-bound relations."""
        if not enabled:
            return super().predict(query_rss)
        network = self.loaded.model_
        network.eval()
        self.relation_encoder.eval()
        axes = []
        # Keep the current wrapper's two-thread inference setting; restore the
        # caller's setting just as official sklearn prediction does.
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for axis, estimator in enumerate(self.estimators):
                results = []
                for x, y, maps in self._bound_views(query_rss, axis):
                    size = estimator.batch_size or len(x)
                    batches = int(np.ceil(len(x)/size))
                    for xb, yb, mb in zip(np.array_split(x, batches), np.array_split(y, batches),
                                          np.array_split(maps, batches)):
                        xt = torch.from_numpy(xb).float().to(estimator.device_)
                        yt = torch.from_numpy(yb).float().to(estimator.device_)
                        with torch.no_grad(), self._scope(mb):
                            # predict_stats applies the original quantile
                            # distribution's monotonic postprocessing.
                            output = network.predict_stats(xt, yt, output_type=['mean'],
                                alphas=None, inference_config=estimator.inference_config_)
                        results.append(output.float().cpu().numpy())
                arr = np.concatenate(results)
                arr = estimator.y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)
                axes.append(arr.mean(axis=0)+self.origin[axis])
        finally:
            torch.set_num_threads(old_threads)
        raw = np.column_stack(axes)
        assert np.isfinite(raw).all()
        return dict(raw=raw, projected=np.clip(raw, self.bounds[0], self.bounds[1]))

    def backward(self, query_rss, query_xy, scaler=None):
        """Accumulate original equal-axis/view/position source pinball gradient.

        Only relation_encoder parameters are trainable. No optimizer step or
        target selection occurs. Zero output initialization means the first
        step can update the output projection while inner layers initially
        have zero gradients; this is not evidence of permanent disconnection.
        """
        query_xy = np.asarray(query_xy, dtype=np.float64)
        _, group, count = np.unique(query_xy, axis=0, return_inverse=True, return_counts=True)
        network = self.loaded.model_
        device = next(network.parameters()).device
        weights = torch.as_tensor(1./(len(count)*count[group]), dtype=torch.float32, device=device)
        network.train()
        self.relation_encoder.train()
        total = 0.
        try:
            for axis, estimator in enumerate(self.estimators):
                views = list(self._bound_views(query_rss, axis))
                view_count = sum(len(x) for x, _, _ in views)
                centered = np.asarray(query_xy[:, axis]-self.origin[axis], dtype=np.float32)
                target = estimator.y_scaler_.transform(centered[:, None]).ravel()
                target = torch.as_tensor(target, dtype=torch.float32, device=device)
                for x, y, maps in views:
                    for view in range(len(x)):
                        amp = torch.autocast('cuda', dtype=torch.float16) if scaler is not None else nullcontext()
                        # Keep the hook live until this view's backward ends.
                        with self._scope(maps[view:view+1]):
                            with amp:
                                output = network(torch.as_tensor(x[view:view+1], dtype=torch.float32, device=device),
                                    torch.as_tensor(y[view:view+1], dtype=torch.float32, device=device),
                                    embed_with_test=False)
                                alpha = torch.linspace(0, 1, output.shape[-1]+2,
                                    dtype=output.dtype, device=device)[1:-1]
                                diff = target[None, :, None]-output
                                per_query = torch.maximum(alpha*diff, (alpha-1)*diff).mean(dim=(0, 2))
                                loss = (per_query*weights).sum()/(2*view_count)
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError('Nonfinite relation source pinball loss')
                            (scaler.scale(loss) if scaler is not None else loss).backward()
                            total += float(loss.detach())
        finally:
            network.eval()
            self.relation_encoder.eval()
        return total

    def backward_system(self, query_rss, source_xy, gp_projected, scaler=None):
        """Accumulate the complete fixed-half system's source distance gradient.

        The first pass obtains both full ensemble coordinates without keeping
        model graphs. The second recomputes each view and applies the SAME
        final-coordinate cotangent, times its axis scale and 1/view_count.
        This is the chain rule for the full ensemble, not per-view MDE. Only
        one view graph stays live, at the cost of an extra forward pass.

        Both passes use the same train-mode path, AMP setting and fixed views;
        this inspected backbone has dropout=0. GP and backbone are frozen,
        and no optimizer operation is permitted between passes. Deployment
        keeps the official inference path; floating-point train/inference
        differences are not claimed to be bitwise absent. The affine readout,
        ensemble averaging, projection and half-combination match deployment.

        Caller must supply legal SOURCE query labels, excluding its reserved
        evaluation positions. This method does not authorize or launch a run.
        Original backward() is the separate single-Tab pinball comparator;
        the two objectives are not silently combined with a new loss weight.
        """
        from context_influence_objective import system_coordinate_feedback

        source_xy = np.asarray(source_xy, dtype=np.float64)
        gp_projected = np.asarray(gp_projected, dtype=np.float64)
        assert source_xy.shape == gp_projected.shape == (len(query_rss), 2)
        network = self.loaded.model_
        assert network.dropout == 0., 'Two-pass derivative requires the inspected deterministic backbone'
        assert not any(parameter.requires_grad for parameter in network.parameters())
        assert network.quantile_dist.crossing_method == 'sort'
        device = next(network.parameters()).device
        _, group, count = np.unique(source_xy, axis=0, return_inverse=True, return_counts=True)
        weights = torch.as_tensor(1./(len(count)*count[group]), dtype=torch.float64, device=device)
        views = [list(self._bound_views(query_rss, axis)) for axis in range(2)]
        counts = [sum(len(x) for x, _, _ in axis_views) for axis_views in views]

        def view_mean(x, y, view):
            amp = torch.autocast('cuda', dtype=torch.float16) if scaler is not None else nullcontext()
            with amp:
                raw = network(torch.as_tensor(x[view:view+1], dtype=torch.float32, device=device),
                    torch.as_tensor(y[view:view+1], dtype=torch.float32, device=device),
                    embed_with_test=False)
                # The official deployment mean averages monotonic quantiles.
                # The installed sort operation is differentiable and preserves
                # the mean; use it explicitly to retain readout semantics.
                result = network.quantile_dist(raw).quantiles.mean(dim=-1)
            return result.float()[0]

        network.train()
        self.relation_encoder.train()
        try:
            axes = []
            for axis, axis_views in enumerate(views):
                means = []
                for x, y, maps in axis_views:
                    for view in range(len(x)):
                        with torch.no_grad(), self._scope(maps[view:view+1]):
                            means.append(view_mean(x, y, view).cpu().numpy())
                arr = np.stack(means)
                # Match the official inverse-scaler and view-mean order.
                arr = self.estimators[axis].y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)
                axes.append(arr.mean(axis=0)+self.origin[axis])
            point = torch.as_tensor(np.column_stack(axes), dtype=torch.float64, device=device)
            loss, feedback = system_coordinate_feedback(point,
                torch.as_tensor(gp_projected, dtype=torch.float64, device=device),
                torch.as_tensor(source_xy, dtype=torch.float64, device=device),
                torch.as_tensor(self.bounds, dtype=torch.float64, device=device), weights)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(feedback).all()):
                raise FloatingPointError('Nonfinite complete-system source loss or coordinate gradient')
            for axis, axis_views in enumerate(views):
                scale = float(self.estimators[axis].y_scaler_.scale_[0])
                coefficient = feedback[:, axis]*scale/counts[axis]
                for x, y, maps in axis_views:
                    for view in range(len(x)):
                        with self._scope(maps[view:view+1]):
                            mean = view_mean(x, y, view)
                            # Scalar VJP surrogate: its VALUE is not an MDE.
                            # Report only the first pass's actual system loss.
                            surrogate = (mean.to(torch.float64)*coefficient).sum()
                            (scaler.scale(surrogate) if scaler is not None else surrogate).backward()
            return float(loss)
        finally:
            network.eval()
            self.relation_encoder.eval()

    def save(self, path, *, source_updates, provenance):
        """Save only the learned relation sidecar, tied to the frozen backbone.

        This is intentionally not an official raw Tab checkpoint. Restore it
        through load_adapter on the SAME trained backbone and refit H/S for
        each deployment. No task-specific relation graph is saved here.
        """
        torch.save(dict(format='gp_column_influence_v2',
            backbone_state_sha256=state_hash(self.loaded.model_),
            config=self.relation_encoder.config, feature_names=list(FEATURE_NAMES),
            moment_names=list(MOMENT_NAMES),
            state_dict={k:v.detach().cpu() for k,v in self.relation_encoder.state_dict().items()},
            source_updates=source_updates, provenance=provenance), path)

    def load_adapter(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['format'] == 'gp_column_influence_v2', 'v1 lacks analytic moment channels; no automatic conversion'
        assert saved['backbone_state_sha256'] == state_hash(self.loaded.model_)
        assert saved['config'] == self.relation_encoder.config
        assert saved['feature_names'] == list(FEATURE_NAMES)
        assert saved['moment_names'] == list(MOMENT_NAMES)
        self.relation_encoder.load_state_dict(saved['state_dict'])
        return self
