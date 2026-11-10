"""Collect AP-bound GP influence features from an existing retained-GP pass.

No fitting, sampling, query input, file I/O or model call is performed here.
The caller constructs a collector from absolute-coordinate raw H and its
matching prepared task/state, then invokes it with each AP's already computed
prior and complete S posterior moments. Raw history insertion/scan order is
preserved, including repeated coordinates; no unique-XY remapping is used.

For each historical row belonging to source j, only that source's persistent
covariance C_j(X_j,S) is used to form A_j=omega_j**2 C_j (K_SS+jitter I)^-1.
K_SS is the original full support prior covariance, including acquisition
drift/intercept. When some source has a field, a missing source/AP still has
Prior(None)'s nonzero background RBF covariance. Only when ALL sources lack
the AP does retained propagation vanish and all C_j become zero. Neither
source coefficients nor these influences are posterior responsibility
probabilities. No new observation noise is added to posterior covariance V.

Output keeps the AP and anchor axes explicitly; the consumer must retain their
binding to the original RSS columns. This collector does not choose an encoder
or pool APs. GP algebra is float64 on the existing task device. Normalized
model features and full V/s**2 are copied to CPU float32; original-coordinate,
raw-RSS and S-statistic metadata remain available for fit-time binding checks.
s=obs.noise_std and length=base.length are fixed source-estimated parameters,
not fitted to query or claimed to be calibrated real-world uncertainty.

This is a derived representation of existing H/S, not new information or a
validated improvement. It does not resolve single-support persistent-versus-
acquisition ambiguity. The current task API requires K>0; no S0 extension or
fallback asset fitting is provided.

The exact own-source contractions d=A_j e/s, u=diag(A_j V A_j^T)/s^2,
and b=diag(B_j A_j^T)/s^2 are also retained before any learned anchor pooling,
where B_j=omega_j^2 C_j(X_j,S). This avoids asking the learned mean-pooling
branch to reconstruct an additive update when K changes. u-b is only the
source's conditional variance correction; neither u nor b is a complete
posterior variance, source responsibility or calibrated target confidence.
The source prior diagonal would be needed for its full variance, and sources
become conditionally correlated. Their individual variances cannot simply be
added to obtain the target variance. B_j A_j^T uses the original jittered
solve exactly, rather than silently substituting A_j K_SS A_j^T.
"""
import numpy as np
import torch

from context_influence_adapter import factorized_influence


FEATURE_NAMES = (
    'A_source', 'A_source_times_innovation_over_noise', 'innovation_over_noise',
    'support_detected_count', 'support_detection_fraction', 'source_weight',
    'support_minus_history_x_over_length', 'support_minus_history_y_over_length',
    'history_detected', 'history_RSS_tab_encoded', 'source_field_present',
)
MOMENT_NAMES = (
    'source_mean_update_over_noise',
    'source_support_posterior_variance_contribution_over_noise_squared',
    'source_prior_conditioning_subtraction_over_noise_squared',
)


class RetainedInfluenceCollector:
    """Callback for the unchanged per-AP M-PMC result.

    collector = RetainedInfluenceCollector(task, state, histories_raw)
    collector(ap, prior, sm, sk, moments)  # moments={'mean': ..., 'cov': ...}
    result = collector.finish()

    task comes from prepare_from_scans/task_inputs with task['origin']; its
    case.sx and fitted History coordinates are already local. histories_raw
    contains {name:{rss[N_j,AP_columns], xy[N_j,2]}} in absolute XY. Its insertion
    order MUST equal task['names'], matching original Tab's H concatenation.
    AP output order is exactly list(task['cases']); callbacks may arrive in
    any order, but every AP must arrive exactly once. The callback must receive
    the actual prior used with the same case/state, not a reconstructed prior
    with another support covariance or already weight-multiplied cross terms.
    """
    def __init__(self, task, state, histories_raw):
        self.task = task
        self.names = list(task['names'])
        assert list(histories_raw) == self.names, 'History insertion order differs from GP source order'
        assert self.names and task['cases']
        self.aps = np.asarray(list(task['cases']), dtype=np.int64)
        self.column = {int(ap): column for column, ap in enumerate(self.aps)}
        self.weights = [float(w) for w in state['weights']]
        assert len(self.weights) == len(self.names)
        self.sigma = float(state['sigma']['intercept_std'])
        self.origin = np.asarray(task['origin'], dtype=np.float64).copy()
        assert self.origin.shape == (2,)
        self.scale = float(task['obs']['noise_std'])
        self.length = float(task['base']['length'])
        assert self.scale > 0 and self.length > 0
        first = next(iter(task['cases'].values()))
        self.sx = first['sx']
        self.k = len(self.sx)
        assert self.k > 0 and self.sx.shape == (self.k, 2) and self.sx.dtype == torch.float64
        self.offsets = [0]
        xy, rss, source_indices, within_source = [], [], [], []
        for j, name in enumerate(self.names):
            h = histories_raw[name]
            x, r = np.asarray(h['xy']), np.asarray(h['rss'])
            assert x.shape == (len(r), 2) and r.ndim == 2
            assert self.aps.min() >= 0 and self.aps.max() < r.shape[1]
            xy.append(np.array(x, dtype=np.float64, copy=True))
            rss.append(np.array(r[:, self.aps], copy=True))
            source_indices.extend([j]*len(r))
            within_source.extend(range(len(r)))
            self.offsets.append(self.offsets[-1]+len(r))
        self.xy = np.concatenate(xy)
        self.rss = np.concatenate(rss)
        self.row_source = np.asarray(source_indices, dtype=np.int64)
        self.row_within_source = np.asarray(within_source, dtype=np.int64)
        n, a = len(self.xy), len(self.aps)
        self.features = torch.empty((n, a, self.k, len(FEATURE_NAMES)), dtype=torch.float32, device='cpu')
        self.covariance = torch.empty((a, self.k, self.k), dtype=torch.float32, device='cpu')
        self.moments = torch.empty((n, a, len(MOMENT_NAMES)), dtype=torch.float32, device='cpu')
        self.present = np.empty((len(self.names), a), dtype=bool)
        self.stats = {key: np.empty((a, self.k), dtype=np.float64) for key in ('n', 'k', 's1', 's2')}
        self.seen = set()

    @torch.no_grad()
    def __call__(self, ap, prior, sm, sk, moments):
        ap = int(ap)
        assert ap in self.column and ap not in self.seen, 'Unknown or repeated AP callback'
        column = self.column[ap]
        case = self.task['cases'][ap]
        sx = case['sx']
        assert torch.equal(sx, self.sx) and sx.dtype == torch.float64
        assert sm.shape == moments['mean'].shape == (self.k,)
        assert sk.shape == moments['cov'].shape == (self.k, self.k)
        assert all(t.dtype == torch.float64 and t.device == sx.device
                   for t in (sm, sk, moments['mean'], moments['cov']))
        assert len(case['histories']) == len(self.names)
        available = [h is not None for h in case['histories']]
        self.present[:, column] = available
        assert float(prior.intercept_std) == self.sigma
        if any(available):
            assert len(prior.sources) == len(self.names) and list(prior.weights) == self.weights
            assert all(p.history is h and p.base['drift'] == 0.
                       for p, h in zip(prior.sources, case['histories']))
        else:
            assert not prior.sources
        n, detected = case['stats']['n'], case['stats']['k']
        assert n.shape == detected.shape == (self.k,) and bool((n > 0).all())
        fraction = detected/n
        for key in self.stats:
            value = case['stats'][key]
            assert value.shape == (self.k,)
            self.stats[key][column] = value.detach().cpu().numpy()
        self.covariance[column].copy_((moments['cov']/self.scale**2).to(device='cpu', dtype=torch.float32))
        for j, name in enumerate(self.names):
            start, stop = self.offsets[j:j+2]
            x = sx.new_tensor(self.xy[start:stop]-self.origin)
            if any(available):
                source = prior.sources[j]
                _, fx = source.features(x)
                _, fs = source.features(sx)
                # In particular, source.history=None is NOT a zero covariance
                # when another source has this AP: retain the original RBF.
                cross = source.covariance(x, sx, fx, fs)
            else:
                cross = sx.new_zeros((len(x), self.k))
            relation = factorized_influence(cross[None], [self.weights[j]], sk, sm,
                                             moments['mean'], moments['cov'])
            a = relation['A'][0]
            e = relation['e']/self.scale
            # Analytic contractions are computed before float32 packing and
            # bypass the encoder's learned mean over K. Full V is retained;
            # replacing it by its diagonal would change u for correlated S.
            weighted_cross = self.weights[j]**2 * cross
            contracted = torch.stack(((a*e[None]).sum(-1),
                ((a @ relation['V'])*a).sum(-1)/self.scale**2,
                (weighted_cross*a).sum(-1)/self.scale**2), dim=-1)
            self.moments[start:stop, column].copy_(contracted.to(device='cpu', dtype=torch.float32))
            raw = sx.new_tensor(self.rss[start:stop, column])
            mask = raw != 100
            encoded = torch.where(mask, (raw+105.)/105., torch.zeros_like(raw))
            relative = (sx[None]-x[:, None])/self.length
            block = torch.stack((a, a*e[None], e.expand_as(a), detected.expand_as(a),
                fraction.expand_as(a), a.new_full(a.shape, self.weights[j]),
                relative[..., 0], relative[..., 1], mask[:, None].expand_as(a),
                encoded[:, None].expand_as(a), a.new_full(a.shape, float(available[j]))), dim=-1)
            self.features[start:stop, column].copy_(block.to(device='cpu', dtype=torch.float32))
        self.seen.add(ap)

    def finish(self):
        """Return complete CPU tensors plus original row/AP binding metadata.

        features: float32 [N_H,AP,K,11]; normalized_covariance: float32 [AP,K,K].
        moments: float32 [N_H,AP,3], ordered by MOMENT_NAMES; no K division.
        raw_history_rss: original values [N_H,AP], with 100 missing sentinel;
        support_stats: float64 numpy arrays [AP,K], n repeated on each AP axis.
        source_field_present: bool [source,AP]; row indices are zero-based.
        source_offsets partitions the exact insertion-order H concatenation.
        No result is returned until every expected AP has supplied its moments.
        """
        assert self.seen == set(self.column), 'Incomplete GP AP callbacks'
        return dict(features=self.features, normalized_covariance=self.covariance,
            moments=self.moments, moment_names=list(MOMENT_NAMES),
            aps=self.aps.copy(), feature_names=list(FEATURE_NAMES),
            source_names=list(self.names), source_offsets=list(self.offsets),
            row_source_indices=self.row_source.copy(), row_within_source=self.row_within_source.copy(),
            row_xy_absolute=self.xy.copy(), raw_history_rss=self.rss.copy(),
            support_xy_absolute=self.sx.detach().cpu().numpy()+self.origin,
            support_stats={key: value.copy() for key, value in self.stats.items()},
            source_field_present=self.present.copy(), source_weights=list(self.weights),
            origin=self.origin.copy(), scale_db=self.scale, length=self.length,
            relation_scope='Own-source persistent influence at each original H row; full S posterior covariance per AP',
            uncertainty_scope='Fixed fitted coefficients and GP moments; not parameter posterior responsibilities or verified target calibration confidence')
