"""Continuous hurdle translation from explicit, permitted scan-cell tables.

No CSV/core/global dataset access, automatic training or resource download.
The original hurdle scripts are deliberately NOT imported: their imports read
the full dataset. Only the pure 55D ObservationTranslationContext is reused.
This implementation has only been statically reviewed, not numerically run.

Table contract: representation='continuous_scan_cells_v1', role='train',
floor/event_id strings; X[N,55], n/nd/mean_detected/variance_detected/weight[N].
Within variance uses denominator nd; mean/variance are zero for nd=0.
Each table weights positions and selected APs equally and sums to one. Table
construction owns source permissions, acquisition isolation and same-target-
floor supervision. No row sampling or class balancing is added. Each estimator
rescales its actual positive fit weights to mean one, preserving their relative
weights while fixing the scale seen by l2_regularization=1. Detection uses the
retained double-label rows; mean uses detected cells; each crossfit mean uses
only its training-floor detected cells; scale uses all held-out detected cells
pooled across both folds. Crossfit splits sorted floors alternately, never cells.
Cell compression preserves the weighted data-loss shape (mean squared error
up to a label-only constant), not an identical raw-scan HGB fit: leaf row counts
and regularization depend on the chosen training representation. The predicted
mean appended to the scale input is float32 in BOTH fitting and preparation;
residual targets and the reported mean field retain float64 predictions.

The fixed reader preserves the upstream robust scores: EPS=.02, BETA=128,
AP-count normalization, shared scan offsets [-8,-4,0,4,8], U_DET=1/60 and
U_MISS=.5. These U constants are robust-score terms, NOT a normalized joint
missing/detected contamination distribution. log_position_mass is normalized
over candidates after tempering; it is not a claim of probability calibration.
H unique coordinates precede empirical S unique coordinates, including H/S
coordinate duplicates. An S candidate uses only that position's actual scans.
No GP, Tab, extra calibrator, learned readout or second localization branch.
"""
import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from observation_translation_tasks import ObservationTranslationContext


TABLE_REPRESENTATION = 'continuous_scan_cells_v1'
REPRESENTATION = 'continuous_observation_translation_v1'
S0 = 3.0
EPS, BETA = .02, 128.0
U_DET, U_MISS = 1.0 / 60.0, .5
OFFSETS = np.array([-8., -4., 0., 4., 8.])


class _Fit:
    """Freeze fit-only columns and normalize this fit's positive weights once."""
    def __init__(self, estimator, X, y, weight):
        self.columns = np.flatnonzero(~np.isnan(X).all(axis=0))
        weight = np.asarray(weight, dtype=np.float64)
        weight = weight * (len(weight) / weight.sum())
        self.estimator = estimator.fit(X[:, self.columns], y, sample_weight=weight)

    def predict(self, X):
        return self.estimator.predict(X[:, self.columns])

    def predict_proba(self, X):
        return self.estimator.predict_proba(X[:, self.columns])


def _parameters(seed, max_iter):
    return dict(max_iter=max_iter, learning_rate=.06, max_leaf_nodes=63,
                min_samples_leaf=40, l2_regularization=1., random_state=seed,
                early_stopping=False)


def _support_candidates(support, aps):
    """Original empirical S smoothing, with no file access or fitted model."""
    xy, rss = np.asarray(support['xy'], dtype=np.float64), np.asarray(support['rss'])[:, aps]
    positions, group = np.unique(xy, axis=0, return_inverse=True)
    group = group.ravel()
    n = np.bincount(group, minlength=len(positions)).astype(np.float64)
    detected = rss != 100
    values = np.where(detected, rss, 0.).astype(np.float64)
    nd = np.zeros((len(positions), len(aps)))
    s1, s2 = np.zeros_like(nd), np.zeros_like(nd)
    np.add.at(nd, group, detected)
    np.add.at(s1, group, values)
    np.add.at(s2, group, values**2)
    mean = s1 / np.maximum(nd, 1)
    variance = np.where(nd > 1, s2 / np.maximum(nd, 1) - mean**2, 0.)
    p = (nd + .5) / (n[:, None] + 1.)
    mean = np.where(nd > 0, mean, -95.)
    variance = np.maximum(variance, 0.) + S0**2 + 4.**2
    return positions, p, mean, variance


class ContinuousObservationTranslationModel:
    """Bernoulli detection, detected mean, and floor-crossfit log-variance."""

    def fit(self, tables, *, seed, max_iter=400):
        tables = list(tables)
        assert tables and all(t['role'] == 'train' and
            t.get('representation') == TABLE_REPRESENTATION for t in tables)
        assert all(t['X'].shape == (len(t['n']), 55) and
            all(np.asarray(t[name]).shape == t['n'].shape for name in
                ('nd', 'mean_detected', 'variance_detected', 'weight')) and
            np.isclose(np.sum(t['weight']), 1.) for t in tables)
        X = np.concatenate([t['X'] for t in tables]).astype(np.float32)
        n, nd, mean, variance, weight = (
            np.concatenate([t[name] for t in tables]).astype(np.float64)
            for name in ('n', 'nd', 'mean_detected', 'variance_detected', 'weight'))
        assert np.all(n > 0) and np.all((nd >= 0) & (nd <= n)) and np.all(weight > 0)
        floor = np.concatenate([np.full(len(t['n']), t['floor']) for t in tables])
        self.source_floors = sorted(set(floor.tolist()))
        assert len(self.source_floors) >= 2, 'Scale crossfit requires at least two source floors'
        self.crossfit_groups = [self.source_floors[::2], self.source_floors[1::2]]
        self.columns = np.flatnonzero(~np.isnan(X).all(axis=0))
        X = X[:, self.columns]
        fraction = nd / n
        detected_weight = weight * fraction
        parameters = _parameters(seed, max_iter)

        # Double labels preserve the cell Bernoulli data loss up to _Fit's scale.
        Xd = np.vstack((X, X))
        yd = np.r_[np.ones(len(X)), np.zeros(len(X))]
        wd = np.r_[detected_weight, weight * (1. - fraction)]
        keep = wd > 0
        assert np.array_equal(np.unique(yd[keep]), [0., 1.])
        self.detection = _Fit(HistGradientBoostingClassifier(loss='log_loss', **parameters),
                              Xd[keep], yd[keep], wd[keep])
        del Xd
        detected = nd > 0
        self.mean = _Fit(HistGradientBoostingRegressor(**parameters),
                         X[detected], mean[detected], detected_weight[detected])

        xs, ys, ws = [], [], []
        for i in (0, 1):
            held = np.isin(floor, self.crossfit_groups[i]) & detected
            other = np.isin(floor, self.crossfit_groups[1-i]) & detected
            assert held.any() and other.any(), 'Each floor group needs detected fit cells'
            crossfit = _Fit(HistGradientBoostingRegressor(**parameters),
                            X[other], mean[other], detected_weight[other])
            prediction = crossfit.predict(X[held])
            xs.append(np.hstack((X[held], prediction[:, None].astype(np.float32))))
            ys.append(np.log(variance[held] + (mean[held] - prediction)**2 + S0**2))
            ws.append(detected_weight[held])
        self.scale = _Fit(HistGradientBoostingRegressor(**parameters),
                          np.vstack(xs), np.concatenate(ys), np.concatenate(ws))
        return self

    def prepare(self, histories, support, target_floor):
        """Fix H/S fields and empirical S candidates before any Query is read."""
        context = ObservationTranslationContext(histories, support, target_floor)
        candidates = np.unique(np.concatenate([
            np.asarray(h['xy'], dtype=np.float64) for h in histories.values()]), axis=0)
        X = context.at(candidates).reshape(-1, 55)[:, self.columns]
        shape = (len(candidates), len(context.aps))
        p = self.detection.predict_proba(X)[:, 1].reshape(shape)
        mean = self.mean.predict(X)
        variance = np.exp(self.scale.predict(np.hstack((X, mean[:, None].astype(np.float32)))))
        mean, variance = mean.reshape(shape), variance.reshape(shape)
        sxy, sp, sm, sv = _support_candidates(support, context.aps)
        xy = np.vstack((candidates, sxy))
        # Match upstream bounds_of: outward rounding on the absolute 2m lattice.
        bounds = np.stack((np.floor(xy.min(0)/2)*2, np.ceil(xy.max(0)/2)*2))
        return PreparedContinuousObservationTranslation(
            xy, context.aps, np.vstack((p, sp)), np.vstack((mean, sm)),
            np.vstack((variance, sv)), bounds)

    def save(self, path):
        joblib.dump(dict(representation=REPRESENTATION,
                         table_representation=TABLE_REPRESENTATION,
                         model=dict(columns=self.columns, detection=self.detection,
                                    mean=self.mean, scale=self.scale,
                                    source_floors=self.source_floors,
                                    crossfit_groups=self.crossfit_groups)), path)

    @classmethod
    def load(cls, path):
        saved = joblib.load(path)
        assert saved.get('representation') == REPRESENTATION
        assert saved['table_representation'] == TABLE_REPRESENTATION
        model = cls()
        for name in ('columns', 'detection', 'mean', 'scale', 'source_floors', 'crossfit_groups'):
            setattr(model, name, saved['model'][name])
        return model


class PreparedContinuousObservationTranslation:
    """H/S-only fields with the fixed shared-offset robust readout.

    p/mean/variance are [candidate, selected AP]; mean is detected RSS dBm,
    variance is detected RSS dB**2. Empirical S candidates are already appended.
    p remains unclipped in storage so ensemble averaging precedes clipping.
    """
    def __init__(self, cand_xy, aps, p, mean, variance, bounds):
        self.cand_xy, self.aps = cand_xy.copy(), aps.copy()
        self.p, self.mean, self.variance = p.copy(), mean.copy(), variance.copy()
        self.bounds = bounds.copy()

    def predict(self, query_rss):
        query = np.asarray(query_rss)
        assert query.ndim == 2 and query.shape[1] == 520
        p = np.clip(self.p, 1e-4, 1 - 1e-4)
        lmiss = np.log((1 - EPS)*(1 - p) + EPS*U_MISS)
        logp, lnorm = np.log(p), -.5*np.log(2*np.pi*self.variance)
        scores = np.empty((len(query), len(self.cand_xy)), dtype=np.float64)
        for i, row in enumerate(query[:, self.aps].astype(np.float64)):
            detected = row != 100
            value = row[detected][None]
            per_offset = [np.logaddexp(
                np.log(1-EPS) + logp[:, detected] + lnorm[:, detected]
                - .5*(value-offset-self.mean[:, detected])**2/self.variance[:, detected],
                np.log(EPS*U_DET)).sum(axis=1) for offset in OFFSETS]
            # One shared offset per complete scan: sum over APs before mixing.
            scores[i] = (logsumexp(np.stack(per_offset), axis=0) - np.log(len(OFFSETS))
                         + lmiss[:, ~detected].sum(axis=1)) * BETA / len(self.aps)
        log_mass = scores - logsumexp(scores, axis=1, keepdims=True)
        projected = np.clip(np.exp(log_mass) @ self.cand_xy, self.bounds[0], self.bounds[1])
        return dict(projected=projected, log_position_mass=log_mass,
                    cand_xy=self.cand_xy.copy())


def prepare_ensemble(models, histories, support, target_floor):
    """Original distribution averaging; no fitting, new weights or extra S rows."""
    prepared = [m.prepare(histories, support, target_floor) for m in models]
    assert prepared
    first = prepared[0]
    ps, means, variances = (np.stack([getattr(p, name) for p in prepared])
                            for name in ('p', 'mean', 'variance'))
    return PreparedContinuousObservationTranslation(
        first.cand_xy, first.aps, ps.mean(0), means.mean(0),
        variances.mean(0) + means.var(0), first.bounds)
