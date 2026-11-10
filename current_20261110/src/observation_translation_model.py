"""Shared conditional distribution of quantized real single-scan RSS.

Class 0 is missing=100; at most 16 detected-RSS intervals are fixed using
training labels and their position/scan/AP weights alone. Interval resolution
is deliberately lost. Localization multiplies AP conditional masses, a working
independence assumption, not a recovered joint scan or GP spatial covariance.
S enters only the 55 conditional features, never a second density/expert.
There is no CSV access, automatic fitting, target selection or model download.
"""
import joblib
import numpy as np
from scipy.special import log_softmax, logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier

from location_decision import geometric_median
from observation_translation_tasks import ObservationTranslationContext


def observation_classes(rss, cuts):
    """Outer detected intervals extend to infinity; missing stays separate."""
    rss = np.asarray(rss)
    return np.where(rss == 100, 0, 1 + np.searchsorted(cuts, rss, side='right'))


class ObservationTranslationModel:
    def fit(self, tables, *, seed, max_iter=200):
        """Fit only source training tables; no random row validation split.

        Tables each have unit total weight. After concatenation, mean-one
        weights preserve equal-task weighting and the HGB regularization scale.
        Only the fitted estimator, cuts and training-defined input columns are
        retained. sklearn 1.9.1's numeric binner fails on all-NaN columns; drop
        only those source-unobserved columns, as in the original translator.
        Partially missing columns retain native HGB missing-value routing.
        """
        tables = list(tables)
        assert tables and all(t['role'] == 'train' for t in tables)
        X = np.concatenate([t['X'] for t in tables])
        y = np.concatenate([t['y'] for t in tables])
        weight = np.concatenate([t['sample_weight'] for t in tables]).astype(np.float64)
        assert X.shape == (len(y), 55) and weight.shape == y.shape
        assert np.all(np.isfinite(y)) and np.all(np.isfinite(weight)) and np.all(weight > 0)
        assert all(np.isclose(np.sum(t['sample_weight']), 1) for t in tables)
        detected = y != 100
        assert detected.any() and (~detected).any(), 'Both missing and detected training outcomes are required'
        assert np.all(y[detected] == np.floor(y[detected])), 'Expected real integer-RSS scan labels'
        values, inverse = np.unique(y[detected], return_inverse=True)
        mass = np.bincount(inverse.ravel(), weights=weight[detected])
        cumulative = np.cumsum(mass)
        indices = np.unique(np.searchsorted(cumulative, cumulative[-1] * np.arange(1, 16) / 16))
        indices = indices[indices < len(values) - 1]
        self.cuts = (values[indices].astype(np.float64) + values[indices + 1]) / 2
        labels = observation_classes(y, self.cuts)
        assert np.array_equal(np.unique(labels), np.arange(len(self.cuts) + 2))
        weight *= len(weight) / weight.sum()
        self.columns = np.flatnonzero(~np.isnan(X).all(axis=0))
        self.estimator = HistGradientBoostingClassifier(
            loss='log_loss', max_iter=max_iter, learning_rate=.06,
            max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1,
            early_stopping=False, class_weight=None, random_state=seed)
        self.estimator.fit(X[:, self.columns], labels, sample_weight=weight)
        assert np.array_equal(self.estimator.classes_, np.arange(len(self.cuts) + 2))
        return self

    def predict_log_mass(self, X):
        """Stable class log-masses from public raw decisions, without flooring.

        For two classes sklearn supplies the class-1 versus class-0 logit;
        [0, logit] is the equivalent two-class softmax representation.
        """
        logits = self.estimator.decision_function(np.asarray(X)[:, self.columns])
        if logits.ndim == 1:
            logits = np.column_stack([np.zeros_like(logits), logits])
        return log_softmax(logits, axis=1)

    def source_nll(self, table):
        """Score a source training/validation table; never update its cuts."""
        assert table['role'] in ('train', 'source_validation')
        labels = observation_classes(table['y'], self.cuts)
        mass = self.predict_log_mass(table['X'])
        return float(-np.average(mass[np.arange(len(labels)), labels],
                                 weights=table['sample_weight']))

    def prepare(self, histories, support, target_floor):
        """Prepare one H/S-only candidate likelihood table without changing weights."""
        context = ObservationTranslationContext(histories, support, target_floor)
        candidates = np.unique(np.concatenate([
            *[np.asarray(h['xy'], dtype=np.float64) for h in histories.values()],
            np.asarray(support['xy'], dtype=np.float64),
        ]), axis=0)
        features = context.at(candidates)
        mass = self.predict_log_mass(features.reshape(-1, 55)).reshape(
            len(candidates), len(context.aps), len(self.cuts) + 2)
        return PreparedObservationTranslation(candidates, context.aps, self.cuts, mass)

    def save(self, path):
        """Save classifier, cuts and column binding; no source tables/contexts."""
        joblib.dump(dict(estimator=self.estimator, cuts=self.cuts, columns=self.columns), path)

    @classmethod
    def load(cls, path):
        saved = joblib.load(path)
        model = cls()
        model.estimator, model.cuts = saved['estimator'], saved['cuts']
        model.columns = saved['columns']
        assert np.array_equal(model.estimator.classes_, np.arange(len(model.cuts) + 2))
        return model


class PreparedObservationTranslation:
    """Immutable-by-convention H/S likelihood table; no estimator needed to score.

    Candidate prior is uniform over unique known H/S coordinates. Geometric
    median minimizes model distance risk approximately with the existing gap
    certificate; this is not a promise of lower true localization error.
    """
    def __init__(self, candidates, aps, cuts, log_mass):
        self.cand_xy, self.aps = candidates.copy(), aps.copy()
        self.cuts, self.log_mass = cuts.copy(), log_mass.copy()

    def predict(self, query_rss):
        query = np.asarray(query_rss)
        assert query.ndim == 2 and query.shape[1] == 520
        labels = observation_classes(query[:, self.aps], self.cuts)
        scores = np.empty((len(query), len(self.cand_xy)), dtype=np.float64)
        ap = np.arange(len(self.aps))
        for row, categories in enumerate(labels):
            scores[row] = self.log_mass[:, ap, categories].sum(axis=1)
        log_position_mass = scores - logsumexp(scores, axis=1, keepdims=True)
        projected = np.empty((len(query), 2), dtype=np.float64)
        gaps = np.empty(len(query), dtype=np.float64)
        for row, log_weights in enumerate(log_position_mass):
            action, gaps[row] = geometric_median(self.cand_xy, np.exp(log_weights))
            projected[row] = np.clip(action, self.cand_xy.min(0), self.cand_xy.max(0))
        return dict(projected=projected, log_position_mass=log_position_mass,
                    cand_xy=self.cand_xy.copy(), decision_gap=gaps)
