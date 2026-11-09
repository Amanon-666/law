"""Frozen Tab marginal distributions and a declared GP-score product rule.

The existing Frozen/EpisodicHistoryRegressor must be fitted normally, on the
unchanged raw H/S context. The exporter uses its official fitted transforms,
eight views, inference configuration and sorted 999-quantile readout. No model,
context, GP posterior or quantile is fitted or altered here. Only query RSS is
needed; query coordinates never enter these interfaces.

Tab's x/y marginals are equal mixtures over all eight views and all quantiles.
A Gaussian KDE with fixed 2 m bandwidth is evaluated on the same absolute 2 m
GP grid. Multiplying the two axis marginals is an explicit independence
approximation, NOT pairing equal-index quantiles into joint samples. The grid
mass is normalized and mixed with 0.001 uniform mass before taking its log.

score_lambda = original_GP_score + lambda * log_Tab_grid_mass is a predictive
fusion rule, not independent evidence or a true Bayesian posterior: both
models consume overlapping H/S information. The only selection is one global
lambda from LAMBDAS on source tasks, performed by the caller; target tasks do
not select it. No bandwidth, floor, seed or GP parameter is selected here.
Zero lambda directly returns the unchanged GP softmax mean. Bitwise agreement
between separately implemented GPU/CPU reductions is not assumed.
"""
import numpy as np
from scipy.special import logsumexp

LAMBDAS = (0., .25, .5, 1., 2.)
BANDWIDTH_METRES = 2.
UNIFORM_MASS = .001


def export_tab_quantiles(model, query_rss):
    """Export absolute-coordinate quantiles [axis, view, query, quantile].

    `model` is an already-fitted original FrozenHistoryRegressor or its
    EpisodicHistoryRegressor subclass. Official raw_quantiles are SORTED by
    quantile_dist; their levels are .001,...,.999. The similarly named
    `quantiles` API instead interpolates at nine default levels and is not used.

    Return mean/projected [Nq,2] and view_mean [2,8,Nq] as well. Native mean is
    computed from the very same sorted quantiles in the same forward pass.
    Its original reduction/float32/inverse-scaling order is retained, rather
    than promising bitwise equality after averaging exported absolute arrays
    in a different precision or order.
    """
    from tabicl_history import features

    assert not model.loaded.model_.training
    query = features(query_rss, model.floor, model.aps)
    quantile_axes, mean_axes, view_axes = [], [], []
    for axis, estimator in enumerate(model.estimators):
        transformed = estimator.X_encoder_.transform(query)
        views = estimator.ensemble_generator_.transform(transformed, mode='both')
        quantiles, means = [], []
        for x, y in views.values():
            # Official regressor._batch_forward calls predict_stats once for
            # both outputs, using its fitted inference_config and batch_size.
            out = estimator._batch_forward(x, y, output_type=['raw_quantiles', 'mean'])
            quantiles.append(out['raw_quantiles'])
            means.append(out['mean'])
        quantiles = np.concatenate(quantiles, axis=0)
        means = np.concatenate(means, axis=0)
        assert quantiles.shape == (8, len(query_rss), 999)
        assert means.shape == quantiles.shape[:2]
        quantiles = estimator.y_scaler_.inverse_transform(
            quantiles.reshape(-1, 1)).reshape(quantiles.shape)
        means = estimator.y_scaler_.inverse_transform(means.reshape(-1, 1)).reshape(means.shape)
        quantile_axes.append(quantiles.astype(np.float64)+model.origin[axis])
        view_axes.append(means.astype(np.float64)+model.origin[axis])
        mean_axes.append(means.mean(axis=0))
    mean = np.column_stack(mean_axes)+model.origin
    return dict(quantiles=np.stack(quantile_axes), view_mean=np.stack(view_axes),
                mean=mean, projected=np.clip(mean, model.bounds[0], model.bounds[1]))


def tab_grid_logprob(quantiles, absolute_grid):
    """Return normalized, uniformly floored Tab log masses [Nq,G].

    Work per query and unique axis coordinates, never materialize
    [Nq,G,8,999]. Every view and quantile has equal mass; axis pairing is absent.
    Grid coordinates and the exported quantiles must use the same metre frame.
    """
    quantiles = np.asarray(quantiles, dtype=np.float64)
    grid = np.asarray(absolute_grid, dtype=np.float64)
    assert quantiles.ndim == 4 and quantiles.shape[0:2] == (2, 8) and quantiles.shape[-1] == 999
    assert grid.ndim == 2 and grid.shape[1] == 2 and len(grid) > 0
    coordinates = [np.unique(grid[:, axis], return_inverse=True) for axis in range(2)]
    n_query = quantiles.shape[2]
    result = np.empty((n_query, len(grid)), dtype=np.float64)
    log_constant = np.log(8*999)+np.log(BANDWIDTH_METRES)+.5*np.log(2*np.pi)
    for query in range(n_query):
        joint = np.zeros(len(grid), dtype=np.float64)
        for axis, (values, inverse) in enumerate(coordinates):
            points = quantiles[axis, :, query, :].reshape(-1)
            difference = (values[:, None]-points[None, :])/BANDWIDTH_METRES
            log_density = logsumexp(-.5*difference*difference, axis=1)-log_constant
            joint += log_density[inverse]
        joint -= logsumexp(joint)
        result[query] = np.logaddexp(np.log1p(-UNIFORM_MASS)+joint,
                                    np.log(UNIFORM_MASS)-np.log(len(grid)))
    return result


def fuse_position_scores(gp_scores, log_tab, absolute_grid):
    """Evaluate all fixed lambdas without selecting one.

    Inputs: unmodified GP full-scan scores [Nq,G], normalized Tab log masses
    [Nq,G], and absolute coordinates [G,2]. Outputs: lambdas [5], predictions
    [5,Nq,2], and gp_mean [Nq,2]. All means use the complete supplied grid.
    """
    scores = np.asarray(gp_scores, dtype=np.float64)
    log_tab = np.asarray(log_tab, dtype=np.float64)
    grid = np.asarray(absolute_grid, dtype=np.float64)
    assert scores.ndim == 2 and scores.shape == log_tab.shape
    assert grid.shape == (scores.shape[1], 2)
    gp_probability = np.exp(scores-logsumexp(scores, axis=1, keepdims=True))
    gp_mean = gp_probability@grid
    predictions = [gp_mean]
    for weight in LAMBDAS[1:]:
        logits = scores+weight*log_tab
        probability = np.exp(logits-logsumexp(logits, axis=1, keepdims=True))
        predictions.append(probability@grid)
    return dict(lambdas=np.asarray(LAMBDAS), predictions=np.stack(predictions), gp_mean=gp_mean)
