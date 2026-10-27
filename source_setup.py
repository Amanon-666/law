"""Source-only initialization and observation-model applicability screen.

Only OfficeP2 training periods 1..11 are read. This is neither localization
evaluation nor a joint-latent observation-model maximum-likelihood fit.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import ndtr

from data import NO_DETECT, SOURCE_TRAIN_PERIODS, load_period


def quantiles(values):
    values = np.asarray(values)
    return {str(p): float(np.percentile(values, p)) for p in (0, 10, 25, 50, 75, 90, 100)}


def cell_values(floor):
    stats = floor["stats"]
    k, n = stats["k"], np.broadcast_to(stats["n"][:, None], stats["k"].shape)
    mean = np.divide(stats["s1"], k, out=np.full_like(k, np.nan), where=k > 0)
    sse = np.maximum(stats["s2"] - np.divide(stats["s1"] ** 2, k,
                                            out=np.zeros_like(k), where=k > 0), 0)
    variance = np.divide(sse, k - 1, out=np.full_like(k, np.nan), where=k >= 2)
    return mean, variance, n, k, sse


def fit_detection(mean, n, k):
    """Binomial plug-in fit conditional on k>=2; initialization only."""
    def objective(parameters):
        alpha, beta, scale = parameters
        p = np.clip(alpha * ndtr((mean - beta) / scale), 1e-12, 1 - 1e-12)
        return float(-np.sum(k * np.log(p) + (n - k) * np.log1p(-p)) / n.sum())

    initial = [0.95, -90., 8.]
    bounds = [(0.05, 0.9999), (-120., -40.), (0.5, 40.)]
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    alpha, beta, scale = result.x
    predicted = alpha * ndtr((mean - beta) / scale)
    return {
        "alpha": float(alpha), "beta_dbm": float(beta), "detect_scale_db": float(scale),
        "success": bool(result.success), "message": str(result.message),
        "iterations": int(result.nit), "binomial_loss_per_trial": float(result.fun),
        "initial_parameters": initial, "bounds": bounds,
        "weighted_absolute_rate_error": float(np.average(np.abs(k / n - predicted), weights=n)),
        "cell_count": len(mean),
        "interpretation": "Plug-in initialization: k>=2 cells only; k=0 cells excluded; detected cell means are noisy and selection-biased proxies. Not a joint latent likelihood fit or evidence of model adequacy.",
    }


def add_candidate_interface(result):
    """Derive the runner interface from saved aggregates, without rereading RSS.

    The pooled visible-cell distribution includes AP-to-AP level variation;
    within-AP spatial SD alone cannot initialize this shared pure-RBF prior.
    """
    q = result["observations"]["detected_cell_mean_quantiles_k_ge_2"]
    center = q["50"]
    global_iqr_scale = (q["75"] - q["25"]) / 1.3489795003921634
    amplitude = max(global_iqr_scale, (center - q["0"]) / 3, (q["100"] - center) / 3)
    spacing = result["coordinate_system"]["nearest_position_distance"]["50"]
    delta2 = result["nearest_position_changes"]["mean_squared_difference_minus_mean_estimation_noise_db2"]
    ratio = delta2 / (2 * amplitude ** 2)
    if not 0 < ratio < 1:
        raise ValueError("saved spatial moments cannot initialize a positive RBF length")
    length = spacing / np.sqrt(-2 * np.log1p(-ratio))
    fit = result["detection_initialization"]
    result["base"] = dict(mean=center, amplitude=amplitude, length=float(length),
                          drift=result["suggested_fixed_initialization"]["drift_amplitude_db"])
    result["obs"] = dict(alpha=0.99, beta=fit["beta_dbm"],
                         detect_scale=fit["detect_scale_db"],
                         noise_std=result["observations"]["conditional_within_cell_sigma_pooled_db"])
    result["candidate_interface"] = {
        "status": "candidate initialization only; not selected, not source-joint-calibrated, not performance-validated",
        "derived_from_existing_source_summary_only": True,
        "raw_data_reread_for_interface": False,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "interface_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mean": "Global visible-cell median, not arithmetic mean: the saved summary contains quantiles, not sufficient moments to reconstruct the exact mean. No AP identity or deployment H is used.",
        "amplitude": "max(global visible-cell IQR/1.34898, maximum absolute departure from global median/3); includes between-AP and within-AP spread and places all source visible-cell means within three prior SDs. This is deliberately broad initialization, not estimated latent variance.",
        "length": "RBF second-increment matching at median nearest-point spacing: ell=d/sqrt(-2*log(1-corrected_increment_MSE/(2*amplitude**2))). Uses a descriptive median-distance/pooled-increment approximation, not fitted covariance or localization search.",
        "alpha": "0.99 starts away from the plugin fit's alpha upper boundary; beta and detect_scale remain biased plugin starting values awaiting joint source ELBO fitting.",
        "previous_suggested_fixed_initialization": "Retained as the original diagnostic output; it misses between-AP variance and is not the recommended runner input.",
        "limits": "Visible cells omit all-nondetection locations. Broadening a pure RBF does not explicitly distinguish a global per-AP level from local spatial variation; joint source calibration is required before claiming this is a calibrated prior.",
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent /
                        "Law-Continuous-Change-20261023/data/OfficeP2.npz")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("source_only_initialization.json"))
    parser.add_argument("--refresh-interface", action="store_true",
                        help="add candidate base/obs using saved output aggregates; no raw data read")
    args = parser.parse_args()
    if args.refresh_interface:
        result = add_candidate_interface(json.loads(args.output.read_text()))
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        print(json.dumps({key: result[key] for key in ("base", "obs", "candidate_interface")},
                         ensure_ascii=False, allow_nan=False))
        return
    if args.data.name != "OfficeP2.npz":
        raise ValueError("this source setup is scoped to OfficeP2.npz")

    floors = [load_period(args.data, period, "training") for period in SOURCE_TRAIN_PERIODS]
    summaries, means, variances, counts, detections, noises, dofs, raw_rss = [], [], [], [], [], [], [], []
    nearest_distances, spatial_differences, spatial_noise, detection_changes = [], [], [], []
    temporal_differences, temporal_noise = [], []
    for period, floor in zip(SOURCE_TRAIN_PERIODS, floors):
        mean, variance, n, k, sse = cell_values(floor)
        eligible = k >= 2
        means.append(mean[eligible]); variances.append(variance[eligible])
        counts.append(n[eligible]); detections.append(k[eligible])
        noises.append(sse[eligible].sum()); dofs.append((k[eligible] - 1).sum())
        raw_rss.append(floor["rss"][floor["rss"] != NO_DETECT])
        active = (k > 0).any(axis=0)
        summaries.append(dict(period=period, scans=len(floor["rss"]),
                              positions=len(floor["positions"]), active_aps=int(active.sum()),
                              repeated_scans=quantiles(floor["stats"]["n"]),
                              zero_detection_cells_on_active_aps=int((k[:, active] == 0).sum()),
                              active_ap_cells=int(k[:, active].size)))
        distances = cdist(floor["positions"], floor["positions"])
        np.fill_diagonal(distances, np.inf)
        neighbors = distances.argmin(axis=1)
        pairs = np.unique(np.sort(np.column_stack((np.arange(len(neighbors)), neighbors)), axis=1), axis=0)
        left, right = pairs.T
        nearest_distances.append(distances[left, right])
        both = eligible[left] & eligible[right]
        spatial_differences.append((mean[right] - mean[left])[both])
        spatial_noise.append((variance[left] / np.maximum(k[left], 1) +
                              variance[right] / np.maximum(k[right], 1))[both])
        detection_changes.append(np.abs(k[left][:, active] / n[left][:, active] -
                                        k[right][:, active] / n[right][:, active]).ravel())

    for previous, current in zip(floors[:-1], floors[1:]):
        # This setup uses the verified common Office training grid; do not
        # silently align different grids by row number in another dataset.
        if not np.array_equal(previous["positions"], current["positions"]):
            raise ValueError("Office source periods no longer share the expected coordinates")
        pm, pv, _, pk, _ = cell_values(previous)
        cm, cv, _, ck, _ = cell_values(current)
        valid = (pk >= 2) & (ck >= 2)
        temporal_differences.append((cm - pm)[valid])
        temporal_noise.append((pv / np.maximum(pk, 1) + cv / np.maximum(ck, 1))[valid])

    mean, variance, n, k = map(np.concatenate, (means, variances, counts, detections))
    rss = np.concatenate(raw_rss)
    fit = fit_detection(mean, n, k)
    bins = []
    for low, high in zip((-np.inf, -95, -90, -85, -80, -75, -70, -65, -60, -50),
                         (-95, -90, -85, -80, -75, -70, -65, -60, -50, np.inf)):
        mask = (mean >= low) & (mean < high)
        if not mask.any():
            continue
        p = fit["alpha"] * ndtr((mean[mask] - fit["beta_dbm"]) / fit["detect_scale_db"])
        residual = k[mask] / n[mask] - p
        observed_mse = np.average(residual ** 2, weights=n[mask])
        binomial_reference = np.average(p * (1 - p) / n[mask], weights=n[mask])
        bins.append(dict(lower_dbm=None if not np.isfinite(low) else low,
                         upper_dbm=None if not np.isfinite(high) else high,
                         cells=int(mask.sum()), pooled_rate=float(k[mask].sum() / n[mask].sum()),
                         cell_rate_quantiles=quantiles(k[mask] / n[mask]),
                         fitted_mean_rate=float(np.average(p, weights=n[mask])),
                         residual_mse=float(observed_mse),
                         binomial_variance_reference=float(binomial_reference),
                         residual_to_binomial_ratio=float(observed_mse / binomial_reference)))

    sigma = float(np.sqrt(sum(noises) / sum(dofs)))
    spatial_delta, temporal_delta = map(np.concatenate, (spatial_differences, temporal_differences))
    spatial_noise, temporal_noise = map(np.concatenate, (spatial_noise, temporal_noise))
    nearest = np.concatenate(nearest_distances)
    spatial_cell_variances = []
    for floor in floors:
        cm, _, _, ck, _ = cell_values(floor)
        for ap in np.flatnonzero((ck >= 2).sum(axis=0) >= 2):
            spatial_cell_variances.append(float(np.var(cm[ck[:, ap] >= 2, ap], ddof=1)))
    history_amplitude = float(np.sqrt(np.median(spatial_cell_variances)))
    drift_amplitude = float(np.sqrt(max(float(np.mean(temporal_delta ** 2 - temporal_noise)), 0)))
    # A few adjacent survey spacings is a transparent smoothness starting point,
    # not an estimated covariance length or an optimized performance parameter.
    length = float(3 * np.median(nearest))
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"dataset": "OfficeP2", "periods": list(SOURCE_TRAIN_PERIODS),
                  "split": "training", "testing_read": False, "validation_period_read": False,
                  "mall_read": False, "raw_data_modified": False,
                  "scan_deduplication": False, "localization_or_neural_training": False},
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "coordinate_system": {
            "minimum": np.min(np.concatenate([f["positions"] for f in floors]), axis=0).tolist(),
            "maximum": np.max(np.concatenate([f["positions"] for f in floors]), axis=0).tolist(),
            "units": "metres in the existing MTLoc experiment convention; no coordinate rescaling; author README itself names coordinates without explicitly stating units",
            "nearest_position_distance": quantiles(nearest)},
        "period_summaries": summaries,
        "observations": {"non_detection_code": NO_DETECT, "rss_units": "dBm",
                         "detected_rss_quantiles": quantiles(rss),
                         "detected_cell_mean_quantiles_k_ge_2": quantiles(mean),
                         "conditional_within_cell_sigma_pooled_db": sigma,
                         "within_cell_sd_quantiles_k_ge_2": quantiles(np.sqrt(variance)),
                         "within_cell_variance_degrees_of_freedom": float(sum(dofs)),
                         "zero_variance_cell_fraction_k_ge_2": float(np.mean(variance == 0)),
                         "source_periods_are_not_independent_buildings": True},
        "detection_initialization": fit,
        "mean_detection_rate_bins_k_ge_2": bins,
        "nearest_position_changes": {
            "paired_ap_cell_count": len(spatial_delta),
            "both_cells_require_at_least_two_detections": True,
            "absolute_detected_mean_difference_db": quantiles(np.abs(spatial_delta)),
            "rms_detected_mean_difference_db": float(np.sqrt(np.mean(spatial_delta ** 2))),
            "mean_squared_difference_minus_mean_estimation_noise_db2": float(np.mean(spatial_delta ** 2 - spatial_noise)),
            "absolute_detection_rate_difference_active_aps": quantiles(np.concatenate(detection_changes))},
        "consecutive_period_changes": {
            "paired_ap_cell_count": len(temporal_delta),
            "absolute_detected_mean_difference_db": quantiles(np.abs(temporal_delta)),
            "rms_detected_mean_difference_db": float(np.sqrt(np.mean(temporal_delta ** 2))),
            "mean_squared_difference_minus_mean_estimation_noise_db2": float(np.mean(temporal_delta ** 2 - temporal_noise)),
            "note": "Conditional on both cells having >=2 detections. Noise subtraction assumes unbiased conditionally independent cell means; drift estimate is an initialization proxy, not an unbiased environmental variance estimate."},
        "suggested_fixed_initialization": {
            "observation_sigma_db": sigma,
            "alpha": fit["alpha"], "beta_dbm": fit["beta_dbm"],
            "detect_scale_db": fit["detect_scale_db"],
            "history_length_coordinate_units": length,
            "drift_length_coordinate_units": length,
            "history_amplitude_db": history_amplitude,
            "drift_amplitude_db": drift_amplitude,
            "history_mean_dbm": fit["beta_dbm"],
            "rationale": "Length=3*median nearest survey spacing, heuristic not kernel fit. History amplitude=sqrt(median within-AP spatial sample variance over source periods). Drift amplitude=sqrt(max(0, consecutive-period mean-square difference minus mean-estimation noise)). History mean=probit threshold as a weak prior convention, not an estimate of never-observed RSS. Shared constants are frozen before source neural development and identical across comparison arms."},
        "limits": ["All diagnostics condition on detected means when intensity is required; k=0 cells lack intensity and cannot validate the detection link.",
                   "Rate-bin overdispersion is a descriptive screen, not a goodness-of-fit p-value; mean estimation, truncation and correlated repeated scans affect the binomial reference.",
                   "A fitted monotone curve does not prove a single latent RSS explains every AP/state. Selection-biased cell means are only initialization proxies.",
                   "Sparse historical acquisition must be masked as no observation, not encoded with the non-detection value."],
    }
    add_candidate_interface(result)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"source_periods": result["scope"]["periods"],
                      "source_scans": sum(p["scans"] for p in summaries),
                      "noise_sigma_db": sigma, "detection_fit": fit,
                      "fixed_initialization": result["suggested_fixed_initialization"],
                      "detection_bins": bins}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
