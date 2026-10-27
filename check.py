"""Bounded structural and source-only adaptation checks; no network training.

Only Office training periods 1 and 2 are opened.  The numerical comparison
holds one H=.25/K=5 episode and one H/S-selected AP fixed across inner budgets.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import draw_episode, load_period
from inference import infer
from run import loss, make_case, predict
from transfer import History, Prior, Transfer, rbf


def structural_checks(base):
    radius = base["length"] / 2
    z = torch.tensor([[0., 0.], [radius, 0.], [-radius, 0.],
                      [0., radius], [0., -radius]])
    covariance = rbf(z, z, 1.2, radius)
    history = History(z, dict(mean=-75 + 3 * z[:, 0], cov=covariance), base)
    x = torch.tensor([[0., 0.], [0.4, 0.8], [-0.7, 0.3], [2.1, -0.5]])
    projection = history.projection(x)
    historical_mean = base["mean"] + projection @ (history.m - base["mean"])
    historical_cov = (history.kernel(x, x) - projection @ history.kernel(z, x)
                      + projection @ covariance @ projection.T)
    drift = rbf(x, x, base["drift"], base["length"])
    errors = {}
    for kind in ("identity", "affine"):
        model = Transfer(kind).double()
        if kind == "affine":
            with torch.no_grad():
                model.a.fill_(1.3)
                model.b.fill_(0.2)
        prior = Prior(model, history, base)
        mean, features = prior.features(x)
        cov = prior.covariance(x, x, features, features)
        expected_mean = (historical_mean if kind == "identity" else
                         -75 + 1.3 * (historical_mean + 75) + 4)
        expected_cov = drift + model.scale() ** 2 * historical_cov
        # History uses 1e-6 Cholesky jitter: its cubature covariance is C+1e-6 I.
        jitter_effect = model.scale() ** 2 * 1e-6 * projection @ projection.T
        torch.testing.assert_close(mean, expected_mean, rtol=1e-12, atol=1e-11)
        torch.testing.assert_close(cov, expected_cov + jitter_effect,
                                   rtol=1e-11, atol=1e-10)
        torch.testing.assert_close(prior.diagonal(x, features), cov.diag(),
                                   rtol=1e-11, atol=1e-10)
        errors[kind] = dict(mean_max_abs=float((mean - expected_mean).abs().max().detach()),
                           covariance_max_abs_before_jitter_accounting=
                           float((cov - expected_cov).abs().max().detach()))

    # A hand-set shape function, not fitted data: read the +x local difference.
    shape = Transfer("shape").double()
    with torch.no_grad():
        shape.network[0].weight.zero_()
        shape.network[0].bias.zero_()
        shape.network[0].weight[0, 1] = 1
        shape.network[-1].weight[0, 0] = 0.3
    prior = Prior(shape, history, base)
    sm, sf = prior.features(x[:2])
    full_mean, full_features = prior.features(x)
    combined_cov = prior.covariance(x, x, full_features, full_features)
    torch.testing.assert_close(sm, full_mean[:2], rtol=0, atol=1e-12)
    support_cov = prior.covariance(x[:2], x[:2], sf, sf)
    torch.testing.assert_close(support_cov, combined_cov[:2, :2], rtol=0, atol=1e-10)
    chunks = [prior.features(row[None]) for row in x]
    chunk_mean = torch.cat([item[0] for item in chunks])
    chunk_features = torch.cat([item[1] for item in chunks])
    torch.testing.assert_close(chunk_mean, full_mean, rtol=0, atol=1e-11)
    torch.testing.assert_close(chunk_features, full_features, rtol=0, atol=1e-11)

    opposite = History(z, dict(mean=-75 - 3 * z[:, 0], cov=covariance), base)
    center = z[:1]
    centers = [h.sample_mean(center).mean() for h in (history, opposite)]
    torch.testing.assert_close(centers[0], centers[1], rtol=0, atol=1e-11)
    outputs = [Prior(shape, h, base).features(center)[0][0] for h in (history, opposite)]
    shape_gap = float((outputs[0] - outputs[1]).detach())
    assert abs(shape_gap) > 0.1, "local shape was lost despite opposite slopes"
    pointwise = Transfer("pointwise").double()
    with torch.no_grad():
        pointwise.network[-1].weight.fill_(0.02)
    point_outputs = [Prior(pointwise, h, base).features(center)[0][0]
                     for h in (history, opposite)]
    torch.testing.assert_close(point_outputs[0], point_outputs[1], rtol=0, atol=1e-10)
    return dict(affine_limits=errors, support_prior_batch_max_abs=
                float((support_cov - combined_cov[:2, :2]).abs().max().detach()),
                same_center_opposite_shape_output_gap_db=shape_gap,
                interpretation="Representational/numerical checks, not learned benefit.")


def source_check(path, base, obs):
    previous, current = (load_period(path, period) for period in (1, 2))
    episode = draw_episode(previous, current, .25, 5, 12, np.random.default_rng(231023))
    counts = episode["S"]["stats"]["k"]
    eligible = np.flatnonzero(episode["historical_positive"] &
                              (counts > 0).any(axis=0) & (counts == 0).any(axis=0))
    j = int(eligible[0])  # Input-only, deterministic selection; no Q-dependent search.
    case = make_case(episode, j, base, obs, h_steps=160)
    torch.manual_seed(231023)
    model = Transfer("shape").double()
    parameter = model.network[-1].weight
    index = (0, 0)  # Fixed before inspecting any gradient or Q outcome.
    records, predictions = [], []
    for steps in (20, 60, 160):
        start = time.perf_counter()
        value = loss(model, case, base, obs, steps, differentiable=True)
        gradient = torch.autograd.grad(value, parameter)[0][index].item()
        differences = {}
        original = parameter[index].item()
        for epsilon in (1e-4, 3e-5):
            values = []
            for sign in (1, -1):
                with torch.no_grad():
                    parameter[index] = original + sign * epsilon
                    values.append(loss(model, case, base, obs, steps).item())
            differences[str(epsilon)] = (values[0] - values[1]) / (2 * epsilon)
        with torch.no_grad():
            parameter[index] = original
            mu, var = predict(model, case, case["qx"], base, obs, steps)
            prior = Prior(model, case["history"], base)
            sm, sf = prior.features(case["sx"])
            sk = prior.covariance(case["sx"], case["sx"], sf, sf)
        posterior = infer(sm, sk, case["stats"], obs, steps)
        for fd in differences.values():
            np.testing.assert_allclose(gradient, fd, rtol=2e-3, atol=2e-5)
        records.append(dict(steps=steps, query_nll=value.item(),
                            negative_support_elbo=posterior["loss"].item(),
                            shape_output_weight_gradient=gradient,
                            finite_difference_gradients=differences,
                            elapsed_seconds=time.perf_counter() - start))
        predictions.append((mu.detach(), var.detach()))
        print(json.dumps(dict(stage="inner_check", **records[-1])), flush=True)
    final_mean, final_var = predictions[-1]
    for record, (mu, var) in zip(records, predictions):
        record["query_mean_max_abs_difference_from_160"] = (mu - final_mean).abs().max().item()
        record["query_variance_max_abs_difference_from_160"] = (var - final_var).abs().max().item()
    return dict(periods=[1, 2], split="training", history_fraction=.25,
                history_positions=len(episode["H"]["positions"]), support_positions=5,
                query_positions=12, seed=231023, ap_column=int(episode["ap_indices"][j]),
                support_detections=counts[:, j].tolist(),
                support_scans=episode["S"]["stats"]["n"].tolist(),
                history_steps=160, shape_weight_index=list(index),
                network_initialization="zero residual output; no parameter optimization",
                records=records,
                scope="Single source case; 160 steps is a comparison, not proven convergence.")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(root.parent /
                        "Law-Continuous-Change-20261023/data/OfficeP2.npz"))
    parser.add_argument("--initialization", default=str(root / "source_only_initialization.json"))
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    init = json.loads(Path(args.initialization).read_text())
    if "base" in init and "obs" in init:
        base, obs, origin = init["base"], init["obs"], "source_only_initialization base/obs"
    else:
        base = dict(mean=-85., amplitude=20., length=3.825, drift=3.84)
        obs = dict(alpha=.98, beta=-95., detect_scale=10., noise_std=2.69)
        origin = "explicit provisional numerical-sensitivity constants; not fitted initialization"
    report = dict(scope="No network training, localization, period 12 or Mall read.",
                  torch_version=torch.__version__, threads=torch.get_num_threads(),
                  initialization_origin=origin, base=base, obs=obs,
                  code_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ("transfer.py", "inference.py", "data.py", "run.py", "check.py")})
    print(json.dumps(dict(stage="initialization", origin=origin, base=base, obs=obs)), flush=True)
    report["structural"] = structural_checks(base)
    print(json.dumps(dict(stage="structural", **report["structural"])), flush=True)
    report["source_adaptation"] = source_check(args.data, base, obs)
    print(json.dumps(dict(stage="complete", **report), indent=2), flush=True)


if __name__ == "__main__":
    main()
