"""Check only the new change-correlation path and one source IFT derivative."""
import json
from pathlib import Path

import numpy as np
import torch

from check import structural_checks
from data import draw_episode, load_period
from inference import gaussian_update, predict_from_posterior
from run import loss, make_case
from transfer import ChangeCorrelation, History, Prior, Transfer, rbf


ROOT = Path(__file__).resolve().parent


def covariance(prior, x, y):
    _, fx = prior.features(x)
    _, fy = prior.features(y)
    return prior.covariance(x, y, fx, fy)


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    settings = json.loads((ROOT / "results/source_transfer_v4_spatial_control/freeze.json").read_text())
    base, obs = settings["base"], settings["obs"]
    report = {"legacy_structural": structural_checks(base)}
    mean_model = Transfer("affine").double()
    mean_model.load_state_dict(torch.load(ROOT / "results/source_transfer_v4_spatial_control/affine.pt",
                                        weights_only=True))
    mean_model.requires_grad_(False)
    z = torch.tensor([[-2., 0.], [2., 0.]])
    hist = History(z, {"mean": torch.tensor([-55., -95.]), "cov": 0.01 * torch.eye(2)}, base)
    x = torch.tensor([[-2., 0.], [2., 0.], [0., 0.], [0.4, .3]])
    old = Prior(mean_model, hist, base)
    cases = {}
    for kind in ("product", "linear", "gram"):
        torch.manual_seed(231024)
        model = ChangeCorrelation(kind).double()
        prior = Prior(mean_model, hist, base, model)
        matrix = covariance(prior, x, x)
        torch.testing.assert_close(matrix, matrix.T, rtol=1e-12, atol=1e-10)
        assert torch.linalg.eigvalsh(matrix).min() > -1e-8
        m, f = prior.features(x)
        torch.testing.assert_close(matrix.diag(), prior.diagonal(x, f), rtol=1e-11, atol=1e-9)
        torch.testing.assert_close(matrix.diag(), covariance(old, x, x).diag(), rtol=1e-11, atol=1e-9)
        torch.testing.assert_close(covariance(prior, x[:1], x[1:]), matrix[:1, 1:], rtol=1e-11, atol=1e-9)
        perm = torch.tensor([3, 1, 0, 2])
        torch.testing.assert_close(covariance(prior, x[perm], x[perm]), matrix[perm][:, perm], rtol=1e-11, atol=1e-9)
        empty = x[:0]
        posterior = dict(mean=empty[:, 0], cov=matrix[:0, :0])
        before = predict_from_posterior(m, matrix.diag(), matrix[:, :0], m[:0], matrix[:0, :0], posterior)
        torch.testing.assert_close(before["mean"], m)
        torch.testing.assert_close(before["variance"], matrix.diag())
        # Empty cross blocks must be defined by the real feature path too.
        assert covariance(prior, empty, x).shape == (0, len(x))
        no_history = Prior(mean_model, None, base, model)
        torch.testing.assert_close(covariance(no_history, x, x), rbf(x, x, base["amplitude"], base["length"]))
        shift = torch.tensor([123., -57.])
        translated = History(z + shift, {"mean": hist.m, "cov": hist.c}, base)
        shifted = Prior(mean_model, translated, base, model)
        torch.testing.assert_close(covariance(shifted, x + shift, x + shift), matrix, rtol=1e-10, atol=1e-8)
        cases[kind] = dict(min_eigenvalue=torch.linalg.eigvalsh(matrix).min().item(),
                           marginal_variance_error=(matrix.diag() - covariance(old, x, x).diag()).abs().max().item())

    # A specified feature function, not a model fitted to make the example pass.
    signed = ChangeCorrelation("linear").double()
    with torch.no_grad():
        signed.encoder.weight.zero_()
        signed.encoder.bias.zero_()
        signed.encoder.weight[0, 0] = 3.
    known = History(z, {"mean": hist.m, "cov": 1e-10 * torch.eye(2)}, base)
    prior = Prior(mean_model, known, base, signed)
    m, f = prior.features(z)
    k = prior.covariance(z, z, f, f)
    assert k[1, 0] < 0
    y = m[:1] + 5
    stats = dict(n=torch.ones(1), k=torch.ones(1), s1=y, s2=y.square())
    posterior = gaussian_update(m[:1], k[:1, :1], stats, obs)
    predicted = predict_from_posterior(m[1:], k.diag()[1:], k[1:, :1], m[:1], k[:1, :1], posterior)
    assert predicted["mean"][0] < m[1]
    zero = dict(n=torch.ones(1), k=torch.ones(1), s1=m[:1], s2=m[:1].square())
    torch.testing.assert_close(gaussian_update(m[:1], k[:1, :1], zero, obs)["mean"], m[:1])
    report["structure"] = cases
    report["known_history_single_anchor"] = dict(change_cross_covariance=k[1, 0].item(),
                                                 query_update_db=(predicted["mean"][0] - m[1]).item())

    path = ROOT.parent / "Law-Continuous-Change-20261023/data/OfficeP2.npz"
    ep = draw_episode(load_period(path, 1), load_period(path, 2), .25, 5, 12, np.random.default_rng(231023))
    count = ep["S"]["stats"]["k"]
    j = int(np.flatnonzero(ep["historical_positive"] & (count > 0).any(0) & (count == 0).any(0))[0])
    case = make_case(ep, j, base, obs, 160)
    report["source_derivatives"] = {}
    for kind in ("product", "linear", "gram"):
        torch.manual_seed(231024)
        model = ChangeCorrelation(kind).double()
        value = loss(mean_model, case, base, obs, 160, differentiable=True, change=model)
        parameters = list(model.parameters())
        gradients = torch.autograd.grad(value, parameters)
        torch.manual_seed(241024)
        direction = [torch.randn_like(p) for p in parameters]
        norm = torch.sqrt(sum(v.square().sum() for v in direction))
        direction = [v / norm for v in direction]
        derivative = sum((g * v).sum() for g, v in zip(gradients, direction)).item()
        original = [p.detach().clone() for p in parameters]
        estimates = {}
        for eps in (1e-3, 3e-4):
            values = []
            for sign in (1, -1):
                with torch.no_grad():
                    for p, initial, v in zip(parameters, original, direction):
                        p.copy_(initial + sign * eps * v)
                values.append(loss(mean_model, case, base, obs, 160, change=model).item())
            estimates[str(eps)] = (values[0] - values[1]) / (2 * eps)
            np.testing.assert_allclose(derivative, estimates[str(eps)], rtol=3e-3, atol=5e-5)
        with torch.no_grad():
            for p, initial in zip(parameters, original):
                p.copy_(initial)
        # A diagnostic of root-dependent cubature, not an asserted exact symmetry.
        history = case["history"]
        rev = torch.arange(len(history.z) - 1, -1, -1)
        reordered = History(history.z[rev], {"mean": history.m[rev], "cov": history.c[rev][:, rev]}, base)
        a = model.kernel(case["qx"], case["sx"], history, base)
        b = model.kernel(case["qx"], case["sx"], reordered, base)
        item = dict(nll=value.item(), directional_derivative=derivative, finite_differences=estimates,
                    h_root_order_change_max_covariance=(a - b).abs().max().item(),
                    h_root_order_change_relative_to_drift_variance=(a - b).abs().max().item() / base["drift"] ** 2)
        report["source_derivatives"][kind] = item
        print(json.dumps({"kind": kind, **item}), flush=True)
    output = ROOT / "results/change_kernel_checks_20261024.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"structure": report["structure"], "single_anchor": report["known_history_single_anchor"]}))


if __name__ == "__main__":
    main()
