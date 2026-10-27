"""Numerical scope audit, not training or a localization performance run."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import quad_vec
from scipy.special import log_ndtr

from data import load_period
from inference import JITTER, gaussian_update, observation_log_prob
from run_change import prepare_validation
from run_change_control import CommonChange
from support_integral import SupportIntegral, conditional_loss
from transfer import ChangeCorrelation, Prior, Transfer


ROOT = Path(__file__).resolve().parent
NODES = [(11, 231024), (13, 231024), (13, 241024)]


def stats(n, k, r=0.):
    return {key: torch.tensor(value) for key, value in
            dict(n=[float(n)], k=[float(k)], s1=[k * r], s2=[k * r * r]).items()}


def reference_1d(mean, covariance, counts, obs):
    """Adaptive integration on standardized [-12,12], split into unit cells.

    Computes the same Gaussian-RSS proposal and original prior conditional as
    the candidate. Unit-cell splits prevent adaptive quadrature from skipping
    a narrow, displaced posterior mode. This is an independent SciPy readout.
    """
    proposal = gaussian_update(mean, covariance, counts, obs)
    center = proposal["mean"].item()
    sd = math.sqrt(proposal["cov"].item() + JITTER)
    n, k = counts["n"].item(), counts["k"].item()
    a, beta, ds, noise = (obs[key] for key in ("alpha", "beta", "detect_scale", "noise_std"))
    A = covariance.item() / (covariance.item() + JITTER)
    query_var = covariance.item() * (1 - A)

    def log_density(z):
        f = center + sd * z
        z_d = (f - beta) / ds
        lp = math.log(a) + log_ndtr(z_d)
        lm = np.logaddexp(math.log1p(-a), math.log(a) + log_ndtr(-z_d))
        return -.5 * (math.log(2 * math.pi) + z * z) + k * lp + (n - k) * lm

    shift = max(log_density(z) for z in np.linspace(-12, 12, 1201))

    def integrand(z):
        f = center + sd * z
        qm = mean.item() + A * (f - mean.item())
        total = query_var + noise ** 2
        observed = -90.
        tilted = qm + query_var / total * (observed - qm)
        tilted_var = query_var * noise ** 2 / total
        lp = (math.log(a) - .5 * (math.log(2 * math.pi * total) + (observed - qm) ** 2 / total)
              + log_ndtr((tilted - beta) / math.sqrt(ds ** 2 + tilted_var)))
        lm = np.logaddexp(math.log1p(-a), math.log(a) + log_ndtr((beta - qm) / math.sqrt(ds ** 2 + query_var)))
        return math.exp(log_density(z) - shift) * np.array([1., f, f * f, math.exp(lp), math.exp(lm)])

    integrals = [quad_vec(integrand, lo, lo + 1, epsabs=1e-12, epsrel=1e-11) for lo in range(-12, 12)]
    total = sum(value for value, _ in integrals)
    normalized = total / total[0]
    return dict(support_mean=normalized[1], support_variance=normalized[2] - normalized[1] ** 2,
                nll=(-np.log(normalized[3:])).tolist(), log_detection_evidence=math.log(total[0]) + shift,
                integration_relative_error=sum(error for _, error in integrals) / total[0])


def synthetic(obs, proposal="rss_gaussian"):
    rows = []
    specifications = [
        ("detected", -85., 8., 7, 6, -78.),
        ("all_missing", -92., 8., 7, 0, 0.),
        ("conflicting_missing", -65., 6., 7, 0, 0.),
        ("rare_conflicting_missing", -60., 5., 10, 0, 0.),
        ("single_detection", -65., 6., 1, 1, -90.),
    ]
    for name, location, sd, n, k, r in specifications:
        mean, covariance = torch.tensor([location]), torch.tensor([[sd ** 2]])
        counts = stats(n, k, r)
        exact = reference_1d(mean, covariance, counts, obs)
        for power, seed in NODES:
            posterior = SupportIntegral(mean, covariance, counts, obs, power, seed, proposal)
            nll = -posterior.log_prob(torch.tensor([-90., 100.]), mean.repeat(2), covariance.diag().repeat(2),
                                     covariance.repeat(2, 1))
            row = dict(case=name, power=power, seed=seed, reference=exact, nll=nll.tolist(), **posterior.diagnostics())
            rows.append(row)
        print(json.dumps(dict(stage="synthetic", case=name, reference=exact,
                              estimates=[dict(nll=r["nll"], mean=r["support_mean"], ess=r["effective_samples"])
                                         for r in rows[-3:]])), flush=True)

    qm, qv, rss = torch.tensor([-90., -85., -75.]), torch.tensor([16., 9., 25.]), torch.tensor([100., -82., -76.])
    empty = SupportIntegral(torch.empty(0), torch.empty(0, 0),
                            {key: torch.empty(0) for key in ("n", "k", "s1", "s2")}, obs, proposal=proposal)
    empty_value = empty.log_prob(rss, qm, qv, torch.empty(3, 0))
    torch.testing.assert_close(empty_value, observation_log_prob(rss, qm, qv, obs), rtol=0, atol=0)
    unobserved = SupportIntegral(torch.tensor([-90.]), torch.tensor([[25.]]), stats(0, 0), obs, proposal=proposal)
    cross = torch.tensor([[2.], [3.], [-1.]])
    torch.testing.assert_close(unobserved.log_prob(rss, qm, qv, cross), empty_value, rtol=0, atol=0)
    posterior = SupportIntegral(torch.tensor([-90.]), torch.tensor([[25.]]), stats(7, 3, -87.), obs, proposal=proposal)
    first = posterior.log_prob(rss, qm, qv, cross, chunk_size=3)
    chunked = posterior.log_prob(rss, qm, qv, cross, chunk_size=1)
    permutation = torch.tensor([2, 0, 1])
    reordered = posterior.log_prob(rss[permutation], qm[permutation], qv[permutation], cross[permutation])
    torch.testing.assert_close(first, chunked, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(first[permutation], reordered, rtol=1e-13, atol=1e-13)
    return dict(cases=rows, empty_support_exact=True, no_factors_exact=True, query_chunks_and_order=True)


def direction_check(mean, change, case, base, obs, which, proposal="rss_gaussian"):
    parameters = list(change.parameters())
    torch.manual_seed(721024)
    direction = [torch.zeros_like(p) if (which == "eta") != (i == 0) else torch.randn_like(p)
                 for i, p in enumerate(parameters)]
    norm = torch.sqrt(sum(d.square().sum() for d in direction))
    direction = [d / norm for d in direction]
    original = [p.detach().clone() for p in parameters]
    rows = []
    for power, seed in NODES:
        objective, _ = conditional_loss(Prior(mean, case["history"], base, change), case, obs, power, seed, proposal=proposal)
        gradients = torch.autograd.grad(objective, parameters)
        auto = sum((g * d).sum() for g, d in zip(gradients, direction)).item()
        offsets = []
        epsilon = 1e-4
        for sign in (-1, 1):
            with torch.no_grad():
                for p, value, d in zip(parameters, original, direction):
                    p.copy_(value + sign * epsilon * d)
                value, _ = conditional_loss(Prior(mean, case["history"], base, change), case, obs, power, seed, proposal=proposal)
                offsets.append(value.item())
        with torch.no_grad():
            for p, value in zip(parameters, original):
                p.copy_(value)
        finite_difference = (offsets[1] - offsets[0]) / (2 * epsilon)
        torch.testing.assert_close(torch.tensor(auto), torch.tensor(finite_difference), rtol=2e-4, atol=2e-7)
        rows.append(dict(power=power, seed=seed, direction=which, autograd=auto,
                         finite_difference=finite_difference, nll=objective.item()))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defensive", action="store_true")
    args = parser.parse_args()
    proposal = "defensive_student" if args.defensive else "rss_gaussian"
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out = ROOT / ("results/support_integral_audit_v2_student" if args.defensive else "results/support_integral_audit_v1")
    out.mkdir(exist_ok=False)
    setting = json.loads((ROOT / "results/change_kernel_v1/freeze.json").read_text())
    base, obs = setting["base"], setting["obs"]
    freeze = dict(role="Numerical audit, no fitting; source Office only", nodes=NODES, proposal=proposal,
                  models=["common", "trained_gram"], base=base, obs=obs,
                  nll_resolution_marker=.002,
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), ROOT / "support_integral.py")})
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2))
    (out / "code").mkdir()
    for name in ("support_integral.py", "check_support_integral.py"):
        shutil.copy2(ROOT / name, out / "code" / name)
    with torch.no_grad():
        result = synthetic(obs, proposal)
    (out / "synthetic.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    mean = Transfer("affine").double()
    mean.load_state_dict(torch.load(setting["mean_checkpoint"], weights_only=True))
    mean.requires_grad_(False)
    change = ChangeCorrelation("gram").double()
    change.load_state_dict(torch.load(ROOT / "results/change_kernel_v1/gram.pt", weights_only=True))
    models = dict(common=CommonChange(base["length"], .5), trained_gram=change)
    path = ROOT.parent / "Law-Continuous-Change-20261023/data/OfficeP2.npz"
    groups = prepare_validation({6: load_period(path, 6)}, path, base, obs)
    rows = []
    with (out / "source.jsonl").open("w") as stream, torch.no_grad():
        for group in groups:
            for ap, case in zip(group["ap_columns"], group["cases"]):
                for name, model in models.items():
                    prior = Prior(mean, case["history"], base, model)
                    for power, seed in NODES:
                        value, posterior = conditional_loss(prior, case, obs, power, seed, proposal=proposal)
                        row = dict(fraction=group["fraction"], k=group["k"], ap=ap, model=name,
                                   power=power, seed=seed, nll=value.item(), **posterior.diagnostics())
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        rows.append(row)
            print(json.dumps(dict(stage="source", fraction=group["fraction"], k=group["k"], rows=len(rows))), flush=True)
    gradients = []
    for group in groups[:2]:
        for which in ("eta", "encoder"):
            checks = direction_check(mean, change, group["cases"][0], base, obs, which, proposal)
            gradients.extend([dict(fraction=group["fraction"], k=group["k"], ap=group["ap_columns"][0], **r)
                              for r in checks])
    (out / "gradients.json").write_text(json.dumps(gradients, indent=2, allow_nan=False))
    differences = []
    for group in groups:
        for ap in group["ap_columns"]:
            for model in models:
                selected = [r for r in rows if r["fraction"] == group["fraction"] and
                            r["k"] == group["k"] and r["ap"] == ap and r["model"] == model]
                values = [r["nll"] for r in selected]
                differences.append(dict(fraction=group["fraction"], k=group["k"], ap=ap, model=model,
                                        nll_range=max(values)-min(values),
                                        min_ess=min(r["effective_samples"] for r in selected)))
    summary = dict(completed=True, synthetic_cases=len(result["cases"]), source_evaluations=len(rows),
                   source_nll_spread=differences, gradients=gradients)
    (out / "completed.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(dict(stage="completed", max_source_nll_range=max(r["nll_range"] for r in differences),
                          marked=sum(r["nll_range"] > .002 for r in differences), total=len(differences))), flush=True)


if __name__ == "__main__":
    main()
