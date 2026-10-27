"""Numerical posterior reference for one observed VI branch-sensitive case.

Integrate only the small Support latent vector after the exact RSS update.
This is a diagnostic, not a production solver or a localization experiment.
"""
import json
from pathlib import Path

import numpy as np
import torch

from data import load_period
from inference import JITTER, _detection_logs, gaussian_update, observation_log_prob
from run import make_case, validation_episode
from run_change_control import CommonChange
from transfer import Prior, Transfer


def quadrature(prior, case, obs, power, seed):
    sx, qx = case["sx"], case["qx"]
    sm, sf = prior.features(sx)
    qm, qf = prior.features(qx)
    sk = prior.covariance(sx, sx, sf, sf)
    cross = prior.covariance(qx, sx, qf, sf)
    diagonal = prior.diagonal(qx, qf)
    eye = torch.eye(len(sx))
    rss = gaussian_update(sm, sk, case["stats"], obs)
    engine = torch.quasirandom.SobolEngine(len(sx), scramble=True, seed=seed)
    uniform = engine.draw_base2(power, dtype=torch.float64)
    # Map finite Sobol endpoints to open-cell midpoints before the normal inverse.
    uniform = (uniform * (2 ** 32 - 1) + .5) / 2 ** 32
    normals = torch.special.ndtri(uniform)
    samples = rss["mean"] + normals @ torch.linalg.cholesky(rss["cov"] + JITTER * eye).T
    detected, missing = _detection_logs(samples, obs)
    stats = case["stats"]
    log_likelihood = (stats["k"] * detected + (stats["n"] - stats["k"]) * missing).sum(-1)
    log_weights = torch.log_softmax(log_likelihood, dim=0)
    weights = log_weights.exp()
    A = torch.cholesky_solve(cross.T, torch.linalg.cholesky(sk + JITTER * eye)).T
    conditional_variance = diagonal - (A * cross).sum(-1)
    if conditional_variance.min() < -1e-8:
        raise RuntimeError("Invalid prior conditional variance")
    conditional_variance = conditional_variance.clamp_min(0)
    conditional_mean = qm[None] + (samples - sm) @ A.T
    components = observation_log_prob(case["rss"][None], conditional_mean[:, case["group"]],
                                      conditional_variance[case["group"]][None], obs)
    logp = torch.logsumexp(log_weights[:, None] + components, dim=0)
    mean = weights @ samples
    variance = weights @ (samples - mean).square()
    return dict(nll=float(-(case["weights"] * logp).sum()), samples=2 ** power,
                effective_samples=float(1 / weights.square().sum()),
                support_mean=mean.tolist(), support_variance=variance.tolist(),
                log_detection_evidence=float(torch.logsumexp(log_likelihood, 0) - np.log(2 ** power)))


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    config = json.loads((root / "results/change_kernel_v1/freeze.json").read_text())
    base, obs = config["base"], config["obs"]
    path = root.parent / "Law-Continuous-Change-20261023/data/OfficeP2.npz"
    mean = Transfer("affine").double()
    mean.load_state_dict(torch.load(config["mean_checkpoint"], weights_only=True))
    mean.requires_grad_(False)
    ep = validation_episode(load_period(path, 6), load_period(path, 12), load_period(path, 12, "testing"),
                            .25, 5, 0, nested_support=True)
    j = int(np.flatnonzero(ep["ap_indices"] == 130)[0])
    case = make_case(ep, j, base, obs, 160)
    rows = []
    with torch.no_grad():
        for eta in (.48731367986356405, .5):
            prior = Prior(mean, case["history"], base, CommonChange(base["length"], eta))
            for power, seed in ((11, 231024), (13, 231024), (13, 241024)):
                row = dict(eta=eta, seed=seed, **quadrature(prior, case, obs, power, seed))
                rows.append(row)
                print(json.dumps(row), flush=True)
    output = root / "results/change_sampling_reference_ap130_20261024.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(dict(role="Post-v1 numerical reference, no training", ap_column=130, rows=rows), indent=2))


if __name__ == "__main__":
    main()
