"""Independent small-S reference using pypmc's established M-PMC update.

No model training. Dispersed mode searches initialize Gaussian components;
importance weighting, not optimizer objective or Query, determines their mass.
The numerical-development interface accepts up to ten Support positions.
Its original five-position experiment is preserved in results snapshots.
"""
import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pypmc
import torch
from pypmc.density.gauss import Gauss
from pypmc.density.mixture import MixtureDensity
from pypmc.mix_adapt.pmc import gaussian_pmc
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp

from data import load_period
from inference import _chol, gaussian_update, observation_log_prob
from run import make_case, validation_episode
from run_change_control import CommonChange
from transfer import Prior, Transfer


ROOT = Path(__file__).resolve().parent


def setup(mean, covariance, counts, obs):
    if len(mean) > 10:
        raise ValueError("This numerical reference is bounded to K <= 10")
    rss = gaussian_update(mean, covariance, counts, obs)
    center, root = rss["mean"].numpy(), _chol(rss["cov"]).numpy()
    n, k = counts["n"].numpy(), counts["k"].numpy()
    alpha, beta, scale = obs["alpha"], obs["beta"], obs["detect_scale"]
    d = len(mean)

    def terms(u):
        f = center + u @ root.T
        z = (f - beta) / scale
        ld = math.log(alpha) + log_ndtr(z)
        lm = np.logaddexp(math.log1p(-alpha), math.log(alpha) + log_ndtr(-z))
        return z, ld, lm

    def log_target(u):
        _, ld, lm = terms(u)
        return -.5 * (d * math.log(2 * math.pi) + (u * u).sum(-1)) + (k * ld + (n - k) * lm).sum(-1)

    def objective(u):
        z, ld, lm = terms(u)
        logphi = -.5 * (math.log(2 * math.pi) + z * z)
        h = np.exp(math.log(alpha) + logphi - ld)
        b = np.exp(math.log(alpha) + logphi - lm)
        gradient = u - root.T @ ((k * h - (n - k) * b) / scale)
        return -float(log_target(u)), gradient

    def curvature(u):
        z, ld, lm = terms(u)
        logphi = -.5 * (math.log(2 * math.pi) + z * z)
        h = np.exp(math.log(alpha) + logphi - ld)
        b = np.exp(math.log(alpha) + logphi - lm)
        second = (-k * h * (z + h) + (n - k) * b * (z - b)) / scale ** 2
        return np.eye(d) - root.T @ (second[:, None] * root)

    missing = np.flatnonzero(n > k)
    modes, components, log_masses = [], [], []
    # Enumerate two plausible starting explanations per missing location.
    # These are optimizer starts, not hard labels or assigned posterior weights.
    for bits in itertools.product((0, 1), repeat=len(missing)):
        initial = center.copy()
        chosen = missing[np.array(bits, dtype=bool)]
        initial[chosen] = np.minimum(initial[chosen], beta - 2 * scale)
        start = np.linalg.solve(root, initial - center)
        optimum = minimize(objective, start, jac=True, method="BFGS", options=dict(gtol=1e-8, maxiter=300))
        if np.max(np.abs(optimum.jac)) > 1e-5:
            raise RuntimeError(f"Nonstationary initialization: {np.max(np.abs(optimum.jac))}")
        if any(np.linalg.norm(optimum.x - mode) < 1e-3 for mode in modes):
            continue
        precision = curvature(optimum.x)
        np.linalg.cholesky(precision)  # fail visibly for a non-mode, do not clip
        covariance_mode = np.linalg.inv(precision)
        modes.append(optimum.x)
        components.append(Gauss(optimum.x, covariance_mode))
        log_masses.append(log_target(optimum.x) + .5 * np.linalg.slogdet(covariance_mode)[1])
    weights = np.exp(log_masses - logsumexp(log_masses))
    return log_target, MixtureDensity(components, weights), center, root, np.array(modes)


def _draw_mixture(proposal, count, rng, trace=False):
    """pypmc 1.2.6 MixtureDensity.propose does not forward rng to components.

    Keep its multinomial mixture sampling, explicitly passing the same local
    generator to each Gaussian. Otherwise the declared seed is incomplete.
    """
    counts = rng.multinomial(count, proposal.weights)
    nodes = np.concatenate([component.propose(int(n), rng=rng)
                            for component, n in zip(proposal.components, counts) if n])
    if trace:
        return nodes, np.repeat(np.arange(len(counts)), counts)
    rng.shuffle(nodes)
    return nodes


def sample_reference(log_target, initial, center, root, seed, defensive_fraction=0.):
    rng = np.random.RandomState(seed)
    proposal = MixtureDensity([Gauss(c.mu.copy(), c.sigma.copy()) for c in initial.components], initial.weights.copy())
    adaptation = []
    for iteration in range(6):
        nodes, labels = _draw_mixture(proposal, 4096, rng, trace=True)
        logw = log_target(nodes) - proposal.multi_evaluate(nodes)
        weights = np.exp(logw - logsumexp(logw))
        adaptation.append(dict(iteration=iteration, ess=float(1 / np.sum(weights ** 2)),
                               component_counts=np.bincount(labels, minlength=len(proposal.components)).tolist(),
                               proposal_weights=proposal.weights.tolist()))
        # The documented pypmc safeguard prevents fitting a covariance from
        # an effectively empty component. This modifies the proposal only;
        # the original posterior density still determines importance weights.
        proposal = gaussian_pmc(nodes, proposal, weights=weights, latent=labels, mincount=20, rb=True)
        # pypmc 1.2.6 can return a one-component weight 1+2e-15; NumPy's
        # multinomial correctly rejects >1. Use the library's normalization.
        proposal.normalize()
        proposal.prune(0.)
    if defensive_fraction:
        # In whitened coordinates target = N(0,I) * detection likelihood <= N(0,I).
        # A fixed prior component therefore bounds unnormalized weights by 1/epsilon.
        proposal = MixtureDensity(list(proposal.components) + [Gauss(np.zeros(len(center)),np.eye(len(center)))],
            np.r_[(1-defensive_fraction)*proposal.weights, defensive_fraction])
    # Fresh final draws: never reuse adaptation samples with the final density.
    nodes = _draw_mixture(proposal, 32768, rng)
    logw = log_target(nodes) - proposal.multi_evaluate(nodes)
    log_weights = logw - logsumexp(logw)
    samples = torch.tensor(center + nodes @ root.T)
    return samples, torch.tensor(log_weights), dict(
        seed=seed, defensive_fraction=defensive_fraction, adaptation=adaptation, log_detection_evidence=float(logsumexp(logw) - math.log(len(logw))),
        effective_samples=float(1 / np.sum(np.exp(log_weights) ** 2)),
        components=len(proposal.components), mixture_weights=proposal.weights.tolist(),
        mixture_means=[c.mu.tolist() for c in proposal.components],
        mixture_covariances=[c.sigma.tolist() for c in proposal.components])


def log_proposal(samples, center, root, info):
    """Frozen proposal log density in physical RSS coordinates."""
    components = [Gauss(np.array(m), np.array(c)) for m, c in
                  zip(info["mixture_means"], info["mixture_covariances"])]
    mixture = MixtureDensity(components, np.array(info["mixture_weights"]))
    whitened = np.linalg.solve(root, (samples.numpy() - center).T).T
    return torch.tensor(mixture.multi_evaluate(whitened) - np.log(np.diag(root)).sum())


def readout(samples, log_weights, sm, sk, qm, qv, cross, rss, weights, obs):
    A = torch.cholesky_solve(cross.T, _chol(sk)).T
    variance = qv - (A * cross).sum(-1)
    if bool((variance < -1e-8).any()):
        raise RuntimeError("Negative prior conditional variance")
    variance = variance.clamp_min(0)
    w = log_weights.exp()
    nll, influence = 0., torch.zeros(len(samples))
    individual = []
    for start in range(0, len(rss), 64):
        sl = slice(start, start + 64)
        mean = qm[sl][None] + (samples - sm) @ A[sl].T
        lp = observation_log_prob(rss[sl][None], mean, variance[sl][None], obs)
        marginal = torch.logsumexp(log_weights[:, None] + lp, 0)
        nll -= (weights[sl] * marginal).sum().item()
        influence -= ((torch.exp(lp - marginal) - 1) * weights[sl]).sum(-1)
        individual.extend((-marginal).tolist())
    posterior_mean = w @ samples
    return dict(nll=nll, per_observation_nll=individual,
                nll_mc_se=torch.sqrt((w.square() * influence.square()).sum()).item(),
                support_mean=posterior_mean.tolist(),
                support_variance=(w @ (samples - posterior_mean).square()).tolist())


def main():
    from check_support_integral import reference_1d, stats
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out = ROOT / "results/support_pmc_reference_v2"
    out.mkdir(exist_ok=False)
    setting = json.loads((ROOT / "results/change_kernel_v1/freeze.json").read_text())
    base, obs = setting["base"], setting["obs"]
    freeze = dict(role="Post fixed-proposal failure: mature M-PMC numerical reference; no training", pypmc=pypmc.__version__,
                  support_limit=5, adaptation_steps=6, adaptation_samples=4096, final_samples=32768,
                  seeds=[231024, 241024], base=base, obs=obs,
                  source_cases=[dict(fraction=.25, k=5, ap=130), dict(fraction=.25, k=5, ap=139),
                                dict(fraction=1., k=5, ap=202)],
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2))
    shutil.copy2(__file__, out / "reference_pmc.py")
    problems = []
    for name, location, sd, n, k, r in (
        ("detected", -85., 8., 7, 6, -78.), ("all_missing", -92., 8., 7, 0, 0.),
        ("conflicting_missing", -65., 6., 7, 0, 0.), ("rare_conflicting_missing", -60., 5., 10, 0, 0.),
        ("single_detection", -65., 6., 1, 1, -90.)):
        sm, sk, counts = torch.tensor([location]), torch.tensor([[sd ** 2]]), stats(n, k, r)
        problems.append(dict(name=name, sm=sm, sk=sk, stats=counts, qm=sm.repeat(2), qv=sk.diag().repeat(2),
                             cross=sk.repeat(2, 1), rss=torch.tensor([-90., 100.]), weights=torch.tensor([.5, .5]),
                             exact=reference_1d(sm, sk, counts, obs)))
    mean = Transfer("affine").double()
    mean.load_state_dict(torch.load(setting["mean_checkpoint"], weights_only=True))
    mean.requires_grad_(False)
    path = ROOT.parent / "Law-Continuous-Change-20261023/data/OfficeP2.npz"
    history, current, query = load_period(path, 6), load_period(path, 12), load_period(path, 12, "testing")
    for spec in freeze["source_cases"]:
        ep = validation_episode(history, current, query, spec["fraction"], spec["k"], 0, nested_support=True)
        case = make_case(ep, int(np.flatnonzero(ep["ap_indices"] == spec["ap"])[0]), base, obs, 160)
        prior = Prior(mean, case["history"], base, CommonChange(base["length"], .5))
        sm, sf = prior.features(case["sx"])
        qm, qf = prior.features(case["qx"])
        sk = prior.covariance(case["sx"], case["sx"], sf, sf)
        cross = prior.covariance(case["qx"], case["sx"], qf, sf)
        qv = prior.diagonal(case["qx"], qf)
        group = case["group"]
        problems.append(dict(name=f"source_H{spec['fraction']}_AP{spec['ap']}", sm=sm, sk=sk, stats=case["stats"],
                             qm=qm[group], qv=qv[group], cross=cross[group], rss=case["rss"], weights=case["weights"]))
    rows = []
    with (out / "results.jsonl").open("w") as stream, torch.no_grad():
        for problem in problems:
            target, initial, center, root, modes = setup(problem["sm"], problem["sk"], problem["stats"], obs)
            for seed in freeze["seeds"]:
                samples, logw, info = sample_reference(target, initial, center, root, seed)
                result = readout(samples, logw, *(problem[key] for key in ("sm", "sk", "qm", "qv", "cross", "rss", "weights")), obs)
                row = dict(name=problem["name"], initial_modes=modes.tolist(), reference=problem.get("exact"), **info, **result)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                print(json.dumps({key: row[key] for key in ("name", "seed", "nll", "nll_mc_se", "effective_samples", "support_mean")}), flush=True)
    (out / "completed.json").write_text(json.dumps(dict(completed=True, rows=len(rows)), indent=2))


if __name__ == "__main__":
    main()
