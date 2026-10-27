"""Bounded source-only development; no target site is opened by this runner."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from data import load_period, draw_episode, group_scans, _subset
from inference import infer, observation_log_prob, predict_from_posterior
from transfer import Transfer, History, Prior, rbf


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


def ap_stats(scans, j):
    return {key: tensor(value if key == "n" else value[:, j])
            for key, value in scans["stats"].items()}


def make_case(ep, j, base, obs, h_steps):
    z, sx, qx = (tensor(ep[name]["positions"]) for name in ("H", "S", "Q"))
    history = None
    if ep["historical_positive"][j]:
        posterior = infer(z.new_full((len(z),), base["mean"]),
                          rbf(z, z, base["amplitude"], base["length"]),
                          ap_stats(ep["H"], j), obs, steps=h_steps)
        history = History(z, posterior, base)
    return dict(history=history, sx=sx, qx=qx, stats=ap_stats(ep["S"], j),
                rss=tensor(ep["Q"]["rss"][:, j]), group=ep["Q"]["group"],
                weights=tensor(ep["Q"]["scan_weights"]))


def predict(model, case, qx, base, obs, steps, differentiable=False, change=None):
    prior = Prior(model, case["history"], base, change)
    sm, sf = prior.features(case["sx"])
    qm, qf = prior.features(qx)
    sk = prior.covariance(case["sx"], case["sx"], sf, sf)
    cross = prior.covariance(qx, case["sx"], qf, sf)
    diagonal = prior.diagonal(qx, qf)
    posterior = infer(sm, sk, case["stats"], obs, steps=steps,
                      differentiable=differentiable)
    result = predict_from_posterior(qm, diagonal, cross, sm, sk, posterior)
    return result["mean"], result["variance"]


def loss(model, case, base, obs, steps, differentiable=False, change=None):
    mu, var = predict(model, case, case["qx"], base, obs, steps, differentiable, change)
    logp = observation_log_prob(case["rss"], mu[case["group"]], var[case["group"]], obs)
    return -(case["weights"] * logp).sum()


def validation_episode(history, current, query, fraction, k, seed, nested_support=False):
    # Q is the author's public testing split. It never determines AP selection.
    # Separate streams make S identical and H nested across coverage fractions.
    s_rng = np.random.default_rng(seed)
    s_ids = np.sort(s_rng.permutation(len(current["positions"]))[:k] if nested_support else
                    s_rng.choice(len(current["positions"]), k, replace=False))
    h_order = np.random.default_rng(seed + 231023).permutation(len(history["positions"]))
    h_ids = np.sort(h_order[:int(np.ceil(fraction * len(h_order)))])
    h_seen = (history["stats"]["k"][h_ids] > 0).any(0)
    s_seen = (current["stats"]["k"][s_ids] > 0).any(0)
    aps = np.flatnonzero(h_seen | s_seen)
    sampled = dict(H=_subset(history, h_ids, aps), S=_subset(current, s_ids, aps),
                   ap_indices=aps, historical_positive=h_seen[aps],
                   h_position_indices=h_ids, s_position_indices=s_ids)
    q = group_scans(query["rss"][:, aps], query["xy"])
    q["scan_weights"] = 1 / (len(q["positions"]) * q["stats"]["n"][q["group"]])
    sampled["Q"] = q
    return sampled


def prepare_source_batches(floors, base, obs, tasks, aps_per_task, history_steps, seed):
    """Shared source episode construction for mean and correlation studies."""
    rng = np.random.default_rng(seed)
    batches = []
    for t in range(tasks):
        a, b = sorted(rng.choice(range(1, 12), 2, replace=False).tolist())
        ep = draw_episode(floors[a], floors[b], (0.25, 1.0)[t % 2],
                          (5, 10)[(t // 2) % 2], 12, rng)
        angle = rng.uniform(0, 2 * np.pi)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        origin = floors[1]["positions"].mean(0)
        for name in ("H", "S", "Q"):
            ep[name]["positions"] = (ep[name]["positions"] - origin) @ rotation
        eligible = np.flatnonzero(ep["historical_positive"])
        aps = rng.choice(eligible, min(aps_per_task, len(eligible)), replace=False)
        batch = [make_case(ep, int(j), base, obs, history_steps) for j in aps]
        for case, j in zip(batch, aps):
            case["episode"] = dict(periods=[a, b], ap_column=int(ep["ap_indices"][j]),
                                   history_fraction=(.25, 1.)[t % 2],
                                   support_positions=ep["s_position_indices"].tolist(),
                                   query_positions=ep["q_position_indices"].tolist())
        batches.append(batch)
        print(json.dumps(dict(stage="prepare", task=t, periods=[a, b], aps=len(batch))), flush=True)
    return batches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--initialization", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--updates", type=int, default=80)
    p.add_argument("--inner-steps", type=int, default=160)
    p.add_argument("--history-steps", type=int, default=160)
    p.add_argument("--tasks", type=int, default=16)
    p.add_argument("--aps-per-task", type=int, default=4)
    p.add_argument("--evaluate", action="store_true")
    args = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    init = json.loads(Path(args.initialization).read_text())
    base, obs = init["base"], init["obs"]
    freeze = dict(vars(args), base=base, obs=obs, source_periods=list(range(1, 12)),
                  validation_period=12, seed=231023,
                  hypothesis="shape-conditioned transfer beats affine/pointwise at identical H/S; source development only",
                  code_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in Path(__file__).parent.glob("*.py")},
                  frozen_unix=time.time())
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2))
    (out / "code").mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(source, out / "code" / source.name)
    floors = {t: load_period(args.data, t) for t in range(1, 12)}
    batches = prepare_source_batches(floors, base, obs, args.tasks, args.aps_per_task,
                                     args.history_steps, 231023)
    models = {kind: Transfer(kind).double() for kind in ("identity", "affine", "pointwise", "linear_shape", "shape")}
    # Initial correction is zero, so all models share the exact same prior.
    for kind in ("affine", "pointwise", "linear_shape", "shape"):
        torch.manual_seed(231023)
        model = models[kind] = Transfer(kind).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        trajectory = []
        for step in range(args.updates):
            batch = batches[step % len(batches)]
            optimizer.zero_grad()
            objective = torch.stack([loss(model, c, base, obs, args.inner_steps, True) for c in batch]).mean()
            objective.backward()
            grad = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters())).item()
            if not np.isfinite(objective.item()) or not np.isfinite(grad):
                raise RuntimeError("nonfinite objective or gradient; stop before evaluation")
            optimizer.step()
            trajectory.append(dict(step=step, nll=objective.item(), grad_norm=grad))
            if step % 10 == 0 or step == args.updates - 1:
                print(json.dumps(dict(stage="train", kind=kind, **trajectory[-1])), flush=True)
        torch.save(model.state_dict(), out / f"{kind}.pt")
        (out / f"{kind}_training.json").write_text(json.dumps(trajectory, indent=2))
    if not args.evaluate:
        return
    # Source validation only, fixed beforehand; no Mall or future Office period.
    current = load_period(args.data, 12)
    query = load_period(args.data, 12, "testing")
    bounds = floors[1]["positions"]
    # Survey geometry is fixed metadata shared by every method, independent of
    # H thinning. Include endpoints so the posterior mean can span the domain.
    grid_spacing = min(1.0, base["length"] / 3)
    axes = [np.linspace(bounds[:, d].min(), bounds[:, d].max(),
                        int(np.ceil(np.ptp(bounds[:, d]) / grid_spacing)) + 1) for d in range(2)]
    grid = tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1, 2))
    models["local_gp"] = Transfer("identity").double()
    records = []
    for fraction in (0.25, 1.0):
        for seed in (0, 1):
            ep = validation_episode(floors[6], current, query, fraction, 10, seed)
            scores = {kind: torch.zeros((len(query["rss"]), len(grid))) for kind in models}
            nlls = {kind: [] for kind in models}
            for j in range(len(ep["ap_indices"])):
                case = make_case(ep, j, base, obs, args.history_steps)
                for kind, model in models.items():
                    model_case = dict(case, history=None) if kind == "local_gp" else case
                    mu, var = predict(model, model_case, torch.cat((grid, case["qx"])), base, obs, args.inner_steps)
                    if not torch.isfinite(mu).all() or not torch.isfinite(var).all() or not (var > 0).all():
                        raise RuntimeError(f"invalid predictive distribution: {kind}, H={fraction}, seed={seed}, AP={j}")
                    scores[kind] += observation_log_prob(case["rss"][:, None], mu[:len(grid)][None], var[:len(grid)][None], obs).detach()
                    qlogp = observation_log_prob(case["rss"], mu[len(grid):][case["group"]], var[len(grid):][case["group"]], obs)
                    nlls[kind].append(-(case["weights"] * qlogp).sum().item())
                if j % 25 == 0:
                    print(json.dumps(dict(stage="validate", fraction=fraction, seed=seed, ap=j)), flush=True)
            predictions = {}
            for kind in models:
                # Continuous posterior expectation over a shared quadrature grid.
                prediction = torch.softmax(scores[kind], dim=1) @ grid
                predictions[kind] = prediction.detach().numpy()
                errors = torch.linalg.vector_norm(prediction - tensor(query["xy"]), dim=1)
                record = dict(kind=kind, fraction=fraction, seed=seed, k=10,
                              nll=float(np.mean(nlls[kind])),
                              mde=float((tensor(ep["Q"]["scan_weights"]) * errors).sum()))
                records.append(record)
                print(json.dumps(dict(stage="result", **record)), flush=True)
            s = ep["S"]["stats"]
            fingerprint = (s["s1"] - 110 * (s["n"][:, None] - s["k"])) / s["n"][:, None]
            scan = np.where(ep["Q"]["rss"] == 100, -110, ep["Q"]["rss"])
            distance = np.sqrt(((scan[:, None] - fingerprint[None]) ** 2).mean(-1))
            nearest = np.argsort(distance, axis=1)[:, :3]
            weights = 1 / np.maximum(np.take_along_axis(distance, nearest, axis=1), 1e-8)
            prediction = (ep["S"]["positions"][nearest] * weights[:, :, None]).sum(1) / weights.sum(1)[:, None]
            records.append(dict(kind="wknn", fraction=fraction, seed=seed, k=10,
                                mde=float((ep["Q"]["scan_weights"] * np.linalg.norm(prediction - query["xy"], axis=1)).sum())))
            predictions["wknn"] = prediction
            np.savez_compressed(out / f"predictions_h{fraction}_seed{seed}.npz",
                                query_xy=query["xy"], support_xy=ep["S"]["positions"],
                                history_xy=ep["H"]["positions"], **predictions)
            (out / "source_validation.json").write_text(json.dumps(records, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
