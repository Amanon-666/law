"""Source-only joint observation/base-GP calibration, including k=0 cells."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from data import load_period
from inference import infer
from transfer import rbf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--initialization", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--updates", type=int, default=64)
    p.add_argument("--inner-steps", type=int, default=160)
    p.add_argument("--start-parameters")
    p.add_argument("--learning-rate", type=float, default=.02)
    args = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    initial = json.loads(Path(args.initialization).read_text())
    noise = initial["observations"]["conditional_within_cell_sigma_pooled_db"]
    drift = initial["suggested_fixed_initialization"]["drift_amplitude_db"]
    out = Path(args.out)
    out.mkdir(exist_ok=False, parents=True)
    rng = np.random.default_rng(231023)
    floors = {t: load_period(args.data, t) for t in range(1, 12)}
    cases, selection = [], []
    for i in range(16):
        t = int(rng.integers(1, 12))
        floor = floors[t]
        active = np.flatnonzero(floor["stats"]["k"].sum(0) > 0)
        j = int(rng.choice(active))
        stats = {name: torch.as_tensor(value if name == "n" else value[:, j])
                 for name, value in floor["stats"].items()}
        cases.append((torch.as_tensor(floor["positions"]), stats))
        selection.append(dict(period=t, ap_column=j, detected=int(stats["k"].sum())))
    # Shared parameters only, no per-AP embedding/offset. The broad initial
    # amplitude covers both between-AP and within-AP variation.
    start = dict(mean=-85., amplitude=20., length=3.825, alpha=.98, beta=-95., detect_scale=10.)
    if args.start_parameters:
        previous = json.loads(Path(args.start_parameters).read_text())
        start.update({key: previous["base"][key] for key in ("mean", "amplitude", "length")})
        start.update({key: previous["obs"][key] for key in ("alpha", "beta", "detect_scale")})
    mean_shift = torch.nn.Parameter(torch.tensor((start["mean"] + 85) / 20))
    logs = torch.nn.Parameter(torch.tensor(np.log([start["amplitude"], start["length"], start["detect_scale"]])))
    beta_shift = torch.nn.Parameter(torch.tensor((start["beta"] + 95) / 20))
    alpha_logit = torch.nn.Parameter(torch.tensor(np.log(start["alpha"] / (1-start["alpha"]))))
    optimizer = torch.optim.Adam([mean_shift, logs, beta_shift, alpha_logit], lr=args.learning_rate)
    freeze = dict(vars(args), source_periods=list(range(1, 12)), selection=selection,
                  start=start,
                  frozen_unix=time.time(),
                  code_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")})
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2))
    (out / "code").mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(source, out / "code" / source.name)
    trajectory = []
    for step in range(args.updates):
        x, stats = cases[step % len(cases)]
        mean, beta = -85 + 20 * mean_shift, -95 + 20 * beta_shift
        amplitude, length, scale = logs.exp().unbind()
        obs = dict(alpha=alpha_logit.sigmoid(), beta=beta, detect_scale=scale, noise_std=noise)
        optimizer.zero_grad()
        posterior = infer(mean.expand(len(x)), rbf(x, x, amplitude, length), stats, obs,
                          steps=args.inner_steps, hyperparameter_fit=True)
        objective = posterior["loss"] / stats["n"].sum()
        objective.backward()
        if not torch.isfinite(objective) or not all(torch.isfinite(q.grad).all() for q in (mean_shift, logs, beta_shift, alpha_logit)):
            failure = dict(step=step, objective=objective.item(),
                           parameters=dict(mean=mean.item(), logs=logs.detach().tolist(), beta=beta.item(), alpha_logit=alpha_logit.item()),
                           gradients=dict(mean_shift=mean_shift.grad.item(), logs=logs.grad.tolist(), beta_shift=beta_shift.grad.item(), alpha_logit=alpha_logit.grad.item()))
            (out / "failure.json").write_text(json.dumps(failure, indent=2))
            (out / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
            raise RuntimeError("joint source fit produced a nonfinite objective/gradient")
        optimizer.step()
        mean, beta = -85 + 20 * mean_shift, -95 + 20 * beta_shift
        trajectory.append(dict(step=step, nll_bound=objective.item(),
                               mean=mean.item(), amplitude=logs[0].exp().item(),
                               length=logs[1].exp().item(), scale=logs[2].exp().item(),
                               beta=beta.item(), alpha=alpha_logit.sigmoid().item()))
        if step % 8 == 0 or step == args.updates - 1:
            print(json.dumps(trajectory[-1]), flush=True)
    result = dict(base=dict(mean=mean.item(), amplitude=logs[0].exp().item(),
                            length=logs[1].exp().item(), drift=drift),
                  obs=dict(alpha=alpha_logit.sigmoid().item(), beta=beta.item(),
                           detect_scale=logs[2].exp().item(), noise_std=noise),
                  scope="alternating full Gaussian VI and shared hyperparameter ELBO on source training1-11; not target tuning",
                  fixed_noise_and_drift="repeat-scan and temporal source estimates",
                  frozen_unix=freeze["frozen_unix"], completed_unix=time.time())
    # Evaluate one common final parameter vector, not an average of objectives
    # from different APs at different intermediate parameter values.
    final_bounds = []
    for x, stats in cases:
        b = result["base"]
        fitted = infer(x.new_full((len(x),), b["mean"]),
                       rbf(x, x, b["amplitude"], b["length"]), stats, result["obs"],
                       steps=args.inner_steps)
        final_bounds.append((fitted["loss"] / stats["n"].sum()).item())
    result["final_fixed_case_nll_bound"] = float(np.mean(final_bounds))
    (out / "parameters.json").write_text(json.dumps(result, indent=2))
    (out / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
