"""Bounded source comparison: frozen mean/variance, learned change relations.

Training and conditional prediction reuse run.py. Only the change kernel is
optimized. Validation is Office period 12 predictive NLL, not localization.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from data import load_period
from run import loss, make_case, prepare_source_batches, validation_episode
from transfer import ChangeCorrelation, Transfer


ROOT = Path(__file__).resolve().parent


def evaluate(mean, models, cases, base, obs):
    rows = []
    with torch.no_grad():
        for group in cases:
            for name, model in models.items():
                values = [loss(mean, case, base, obs, 160, change=model).item()
                          for case in group["cases"]]
                if not np.isfinite(values).all():
                    raise RuntimeError(f"nonfinite source validation: {name}")
                rows.append({key: value for key, value in group.items() if key != "cases"} |
                            dict(model=name, nll=float(np.mean(values)), per_ap_nll=values))
    return rows


def prepare_validation(floors, data_path, base, obs):
    current = load_period(data_path, 12)
    query = load_period(data_path, 12, "testing")
    validation = []
    for fraction in (.25, 1.0):
        for k in (5, 10):
            ep = validation_episode(floors[6], current, query, fraction, k, 0, nested_support=True)
            rng = np.random.default_rng(231024 + int(100 * fraction))
            eligible = np.flatnonzero(ep["historical_positive"])
            indices = np.sort(rng.choice(eligible, min(8, len(eligible)), replace=False))
            validation.append(dict(fraction=fraction, k=k, seed=0,
                                   ap_columns=ep["ap_indices"][indices].tolist(),
                                   cases=[make_case(ep, int(j), base, obs, 160) for j in indices]))
            print(json.dumps(dict(stage="validation_prepare", fraction=fraction, k=k)), flush=True)
    return validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results/change_kernel_v1")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    origin = ROOT / "results/source_transfer_v4_spatial_control"
    setting = json.loads((origin / "freeze.json").read_text())
    base, obs = setting["base"], setting["obs"]
    data_path = ROOT / setting["data"]
    mean = Transfer("affine").double()
    mean.load_state_dict(torch.load(origin / "affine.pt", weights_only=True))
    mean.requires_grad_(False)
    models = {"spatial": None}
    for kind in ("product", "linear", "gram"):
        torch.manual_seed(231024)
        models[kind] = ChangeCorrelation(kind).double()
    freeze = dict(started_unix=time.time(), base=base, obs=obs,
                  hypothesis="At fixed pre-S marginals, learned H-conditioned change relations improve post-S predictive density.",
                  scope="Office source development only; not localization or unseen-site evidence",
                  mean_checkpoint=str(origin / "affine.pt"),
                  mean_sha256=hashlib.sha256((origin / "affine.pt").read_bytes()).hexdigest(),
                  data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  source_periods=list(range(1, 12)), tasks=16, aps_per_task=4,
                  training_seed=231023, model_seed=231024, learning_rate=.002, updates=80,
                  inner_maxiter=160, validation_period=12, validation_history=6,
                  validation_max_aps=8, validation_seed=0, validation_H=[.25, 1.0], validation_K=[5, 10],
                  validation_support_nested=True, validation_ap_selection="H-only; same columns across K within H fraction",
                  selection="Initial and fixed final update; no validation checkpoint selection",
                  torch_version=torch.__version__,
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")})
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2))
    (out / "code").mkdir()
    for p in ROOT.glob("*.py"):
        shutil.copyfile(p, out / "code" / p.name)
    floors = {period: load_period(data_path, period) for period in range(1, 12)}
    batches = prepare_source_batches(floors, base, obs, 16, 4, 160, 231023)
    validation = prepare_validation(floors, data_path, base, obs)
    initial = evaluate(mean, models, validation, base, obs)
    (out / "initial_validation.json").write_text(json.dumps(initial, indent=2, allow_nan=False))
    print(json.dumps(dict(stage="initial_validation", rows=[{k: v for k, v in r.items() if k != "per_ap_nll"}
                                                           for r in initial])), flush=True)
    for kind in ("product", "linear", "gram"):
        model = models[kind]
        torch.save(model.state_dict(), out / f"{kind}_initial.pt")
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        trajectory = []
        started = time.perf_counter()
        for step in range(80):
            batch = batches[step % len(batches)]
            optimizer.zero_grad()
            try:
                value = torch.stack([loss(mean, case, base, obs, 160, differentiable=True, change=model)
                                     for case in batch]).mean()
                value.backward()
                norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())).item()
                if not np.isfinite(value.item()) or not np.isfinite(norm):
                    raise RuntimeError("nonfinite objective/gradient")
            except Exception:
                torch.save(dict(kind=kind, step=step, model=model.state_dict(), batch=batch), out / "failure.pt")
                raise
            optimizer.step()
            row = dict(step=step, nll=value.item(), grad_norm=norm,
                       local_fraction=torch.sigmoid(model.logit_local).item(),
                       seconds=time.perf_counter() - started)
            trajectory.append(row)
            (out / f"{kind}_training.json").write_text(json.dumps(trajectory, indent=2, allow_nan=False))
            if step % 10 == 0 or step == 79:
                print(json.dumps(dict(stage="train", kind=kind, **row)), flush=True)
        torch.save(model.state_dict(), out / f"{kind}.pt")
        rows = evaluate(mean, {kind: model}, validation, base, obs)
        (out / f"{kind}_validation.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
        print(json.dumps(dict(stage="final_validation", kind=kind,
                              rows=[{k: v for k, v in r.items() if k != "per_ap_nll"} for r in rows])), flush=True)
    (out / "completed.json").write_text(json.dumps(dict(completed_unix=time.time(), models=["product", "linear", "gram"])))


if __name__ == "__main__":
    main()
