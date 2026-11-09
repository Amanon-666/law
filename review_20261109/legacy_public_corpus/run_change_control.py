"""Post-v1 mature controls: independently fitted change length/global offset.

Uses the original source episodes to select common kernel hyperparameters.
Does not retrain neural models or choose parameters from period 12.
"""
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from data import load_period
from run import loss, prepare_source_batches
from run_change import evaluate, prepare_validation
from transfer import ChangeCorrelation, Transfer, rbf


ROOT = Path(__file__).resolve().parent


class CommonChange:
    def __init__(self, length, local_fraction):
        self.length, self.local_fraction = length, local_fraction

    def kernel(self, x, y, history, base):
        constant = x.new_ones((len(x), len(y)))
        spatial = (constant if self.length is None else rbf(x, y, 1.0, self.length))
        return base["drift"] ** 2 * (self.local_fraction * spatial +
                                    (1 - self.local_fraction) * constant)


def attribute():
    """Post-run replacement: learned feature relations vs a uniform relation.

    Keep each trained model's local mixture weight. No parameter fitting.
    """
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    origin = ROOT / "results/change_kernel_v1"
    setting = json.loads((origin / "freeze.json").read_text())
    base, obs = setting["base"], setting["obs"]
    data_path = ROOT.parent / "Law-Continuous-Change-20261023/data/OfficeP2.npz"
    mean = Transfer("affine").double()
    mean.load_state_dict(torch.load(setting["mean_checkpoint"], weights_only=True))
    mean.requires_grad_(False)
    validation = prepare_validation({6: load_period(data_path, 6)}, data_path, base, obs)
    models, summaries = {}, []
    for kind in ("linear", "gram"):
        model = ChangeCorrelation(kind).double()
        model.load_state_dict(torch.load(origin / f"{kind}.pt", weights_only=True))
        local = torch.sigmoid(model.logit_local).item()
        models[kind] = model
        models[f"{kind}_uniform_relation"] = CommonChange(base["length"], local)
        with torch.no_grad():
            for group in validation:
                relations = []
                for case in group["cases"]:
                    x, y = case["qx"], case["sx"]
                    relation = (model.kernel(x, y, case["history"], base) / base["drift"] ** 2 -
                                local * rbf(x, y, 1., base["length"])) / (1 - local)
                    relations.append(relation.flatten())
                values = torch.cat(relations)
                summaries.append(dict(kind=kind, fraction=group["fraction"], k=group["k"],
                                      local_fraction=local, min=values.min().item(),
                                      mean=values.mean().item(), max=values.max().item(),
                                      negative_fraction=(values < 0).double().mean().item()))
    rows = evaluate(mean, models, validation, base, obs)
    result = dict(role="Post-v1 mechanism attribution; no fitting or checkpoint selection",
                  completed_unix=time.time(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  relations=summaries, validation=rows)
    path = ROOT / "results/change_relation_attribution_20261024.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(relations=summaries, validation=[{k: v for k, v in r.items()
                                                          if k != "per_ap_nll"} for r in rows])), flush=True)


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out = ROOT / "results/change_kernel_mature_control_v1"
    out.mkdir(exist_ok=False)
    old = ROOT / "results/source_transfer_v4_spatial_control"
    setting = json.loads((old / "freeze.json").read_text())
    base, obs = setting["base"], setting["obs"]
    data_path = ROOT / setting["data"]
    mean = Transfer("affine").double()
    mean.load_state_dict(torch.load(old / "affine.pt", weights_only=True))
    mean.requires_grad_(False)
    configs = [(base["length"] * factor, local) for factor in (.5, 1, 2, 4, 8, 16)
               for local in (.5, 1.)] + [(None, 0.)]
    freeze = dict(started_unix=time.time(), role="Post-v1 control prompted by untrained improvement; not preregistered in v1",
                  selection="Minimum source-episode conditional NLL, before period-12 evaluation",
                  configs=configs, source_periods=list(range(1, 12)), base=base, obs=obs,
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")})
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2))
    (out / "code").mkdir()
    for p in ROOT.glob("*.py"):
        shutil.copyfile(p, out / "code" / p.name)
    floors = {t: load_period(data_path, t) for t in range(1, 12)}
    batches = prepare_source_batches(floors, base, obs, 16, 4, 160, 231023)
    fits = []
    with torch.no_grad():
        for length, local in configs:
            model = CommonChange(length, local)
            values = [np.mean([loss(mean, c, base, obs, 160, change=model).item() for c in batch])
                      for batch in batches]
            if not np.isfinite(values).all():
                raise RuntimeError("Nonfinite source control objective")
            row = dict(length=length, local_fraction=local, source_nll=float(np.mean(values)))
            fits.append(row)
            print(json.dumps(dict(stage="source_control_fit", **row)), flush=True)
    selected = min(fits, key=lambda r: r["source_nll"])
    selected_rbf = min((r for r in fits if r["local_fraction"] == 1), key=lambda r: r["source_nll"])
    (out / "source_selection.json").write_text(json.dumps(dict(candidates=fits, selected=selected,
                                                               selected_rbf=selected_rbf), indent=2))
    validation = prepare_validation(floors, data_path, base, obs)
    original = json.loads((ROOT / "results/change_kernel_v1/initial_validation.json").read_text())
    for group in validation:
        reference = next(r for r in original if r["model"] == "spatial" and
                         r["fraction"] == group["fraction"] and r["k"] == group["k"])
        assert reference["ap_columns"] == group["ap_columns"]
    models = dict(spatial=None,
                  constant_local_half=CommonChange(base["length"], .5),
                  source_selected_rbf=CommonChange(selected_rbf["length"], 1.),
                  source_selected_common=CommonChange(selected["length"], selected["local_fraction"]))
    rows = evaluate(mean, models, validation, base, obs)
    (out / "validation.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    (out / "completed.json").write_text(json.dumps(dict(completed_unix=time.time())))
    print(json.dumps(dict(stage="complete", selected=selected, selected_rbf=selected_rbf,
                          rows=[{k: v for k, v in r.items() if k != "per_ap_nll"} for r in rows])), flush=True)


if __name__ == "__main__":
    main()
