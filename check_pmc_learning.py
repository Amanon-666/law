"""Source-only audit of fixed-proposal gradients and K=10 posterior scope."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from data import load_period
from reference_pmc import setup, sample_reference, log_proposal
from run import make_case, validation_episode
from support_integral import importance_loss
from transfer import Prior, ChangeCorrelation, Transfer


ROOT = Path(__file__).resolve().parent


def check_gradient(mean, model, case, base, obs, samples, logq):
    parameters = list(model.parameters())
    value, posterior = importance_loss(Prior(mean, case["history"], base, model), case, obs, samples, logq)
    derivatives = torch.autograd.grad(value, parameters)
    flattened = torch.cat([g.flatten() for g in derivatives])
    original = [p.detach().clone() for p in parameters]
    checks = []
    for which in ("eta", "encoder"):
        torch.manual_seed(721024)
        direction = [torch.randn_like(p) if (i == 0) == (which == "eta") else torch.zeros_like(p)
                     for i, p in enumerate(parameters)]
        norm = torch.sqrt(sum(d.square().sum() for d in direction))
        direction = [d / norm for d in direction]
        derivative = sum((g * d).sum() for g, d in zip(derivatives, direction)).item()
        values = []
        for sign in (-1, 1):
            with torch.no_grad():
                for p, initial, d in zip(parameters, original, direction):
                    p.copy_(initial + sign * 1e-4 * d)
                nearby, _ = importance_loss(Prior(mean, case["history"], base, model), case, obs, samples, logq)
                values.append(nearby.item())
        with torch.no_grad():
            for p, initial in zip(parameters, original):
                p.copy_(initial)
        finite = (values[1] - values[0]) / 2e-4
        torch.testing.assert_close(torch.tensor(derivative), torch.tensor(finite), rtol=2e-4, atol=2e-7)
        checks.append(dict(direction=which, autograd=derivative, finite_difference=finite))
    return dict(nll=value.item(), gradient=flattened.tolist(), gradient_norm=flattened.norm().item(),
                directions=checks, posterior=posterior.diagnostics()), posterior.log_weights.detach()


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out = ROOT / "results/pmc_learning_audit_v1"
    out.mkdir(exist_ok=False)
    source = ROOT / "results/change_kernel_v1"
    settings = json.loads((source / "freeze.json").read_text())
    base, obs = settings["base"], settings["obs"]
    specs = [dict(fraction=fraction, k=k, ap=ap)
             for fraction, ap in ((.25, 130), (.25, 139), (1., 202)) for k in (5, 10)]
    freeze = dict(role="Fixed-proposal gradient and K10 scope audit; no training", specs=specs,
                  seeds=[231024, 241024], model="trained_gram", started_unix=time.time(),
                  derivative_target="Same-sample self-normalized original posterior; frozen proposal density",
                  base=base, obs=obs, code_sha256={name:hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ("support_integral.py", "reference_pmc.py", "check_pmc_learning.py")})
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2))
    (out / "code").mkdir()
    for name in freeze["code_sha256"]:
        shutil.copy2(ROOT / name, out / "code" / name)
    mean = Transfer("affine").double()
    mean.load_state_dict(torch.load(settings["mean_checkpoint"], weights_only=True))
    mean.requires_grad_(False)
    model = ChangeCorrelation("gram").double()
    model.load_state_dict(torch.load(source / "gram.pt", weights_only=True))
    path = ROOT.parent / "Law-Continuous-Change-20261023/data/OfficeP2.npz"
    history, current, query = load_period(path, 6), load_period(path, 12), load_period(path, 12, "testing")
    comparisons = []
    with (out / "results.jsonl").open("w") as stream:
        for spec in specs:
            ep = validation_episode(history, current, query, spec["fraction"], spec["k"], 0, nested_support=True)
            case = make_case(ep, int(np.flatnonzero(ep["ap_indices"] == spec["ap"])[0]), base, obs, 160)
            with torch.no_grad():
                prior = Prior(mean, case["history"], base, model)
                sm, sf = prior.features(case["sx"])
                sk = prior.covariance(case["sx"], case["sx"], sf, sf)
                target, initial, center, root, modes = setup(sm, sk, case["stats"], obs)
            print(json.dumps(dict(stage="modes", **spec, modes=len(modes))), flush=True)
            rows = []
            for seed in freeze["seeds"]:
                with torch.no_grad():
                    samples, reference_logw, info = sample_reference(target, initial, center, root, seed)
                    logq = log_proposal(samples, center, root, info)
                result, actual_logw = check_gradient(mean, model, case, base, obs, samples, logq)
                # Detect double-counted RSS, missed proposal Jacobian/normalization.
                torch.testing.assert_close(actual_logw, reference_logw, rtol=1e-10, atol=1e-10)
                row = dict(**spec, seed=seed, initial_modes=len(modes), **result)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                print(json.dumps({key:row[key] for key in ("fraction", "k", "ap", "seed", "nll", "gradient_norm")}), flush=True)
            g1, g2 = (np.array(r["gradient"]) for r in rows)
            comparisons.append(dict(**spec, nll_difference=abs(rows[0]["nll"]-rows[1]["nll"]),
                                    gradient_cosine=float(g1 @ g2 / (np.linalg.norm(g1)*np.linalg.norm(g2))),
                                    gradient_relative_difference=float(np.linalg.norm(g1-g2)/max(np.linalg.norm(g1),np.linalg.norm(g2)))))
    (out / "completed.json").write_text(json.dumps(dict(completed=True, comparisons=comparisons), indent=2))
    print(json.dumps(dict(stage="completed", comparisons=comparisons)), flush=True)


if __name__ == "__main__":
    main()
