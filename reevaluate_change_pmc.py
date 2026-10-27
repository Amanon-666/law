"""Re-evaluate fixed v1 relation models with the reviewed posterior reference.

No fitting, selection, new tasks, or localization. The original 32 source
validation AP-cases and all initial/final model weights stay fixed.
"""
import hashlib
import json
import shutil
import time

import numpy as np
import torch

from data import load_period
from reference_pmc import ROOT, readout, sample_reference, setup
from run_change import prepare_validation
from run_change_control import CommonChange
from transfer import ChangeCorrelation, Prior, Transfer


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out = ROOT / "results/change_pmc_reevaluation_v3"
    out.mkdir(exist_ok=False)
    origin = ROOT / "results/change_kernel_v1"
    settings = json.loads((origin / "freeze.json").read_text())
    base, obs = settings["base"], settings["obs"]
    mean = Transfer("affine").double()
    mean.load_state_dict(torch.load(settings["mean_checkpoint"], weights_only=True))
    mean.requires_grad_(False)
    torch.manual_seed(231024)
    initial = ChangeCorrelation("gram").double()
    trained = ChangeCorrelation("gram").double()
    trained.load_state_dict(torch.load(origin / "gram.pt", weights_only=True))
    models = dict(gram_initial=initial, gram_trained=trained,
                  uniform_trained_eta=CommonChange(base["length"], torch.sigmoid(trained.logit_local).item()))
    freeze = dict(role="Fixed-model posterior reevaluation; post numerical-audit; no training",
                  started_unix=time.time(), models=list(models), seeds=[231024,241024],
                  base=base, obs=obs, source_history=6, source_validation=12, H=[.25,1.],K=[5,10],
                  initialization_seed=231024, final_checkpoint=str(origin / "gram.pt"),
                  final_sha256=hashlib.sha256((origin / "gram.pt").read_bytes()).hexdigest(),
                  adaptation_steps=6, adaptation_samples=4096, final_samples=32768,
                  proposal_min_component_draws=20,
                  estimands=["final vs initial, including eta", "final vs uniform with identical final eta"],
                  code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ("reference_pmc.py","reevaluate_change_pmc.py","transfer.py")})
    (out / "freeze.json").write_text(json.dumps(freeze,indent=2))
    (out / "code").mkdir()
    for name in freeze["code_sha256"]:
        shutil.copy2(ROOT / name, out / "code" / name)
    path = ROOT.parent / "Law-Continuous-Change-20261023/data/OfficeP2.npz"
    groups = prepare_validation({6:load_period(path,6)},path,base,obs)
    rows=[]
    with (out / "results.jsonl").open("w") as stream, torch.no_grad():
        for group in groups:
            for ap,case in zip(group["ap_columns"],group["cases"]):
                for name,model in models.items():
                    prior=Prior(mean,case["history"],base,model)
                    sm,sf=prior.features(case["sx"])
                    qm,qf=prior.features(case["qx"])
                    sk=prior.covariance(case["sx"],case["sx"],sf,sf)
                    cross=prior.covariance(case["qx"],case["sx"],qf,sf)
                    qv=prior.diagonal(case["qx"],qf)
                    target,proposal,center,root,modes=setup(sm,sk,case["stats"],obs)
                    for seed in freeze["seeds"]:
                        samples,logw,info=sample_reference(target,proposal,center,root,seed)
                        index=case["group"]
                        result=readout(samples,logw,sm,sk,qm[index],qv[index],cross[index],case["rss"],case["weights"],obs)
                        # Keep per-AP summaries; raw query scans remain in read-only data.
                        result.pop("per_observation_nll")
                        row=dict(fraction=group["fraction"],k=group["k"],ap=ap,model=name,seed=seed,
                                 modes=len(modes),ess=info["effective_samples"],**result)
                        stream.write(json.dumps(row,allow_nan=False)+"\n")
                        stream.flush()
                        rows.append(row)
                print(json.dumps(dict(stage="ap",fraction=group["fraction"],k=group["k"],ap=ap,rows=len(rows))),flush=True)
    summaries=[]
    for group in groups:
        for name in models:
            selected=[r for r in rows if r["fraction"]==group["fraction"] and r["k"]==group["k"] and r["model"]==name]
            by_seed=[np.mean([r["nll"] for r in selected if r["seed"]==seed]) for seed in freeze["seeds"]]
            summaries.append(dict(fraction=group["fraction"],k=group["k"],model=name,
                                  mean_nll=float(np.mean(by_seed)),seed_mean_nll=by_seed))
    (out / "completed.json").write_text(json.dumps(dict(completed=True,rows=len(rows),summaries=summaries),indent=2))
    print(json.dumps(dict(stage="completed",summaries=summaries)),flush=True)


if __name__=="__main__":
    main()
