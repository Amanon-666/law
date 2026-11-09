"""Matched calibration-prior development: prepare, audit, train, evaluate.

All stages use ordinary historical/current scan sets, never fabricated paired
time series. These are GP/conditional-kernel controls, not AcMu reproductions.
No unseen-site or localization experiment is performed by this driver.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from calibration_relation import CalibrationRelation
from data import load_period
from reference_pmc import ROOT, log_proposal, readout, sample_reference, setup
from run import prepare_source_batches
from run_change import prepare_validation
from support_integral import importance_loss
from transfer import Prior, Transfer


KINDS = ("uniform", "strength_rbf", "pointwise", "linear", "shape")
CODE = ("run_calibration_learning.py", "calibration_relation.py", "reference_pmc.py",
        "support_integral.py", "transfer.py", "inference.py", "data.py", "run.py", "run_change.py")
DATA = ROOT.parent / "Law-Continuous-Change-20261023/data/OfficeP2.npz"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def load_mean(setting):
    model = Transfer("affine").double()
    model.load_state_dict(torch.load(setting["mean_checkpoint"], weights_only=True))
    model.requires_grad_(False)
    return model


def fit_samples(mean, relation, case, base, obs, seed):
    # The sampler only sees H-derived prior and S. It has no Query interface.
    with torch.no_grad():
        prior = Prior(mean, case["history"], base, relation)
        sm, sf = prior.features(case["sx"])
        sk = prior.covariance(case["sx"], case["sx"], sf, sf)
        target, initial, center, root, modes = setup(sm, sk, case["stats"], obs)
        samples, logw, info = sample_reference(target, initial, center, root, seed)
        logq = log_proposal(samples, center, root, info)
    return samples, logq, logw, dict(modes=len(modes), ess=info["effective_samples"])


def score(mean, relation, case, base, obs, samples, logw):
    with torch.no_grad():
        prior = Prior(mean, case["history"], base, relation)
        sm, sf = prior.features(case["sx"])
        qm, qf = prior.features(case["qx"])
        sk = prior.covariance(case["sx"], case["sx"], sf, sf)
        cross = prior.covariance(case["qx"], case["sx"], qf, sf)
        qv = prior.diagonal(case["qx"], qf)
        ix = case["group"]
        result = readout(samples, logw, sm, sk, qm[ix], qv[ix], cross[ix], case["rss"], case["weights"], obs)
    return {key: result[key] for key in ("nll", "nll_mc_se")}


def prepare(out):
    out.mkdir(exist_ok=False)
    previous = json.loads((ROOT / "results/change_kernel_v1/freeze.json").read_text())
    setting = dict(role="Bounded shared calibration prior study; source Office only", started_unix=time.time(),
                   hypothesis="Historical neighborhood information improves calibration beyond shared global/local and point-value relations.",
                   base=previous["base"], obs=previous["obs"], mean_checkpoint=previous["mean_checkpoint"],
                   source_periods=list(range(1,12)), tasks=16, aps_per_task=4, episode_seed=231023,
                   kinds=KINDS, model_seed=231024, updates=80, hyper_lr=.02, encoder_lr=.002,
                   posterior="Fresh M-PMC fit for every case before every update; fixed-proposal density-ratio gradient",
                   sampler_seed=241024, final_samples=32768, adaptation_steps=6, adaptation_samples=4096,
                   selection="Initial and update 80 only; no validation checkpoint selection",
                   validation_period=12, validation_seeds=[231024,241024],
                   limitations="One training seed and one source environment; neither converged optimum nor cross-site evidence",
                   data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                   mean_sha256=hashlib.sha256(Path(previous["mean_checkpoint"]).read_bytes()).hexdigest(),
                   code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in CODE})
    write(out / "freeze.json", setting)
    (out / "code").mkdir()
    for name in CODE:
        shutil.copy2(ROOT/name, out/"code"/name)
    for kind in KINDS:
        torch.manual_seed(setting["model_seed"])
        model = CalibrationRelation(kind, setting["base"]["length"]).double()
        torch.save(model.state_dict(), out/f"{kind}_initial.pt")
    floors = {period:load_period(DATA,period) for period in setting["source_periods"]}
    batches = prepare_source_batches(floors,setting["base"],setting["obs"],16,4,160,setting["episode_seed"])
    torch.save(batches, out/"source_cases.pt")
    write(out/"source_manifest.json", [[case["episode"] for case in batch] for batch in batches])
    write(out/"prepared.json",dict(completed=True, cases=sum(map(len,batches))))


def audit(out, setting, batches, mean):
    base,obs=setting["base"],setting["obs"]
    rows=[]
    with (out/"audit.jsonl").open("x") as stream:
        for kind in KINDS:
            model=CalibrationRelation(kind,base["length"]).double()
            model.load_state_dict(torch.load(out/f"{kind}_initial.pt",weights_only=True))
            # All source cases under the ordinary calibration baseline; other
            # families' first actual training batch supplements prior checks.
            selected=batches if kind=="uniform" else batches[:1]
            for t,batch in enumerate(selected):
                for j,case in enumerate(batch):
                    samples,logq,logw,info=fit_samples(mean,model,case,base,obs,setting["sampler_seed"]+4*t+j)
                    value,posterior=importance_loss(Prior(mean,case["history"],base,model),case,obs,samples,logq)
                    gradients=torch.autograd.grad(value,list(model.parameters()))
                    norm=torch.sqrt(sum(g.square().sum() for g in gradients)).item()
                    if not np.isfinite(value.item()) or not np.isfinite(norm):
                        raise RuntimeError(f"Nonfinite audit objective/gradient: {kind}, {t}, {j}")
                    diagnostic=score(mean,model,case,base,obs,samples,logw)
                    row=dict(kind=kind,task=t,ap_index=j,episode=case["episode"],grad_norm=norm,**info,**diagnostic)
                    stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush();rows.append(row)
                print(json.dumps(dict(stage="audit",kind=kind,task=t,rows=len(rows))),flush=True)
    write(out/"audit_completed.json",dict(completed=True,rows=len(rows),
                                        max_function_mc_se=max(r["nll_mc_se"] for r in rows)))


def train(out, setting, batches, mean, kind):
    if not (out/"audit_completed.json").exists():
        raise RuntimeError("Source audit has not completed")
    base,obs=setting["base"],setting["obs"]
    model=CalibrationRelation(kind,base["length"]).double()
    model.load_state_dict(torch.load(out/f"{kind}_initial.pt",weights_only=True))
    groups=[dict(params=[p for n,p in model.named_parameters() if not n.startswith("encoder.")],lr=setting["hyper_lr"])]
    if hasattr(model,"encoder"):
        groups.append(dict(params=model.encoder.parameters(),lr=setting["encoder_lr"]))
    optimizer=torch.optim.Adam(groups)
    started=time.perf_counter()
    with (out/f"{kind}_training.jsonl").open("x") as stream:
        for step in range(setting["updates"]):
            batch=batches[step%len(batches)]; optimizer.zero_grad()
            values=[]; diagnostics=[]
            try:
                for j,case in enumerate(batch):
                    seed=setting["sampler_seed"]+4*step+j
                    samples,logq,_,info=fit_samples(mean,model,case,base,obs,seed)
                    value,_=importance_loss(Prior(mean,case["history"],base,model),case,obs,samples,logq)
                    (value/len(batch)).backward()
                    values.append(value.item());diagnostics.append(info)
                norm=torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())).item()
                if not np.isfinite(values).all() or not np.isfinite(norm):
                    raise RuntimeError("Nonfinite training objective/gradient")
                optimizer.step()
            except Exception as error:
                torch.save(dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict()),out/f"{kind}_failure.pt")
                write(out/f"{kind}_failure.json",dict(step=step,error=repr(error),task=step%len(batches)))
                raise
            row=dict(step=step,nll=float(np.mean(values)),grad_norm=norm,
                     local_fraction=torch.sigmoid(model.logit_local).item(),local_length=model.log_length.exp().item(),
                     min_ess=min(x["ess"] for x in diagnostics),seconds=time.perf_counter()-started)
            stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush()
            if step%10==0 or step==setting["updates"]-1:
                torch.save(dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict()),out/f"{kind}_latest.pt")
                print(json.dumps(dict(stage="train",kind=kind,**row)),flush=True)
    torch.save(model.state_dict(),out/f"{kind}.pt")
    write(out/f"{kind}_completed.json",dict(completed=True,updates=setting["updates"]))


def evaluate(out,setting,mean,kind):
    base,obs=setting["base"],setting["obs"]
    groups=prepare_validation({6:load_period(DATA,6)},DATA,base,obs)
    rows=[]
    with (out/f"{kind}_evaluation.jsonl").open("x") as stream,torch.no_grad():
        for checkpoint in ("initial","final"):
            model=CalibrationRelation(kind,base["length"]).double()
            path=out/(f"{kind}_initial.pt" if checkpoint=="initial" else f"{kind}.pt")
            model.load_state_dict(torch.load(path,weights_only=True))
            for group in groups:
                for ap,case in zip(group["ap_columns"],group["cases"]):
                    for seed in setting["validation_seeds"]:
                        samples,_,logw,info=fit_samples(mean,model,case,base,obs,seed)
                        row=dict(kind=kind,checkpoint=checkpoint,fraction=group["fraction"],k=group["k"],ap=ap,seed=seed,
                                 **info,**score(mean,model,case,base,obs,samples,logw))
                        stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush();rows.append(row)
                print(json.dumps(dict(stage="evaluate",kind=kind,checkpoint=checkpoint,fraction=group["fraction"],k=group["k"])),flush=True)
    write(out/f"{kind}_evaluation_completed.json",dict(completed=True,rows=len(rows)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","audit","train","evaluate"))
    parser.add_argument("--out",type=Path,default=ROOT/"results/calibration_learning_v1")
    parser.add_argument("--kind",choices=KINDS)
    args=parser.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    if args.stage=="prepare":
        prepare(args.out);return
    setting=json.loads((args.out/"freeze.json").read_text())
    for name,digest in setting["code_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Implementation changed after freeze: {name}")
    mean=load_mean(setting)
    if args.stage=="evaluate":
        evaluate(args.out,setting,mean,args.kind);return
    # Trusted cache produced locally by prepare(), containing History objects.
    batches=torch.load(args.out/"source_cases.pt",weights_only=False)
    if args.stage=="audit":
        audit(args.out,setting,batches,mean)
    else:
        train(args.out,setting,batches,mean,args.kind)


if __name__=="__main__":
    main()
