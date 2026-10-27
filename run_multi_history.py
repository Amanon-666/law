"""One bounded multi-history Office study; prepare -> audit -> train -> evaluate.

All outputs are new. Observation/mean/history priors are frozen; only the
existing three strength-kernel parameters receive 32 source updates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from calibration_relation import CalibrationRelation
from data import group_scans, load_period
from inference import observation_log_prob, predict_from_posterior
from multi_history import GaussianStrengthRelation, MultiHistory, collect_histories, single_history, source_calendar
from reference_pmc import readout
from run import ap_stats, tensor
from run_calibration_learning import fit_samples, load_mean
from support_integral import importance_loss
from transfer import Prior


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "Law-Continuous-Change-20261023/data/OfficeP2.npz"
OLD = ROOT / "results/calibration_learning_v1"
CODE = ("multi_history.py", "run_multi_history.py", "check_multi_history.py",
        "data.py", "inference.py", "transfer.py", "calibration_relation.py",
        "support_integral.py", "reference_pmc.py", "run_calibration_learning.py", "run.py")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(**row):
    print(json.dumps(row, allow_nan=False), flush=True)


def relation(setting, path, legacy=False):
    cls = CalibrationRelation if legacy else GaussianStrengthRelation
    model = cls("strength_rbf", setting["base"]["length"]).double()
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def take_positions(floor, ids):
    rows = np.flatnonzero(np.isin(floor["group"], ids))
    return rows, group_scans(floor["rss"][rows], floor["xy"][rows])


def build_event(floors, current_period, history_periods, seed, aps_count, setting, validation=False):
    rng = np.random.default_rng(seed)
    selections = []
    for period in history_periods:
        ids = np.sort(rng.choice(len(floors[period]["positions"]), 25, replace=False))
        rows, _ = take_positions(floors[period], ids)
        selections.append((period, rows))
    histories = collect_histories(floors, selections)
    current = floors[current_period]
    order = rng.permutation(len(current["positions"]))
    s_ids = np.sort(order[:5])
    s_rows, support = take_positions(current, s_ids)
    if validation:
        query_all = load_period(DATA, 12, "testing")
        overlap = (query_all["xy"][:, None] == support["positions"][None]).all(-1).any(-1)
        q_rows = np.flatnonzero(~overlap)
        query = group_scans(query_all["rss"][q_rows], query_all["xy"][q_rows])
    else:
        q_rows, query = take_positions(current, np.sort(order[5:17]))
    query["scan_weights"] = 1 / (len(query["positions"]) * query["stats"]["n"][query["group"]])
    # Same AP rights for every arm; only newest observed history chooses columns.
    latest = histories[-1]
    eligible = np.flatnonzero((latest["scans"]["stats"]["k"] > 0).any(0))
    aps = np.sort(rng.choice(eligible, aps_count, replace=False))
    cases = []
    for ap in aps:
        multi = MultiHistory(histories, int(ap), setting["base"], setting["obs"], setting["calendar"])
        single = single_history(latest, int(ap), setting["base"], setting["obs"])
        cases.append(dict(history=multi, single=single, ap=int(ap), sx=tensor(support["positions"]),
                          qx=tensor(query["positions"]), stats=ap_stats(support, int(ap)),
                          rss=tensor(query["rss"][:, ap]), group=query["group"],
                          weights=tensor(query["scan_weights"])))
        log(stage="history_prepared", current=current_period, seed=seed, ap=int(ap))
    manifest = dict(current=current_period, history_periods=history_periods, seed=seed,
                    historical_rows={str(r["period"]): r["row_ids"].tolist() for r in histories},
                    historical_positions={str(r["period"]): r["scans"]["positions"].tolist() for r in histories},
                    historical_scans={str(r["period"]): len(r["row_ids"]) for r in histories},
                    support_rows=s_rows.tolist(), support_positions=support["positions"].tolist(),
                    support_scans=len(s_rows), query_rows=q_rows.tolist(), query_scans=len(q_rows),
                    query_positions=len(query["positions"]), ap_columns=aps.tolist(),
                    query_split="testing12, S-coordinate excluded" if validation else "training, S-position disjoint")
    return dict(manifest=manifest, cases=cases, support=support, query=query)


def prepare(out):
    out.mkdir(exist_ok=False)
    old = json.loads((OLD / "freeze.json").read_text())
    setting = dict(scope="Office source development only; selected AP subset", started_unix=time.time(),
                   base=old["base"], obs=old["obs"], mean_checkpoint=old["mean_checkpoint"],
                   calendar=source_calendar(DATA, old["base"]), k=5, history_positions_per_period=25,
                   source_events=[5, 7, 9, 11], historical_offsets=[-4, -2, -1],
                   train_aps_per_event=4, train_query_positions=12, training_seed=251025,
                   validation_period=12, validation_histories=[8, 10, 11],
                   validation_seeds=[0, 1], validation_aps=8,
                   updates=32, optimizer="Adam", learning_rate=.02,
                   trainable="existing CalibrationRelation strength_rbf: 3 parameters",
                   selection="fixed final step, no validation model selection",
                   training_budget="128 AP-case visits, below prior 80*4=320 per model",
                   target_network_updates=0,
                   historical_inference="existing joint Gaussian VI, maxiter160; no H gradients",
                   current_inference="existing M-PMC:6*4096 adaptation,32768 final",
                   location_readout="existing Gaussian moment / sum AP log likelihood / grid posterior mean",
                   area="existing Office period1 surveyed bounding rectangle; shared and disclosed",
                   source_files={str(p):digest(p) for p in [DATA,OLD/"strength_rbf.pt",Path(old["mean_checkpoint"])]},
                   code_sha256={name:digest(ROOT/name) for name in CODE},
                   design_sha256=digest(ROOT/"MULTI_HISTORY_DESIGN.md"))
    write(out / "freeze.json", setting)
    (out / "code").mkdir()
    for name in CODE:
        shutil.copy2(ROOT/name, out/"code"/name)
    shutil.copy2(ROOT/"MULTI_HISTORY_DESIGN.md", out/"design.md")
    shutil.copy2(OLD/"strength_rbf.pt", out/"initial.pt")
    floors = {t: load_period(DATA, t) for t in range(1, 13)}
    training = [build_event(floors, t, [t-4, t-2, t-1], setting["training_seed"]+t, 4, setting)
                for t in setting["source_events"]]
    validation = [build_event(floors, 12, [8,10,11], seed, 8, setting, True) for seed in (0,1)]
    bounds = floors[1]["positions"]
    spacing = min(1., setting["base"]["length"] / 3)
    axes = [np.linspace(bounds[:,d].min(), bounds[:,d].max(), int(np.ceil(np.ptp(bounds[:,d])/spacing))+1)
            for d in range(2)]
    grid = tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1,2))
    torch.save(dict(training=training, validation=validation, grid=grid), out/"cases.pt")
    write(out/"manifest.json",dict(training=[e["manifest"] for e in training],
                                   validation=[e["manifest"] for e in validation],
                                   grid_bounds=[bounds.min(0).tolist(),bounds.max(0).tolist()],grid_points=len(grid)))
    write(out/"prepared.json",dict(completed=True,cases_sha256=digest(out/"cases.pt")))


def audit(out, setting, cache, mean):
    model = relation(setting, out/"initial.pt")
    case = cache["training"][0]["cases"][0]
    samples, logq, _, info = fit_samples(mean, model, case, setting["base"], setting["obs"], 251025)
    loss, post = importance_loss(Prior(mean,case["history"],setting["base"],model),case,
                                 setting["obs"],samples,logq)
    params = list(model.parameters())
    grads = torch.autograd.grad(loss,params)
    finite=[]
    for param, grad in zip(params,grads):
        old=param.detach().clone(); values=[]
        for sign in (1,-1):
            with torch.no_grad(): param.copy_(old+sign*1e-4)
            value,_=importance_loss(Prior(mean,case["history"],setting["base"],model),case,
                                     setting["obs"],samples,logq)
            values.append(value.item())
        with torch.no_grad(): param.copy_(old)
        fd=(values[0]-values[1])/2e-4
        if abs(fd-grad.item())>2e-5: raise RuntimeError("conditional gradient finite difference mismatch")
        finite.append(dict(autograd=grad.item(),finite_difference=fd))
    # The deployment prior contains H only; neither S labels nor Q appear in it.
    prior=Prior(mean,case["history"],setting["base"],model)
    x=torch.cat((case["sx"],case["qx"]))
    m,f=prior.features(x); cov=prior.covariance(x,x,f,f)
    torch.linalg.cholesky(cov+1e-6*torch.eye(len(x)))
    write(out/"audit_completed.json",dict(completed=True,source_case=0,nll=loss.item(),
          finite_difference=finite,min_cov_eigenvalue=torch.linalg.eigvalsh(cov).min().item(),**info))
    log(stage="audit_passed",loss=loss.item())


def train(out, setting, cache, mean):
    if not (out/"audit_completed.json").exists(): raise RuntimeError("audit must precede training")
    model=relation(setting,out/"initial.pt")
    optimizer=torch.optim.Adam(model.parameters(),lr=setting["learning_rate"])
    started=time.perf_counter()
    with (out/"training.jsonl").open("x") as stream:
        for step in range(setting["updates"]):
            event=cache["training"][step%len(cache["training"])]
            optimizer.zero_grad(); values=[]; ess=[]
            for j,case in enumerate(event["cases"]):
                samples,logq,_,info=fit_samples(mean,model,case,setting["base"],setting["obs"],251025+4*step+j)
                value,_=importance_loss(Prior(mean,case["history"],setting["base"],model),case,
                                       setting["obs"],samples,logq)
                (value/len(event["cases"])).backward();values.append(value.item());ess.append(info["ess"])
            if not np.isfinite(values).all() or not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("nonfinite conditional objective or gradient")
            optimizer.step()
            row=dict(step=step,current=event["manifest"]["current"],nll=float(np.mean(values)),
                     min_ess=min(ess),seconds=time.perf_counter()-started,
                     parameters={k:v.item() for k,v in model.state_dict().items()})
            stream.write(json.dumps(row)+"\n");stream.flush()
            if step%4==0 or step==setting["updates"]-1: log(stage="training",**row)
    torch.save(model.state_dict(),out/"trained.pt")
    write(out/"training_completed.json",dict(completed=True,updates=setting["updates"],
         parameters={k:v.item() for k,v in model.state_dict().items()},seconds=time.perf_counter()-started))


def evaluate(out, setting, cache, mean):
    models={"single_legacy":relation(setting,out/"initial.pt",True),
            "single":relation(setting,out/"initial.pt"),"multi_frozen":relation(setting,out/"initial.pt"),
            "multi_learned":relation(setting,out/"trained.pt")}
    base,obs,grid=setting["base"],setting["obs"],cache["grid"]
    rows=[]; summaries=[]
    with (out/"evaluation.jsonl").open("x") as stream, torch.no_grad():
        for e,event in enumerate(cache["validation"]):
            query,support=event["query"],event["support"]
            scores={name:torch.zeros((len(query["rss"]),len(grid))) for name in models}
            saved={}; nlls={name:[] for name in models}
            for case in event["cases"]:
                for name,model in models.items():
                    used=dict(case,history=case["single"]) if name.startswith("single") else case
                    samples,_,logw,info=fit_samples(mean,model,used,base,obs,251025+e)
                    prior=Prior(mean,used["history"],base,model)
                    sm,sf=prior.features(used["sx"]); sk=prior.covariance(used["sx"],used["sx"],sf,sf)
                    qm,qf=prior.features(used["qx"]); qv=prior.diagonal(used["qx"],qf)
                    cross=prior.covariance(used["qx"],used["sx"],qf,sf)
                    group=used["group"]
                    diag=readout(samples,logw,sm,sk,qm[group],qv[group],cross[group],used["rss"],used["weights"],obs)
                    nlls[name].append(diag["nll"])
                    w=logw.exp(); pm=w@samples; centered=samples-pm
                    posterior=dict(mean=pm,cov=(centered.T*w)@centered)
                    points=torch.cat((grid,used["qx"]))
                    gm,gf=prior.features(points); gv=prior.diagonal(points,gf)
                    gc=prior.covariance(points,used["sx"],gf,sf)
                    prediction=predict_from_posterior(gm,gv,gc,sm,sk,posterior)
                    scores[name]+=observation_log_prob(used["rss"][:,None],prediction["mean"][:len(grid)][None],
                                                       prediction["variance"][:len(grid)][None],obs)
                    hmean=gm[len(grid):]; hvar=gv[len(grid):]
                    predmean=prediction["mean"][len(grid):]; predvar=prediction["variance"][len(grid):]
                    prefix=f"{name}_ap{case['ap']}"
                    for key,value in dict(prior_mean=hmean,prior_variance=hvar,mean=predmean,variance=predvar,
                                          scan_nll=tensor(diag["per_observation_nll"])).items():
                        saved[f"{prefix}_{key}"]=value.numpy()
                    row=dict(seed=e,kind=name,ap=case["ap"],nll=diag["nll"],nll_mc_se=diag["nll_mc_se"],**info)
                    rows.append(row);stream.write(json.dumps(row)+"\n");stream.flush()
                log(stage="evaluation_ap",seed=e,ap=case["ap"])
            for name in models:
                pred=(torch.softmax(scores[name],1)@grid).numpy()
                error=np.linalg.norm(pred-query["xy"],axis=1)
                saved[f"{name}_xy"]=pred
                row=dict(seed=e,kind=name,nll=float(np.mean(nlls[name])),
                         mde=float(query["scan_weights"]@error),median=float(np.median(error)),
                         p90=float(np.quantile(error,.9)),ap_count=len(event["cases"]))
                summaries.append(row);log(stage="result",**row)
            aps=event["manifest"]["ap_columns"]
            s=support["stats"]
            fingerprint=(s["s1"][:,aps]-110*(s["n"][:,None]-s["k"][:,aps]))/s["n"][:,None]
            qr=np.where(query["rss"][:,aps]==100,-110,query["rss"][:,aps])
            distance=np.sqrt(((qr[:,None]-fingerprint[None])**2).mean(-1))
            nearest=np.argsort(distance,axis=1)[:,:3]
            weights=1/np.maximum(np.take_along_axis(distance,nearest,axis=1),1e-8)
            pred=(support["positions"][nearest]*weights[:,:,None]).sum(1)/weights.sum(1)[:,None]
            error=np.linalg.norm(pred-query["xy"],axis=1);saved["wknn_xy"]=pred
            summaries.append(dict(seed=e,kind="wknn",mde=float(query["scan_weights"]@error),
                                  median=float(np.median(error)),p90=float(np.quantile(error,.9)),ap_count=len(aps)))
            np.savez_compressed(out/f"predictions_seed{e}.npz",**saved,query_xy=query["xy"],
                                query_positions=query["positions"],query_group=query["group"],
                                support_xy=support["positions"],ap_columns=aps,
                                query_rss=query["rss"][:,aps])
    write(out/"summary.json",dict(completed=True,scope=setting["scope"],results=summaries,
           evaluation_rows=len(rows),training_updates=setting["updates"],
           caveat="8 selected APs per draw, 2 calibration draws, one Office development period; Gaussian moment localization"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","audit","train","evaluate"))
    parser.add_argument("--out",type=Path,default=ROOT/"results/multi_history_v1")
    args=parser.parse_args();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    if args.stage=="prepare": prepare(args.out);return
    setting=json.loads((args.out/"freeze.json").read_text())
    for name,sha in setting["code_sha256"].items():
        if digest(ROOT/name)!=sha: raise RuntimeError(f"code changed after freeze: {name}")
    if digest(args.out/"cases.pt")!=json.loads((args.out/"prepared.json").read_text())["cases_sha256"]:
        raise RuntimeError("prepared cases changed")
    cache=torch.load(args.out/"cases.pt",weights_only=False)  # locally generated trusted History cache
    mean=load_mean(setting)
    {"audit":audit,"train":train,"evaluate":evaluate}[args.stage](args.out,setting,cache,mean)


if __name__=="__main__":
    main()
