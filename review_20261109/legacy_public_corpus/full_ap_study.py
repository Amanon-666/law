"""Frozen all-permitted-AP Office comparison; no learning in phases A/B.

Prepare immutable event manifests, run each independently, summarize completed
events. Per-AP checkpoints allow completion after interruption without redoing
successful inference. A failed AP is never omitted from a method's ranking.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import inference
from data import group_scans, load_period
from inference import observation_log_prob, predict_from_posterior
from multi_history import MultiHistory, collect_histories, single_history
from reference_pmc import readout
from run import ap_stats, tensor
from run_calibration_learning import fit_samples, load_mean
from run_multi_history import DATA, digest, relation, take_positions, write
from transfer import Prior


ROOT=Path(__file__).resolve().parent
V1=ROOT/"results/multi_history_v1"
METHODS=("single","pooled","multi")


def log(**data):
    print(json.dumps(data,allow_nan=False),flush=True)


def event_data(meta):
    periods=meta["history_periods"]+[meta["current"]]
    floors={t:load_period(DATA,t) for t in periods}
    records=collect_histories(floors,[(int(t),rows) for t,rows in meta["historical_rows"].items()])
    current=floors[meta["current"]];rows=np.array(meta["support_rows"])
    support=group_scans(current["rss"][rows],current["xy"][rows])
    query_all=load_period(DATA,meta["current"],"testing");rows=np.array(meta["query_rows"])
    query=group_scans(query_all["rss"][rows],query_all["xy"][rows])
    query["scan_weights"]=1/(len(query["positions"])*query["stats"]["n"][query["group"]])
    return records,support,query


def prepare(out):
    out.mkdir(exist_ok=False)
    old=json.loads((V1/"freeze.json").read_text())
    setting={k:old[k] for k in ("base","obs","mean_checkpoint","calendar")}
    setting.update(scope="Office full-permitted-AP source development, zero new learning",
        current_periods=[8,10,12],support_draws=[0,1],k=5,scans_per_position=7,
        methods=list(METHODS)+["wknn_raw","wknn_position_mean"],
        ap_rule="union detected in selected histories or Support; never Query-selected",
        source_information="8/10 used in earlier source fitting, 12 repeatedly viewed development",
        history_solver=dict(maxiter=160,diagnostics="actual SciPy status/jac captured"),
        support_solver="unchanged M-PMC 6x4096 adaptation + 32768 final",
        location="unchanged Gaussian posterior moments, summed AP log score, grid posterior expectation",
        coordinate_unit="Office raw coordinate distance; physical unit unverified",
        failure_threshold_raw_units=10.,wknn_neighbours=3,
        parameter_updates=0,created_unix=time.time(),
        code_sha256={p.name:digest(p) for p in ROOT.glob("*.py")},
        inputs={str(p):digest(p) for p in [DATA,V1/"initial.pt",Path(old["mean_checkpoint"]),V1/"manifest.json"]},
        protocol_sha256=digest(ROOT/"FULL_AP_PROTOCOL.md"),
        environment={p:importlib.metadata.version(p) for p in ["torch","numpy","scipy","pypmc"]})
    write(out/"freeze.json",setting)
    (out/"code").mkdir()
    for p in ROOT.glob("*.py"):shutil.copy2(p,out/"code"/p.name)
    shutil.copy2(ROOT/"FULL_AP_PROTOCOL.md",out/"protocol.md")
    shutil.copy2(V1/"initial.pt",out/"initial.pt")
    floors={t:load_period(DATA,t) for t in range(1,13)}
    old_manifest=json.loads((V1/"manifest.json").read_text())
    events=[]
    for current in [8,10,12]:
        for seed in [0,1]:
            history_periods=[current-4,current-2,current-1]
            if current==12:
                old_event=old_manifest["validation"][seed]
                historical_rows=old_event["historical_rows"]
                support_rows=old_event["support_rows"]
                query_rows=old_event["query_rows"]
            else:
                rng=np.random.default_rng(seed);historical_rows={}
                for period in history_periods:
                    ids=np.sort(rng.choice(len(floors[period]["positions"]),25,replace=False))
                    rows,_=take_positions(floors[period],ids);historical_rows[str(period)]=rows.tolist()
                ids=np.sort(rng.permutation(len(floors[current]["positions"]))[:5])
                rows,support=take_positions(floors[current],ids);support_rows=rows.tolist()
                q=load_period(DATA,current,"testing")
                overlap=(q["xy"][:,None]==support["positions"][None]).all(-1).any(-1)
                query_rows=np.flatnonzero(~overlap).tolist()
            meta=dict(id=f"p{current}_s{seed}",current=current,seed=seed,
                      history_periods=history_periods,historical_rows=historical_rows,
                      support_rows=support_rows,query_rows=query_rows)
            histories,support,query=event_data(meta)
            h_seen=np.stack([(r["scans"]["stats"]["k"]>0).any(0) for r in histories])
            s_seen=(support["stats"]["k"]>0).any(0)
            aps=np.flatnonzero(h_seen.any(0)|s_seen)
            meta.update(ap_columns=aps.tolist(),ap_count=len(aps),
                ap_seen_latest=h_seen[-1,aps].tolist(),ap_seen_any_history=h_seen[:,aps].any(0).tolist(),
                ap_seen_support=s_seen[aps].tolist(),
                history_scans={str(r["period"]):len(r["row_ids"]) for r in histories},
                support_scans=len(support_rows),query_scans=len(query_rows),
                support_positions=support["positions"].tolist(),
                query_all_missing_count=int((query["rss"][:,aps]==100).all(1).sum()),
                query_detected_ap_counts=(query["rss"][:,aps]!=100).sum(1).tolist())
            # Structural input isolation, not a statistical test or seed search.
            assert not (query["xy"][:,None]==support["positions"][None]).all(-1).any()
            if current==12:
                assert historical_rows==old_event["historical_rows"] and support_rows==old_event["support_rows"]
            events.append(meta)
    bounds=floors[1]["positions"];spacing=min(1.,setting["base"]["length"]/3)
    axes=[np.linspace(bounds[:,d].min(),bounds[:,d].max(),int(np.ceil(np.ptp(bounds[:,d])/spacing))+1) for d in range(2)]
    grid=np.stack(np.meshgrid(*axes),-1).reshape(-1,2)
    np.save(out/"grid.npy",grid)
    write(out/"manifest.json",dict(events=events,grid_source="period1 source survey bounds",
                                   grid_bounds=[bounds.min(0).tolist(),bounds.max(0).tolist()],grid_points=len(grid)))
    write(out/"prepared.json",dict(completed=True,manifest_sha256=digest(out/"manifest.json"),grid_sha256=digest(out/"grid.npy")))
    log(stage="prepared",events=[dict(id=e["id"],aps=e["ap_count"],queries=e["query_scans"],all_missing=e["query_all_missing_count"]) for e in events])


def metrics(pred,query):
    error=np.linalg.norm(pred-query["xy"],axis=1)
    return dict(mde=float(query["scan_weights"]@error),median=float(np.median(error)),
                p90=float(np.quantile(error,.9)),p95=float(np.quantile(error,.95)),
                max_error=float(error.max()),fraction_error_gt10=float((error>10).mean()),
                nonfinite_predictions=int((~np.isfinite(pred).all(1)).sum()))


def wknn(support,query,aps,mean_by_position):
    if mean_by_position:
        s=support["stats"]
        refs=(s["s1"][:,aps]-110*(s["n"][:,None]-s["k"][:,aps]))/s["n"][:,None]
        xy=support["positions"]
    else:
        refs=np.where(support["rss"][:,aps]==100,-110,support["rss"][:,aps])
        xy=support["xy"]
    rss=np.where(query["rss"][:,aps]==100,-110,query["rss"][:,aps])
    distance=np.sqrt(((rss[:,None]-refs[None])**2).mean(-1))
    nearest=np.argsort(distance,axis=1)[:,:3]
    weights=1/np.maximum(np.take_along_axis(distance,nearest,axis=1),1e-8)
    return (xy[nearest]*weights[:,:,None]).sum(1)/weights.sum(1)[:,None]


def check_new_paths():
    # One case that would fail if raw-scan WKNN silently averaged by position.
    support=group_scans(np.array([[-50],[-50],[-100],[-70],[-70],[-70]]),
                        np.array([[0.,0.]]*3+[[10.,0.]]*3))
    query=group_scans(np.array([[-50]]),np.array([[0.,0.]]))
    raw=wknn(support,query,[0],False);averaged=wknn(support,query,[0],True)
    assert raw[0,0]<1e-6 and averaged[0,0]>4
    # Pooled ordinary likelihood aggregates real rows, not mean-of-map densities.
    pooled=group_scans(np.array([[-50],[-70],[-70]]),np.array([[0.,0.]]*3))
    assert pooled["stats"]["n"][0]==3 and pooled["stats"]["s1"][0,0]==-190
    log(stage="new_paths_checked",raw=raw.tolist(),position_mean=averaged.tolist())


def run_event(out,setting,meta):
    destination=out/meta["id"];destination.mkdir(exist_ok=True)
    if (destination/"completed.json").exists():
        log(stage="already_completed",event=meta["id"]);return
    histories,support,query=event_data(meta)
    # The stationary pooled comparator uses EXACTLY the same unique acquisitions.
    pooled_scans=group_scans(np.concatenate([r["scans"]["rss"] for r in histories]),
                            np.concatenate([r["scans"]["xy"] for r in histories]))
    pooled_record=dict(scans=pooled_scans)
    model=relation(setting,out/"initial.pt");mean=load_mean(setting)
    base,obs=setting["base"],setting["obs"]
    grid=tensor(np.load(out/"grid.npy"));points=torch.cat((grid,tensor(query["positions"])))
    real_minimize=inference.minimize;current_diagnostics=[]
    def instrument(fun,x0,*args,**kwargs):
        result=real_minimize(fun,x0,*args,**kwargs)
        current_diagnostics.append(dict(success=bool(result.success),status=int(result.status),
            message=str(result.message),nit=int(result.nit),variables=len(result.x),
            gradient_max=float(np.max(np.abs(result.jac)))))
        return result
    inference.minimize=instrument
    started=time.perf_counter()
    for index,ap in enumerate(meta["ap_columns"]):
        target=destination/f"ap{ap}.npz";record=destination/f"ap{ap}.json"
        if target.exists() and record.exists():continue
        saved={};rows=[]
        try:
            for name in METHODS:
                current_diagnostics.clear();begin=time.perf_counter()
                limit=setting["history_solver"]["maxiter"]
                if name=="single":history=single_history(histories[-1],ap,base,obs,steps=limit)
                elif name=="pooled":history=single_history(pooled_record,ap,base,obs,steps=limit)
                else:history=MultiHistory(histories,ap,base,obs,setting["calendar"],steps=limit)
                history_seconds=time.perf_counter()-begin
                historical_solver=list(current_diagnostics)
                if any(not r["success"] for r in historical_solver):
                    raise RuntimeError(f"historical VI did not converge: {historical_solver}")
                case=dict(history=history,sx=tensor(support["positions"]),qx=tensor(query["positions"]),
                          stats=ap_stats(support,ap),rss=tensor(query["rss"][:,ap]),
                          group=query["group"],weights=tensor(query["scan_weights"]))
                begin=time.perf_counter()
                with torch.no_grad():
                    samples,_,logw,info=fit_samples(mean,model,case,base,obs,251025+meta["seed"])
                    support_seconds=time.perf_counter()-begin;begin=time.perf_counter()
                    prior=Prior(mean,history,base,model)
                    sm,sf=prior.features(case["sx"]);sk=prior.covariance(case["sx"],case["sx"],sf,sf)
                    gm,gf=prior.features(points);gv=prior.diagonal(points,gf)
                    gc=prior.covariance(points,case["sx"],gf,sf)
                    w=logw.exp();pm=w@samples;centered=samples-pm
                    posterior=dict(mean=pm,cov=(centered.T*w)@centered)
                    pred=predict_from_posterior(gm,gv,gc,sm,sk,posterior)
                    if not torch.isfinite(pred["mean"]).all() or not torch.isfinite(pred["variance"]).all() or (pred["variance"]<0).any():
                        raise RuntimeError("invalid predicted marginal")
                    g=case["group"];sl=slice(len(grid),None)
                    diag=readout(samples,logw,sm,sk,gm[sl][g],gv[sl][g],gc[sl][g],case["rss"],case["weights"],obs)
                    for key,value in dict(grid_mean=pred["mean"][:len(grid)],grid_variance=pred["variance"][:len(grid)],
                          query_mean=pred["mean"][sl],query_variance=pred["variance"][sl],
                          prior_query_mean=gm[sl],prior_query_variance=gv[sl],scan_nll=tensor(diag["per_observation_nll"])).items():
                        saved[f"{name}_{key}"]=value.numpy()
                rows.append(dict(method=name,ap=ap,history_seconds=history_seconds,support_seconds=support_seconds,
                                 readout_seconds=time.perf_counter()-begin,history_solver=historical_solver,
                                 nll=diag["nll"],nll_mc_se=diag["nll_mc_se"],**info))
            np.savez_compressed(target,**saved);write(record,dict(ap=ap,methods=rows))
        except Exception as error:
            write(destination/"failure.json",dict(ap=ap,index=index,error=repr(error),history_solver=current_diagnostics,
                                                 completed_aps=len(list(destination.glob("ap*.json")))))
            raise
        if index%10==0 or index==len(meta["ap_columns"])-1:
            log(stage="ap_completed",event=meta["id"],ap_index=index+1,total=meta["ap_columns"].__len__(),elapsed=time.perf_counter()-started)
    inference.minimize=real_minimize
    finish_event(out,setting,meta,support,query,destination,started)


def finish_event(out,setting,meta,support,query,destination,started):
    grid=tensor(np.load(out/"grid.npy"));obs=setting["obs"]
    scores={name:torch.zeros((len(query["rss"]),len(grid))) for name in METHODS}
    aggregate={name:[] for name in METHODS};saved={};summary=[]
    score_seconds={name:0. for name in METHODS}
    with torch.no_grad():
        for ap in meta["ap_columns"]:
            records=json.loads((destination/f"ap{ap}.json").read_text())["methods"]
            for row in records:aggregate[row["method"]].append(row)
            with np.load(destination/f"ap{ap}.npz") as arrays:
                for name in METHODS:
                    now=time.perf_counter()
                    scores[name]+=observation_log_prob(tensor(query["rss"][:,ap])[:,None],
                          tensor(arrays[f"{name}_grid_mean"])[None],tensor(arrays[f"{name}_grid_variance"])[None],obs)
                    score_seconds[name]+=time.perf_counter()-now
        for name in METHODS:
            if not torch.isfinite(scores[name]).all():raise RuntimeError("nonfinite localization score")
            pred=(torch.softmax(scores[name],1)@grid).numpy()
            saved[f"{name}_xy"]=pred;saved[f"{name}_scores"]=scores[name].numpy()
            rows=aggregate[name]
            summary.append(dict(method=name,**metrics(pred,query),nll=float(np.mean([r["nll"] for r in rows])),
                 offline_history_seconds=sum(r["history_seconds"] for r in rows),
                 support_adaptation_seconds=sum(r["support_seconds"] for r in rows),
                 predictive_readout_seconds=sum(r["readout_seconds"] for r in rows),score_seconds=score_seconds[name],
                 min_effective_samples=min(r["ess"] for r in rows)))
    for name,avg in [("wknn_raw",False),("wknn_position_mean",True)]:
        now=time.perf_counter();pred=wknn(support,query,meta["ap_columns"],avg)
        saved[f"{name}_xy"]=pred
        summary.append(dict(method=name,**metrics(pred,query),seconds=time.perf_counter()-now))
    np.savez_compressed(destination/"predictions.npz",**saved,query_xy=query["xy"],query_group=query["group"],
        query_rss=query["rss"][:,meta["ap_columns"]],query_positions=query["positions"],
        support_xy=support["positions"],support_scan_xy=support["xy"],support_rss=support["rss"][:,meta["ap_columns"]],
        grid=grid.numpy(),ap_columns=meta["ap_columns"])
    write(destination/"completed.json",dict(completed=True,event=meta["id"],current=meta["current"],seed=meta["seed"],
          ap_count=meta["ap_count"],all_missing_query_fraction=meta["query_all_missing_count"]/meta["query_scans"],
          query_outside_grid=int(((query["xy"]<grid.numpy().min(0))|(query["xy"]>grid.numpy().max(0))).any(1).sum()),
          results=summary,elapsed_since_resume=time.perf_counter()-started,
          predictions_sha256=digest(destination/"predictions.npz")))
    log(stage="event_completed",event=meta["id"],results=summary)


def summarize(out,setting,manifest):
    results=[]
    for meta in manifest["events"]:
        p=out/meta["id"]/"completed.json"
        if not p.exists():raise RuntimeError(f"event incomplete: {meta['id']}")
        result=json.loads(p.read_text());results.append(result)
    means={name:{key:float(np.mean([next(r for r in e["results"] if r["method"]==name)[key] for e in results]))
                 for key in ["mde","p90","fraction_error_gt10"]} for name in setting["methods"]}
    write(out/"summary.json",dict(completed=True,events=results,equal_event_means=means,
          coordinate_unit=setting["coordinate_unit"],parameter_updates=0,
          scope="source development; same-data pooled vs temporal multi; all permitted APs"))
    log(stage="all_completed",means=means)


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["check","prepare","event","summarize"])
    p.add_argument("--event");p.add_argument("--out",type=Path,default=ROOT/"results/full_ap_v1")
    args=p.parse_args();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    if args.stage=="check":check_new_paths();return
    if args.stage=="prepare":prepare(args.out);return
    setting=json.loads((args.out/"freeze.json").read_text());manifest=json.loads((args.out/"manifest.json").read_text())
    amendment=args.out/"numerical_amendment.json"
    if amendment.exists():
        revised=json.loads(amendment.read_text())
        setting["history_solver"]["maxiter"]=revised["history_maxiter"]
        setting["code_sha256"]["full_ap_study.py"]=revised["runner_sha256"]
    frozen=json.loads((args.out/"prepared.json").read_text())
    for name,sha in setting["code_sha256"].items():
        if digest(ROOT/name)!=sha:raise RuntimeError(f"frozen code changed: {name}")
    if digest(args.out/"manifest.json")!=frozen["manifest_sha256"]:raise RuntimeError("manifest changed")
    if args.stage=="event":run_event(args.out,setting,next(e for e in manifest["events"] if e["id"]==args.event))
    else:summarize(args.out,setting,manifest)


if __name__=="__main__":main()
