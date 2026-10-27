"""Post-run numerical audit of frozen historical VI, no shared training.

Replay the 64 historical solves at their original 160-iteration limit and
capture SciPy's actual status/gradient. Then check at most four cases chosen
by solver residual (one per split/arm), at 640 iterations on the SAME model.
Old cached histories, weights and performance results are never overwritten.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import inference
from data import load_period
from multi_history import MultiHistory, collect_histories, single_history


ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/multi_history_v1"


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    setting=json.loads((OUT/"freeze.json").read_text())
    for name,expected in setting["code_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f"frozen implementation changed: {name}")
    cache=torch.load(OUT/"cases.pt",weights_only=False)
    manifest=json.loads((OUT/"manifest.json").read_text())
    floors={t:load_period(ROOT.parent/"Law-Continuous-Change-20261023/data/OfficeP2.npz",t)
            for t in range(1,12)}
    real_minimize=inference.minimize
    captured=[]

    def measured_minimize(fun,x0,*args,**kwargs):
        solved=real_minimize(fun,x0,*args,**kwargs)
        # Objective uses full scan likelihood, so retain both raw and per-scan residuals.
        projected=solved.jac.copy()
        for j,(lo,hi) in enumerate(kwargs["bounds"]):
            if lo is not None and solved.x[j]<=lo+1e-12 and projected[j]>0:projected[j]=0
            if hi is not None and solved.x[j]>=hi-1e-12 and projected[j]<0:projected[j]=0
        captured.append(dict(success=bool(solved.success),status=int(solved.status),message=str(solved.message),
             nit=int(solved.nit),nfev=int(solved.nfev),limit=kwargs["options"]["maxiter"],
             objective=float(solved.fun),raw_gradient_max=float(np.max(np.abs(solved.jac))),
             projected_gradient_max=float(np.max(np.abs(projected))),
             optimizer_variables=len(solved.x)))
        return solved

    inference.minimize=measured_minimize
    auditdir=OUT/"numerical_audit"
    auditdir.mkdir(exist_ok=False)
    records=[]; lookup={}
    with (auditdir/"replay.jsonl").open("x") as stream:
        for split in ["training","validation"]:
            for event_id,(event,meta) in enumerate(zip(cache[split],manifest[split])):
                sources=collect_histories(floors,[(int(t),rows) for t,rows in meta["historical_rows"].items()])
                for case in event["cases"]:
                    for arm in ["single","multi"]:
                        def build(steps,arm=arm,ap=case["ap"],sources=sources):
                            if arm=="single":return single_history(sources[-1],ap,setting["base"],setting["obs"],steps)
                            return MultiHistory(sources,ap,setting["base"],setting["obs"],setting["calendar"],steps)
                        old=case["single"] if arm=="single" else case["history"]
                        new=build(160)
                        scans=len(sources[-1]["row_ids"]) if arm=="single" else sum(len(r["row_ids"]) for r in sources)
                        row=dict(split=split,event=event_id,current=meta["current"],ap=case["ap"],arm=arm,
                             scans=scans,**captured[-1],
                             max_cached_mean_difference=(old.m-new.m).abs().max().item(),
                             max_cached_cov_difference=(old.c-new.c).abs().max().item())
                        row["gradient_per_scan"]=row["projected_gradient_max"]/scans
                        records.append(row);lookup[split,event_id,case["ap"],arm]=(build,new,case)
                        stream.write(json.dumps(row)+"\n");stream.flush()
                print(json.dumps(dict(stage="historical_replay",split=split,event=event_id,solves=len(records))),flush=True)
    selected=[]
    for split in ["training","validation"]:
        for arm in ["single","multi"]:
            group=[r for r in records if r["split"]==split and r["arm"]==arm]
            worst=max(group,key=lambda r:r["gradient_per_scan"])
            build,old,case=lookup[split,worst["event"],worst["ap"],arm]
            new=build(640);diagnostic=captured[-1]
            points=case["qx"]
            om=old.sample_mean(points).mean(1);nm=new.sample_mean(points).mean(1)
            row=dict(selection="largest original per-scan projected gradient within split/arm",
                     original=worst,refined=diagnostic,
                     objective_decrease=worst["objective"]-diagnostic["objective"],
                     max_mean_change_at_history=(old.m-new.m).abs().max().item(),
                     max_cov_change_at_history=(old.c-new.c).abs().max().item(),
                     max_mean_change_at_query_positions=(om-nm).abs().max().item())
            selected.append(row)
            print(json.dumps(dict(stage="bounded_refinement",split=split,arm=arm,ap=worst["ap"],
                                  old_residual=worst["projected_gradient_max"],new_residual=diagnostic["projected_gradient_max"],
                                  max_query_mean_change=row["max_mean_change_at_query_positions"])),flush=True)
    inference.minimize=real_minimize
    aggregate={}
    for arm in ["single","multi"]:
        group=[r for r in records if r["arm"]==arm]
        aggregate[arm]=dict(solves=len(group),successes=sum(r["success"] for r in group),
             hit_iteration_limit=sum(r["nit"]>=160 for r in group),
             max_projected_gradient=max(r["projected_gradient_max"] for r in group),
             max_gradient_per_scan=max(r["gradient_per_scan"] for r in group),
             max_cached_mean_difference=max(r["max_cached_mean_difference"] for r in group),
             max_cached_cov_difference=max(r["max_cached_cov_difference"] for r in group))
    result=dict(scope="post-run numerical diagnostic; same data/model, no training or performance selection",
                original_iteration_limit=160,bounded_refinement_limit=640,refined_cases=4,
                original=aggregate,refinement=selected,
                limits="success indicates numerical termination, not global optimum or correctness of Gaussian VI")
    (auditdir/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(aggregate),flush=True)


if __name__=="__main__":main()
