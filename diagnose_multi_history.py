"""Post-hoc explanations of saved predictions; no model rerun or selection."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from data import load_period


ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT/"results/multi_history_v1")
    args=parser.parse_args();out=args.out
    summary=json.loads((out/"summary.json").read_text())
    manifest=json.loads((out/"manifest.json").read_text())
    floors={t:load_period(ROOT.parent/"Law-Continuous-Change-20261023/data/OfficeP2.npz",t) for t in [8,10,11,12]}
    results=summary["results"]
    means={name:{key:float(np.mean([r[key] for r in results if r["kind"]==name]))
                  for key in ("mde","p90")}
           for name in ["single_legacy","single","multi_frozen","multi_learned","wknn"]}
    for name in means:
        if name!="wknn": means[name]["nll"]=float(np.mean([r["nll"] for r in results if r["kind"]==name]))
    explanations=[]; pair_counts={}; source_counts={}
    for arm_a,arm_b in [("single_legacy","single"),("single","multi_frozen"),("multi_frozen","multi_learned")]:
        key=f"{arm_a}_to_{arm_b}"
        gains=[]; candidates=[]; loc_examples=[]
        for event in manifest["validation"]:
            seed=event["seed"]
            with np.load(out/f"predictions_seed{seed}.npz") as arrays:
                xy=arrays["query_xy"]; positions=arrays["query_positions"]; groups=arrays["query_group"]
                err_a=np.linalg.norm(arrays[f"{arm_a}_xy"]-xy,axis=1)
                err_b=np.linalg.norm(arrays[f"{arm_b}_xy"]-xy,axis=1)
                for tag,i in [("best",int(np.argmax(err_a-err_b))),("worst",int(np.argmin(err_a-err_b)))]:
                    loc_examples.append(dict(seed=seed,case=tag,xy=xy[i].tolist(),
                         before_xy=arrays[f"{arm_a}_xy"][i].tolist(),after_xy=arrays[f"{arm_b}_xy"][i].tolist(),
                         before_error=float(err_a[i]),after_error=float(err_b[i])))
                for j,ap in enumerate(event["ap_columns"]):
                    before=arrays[f"{arm_a}_ap{ap}_scan_nll"]
                    after=arrays[f"{arm_b}_ap{ap}_scan_nll"]
                    gains.append(float((before-after).mean()))
                    for i in [int(np.argmax(before-after)),int(np.argmin(before-after))]:
                        g=int(groups[i]);target=xy[i]
                        historical=[]
                        for t in [8,10,11]:
                            selected=np.array(event["historical_rows"][str(t)])
                            coords=floors[t]["xy"][selected]
                            dist=np.linalg.norm(coords-target,axis=1); nearest=coords[dist.argmin()]
                            rows=selected[(coords==nearest).all(1)]
                            rss=floors[t]["rss"][rows,ap];visible=rss!=100
                            historical.append(dict(period=t,position=nearest.tolist(),distance=float(dist.min()),
                                 scans=len(rows),detected=int(visible.sum()),
                                 detected_mean=float(rss[visible].mean()) if visible.any() else None))
                        row=dict(comparison=key,seed=seed,ap=ap,xy=target.tolist(),
                                 observed_rss=float(arrays["query_rss"][i,j]),nll_gain=float(before[i]-after[i]),
                                 before_nll=float(before[i]),after_nll=float(after[i]),
                                 support_distance=float(np.linalg.norm(np.array(event["support_positions"])-target,axis=1).min()),
                                 nearest_history=historical)
                        sr=np.array(event["support_rows"]);sc=floors[12]["xy"][sr]
                        nearest=sc[np.linalg.norm(sc-target,axis=1).argmin()]
                        row["nearest_support"]=dict(position=nearest.tolist(),
                            rss=floors[12]["rss"][sr[(sc==nearest).all(1)],ap].tolist())
                        for name in [arm_a,arm_b]:
                            row[name]={field:float(arrays[f"{name}_ap{ap}_{field}"][g])
                                       for field in ["prior_mean","prior_variance","mean","variance"]}
                        candidates.append(row)
        order=sorted(candidates,key=lambda x:x["nll_gain"])
        explanations.extend([dict(role="largest_degradation",**order[0]),dict(role="largest_improvement",**order[-1])])
        pair_counts[key]=dict(improved_ap_cases=int(np.sum(np.array(gains)>0)),total_ap_cases=len(gains),
                             mean_ap_nll_gain=float(np.mean(gains)),localization_examples=loc_examples)
    for name in ["training","validation"]:
        unique_h=set(); unique_s=set(); unique_q=set()
        for e in manifest[name]:
            unique_h.update((int(t),row) for t,rows in e["historical_rows"].items() for row in rows)
            unique_s.update((e["current"],row) for row in e["support_rows"])
            unique_q.update((e["current"],row) for row in e["query_rows"])
        source_counts[name]=dict(unique_historical_acquisitions=len(unique_h),
                                 unique_support_acquisitions=len(unique_s),unique_query_acquisitions=len(unique_q))
    scan_information=[]
    for event in manifest["validation"]:
        with np.load(out/f"predictions_seed{event['seed']}.npz") as arrays:
            detected=(arrays["query_rss"]!=100).sum(1)
            groups=[]
            for mask,name in [(detected==0,"all_selected_APs_missing"),(detected>0,"some_selected_AP_detected")]:
                if mask.any():
                    groups.append(dict(group=name,scans=int(mask.sum()),
                        errors={arm:float(np.linalg.norm(arrays[f"{arm}_xy"][mask]-arrays["query_xy"][mask],axis=1).mean())
                                for arm in ["single","multi_frozen","multi_learned","wknn"]}))
            scan_information.append(dict(seed=event["seed"],detected_AP_histogram=np.bincount(detected).tolist(),groups=groups))
    rows=[json.loads(s) for s in (out/"evaluation.jsonl").read_text().splitlines()]
    report=dict(scope="Post-hoc descriptive diagnosis, extremes selected after fixed evaluation; no parameter change",
                equal_draw_means=means,paired_ap_cases=pair_counts,examples=explanations,
                acquisition_counts=source_counts,selected_scan_information=scan_information,
                min_effective_samples=min(r["ess"] for r in rows),
                max_scan_nll_mc_se=max(r["nll_mc_se"] for r in rows),
                uncertainty="2 draws change H, S and AP subset jointly, not a confidence interval over environments")
    (out/"diagnostics.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    environment={x:importlib.metadata.version(x) for x in ["torch","numpy","scipy","pypmc"]}
    (out/"environment.json").write_text(json.dumps(environment,indent=2))
    print(json.dumps(dict(means=means,counts=source_counts,paired={k:{a:b for a,b in v.items() if a!="localization_examples"} for k,v in pair_counts.items()})))


if __name__=="__main__": main()
