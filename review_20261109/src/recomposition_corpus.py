"""Reuse the already prepared UAH/Pisa corpus for the manuscript learner.
No new data, history fit, target selection, model component or loss is introduced.
"""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import torch
from recomposition_trial import ROOT,OLD,write,digest,now
from recomposition import prepare,prior,Parameters,evaluate
from uji import tensor,load_history,ap_stats

PREP=OLD/"results/source_residual_probe_v1_20261026/preparation"
MOMENT=OLD/"results/moment_multisource_prepared_20261031"
UJI=ROOT/"results/recomposition_prepared_20261103"
OUT=ROOT/"results/recomposition_corpus_prepared_20261103"
TRAIN=ROOT/"results/recomposition_corpus_training_20261103"

def prepare_corpus(resume=False):
    if resume:
        assert OUT.is_dir()
        assert digest(ROOT/"recomposition_corpus_initial.py")==json.loads((OUT/"freeze.json").read_text())["code_sha256"]
    else:
        OUT.mkdir(parents=True,exist_ok=False)
    ready={}
    if resume:
        for line in (ROOT/"results/recomposition_corpus_prepared_20261103.log").read_text().splitlines():
            item=json.loads(line)
            assert Path(item["path"])==OUT/(item["id"]+".pt")
            assert digest(item["path"])==item["sha256"]
            ready[item["id"]]=item
    oldmanifest=json.loads((MOMENT/"complete.json").read_text())
    selected=[r for r in oldmanifest["episodes"] if r["building"] in ("UAH2016-b20","Pisa2019-b60")]
    write(OUT/("resume_freeze.json" if resume else "freeze.json"),dict(created=now(),code_sha256=digest(Path(__file__)),
       previous_attempt_exit=124 if resume else None,
       previous_freeze_sha256=digest(OUT/"freeze.json") if resume else None,
       completed_reused=list(ready),
       recomposition_sha256=digest(ROOT/"recomposition.py"),source_preparation_sha256=digest(PREP/"freeze.json"),
       source_manifest_sha256=digest(MOMENT/"complete.json"),public_events=selected,
       derivation="The same dimensionless shared rule is learned across existing environments instead of inferring generality from B0-only fitting.",
       reverse_audit="All H rows, S/Q split, APs, grid, history posteriors, source weights/intercept and proposal density are reused and checked. Q labels only supervise outer localization. Only training H normalizes features.",
       counterexamples="Different coordinate systems keep their own existing kernels/grids; the dimensionless theta alone is shared. Public trajectories are R=1, not invented static repeats. Correlated APs/tasks are not independent environments.",
       limits=["B0-fitted base/obs remain inherited approximations for public sites.",
               "UJI length uses original units; public sites use metres. No conversion is inferred and no radii are learned/shared by this trial.",
               "B1 is repeatedly viewed development data, not a fresh test."],
       reuse_only=True,history_refits=0,training_updates=0,wall_limit_seconds=900))
    sys.path.append(str(OLD))
    from public_corpus import probe_episodes as source
    manifest=source.verify_freeze(PREP)
    byid={e["id"]:e for e in manifest["events"]}
    records=[]
    for row in json.loads((UJI/"complete.json").read_text())["episodes"]:
        assert digest(UJI/row["path"])==row["sha256"]
        records.append(dict(row,path=str(UJI/row["path"])))
    for row in selected:
        if row["id"] in ready:
            records.append(ready[row["id"]]);continue
        t=time.monotonic();p=OLD/row["path"];assert digest(p)==row["sha256"]
        old=torch.load(p,weights_only=False);meta=byid[row["id"]];task=source.task_for(meta)
        assert task["S"]["scan_ids"].tolist()==meta["support_scan_ids"]
        assert task["Q"]["scan_ids"].tolist()==meta["query_scan_ids"]
        assert task["ap_indices"].tolist()==meta["AP_indices"]
        assert source.grid_definition(task)==meta["grid"]
        assert old["base"]==manifest["base"] and old["obs"]==manifest["obs"]
        s,q=task["S"],task["Q"]
        ep=dict(id=row["id"],role="train",grid=old["grid"],base=old["base"],obs=old["obs"],
                weights=tensor(old["weights"]),truth=tensor(q["xy"]),query_weights=tensor(q["scan_weights"]),cases={})
        ep["nearest"]=torch.cdist(ep["truth"],ep["grid"]).argmin(1)
        positions=[tensor(h["positions"]) for h in task["H"]]
        delta=[meta["target"]-f for f in meta["histories"]]
        maxerr=0.;inputs={}
        for col,ap in enumerate(meta["AP_indices"]):
            c=old["cases"][ap];stats=ap_stats(s,col)
            assert all(torch.equal(stats[k],c["stats"][k]) for k in stats)
            assert torch.equal(tensor(q["rss"][:,col]),c["rss"])
            assert torch.equal(ep["truth"],c["qx"][c["group"]])
            assert torch.equal(ep["query_weights"],c["query_weights"])
            hs=[]
            for f in meta["histories"]:
                info=meta["history_metadata"][str(f)]
                hp=Path(info["cache"])/f"ap{ap}.npz";inputs[str(hp)]=digest(hp)
                shift=tuple(np.array(info["cache_origin"])-np.array(meta["origin"]))
                hs.append(load_history(str(hp),shift,tuple(old["base"].items())))
            b=prepare(hs,positions,delta,tensor(s["positions"]),ep["grid"],ep["base"],old["calibration"]["intercept_std"])
            zero=Parameters(torch.zeros(9),torch.ones(9));v=prior(zero,b,ep["weights"])
            expected=dict(sm=c["blocks"]["sm"],gm=c["blocks"]["gm"],sk=c["blocks"]["ss"]["base"],
                          cross=c["blocks"]["gs"]["base"],gv=c["blocks"]["gg"]["base"])
            error=max(float((v[k]-x).abs().max().detach()) for k,x in expected.items())
            assert error<1e-7,(row["id"],ap,error);maxerr=max(maxerr,error)
            ep["cases"][ap]=dict(blocks=b,stats=stats,rss=c["rss"],proposal=old["proposals"][ap])
        output=OUT/(row["id"]+".pt");torch.save(ep,output)
        rec=dict(id=row["id"],role="train",path=str(output),sha256=digest(output),identity_error=maxerr,seconds=time.monotonic()-t)
        records.append(rec);write(OUT/(row["id"]+"_inputs.json"),dict(history_sha256=inputs,metadata=meta))
        print(json.dumps(rec),flush=True);del ep,old;load_history.cache_clear()
    means=[];seconds=[]
    for row in records:
        if row["role"]!="train":continue
        ep=torch.load(row["path"],weights_only=False)
        z=torch.cat([c["blocks"]["za"].reshape(-1,9) for c in ep["cases"].values()])
        means.append(z.mean(0));seconds.append(z.square().mean(0));del ep,z
    center=torch.stack(means).mean(0);v=torch.stack(seconds).mean(0)-center.square()
    assert v.min()>-1e-10
    scale=v.clamp_min(0).sqrt();scale[v.abs()<1e-12]=0
    torch.save(dict(center=center,scale=scale),OUT/"normalization.pt")
    write(OUT/"complete.json",dict(complete=True,created=now(),episodes=records,
        normalization_sha256=digest(OUT/"normalization.pt")))

def train_corpus():
    TRAIN.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((OUT/"complete.json").read_text())
    for r in manifest["episodes"]:assert digest(r["path"])==r["sha256"]
    assert digest(OUT/"normalization.pt")==manifest["normalization_sha256"]
    trainids=[r["id"] for r in manifest["episodes"] if r["role"]=="train"]
    validids=[r["id"] for r in manifest["episodes"] if r["role"]=="validation"]
    assert len(trainids)==10 and len(validids)==2
    rng=np.random.default_rng(11304);schedule=[trainids[int(i)] for _ in range(6) for i in rng.permutation(10)]
    write(TRAIN/"freeze.json",dict(created=now(),code_sha256=digest(Path(__file__)),
        prepared_sha256=digest(OUT/"complete.json"),recomposition_sha256=digest(ROOT/"recomposition.py"),
        arms=["full"],steps=60,checkpoints=[0,30,60],schedule=schedule,learning_rate=.001,
        optimizer="Adam",regularization=.001,score_B=1.,eta_bound=.5,log_scale_bound="log(2)",
        objective="Same full-scan nearest-grid location NLL; equal tasks and equal Q locations within task.",
        selection="Minimum B1 development MDE at fixed checkpoints, including identity.",
        changes_from_first_trial="Source environments and H-only normalization only; no new components or losses. Not a matched-budget data-effect study.",
        final_test_events=0,threads=2,wall_limit_seconds=1200))
    eps={r["id"]:torch.load(r["path"],weights_only=False) for r in manifest["episodes"]}
    norm=torch.load(OUT/"normalization.pt",weights_only=True)
    model=Parameters(**norm);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    measurements={};t=time.monotonic()
    def checkpoint(step):
        measurements[step]=dict(source={i:evaluate(model,eps[i]) for i in trainids},
             validation={i:evaluate(model,eps[i]) for i in validids})
        phi,eta,lg,ls=model.values()
        measurements[step]["parameters"]=dict(phi=phi.detach().tolist(),eta=float(eta.detach()),g=float(lg.exp().detach()),s=float(ls.exp().detach()))
        torch.save(model.state_dict(),TRAIN/f"full_{step}.pt");write(TRAIN/"measurements.json",measurements)
        print(json.dumps(dict(step=step,source_nll=np.mean([v["nll"] for v in measurements[step]["source"].values()]),
           validation_mde=np.mean([v["mde"] for v in measurements[step]["validation"].values()]),seconds=time.monotonic()-t)),flush=True)
    checkpoint(0)
    with (TRAIN/"train.jsonl").open("w") as trace:
        for step,event in enumerate(schedule,1):
            optimizer.zero_grad(set_to_none=True);metrics=evaluate(model,eps[event],backward=True)
            if not torch.isfinite(model.theta.grad).all():raise RuntimeError("Nonfinite gradient")
            optimizer.step();trace.write(json.dumps(dict(step=step,event=event,**metrics))+"\n");trace.flush()
            if step in (30,60):checkpoint(step)
    chosen=min(measurements,key=lambda k:(np.mean([v["mde"] for v in measurements[k]["validation"].values()]),k))
    write(TRAIN/"complete.json",dict(complete=True,created=now(),selected=chosen,steps=60,
        validation_mde=np.mean([v["mde"] for v in measurements[chosen]["validation"].values()]),
        identity_validation_mde=np.mean([v["mde"] for v in measurements[0]["validation"].values()]),
        final_test_events=0,seconds=time.monotonic()-t,
        interpretation="Three source environments, one repeatedly viewed development building. Does not establish general gain or a K curve."))

def main():
    import socket
    if socket.gethostname()!="ray-web":raise RuntimeError("Server only")
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","resume","train"));args=p.parse_args()
    output=TRAIN if args.stage=="train" else OUT
    try:
        if args.stage!="train":prepare_corpus(resume=args.stage=="resume")
        else:train_corpus()
    except Exception as e:
        if output.exists():write(output/"failure.json",dict(created=now(),error=f"{type(e).__name__}: {e}"))
        raise
if __name__=="__main__":main()

