"""Server-only frozen development trial of the complete deterministic ReFi prior.
Existing B0 train / B1 development events are reused; no final test is opened.
"""
import argparse,hashlib,json,socket,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from uji import load_uji,group_scans,ap_stats,load_history,tensor
from recomposition import prepare,prior,Parameters,evaluate
from check_recomposition import checks

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/"LAW-source-resume-20261025-01a164d0/Law-Probabilistic-Transfer-20261023"
CACHED=OLD/"results/moment_pilot_prepared_20261031"
PLAN=OLD/"results/decision_local_pilot_20261030/plan.json"
CSV=ROOT.parent/"UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv"

def digest(path):
    with open(path,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
def now():return datetime.now(timezone.utc).isoformat()
def subset(raw,rows,origin):
    lookup={int(v):i for i,v in enumerate(raw["row_ids"])}
    idx=[lookup[int(v)] for v in rows]
    return group_scans(raw["rss"][idx],raw["xy"][idx]-origin)

def prepare_all(out):
    out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(PLAN.read_text());manifest=json.loads((CACHED/"complete.json").read_text())
    events={e["id"]:e for e in plan["events"] if e["role"] in ("train","validation")}
    expected=next(v for k,v in plan["input_sha256"].items() if Path(k).name==CSV.name)
    assert digest(CSV)==expected
    numerical=checks();write(out/"algebra.json",numerical)
    write(out/"prepare_freeze.json",dict(created=now(),csv_sha256=expected,plan_sha256=digest(PLAN),
        old_manifest_sha256=digest(CACHED/"complete.json"),events=list(events),
        code={p:digest(ROOT/p) for p in ("recomposition.py","recomposition_trial.py","check_recomposition.py")},
        derivation=dict(object="12 shared parameters: nine local source gates, one deterministic shape parameter, separate historical mean/covariance scales",
          benefit="A history-informed prior can learn local source reliability and how S corrections propagate, optimized for full-scan localization.",
          condition="Shared horizontal coordinates and AP correspondences; training environments must support transferable relations.",
          falsifier="No held-environment localization gain over identity under matched information, or gain confined to already-seen source tasks.",
          limits="No coordinate warp; no resurrection of zero-weight sources; CE need not improve expected coordinate error; all outputs restricted to grid convex hull."),
        reverse_audit=dict(H="Only allowed history rows form GP posteriors, anchors and nine features.",
          S="Only S fits frozen source weights and intercept; exact stats compared with retained preparation.",
          training="Training H anchors determine normalization; train Q coordinates enter outer location CE only.",
          deployment="Frozen theta and normalization; target H/S produce prior and S posterior, query RSS only enters readout.",
          numerical="Stored proposals and their original density are numerical auxiliaries, not the stochastic-moment prior; reweight under the current deterministic prior, ESS >=1000.",
          covariance="Same transformed kernel at SS, grid-S and grid diagonal; residual/intercept added once.",
          missing="Partially missing H uses base GP; all missing bypasses theta; AP set fixed by H/S."),
        limitations=["B0 base/observation parameters were fitted to all B0 locations, including source queries.",
                    "B1 validation has already been inspected in earlier work: development only.",
                    "K=5,R=3 and two historical floors do not establish the K curve or general cross-environment gain.",
                    "Coordinate units are original UJI units, not verified metres."],
        training_updates=0,query_scores=0))
    floors=load_uji(CSV);origin=np.array(plan["origin"]);records=[];means=[];seconds=[]
    for row in manifest["episodes"]:
        t=time.monotonic();event=events[row["id"]];path=CACHED/row["path"]
        assert digest(path)==row["sha256"]
        old=torch.load(path,weights_only=False)
        meta=json.loads((OLD/"results/decision_local_pilot_20261030"/event["id"]/"meta.json").read_text())
        entries={(r["floor"],r["ap"]):r for r in meta["history_records"]}
        s=subset(floors[event["floor"]],event["support_rows"],origin)
        q=subset(floors[event["floor"]],event["query_rows"],origin)
        assert not (s["positions"][:,None]==q["positions"][None]).all(-1).any()
        positions=[tensor(subset(floors[f],event["history_rows"][f],origin)["positions"]) for f in event["histories"]]
        delta=[int(event["floor"].split("F")[1])-int(f.split("F")[1]) for f in event["histories"]]
        ep=dict(id=event["id"],role=event["role"],grid=old["grid"],base=old["base"],obs=old["obs"],
                weights=tensor(old["weights"]),truth=tensor(q["xy"]),cases={})
        ep["nearest"]=torch.cdist(ep["truth"],ep["grid"]).argmin(1)
        ep["query_weights"]=tensor(1/(len(q["positions"])*q["stats"]["n"][q["group"]]))
        seen=(s["rss"]!=100).any(0)
        for f in event["histories"]:seen|=(floors[f]["rss"]!=100).any(0)
        assert set(np.flatnonzero(seen))==set(old["cases"])
        maxerr=0.;history_hashes={}
        for ap,c in old["cases"].items():
            stats=ap_stats(s,ap)
            assert all(torch.equal(stats[k],c["stats"][k]) for k in stats)
            assert torch.equal(tensor(q["rss"][:,ap]),c["rss"])
            assert torch.equal(tensor(q["xy"]),c["qx"][c["group"]])
            assert torch.equal(ep["query_weights"],c["query_weights"])
            hs=[]
            for f in event["histories"]:
                rec=entries[(f,ap)]
                if "reuse_path" in rec:
                    hp=OLD/rec["reuse_path"].split("/Law-Probabilistic-Transfer-20261023/",1)[1]
                else:hp=OLD/"results/decision_local_pilot_20261030/histories"/f/f"ap{ap}.npz"
                sha=digest(hp);assert sha==rec["sha256"];history_hashes[str(hp)]=sha
                with np.load(hp) as arrays:shift=tuple(arrays["origin"]-origin)
                hs.append(load_history(str(hp),shift,tuple(old["base"].items())))
            b=prepare(hs,positions,delta,tensor(s["positions"]),old["grid"],old["base"],old["calibration"]["intercept_std"])
            zero=Parameters(torch.zeros(9),torch.ones(9));v=prior(zero,b,ep["weights"])
            # Old deterministic base blocks provide independent real-data identity parity.
            expected=dict(sm=c["blocks"]["sm"],gm=c["blocks"]["gm"],
                sk=c["blocks"]["ss"]["base"],cross=c["blocks"]["gs"]["base"],gv=c["blocks"]["gg"]["base"])
            err=max(float((v[k]-x).abs().max().detach()) for k,x in expected.items())
            maxerr=max(maxerr,err);assert err<1e-7,(event["id"],ap,err)
            ep["cases"][ap]=dict(blocks=b,stats=stats,rss=c["rss"],proposal=old["proposals"][ap])
        if event["role"]=="train":
            # Equal task, source/AP/anchor rows within each task.
            rows=torch.cat([c["blocks"]["za"].reshape(-1,9) for c in ep["cases"].values()])
            means.append(rows.mean(0));seconds.append(rows.square().mean(0))
        output=out/(event["id"]+".pt");torch.save(ep,output)
        record=dict(id=event["id"],role=event["role"],path=output.name,sha256=digest(output),
            APs=len(ep["cases"]),identity_error=maxerr,seconds=time.monotonic()-t)
        records.append(record)
        write(out/(event["id"]+"_inputs.json"),dict(history_sha256=history_hashes,event=event))
        print(json.dumps(record),flush=True)
        del old,ep
    center=torch.stack(means).mean(0);variance=torch.stack(seconds).mean(0)-center.square()
    assert variance.min()>-1e-10
    scale=variance.clamp_min(0).sqrt()
    # Mathematical constants have exactly zero variance; roundoff is not variation.
    scale[variance.abs()<1e-12]=0.
    torch.save(dict(center=center,scale=scale),out/"normalization.pt")
    write(out/"complete.json",dict(created=now(),complete=True,episodes=records,
        normalization_sha256=digest(out/"normalization.pt"),source_roles_only=True))

def train_all(prepared,out,steps):
    assert steps in (48,96)
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((prepared/"complete.json").read_text())
    assert manifest["complete"]
    for r in manifest["episodes"]:assert digest(prepared/r["path"])==r["sha256"]
    norm=torch.load(prepared/"normalization.pt",weights_only=True)
    assert digest(prepared/"normalization.pt")==manifest["normalization_sha256"]
    trainids=[r["id"] for r in manifest["episodes"] if r["role"]=="train"]
    validids=[r["id"] for r in manifest["episodes"] if r["role"]=="validation"]
    rng=np.random.default_rng(11303);schedule=[trainids[int(i)] for _ in range(steps//6) for i in rng.permutation(6)]
    freeze=dict(created=now(),prepared_sha256=digest(prepared/"complete.json"),arms=["scales","shape_scales","full"],
       schedule=schedule,checkpoints=[0,steps//2,steps],optimizer="Adam",learning_rate=.001,
       bounds=dict(score_B=1.,eta=[-.5,.5],g=[.5,2.],s=[.5,2.]),regularization=.001,min_ess=1000,
       threads=2,wall_limit_seconds=2700,
       code={p:digest(ROOT/p) for p in ("recomposition.py","recomposition_trial.py","observations.py","posterior.py")},
       selection="Minimum B1 development MDE at fixed checkpoints including identity; ties earliest. No claim of unbiased performance.",
       objective="Full H/S-defined AP scan location NLL at nearest grid point, including non-detections; no RSS-only training.")
    write(out/"freeze.json",freeze)
    eps={r["id"]:torch.load(prepared/r["path"],weights_only=False) for r in manifest["episodes"]}
    summary={};identity=None;t0=time.monotonic()
    for arm in freeze["arms"]:
        model=Parameters(**norm,mode=arm);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        measurements={}
        def checkpoint(step):
            nonlocal identity
            if step==0 and identity is not None:measurements[step]=identity
            else:
                measurements[step]=dict(source={i:evaluate(model,eps[i]) for i in trainids},
                      validation={i:evaluate(model,eps[i]) for i in validids})
                if step==0:identity=measurements[step]
            phi,eta,lg,ls=model.values()
            measurements[step]=dict(measurements[step],parameters=dict(phi=phi.detach().tolist(),eta=float(eta.detach()),
                g=float(lg.exp().detach()),s=float(ls.exp().detach())))
            torch.save(model.state_dict(),out/f"{arm}_{step}.pt");write(out/f"{arm}_measurements.json",measurements)
            print(json.dumps(dict(arm=arm,step=step,source_nll=np.mean([x["nll"] for x in measurements[step]["source"].values()]),
               source_mde=np.mean([x["mde"] for x in measurements[step]["source"].values()]),
               validation_mde=np.mean([x["mde"] for x in measurements[step]["validation"].values()]),
               seconds=time.monotonic()-t0)),flush=True)
        checkpoint(0)
        with (out/f"{arm}_train.jsonl").open("w") as trace:
            for step,i in enumerate(schedule,1):
                optimizer.zero_grad(set_to_none=True);metrics=evaluate(model,eps[i],backward=True)
                if not torch.isfinite(model.theta.grad).all():raise RuntimeError("Nonfinite gradient")
                optimizer.step()
                trace.write(json.dumps(dict(step=step,event=i,**metrics))+"\n");trace.flush()
                if step in freeze["checkpoints"]:checkpoint(step)
        selected=min(measurements,key=lambda k:(np.mean([v["mde"] for v in measurements[k]["validation"].values()]),k))
        summary[arm]=dict(selected=selected,validation_mde=np.mean([v["mde"] for v in measurements[selected]["validation"].values()]),
            final_validation_mde=np.mean([v["mde"] for v in measurements[steps]["validation"].values()]))
    write(out/"complete.json",dict(created=now(),complete=True,arms=summary,
          identity_validation_mde=np.mean([v["mde"] for v in identity["validation"].values()]),
          final_test_events=0,seconds=time.monotonic()-t0,interpretation="Development only. No general performance or K-curve conclusion."))

def main():
    if socket.gethostname()!="ray-web":raise RuntimeError("Server only")
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(11303)
    p=argparse.ArgumentParser();p.add_argument("action",choices=("prepare","train"))
    p.add_argument("--prepared",type=Path);p.add_argument("--output",type=Path,required=True);p.add_argument("--steps",type=int,default=48)
    args=p.parse_args()
    try:
        if args.action=="prepare":prepare_all(args.output)
        else:train_all(args.prepared,args.output,args.steps)
    except Exception as e:
        if args.output.exists():write(args.output/"failure.json",dict(created=now(),error=f"{type(e).__name__}: {e}"))
        raise
if __name__=="__main__":main()

