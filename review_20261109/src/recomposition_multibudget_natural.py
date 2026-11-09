"""Nested-budget ReFi tasks: retain H, hold Q fixed, refit using only S_K.

Uses H union current S_K APs/domain. Stores support state and integration nodes. The large,
immutable historical fields are reused across K and across learned models.
"""
import argparse,hashlib,json,socket,sys,time
from pathlib import Path
import numpy as np
import torch
from recomposition_trial import ROOT,OLD,CSV,PLAN,subset,digest,write,now
from recomposition_corpus import OUT as CORPUS
from recomposition import Parameters,evaluate
from recomposition_predict import refresh_case
from adaptation import fit_source_weights,marginal_loglik,SIGMA
from fields import rbf
from uji import load_uji,tensor,ap_stats

OUT=ROOT/"results/recomposition_multibudget_natural_20261104"
KS=(1,3,5)

def seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],"little")

def raw_inputs(row,floors):
    path=Path(row["path"])
    info=json.loads(path.with_name(row["id"]+"_inputs.json").read_text())
    if "event" in info:
        event=info["event"];origin=np.array(json.loads(PLAN.read_text())["origin"])
        s=subset(floors[event["floor"]],event["support_rows"],origin)
        q=subset(floors[event["floor"]],event["query_rows"],origin)
        hs=[subset(floors[f],event["history_rows"][f],origin) for f in event["histories"]]
        aps=np.arange(s["rss"].shape[1])
        provenance=dict(support_rows=event["support_rows"],query_rows=event["query_rows"],
                        history_rows=event["history_rows"],origin=origin.tolist())
    else:
        sys.path.append(str(OLD))
        from public_corpus import probe_episodes as source
        meta=info["metadata"];task=source.task_for(meta)
        s,q,hs,aps=task["S"],task["Q"],task["H"],task["ap_indices"]
        assert s["scan_ids"].tolist()==meta["support_scan_ids"]
        assert q["scan_ids"].tolist()==meta["query_scan_ids"]
        provenance=dict(support_scan_ids=meta["support_scan_ids"],query_scan_ids=meta["query_scan_ids"],
                        history_scan_ids=[h["scan_ids"].tolist() for h in hs])
    return s,q,hs,aps,provenance

def grid_subset(b,idx):
    out=dict(b)
    for key,axis in (("zg",1),("gm",1),("gd",1),("gs",2),("gv",2),("rg",0),("rv",0)):
        out[key]=b[key].index_select(axis,idx)
    return out

def restrict_support(b,indices,sx,grid,base,intercept):
    out=dict(b)
    for key in ("zs","sm","sd"):
        out[key]=b[key].index_select(1,indices)
    out["ss"]=b["ss"].index_select(2,indices).index_select(3,indices)
    out["gs"]=b["gs"].index_select(3,indices)
    amplitude=base["drift"] if b["active"] else base["amplitude"]
    x=sx[indices]
    out["rs"]=rbf(x,x,amplitude,base["length"])+intercept**2
    out["rg"]=rbf(grid,x,amplitude,base["length"])+intercept**2
    out["rv"]=grid.new_full((len(grid),),amplitude**2+intercept**2)
    return out

def adapt_blocks(cases,obs):
    means=[];covs=[];residual=[]
    for c in cases.values():
        b=c["blocks"]
        means.append(b["sm"] if b["active"] else torch.full_like(b["sm"],b["base_mean"]))
        covs.append(b["ss"][:,0] if b["active"] else torch.zeros_like(b["ss"][:,0]))
        residual.append(b["rs"])
    stats={key:torch.stack([c["stats"][key] for c in cases.values()]) for key in ("n","k","s1","s2")}
    weights,trials=fit_source_weights(torch.stack(means),torch.stack(covs),torch.stack(residual),stats,obs)
    w=tensor(weights);scores=torch.zeros(len(SIGMA))
    for mean,cov,rr,c in zip(means,covs,residual,cases.values()):
        m=(w[:,None]*mean).sum(0)
        v=(w[:,None].square()*cov.diagonal(dim1=-2,dim2=-1)).sum(0)+rr.diag()
        for i,sigma in enumerate(SIGMA):
            scores[i]+=marginal_loglik(m,v+sigma**2,**c["stats"],obs=obs).sum()
    return dict(weights=w,intercept=SIGMA[int(scores.argmax())],trials=trials,
                intercept_scores=scores.tolist())

def wknn(s,q,hs,columns,indices,grid):
    keep=np.isin(s["group"],indices.numpy())
    rss=np.concatenate([*[h["rss"][:,columns] for h in hs],s["rss"][keep][:,columns]])
    xy=np.concatenate([*[h["xy"] for h in hs],s["xy"][keep]])
    qr=q["rss"][:,columns]
    a=np.where(rss==100,-110,rss).astype(float);b=np.where(qr==100,-110,qr).astype(float)
    d=np.sqrt(((b[:,None]-a[None])**2).mean(-1))
    chosen=np.argsort(d,axis=1,kind="stable")[:,:3]
    w=1/np.maximum(np.take_along_axis(d,chosen,1),1e-8);w/=w.sum(1,keepdims=True)
    prediction=np.einsum("qk,qkd->qd",w,xy[chosen])
    prediction=np.clip(prediction,grid.min(0).values.numpy(),grid.max(0).values.numpy())
    errors=np.linalg.norm(prediction-q["xy"],axis=1)
    qw=1/(len(q["positions"])*q["stats"]["n"][q["group"]])
    return dict(mde=float(qw@errors),errors=errors.tolist(),k=3,missing=-110)

def prepare_all(limit):
    manifest=json.loads((CORPUS/"complete.json").read_text())
    limited=ROOT/"results/recomposition_multibudget_20261104"
    previous=json.loads((limited/"complete.json").read_text())
    previous_by_id={r["id"]:r for r in previous["records"]}
    if not OUT.exists():
        OUT.mkdir()
        write(OUT/"freeze.json",dict(created=now(),code_sha256=digest(Path(__file__)),
          corpus_manifest_sha256=digest(CORPUS/"complete.json"),
          previous_preparation_sha256=digest(limited/"complete.json"),budgets=list(KS),
          derivation="Learn a shared historical recomposition rule across nested actual support budgets. AP catalogue and candidate domain use H union CURRENT S_K, as in deployment; no future support or query input.",
          reason_for_successor="The H-only comparison discarded usable target APs and spatial coverage, materially for Pisa. Correct the task scope before learned performance is available.",
          H="Reuse original frozen H fields and H-only normalization once per parent. Retain every source.",
          S="Same nested actual positions/repeats as the earlier preparation. Refit H/S_K state if the permitted AP set changed. A different query grid alone does not change the S posterior.",
          reuse="Original K5 episodes already have correct H union S5 scope. Reuse their source weights, intercept and true-density proposals. For K1/3 reuse the completed H-only S state ONLY when APs, support indices and statistics are identical.",
          Q="Original Q remains fixed. Coordinates enter outer learning/scoring only. No cropping of target queries.",
          reverse_audit="Full scan readout -> transformed H prior conditioned on S_K -> per-budget H/S state and globally learned theta. New target-only APs use the existing all-missing-H branch; not silently discarded.",
          counterexamples=["An AP first seen at K3 is absent at K1, present at K3/K5.",
             "A K5 point outside historical coverage cannot expand the K1 domain unless already selected at K1.",
             "Proposal samples are never sliced with a joint log density.",
             "New APs with missing H bypass theta but still contribute to support calibration and full scan location probabilities.",
             "No promise of monotone realized MDE or arbitrary coordinate deformation.",
             "Existing core algebra/gradient checks and query-reuse equivalence remain applicable; no changes to the stochastic model."],
          limits=["K1/3/5; K10/20 remains outstanding.","B1 development, not independent confirmation.",
                  "Inherited B0 base/observation parameters remain approximations on public sites.",
                  "Correlated nested tasks are not new independent environments."],
          preparation_wall_seconds=1800,threads=2,network_downloads=0))
        write(OUT/"progress.json",dict(records=[]))
    freeze=json.loads((OUT/"freeze.json").read_text())
    assert freeze["code_sha256"]==digest(Path(__file__))
    assert freeze["corpus_manifest_sha256"]==digest(CORPUS/"complete.json")
    progress=json.loads((OUT/"progress.json").read_text())
    done={r["id"] for r in progress["records"]}
    for r in progress["records"]:assert digest(OUT/r["path"])==r["sha256"]
    plan=json.loads(PLAN.read_text())
    expected=next(v for k,v in plan["input_sha256"].items() if Path(k).name==CSV.name)
    assert digest(CSV)==expected
    sys.path.append(str(OLD))
    from public_corpus import probe_episodes as source
    source.verify_freeze(source.PREPARATION)
    floors=load_uji(CSV);started=time.monotonic();count=0
    for row in manifest["episodes"]:
        if row["id"] in done:continue
        if count>=limit or time.monotonic()-started>1400:break
        assert digest(row["path"])==row["sha256"]
        old=torch.load(row["path"],weights_only=False)
        prior_row=previous_by_id[row["id"]]
        assert digest(limited/prior_row["path"])==prior_row["sha256"]
        prior_payload=torch.load(limited/prior_row["path"],weights_only=False)
        s,q,hs,aps,provenance=raw_inputs(row,floors)
        sx=tensor(s["positions"]);assert len(sx)==5
        assert torch.equal(tensor(q["xy"]),old["truth"])
        assert not (s["positions"][:,None]==q["positions"][None]).all(-1).any()
        hseen=np.logical_or.reduce([(h["rss"]!=100).any(0) for h in hs])
        haps=[int(aps[c]) for c in np.flatnonzero(hseen)]
        assert haps==prior_payload["APs"] and torch.equal(sx,prior_payload["support_positions"])
        for col,ap in enumerate(aps):
            if int(ap) not in old["cases"]:continue
            c=old["cases"][int(ap)]
            assert all(torch.equal(v,c["stats"][key]) for key,v in ap_stats(s,col).items())
            assert torch.equal(tensor(q["rss"][:,col]),c["rss"])
        payload=dict(id=row["id"],role=row["role"],base_path=row["path"],base_sha256=row["sha256"],
            support_positions=sx,normalization=prior_payload["normalization"],
            provenance=provenance,states={})
        hx=tensor(np.concatenate([h["positions"] for h in hs]))
        order=np.random.default_rng(seed(row["id"]+"/nested")).permutation(5)
        zero=Parameters(torch.zeros(9),torch.ones(9))
        for k in KS:
            indices=torch.tensor(np.sort(order[:k]),dtype=torch.long)
            keep=np.isin(s["group"],indices.numpy())
            columns=np.flatnonzero(hseen|(s["rss"][keep]!=100).any(0))
            selected=[int(aps[c]) for c in columns]
            xy=torch.cat((hx,sx[indices]));low=torch.floor(xy.min(0).values/2)*2;high=torch.ceil(xy.max(0).values/2)*2
            gi=torch.nonzero(((old["grid"]>=low)&(old["grid"]<=high)).all(1)).flatten()
            grid=old["grid"][gi]
            assert len(grid)==int(torch.prod((high-low)/2+1))
            assert torch.equal(grid.min(0).values,low) and torch.equal(grid.max(0).values,high)
            previous_state=prior_payload["states"][k]
            assert torch.equal(indices,previous_state["indices"])
            if k==5:
                assert selected==list(old["cases"])
                assert torch.equal(gi,torch.arange(len(old["grid"])))
                b=next(iter(old["cases"].values()))["blocks"]
                amplitude=old["base"]["drift"] if b["active"] else old["base"]["amplitude"]
                variance=float(b["rv"][0])-amplitude**2
                intercept=min(SIGMA,key=lambda value:abs(value**2-variance))
                assert abs(intercept**2-variance)<1e-7
                for c in old["cases"].values():
                    b=c["blocks"];a=old["base"]["drift"] if b["active"] else old["base"]["amplitude"]
                    assert float((b["rv"]-a**2-intercept**2).abs().max())<1e-7
                state=dict(weights=old["weights"],intercept=intercept,indices=indices,
                           proposals={ap:c["proposal"] for ap,c in old["cases"].items()},
                           numerical={},reuse="Original verified K5 H union S5 prior/proposals, unchanged S indices/APs/grid.")
            elif selected==prior_payload["APs"]:
                state=dict(previous_state,reuse="Identical H/S/APs to previous preparation; only candidate grid may expand.")
            else:
                cases={ap:dict(blocks=restrict_support(grid_subset(old["cases"][ap]["blocks"],gi),
                         indices,sx,grid,old["base"],0.),
                         stats={n:v[indices] for n,v in old["cases"][ap]["stats"].items()}) for ap in selected}
                state=adapt_blocks(cases,old["obs"])
                state.update(indices=indices,proposals={},numerical={},reuse="None: AP scope changed; refit using actual H union S_K.")
                for ap,c in cases.items():
                    for key in ("rs","rg","rv"):c["blocks"][key]+=state["intercept"]**2
                    fresh,info=refresh_case(zero,c,state["weights"],old["obs"],seed(f"natural-budget/{row['id']}/{k}/{ap}"))
                    state["proposals"][ap]=fresh["proposal"];state["numerical"][ap]=info
                del cases
            state.update(APs=selected,grid_indices=gi)
            if row["role"]=="validation":state["wknn"]=wknn(s,q,hs,columns,indices,grid)
            payload["states"][k]=state
            print(json.dumps(dict(id=row["id"],K=k,APs=len(selected),grid=len(grid),reuse=state["reuse"])),flush=True)
        output=OUT/(row["id"]+".pt");torch.save(payload,output)
        rec=dict(id=row["id"],role=row["role"],path=output.name,sha256=digest(output),
                 APs_by_K={str(k):len(v["APs"]) for k,v in payload["states"].items()},
                 grid_by_K={str(k):len(v["grid_indices"]) for k,v in payload["states"].items()})
        progress["records"].append(rec);write(OUT/"progress.json",progress)
        del old,prior_payload,payload,state
        count+=1
    if len(progress["records"])==len(manifest["episodes"]):
        # Normalization was already computed solely from the same training H.
        norm=torch.load(limited/"normalization.pt",weights_only=True)
        assert digest(limited/"normalization.pt")==previous["normalization_sha256"]
        torch.save(norm,OUT/"normalization.pt")
        write(OUT/"complete.json",dict(complete=True,created=now(),records=progress["records"],
            normalization_sha256=digest(OUT/"normalization.pt"),train_tasks=30,validation_tasks=6,
            AP_and_domain_scope="H union current S_K"))

def load_parent(row):
    assert digest(OUT/row["path"])==row["sha256"]
    p=torch.load(OUT/row["path"],weights_only=False)
    assert digest(p["base_path"])==p["base_sha256"]
    old=torch.load(p["base_path"],weights_only=False)
    old["cases"]={ap:dict(stats=c["stats"],rss=c["rss"],blocks=c["blocks"])
                  for ap,c in old["cases"].items()}
    return old,p

def load_parents():
    ready=json.loads((OUT/"complete.json").read_text());assert ready["complete"]
    return {r["id"]:load_parent(r) for r in ready["records"]}

def episode(parent,k):
    old,p=parent;state=p["states"][k];u=state["indices"];gi=state["grid_indices"];grid=old["grid"][gi]
    cases={ap:dict(blocks=restrict_support(grid_subset(old["cases"][ap]["blocks"],gi),
         u,p["support_positions"],grid,old["base"],state["intercept"]),
         stats={n:v[u] for n,v in old["cases"][ap]["stats"].items()},
         rss=old["cases"][ap]["rss"],proposal=state["proposals"][ap]) for ap in state["APs"]}
    return dict(old,grid=grid,nearest=torch.cdist(old["truth"],grid).argmin(1),weights=state["weights"],cases=cases)

def train(steps):
    assert steps%30==0
    target=OUT/f"training_{steps}";target.mkdir(exist_ok=False)
    ready=json.loads((OUT/"complete.json").read_text())
    ids=[r["id"] for r in ready["records"] if r["role"]=="train"]
    valid=[r["id"] for r in ready["records"] if r["role"]=="validation"]
    tasks=[(i,k) for i in ids for k in KS]
    rng=np.random.default_rng(11304)
    schedule=[tasks[int(j)] for _ in range(steps//30) for j in rng.permutation(30)]
    checkpoints=[0,steps//3,2*steps//3,steps]
    arms=["scales","shape_scales","full"]
    freeze=dict(created=now(),code_sha256=digest(Path(__file__)),core_sha256=digest(ROOT/"recomposition.py"),
       prepared_sha256=digest(OUT/"complete.json"),arms=arms,schedule=schedule,checkpoints=checkpoints,
       objective="Equal parent tasks and K; full-scan nearest-grid localization NLL; original .001 prior penalty.",
       optimizer="Adam",learning_rate=.001,selection="Development mean MDE over two B1 draws and three K; retain identity. No independent test.",
       reverse_review="Same cached historical blocks, same per-K S-only states, proposals, training order and evaluation events in all arms. Each arm starts at identity. Differentiation includes full transformed prior and S posterior.",
       counterexamples="Low ESS or invalid moments stop, never silently resample with query labels. More parameters are not guaranteed useful. Source NLL decline is not a localization result.",
       wall_limit_seconds=4800,threads=2,independent_test=False)
    write(target/"freeze.json",freeze)
    parents=load_parents()
    assert digest(OUT/"normalization.pt")==ready["normalization_sha256"]
    norm=torch.load(OUT/"normalization.pt",weights_only=True)
    reference=None;summary={};start=time.monotonic()
    for arm in arms:
        model=Parameters(**norm,mode=arm);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        measurements={}
        def checkpoint(step):
            nonlocal reference
            vals=reference if step==0 and reference is not None else {
                f"{i}/K{k}":evaluate(model,episode(parents[i],k)) for i in valid for k in KS}
            if step==0:reference=vals
            phi,eta,lg,ls=model.values()
            measurements[step]=dict(validation=vals,by_K={str(k):float(np.mean([vals[f"{i}/K{k}"]["mde"] for i in valid])) for k in KS},
              parameters=dict(phi=phi.detach().tolist(),eta=float(eta.detach()),g=float(lg.exp().detach()),s=float(ls.exp().detach())))
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step),target/f"{arm}_{step}.pt")
            write(target/f"{arm}.json",measurements)
            print(json.dumps(dict(arm=arm,step=step,by_K=measurements[step]["by_K"],seconds=time.monotonic()-start)),flush=True)
        checkpoint(0)
        with (target/f"{arm}_train.jsonl").open("w") as log:
            for step,(i,k) in enumerate(schedule,1):
                optimizer.zero_grad(set_to_none=True);metric=evaluate(model,episode(parents[i],k),backward=True)
                if not torch.isfinite(model.theta.grad).all():raise RuntimeError("Nonfinite gradient")
                optimizer.step();log.write(json.dumps(dict(step=step,id=i,K=k,**metric))+"\n");log.flush()
                if step in checkpoints:checkpoint(step)
        chosen=min(measurements,key=lambda s:(np.mean(list(measurements[s]["by_K"].values())),s))
        summary[arm]=dict(selected=chosen,by_K=measurements[chosen]["by_K"],
                          final_by_K=measurements[steps]["by_K"])
    write(target/"complete.json",dict(complete=True,created=now(),arms=summary,
       identity_by_K={str(k):float(np.mean([reference[f"{i}/K{k}"]["mde"] for i in valid])) for k in KS},
       wknn_by_K={str(k):float(np.mean([parents[i][1]["states"][k]["wknn"]["mde"] for i in valid])) for k in KS},
       seconds=time.monotonic()-start,development_only=True))

if __name__=="__main__":
    if socket.gethostname()!="ray-web":raise RuntimeError("Server only")
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("prepare","train"))
    parser.add_argument("--limit",type=int,default=12);parser.add_argument("--steps",type=int,default=90)
    args=parser.parse_args()
    try:
        if args.action=="prepare":prepare_all(args.limit)
        else:train(args.steps)
    except Exception as e:
        if OUT.exists():write(OUT/("failure_"+str(time.time_ns())+".json"),dict(created=now(),action=args.action,error=repr(e)))
        raise


