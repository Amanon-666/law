"""Paired spatial-block diagnostic of already frozen F0 official draw0 VI500 predictions.
Usage: python -m research.r1_single_scan.analyze_highk_bridge RAW_ARTIFACT.zip OUT.json
No training, model selection, or target-Q parameter tuning.
"""
import io,zipfile,json,sys
import numpy as np
from pathlib import Path

KS=(5,10,20)
ARMS=("gp","seven","nonlinear")
PAIRS=([(20,m,5,m) for m in ARMS]+
       [(10,m,5,m) for m in ARMS]+
       [(20,"seven",20,"gp"),(10,"seven",10,"gp"),(5,"seven",5,"gp"),(20,"nonlinear",20,"seven")])
def main(inp,out):
    scores={}
    with zipfile.ZipFile(inp) as z:
        for k in KS:
            a=np.load(io.BytesIO(z.read(f"official_F0_0_K{k}.npz")),allow_pickle=False)
            if k==5:
                xy=a["query_xy"].copy()
                assert len(xy)==30 and len(np.unique(xy,axis=0))==30
            assert np.array_equal(a["query_xy"],xy)
            for m in ARMS:
                pred=a[f"K{k}_{m}"]
                assert pred.shape==xy.shape and np.isfinite(pred).all()
                scores[(k,m)]=np.linalg.norm(pred-xy,axis=1)
    block=np.floor(xy/8).astype(np.int64)
    unique,ix=np.unique(block,axis=0,return_inverse=True)
    cells=[np.flatnonzero(ix==i) for i in range(len(unique))]
    rng=np.random.default_rng(261103)
    samples=[np.concatenate([cells[j] for j in rng.integers(len(cells),size=len(cells))])
             for _ in range(4000)]
    result={"scope":"Single B1 F0 official draw0; 30 exact positions; conditional spatial-block bootstrap only",
            "seed":261103,"block_size_U":8,"block_count":len(cells),"replicates":len(samples),
            "comparisons":{}}
    for ka,ma,kb,mb in PAIRS:
        diff=scores[(ka,ma)]-scores[(kb,mb)]
        draw=np.asarray([diff[ii].mean() for ii in samples])
        result["comparisons"][f"K{ka}_{ma} minus K{kb}_{mb}"]={
           "estimate":float(diff.mean()),
           "ci95":[float(x) for x in np.quantile(draw,[.025,.975])]}
    Path(out).write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))

if __name__=="__main__":
    main(sys.argv[1],sys.argv[2])
