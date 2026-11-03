"""Independent, NumPy-only audit of original R1 F0 draw0 high-K Gaussian VI bridge.

Invocation: python -m research.r1_single_scan.verify_highk_bridge OUTDIR OLD_K5_DIR
Never selects a model; diagnostics distinguish numerical backend from learned gain.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np

KS=(5,10,20)
ARMS=("gp","seven","nonlinear")
MANIFEST="5f4fb0ceccc5a64aa7411cc287715e6fa9a39d593da6713910920a3fda67c613"
DEVICE="76763a87569d885705dccc54cbd72e878f7a711cbb5343fbfee4517caf2482e6"
NONLINEAR="53061b01164f78def406e7b1395c5e62a295adb9a9a3570a2b91efed7b71e7c7"
SEVEN="cd5c4972f4ad418a58c9c73f88d9c65c245f24ecf5c35842bcdc8535114caf58"

def metric(pred,truth):
    _,inverse,count=np.unique(truth,axis=0,return_inverse=True,return_counts=True)
    weights=1/count[inverse]
    weights=weights/weights.sum()
    distances=np.linalg.norm(pred-truth,axis=1)
    order=np.argsort(distances)
    return dict(mde=float(weights@distances),
                p90=float(distances[order][np.searchsorted(np.cumsum(weights[order]),.9)]))

def main(folder,old):
    folder,old=Path(folder),Path(old)
    manifest=json.loads((folder/"highk_manifest.json").read_text())
    oldmeta=json.loads((old/"official_F0_0.json").read_text())
    with np.load(old/"official_F0_0.npz",allow_pickle=False) as zz:
        oldpred={k:zz[k].copy() for k in zz.files}
    assert manifest["historical_floor_ids"]==[1,2,3]
    assert (manifest["building"],manifest["floor"],manifest["draw"],manifest["R"])==(1,0,0,1)
    assert manifest["manifest_hash"]==MANIFEST and manifest["device_hash"]==DEVICE
    assert manifest["affine_source_sha256"]==SEVEN and manifest["model_source_sha256"]==NONLINEAR
    assert manifest["query_rows_sha256"]==oldmeta["query_rows_sha256"]
    assert manifest["support_order_sha256"]==oldmeta["results"][0]["support_order_sha256"] if "support_order_sha256" in oldmeta["results"][0] else manifest["support_order_sha256"]==oldmeta["support_order_sha256"]
    assert len(oldpred["query_ids"])==30
    recorded=dict(scope="B1 F0 official draw0 only, K5 VI bridge + K10/20 VI; not complete R1",backend="GaussianVI500",comparison={},failures=[],source_run="41443707993")
    previous_ids=None
    for k in KS:
        path=folder/f"official_F0_0_K{k}.json"
        if not path.exists():
            recorded["failures"].append(f"K{k}: missing JSON");continue
        entry=json.loads(path.read_text())
        if entry.get("K")!=k or entry.get("backend")!="VI500":
            recorded["failures"].append(f"K{k}: backend or budget mismatch");continue
        names=entry.get("completed_modes",[])
        if names!=list(ARMS) or entry.get("failures"):
            recorded["failures"].append(f"K{k}: incomplete modes {names}, failures {entry.get('failures')}")
        arrpath=folder/f"official_F0_0_K{k}.npz"
        if not arrpath.exists():recorded["failures"].append(f"K{k}: missing predictions");continue
        with np.load(arrpath,allow_pickle=False) as z:dat={n:z[n].copy() for n in z.files}
        assert np.array_equal(dat["query_ids"],oldpred["query_ids"])
        assert np.array_equal(dat["query_xy"],oldpred["query_xy"])
        ids=entry["support_ids"]
        assert len(ids)==k and len(set(ids))==k
        assert entry["support_hash"]==hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest()
        assert previous_ids is None or ids[:len(previous_ids)]==previous_ids
        previous_ids=ids
        assert entry["ap_count"]>=oldmeta["results"][2]["ap_count"]
        for arm in ARMS:
            key=f"K{k}_{arm}"
            if key not in dat:
                recorded["failures"].append(f"{key} missing");continue
            coords=dat[key]
            assert coords.shape==(30,2) and np.isfinite(coords).all()
            independently=metric(coords,dat["query_xy"])
            if arm not in entry.get("metrics",{}):
                recorded["failures"].append(f"{key}: missing metrics");continue
            saved=entry["metrics"][arm]
            assert abs(independently["mde"]-saved["mde"])<1e-8,(key,"MDE",independently,saved)
            assert abs(independently["p90"]-saved["p90"])<1e-8,(key,"P90",independently,saved)
            assert np.isfinite(entry["vi_min_diagonal"][arm]) and entry["vi_min_diagonal"][arm]>0
            comparison=dict(vi=independently,vi_min_diagonal=entry["vi_min_diagonal"][arm],
                            seconds=entry["mode_seconds"][arm])
            if k==5:
                oldk5=oldmeta["results"][2]
                assert entry["ap_count"]==oldk5["ap_count"] and entry["grid"]==oldk5["grid"]
                assert ids==oldk5["support_ids"]
                assert np.allclose(entry["weights"],oldk5["weights"],atol=1e-10,rtol=1e-9)
                oldcoords=oldpred[key]
                oldmetric=metric(oldcoords,dat["query_xy"])
                assert abs(oldmetric["mde"]-oldk5["metrics"][arm]["mde"])<1e-8
                comparison.update(mpmc=oldmetric,delta_vi_minus_mpmc_mde=independently["mde"]-oldmetric["mde"],
                                  mean_coordinate_shift=float(np.linalg.norm(coords-oldcoords,axis=1).mean()),
                                  max_coordinate_shift=float(np.linalg.norm(coords-oldcoords,axis=1).max()))
            recorded["comparison"][key]=comparison
    resultfile=folder/"INDEPENDENT_BRIDGE_CHECK.json"
    resultfile.write_text(json.dumps(recorded,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps(recorded,indent=2,ensure_ascii=False,allow_nan=False))
    assert len(recorded["comparison"])==9 and not recorded["failures"],"Incomplete high-K locked numerical bridge"

if __name__=="__main__":
    main(sys.argv[1],sys.argv[2])
