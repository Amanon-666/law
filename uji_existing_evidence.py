"""Re-tabulate existing UJI results without inference or retraining.

Keep protocol families and internal/official queries separate. This is an
evidence index, not a new benchmark or a ranking across unequal information.
"""
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
MIG=WORK/"迁移学习"
OLD=WORK/"定位迁移研究"


def csv_rows(path):
    return list(csv.DictReader(path.open()))


def write_csv(path,rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    out=ROOT/"results/uji_existing_evidence";out.mkdir(exist_ok=False)
    sources=[];protocols=[]
    rows=[]
    for b in range(3):
        path=MIG/f"law_audit_20261022/building{b}.json";sources.append(path)
        rows.extend(json.loads(path.read_text())["rows"])
    for split in ["internal","official"]:
        groups=defaultdict(list)
        for r in rows:
            if r["split"]==split:groups[r["method"],str(r["K"]),r["building"],r["floor"]].append(r["mde"])
        floors=defaultdict(list)
        for (method,k,b,f),values in groups.items():floors[method,k,b].append(float(np.mean(values)))
        buildings=defaultdict(list)
        for (method,k,b),values in floors.items():buildings[method,k].append(float(np.mean(values)))
        write_csv(out/f"law_{split}.csv",[dict(method=m,K=k,mean_error=float(np.mean(v)),
            aggregation="equal buildings, equal floors, then draws",query=split)
            for (m,k),v in sorted(buildings.items())])
    protocols.append(dict(family="LAW audit",tables=["law_internal.csv","law_official.csv"],
       source="other two buildings' adjacent-floor pairs for each held-out building",targets="10 upper floors, rotating held-out building",
       K="0/5/10/30/full as in archived rows",scans="3 records per S position for finite K",
       dedup="historical cells deduplicate identical coordinate+RSS; extra support draws sample raw records; original split row semantics not re-audited here",
       ap="all 520 schema columns",deployment="corresponding lower-floor historical fingerprints plus S",
       query="internal fixed position-disjoint pool and all official validation separately; official coordinate overlap allowed",
       metric="archived raw projected coordinate distance",limits="not rankable with no-H methods on different target subsets/budgets"))

    path=OLD/"dkt_official_validation/summary.csv";sources.append(path);summary=csv_rows(path)
    for split,key in [("internal","old_training_query_mde_m"),("official","official_mde_m")]:
        rows=[dict(floor=r["target"],method=r["method"],K=10,S_scans=30,mean_error=float(r[key]),
                   query=split,source_training="DKT:10 floors; frozen TabICL:external pretraining only; WKNN:none") for r in summary]
        if split=="official":
            p=OLD/"tabicl_source_final/outputs/floor_summary.csv";sources.append(p)
            rows.extend(dict(floor=r["target"],method="TabICLv2_source_trained",K=10,S_scans=30,
                        mean_error=float(r["trained_mde_m"]),query=split,source_training="7 floors, one training seed, dev-selected700 scalar updates") for r in csv_rows(p))
        write_csv(out/f"fewshot_K10_{split}.csv",rows)
    protocols.append(dict(family="DKT / frozen TabICL / source-trained TabICL",tables=["fewshot_K10_internal.csv","fewshot_K10_official.csv"],
       source="DKT10 source floors vs TabICL7 source floors; three top floors held out, other floors of each target building in source",
       targets="B0F3/B1F3/B2F4",K=10,scans="3 distinct fingerprints per position,30 total",
       dedup="support candidates deduplicate identical fingerprints within position; query keeps raw records",
       ap="520 fixed schema columns; TabICL official preprocessing filters S-constant columns",
       deployment="no H; zero target network updates",query="internal all other-position training scans; official all validation rows, B1F3 overlap in11/60 tasks",
       metric="scan MDE; three sampling seeds x20 tasks; not 180 independent environments",
       limits="paired target evidence; unequal source data/number of training seeds prevents attributing cross-model rank solely to architecture"))

    groups=defaultdict(list)
    for folder in ["tabicl_target_k10","tabicl_target_other","rnp_full_seed0","rnp_full_seed1","rnp_full_seed2"]:
        p=MIG/f"RNP-UJI/outputs/{folder}/results.csv";sources.append(p)
        for r in csv_rows(p):
            if r["method"] in ["TabICLv2","WKNN"]:groups[r["target"],int(r["shots"]),r["method"]].append(float(r["mde_m"]))
    write_csv(out/"frozen_tabicl_internal_budgets.csv",[dict(floor=f,K=k,method=m,tasks=len(v),mean_error=float(np.mean(v)))
              for (f,k,m),v in sorted(groups.items())])

    baseline=MIG/"ConditionalFingerprint-20261023/outputs/uji_v0_20261023/evaluation/curve.json"
    neural=MIG/"ConditionalFingerprint-20261023/outputs/uji_budget20000_20261023/evaluation/curve.json"
    sources.extend([baseline,neural]);combined=[]
    for p in [baseline,neural]:
        for r in json.loads(p.read_text())["results"]:
            if (p==baseline and r["method"]!="conditional") or (p==neural and r["method"]=="conditional"):
                combined.append(r)
    groups=defaultdict(list)
    for r in combined:groups[r["method"],r["budget"]].append(r["mean_error"])
    write_csv(out/"conditional_B2_official.csv",[dict(method=m,K=k,mean_error=float(np.mean(v)),floor_draw_cases=len(v),
                      aggregation="five floors equal, three draws (full uses one)") for (m,k),v in sorted(groups.items())])
    raw=OLD/"mcel-reproduction/server-evidence/upstream/datasets/uji/trainingData.csv"
    coords=np.loadtxt(raw,delimiter=",",skiprows=1,usecols=[520,521,522,523])
    scan_budgets=[]
    for f in range(5):
        xy=coords[(coords[:,2]==f)&(coords[:,3]==2),:2]
        positions,counts=np.unique(xy,axis=0,return_counts=True)
        for seed in [0,1,2]:
            order=np.random.default_rng(seed).permutation(len(positions))
            for k in [5,10,20,40,80]:
                if k<=len(positions):
                    n=counts[order[:k]];scan_budgets.append(dict(floor=f"B2F{f}",seed=seed,K=k,
                       S_scans=int(n.sum()),scans_per_position_min=int(n.min()),scans_per_position_max=int(n.max())))
    write_csv(out/"conditional_actual_scan_budgets.csv",scan_budgets)
    protocols.append(dict(family="ConditionalFingerprint20000 + local controls",tables=["conditional_B2_official.csv","conditional_actual_scan_budgets.csv"],
       source="select B0, develop B1, refit B0+B1; evaluate all B2 floors",targets="B2F0..F4",K="5/10/20/40/80/full",
       scans="all actual scans at each selected position, variable count; reconstructed in separate table",
       dedup="raw duplicates retained",ap="Support-detected APs only",deployment="no H; dense target reference coordinates used for grid bounds",
       query="all official target rows; Support-coordinate overlaps not excluded",metric="equal position mean scan error; not identical to scan-MDE family",
       limits="GP/source-selected hyperparameters and WKNN choices reused; full-H LAW not an information-matched comparison"))
    (out/"protocol_families.json").write_text(json.dumps(protocols,indent=2,ensure_ascii=False))
    evidence=dict(kind="existing saved result re-tabulation only; no new inference",sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  scan_budget_source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"provenance.json").write_text(json.dumps(evidence,indent=2));print(json.dumps(dict(completed=True,tables=[p.name for p in out.glob('*.csv')]),ensure_ascii=False))


if __name__=="__main__":main()
