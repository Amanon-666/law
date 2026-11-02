"""Post-hoc, read-only macro and paired spatial-block audit of the locked R1 draw-0 ZIP.

Requires all eight original shards, no re-fitting and no target-model selection.
Run: python -m research.r1_single_scan.analyze_locked /path/to/flat/shards
"""
import argparse
import json
from pathlib import Path

import numpy as np
from research.r1_single_scan.verify_locked import FLOORS, REGIMES, KS, ARMS, load

def per_position_errors(z, k, arm):
    xy = z["query_xy"]
    points, inverse = np.unique(xy, axis=0, return_inverse=True)
    e = np.linalg.norm(z[f"K{k}_{arm}"] - xy, axis=1)
    sums = np.bincount(inverse, weights=e, minlength=len(points))
    counts = np.bincount(inverse, minlength=len(points))
    return points, sums / counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--bootstraps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261102)
    args = ap.parse_args()
    data = {(reg, f): load(args.directory, reg, f)
            for reg in REGIMES for f in FLOORS}
    result = {
        "scope": "Fixed B1 draw 0; K1/3/5 one real scan; eight shards, 24 cells; not all 400 R1 events",
        "verified_shards": len(data), "verified_cells": len(data) * len(KS),
        "paired_CI_limit": "Conditional on four previously examined B1 floors and one support draw; 8-U spatial blocks paired across methods, 4000 replicates; not a cross-building or support-draw interval",
        "regimes": {},
    }
    rng = np.random.default_rng(args.seed)
    for reg in REGIMES:
        by_k, block_matrices, point_counts, block_counts = {}, [], [], []
        keys = [(k, a) for k in KS for a in ARMS]
        for f in FLOORS:
            j, z = data[(reg, f)]
            xy = z["query_xy"]
            points = np.unique(xy, axis=0)
            groups, bi = np.unique(np.floor(points / 8).astype("int64"), axis=0, return_inverse=True)
            errors = np.column_stack([per_position_errors(z, k, a)[1] for k, a in keys])
            block_matrices.append((
                np.asarray([errors[bi == u].sum(axis=0) for u in range(len(groups))]),
                np.bincount(bi, minlength=len(groups))
            ))
            point_counts.append(len(points))
            block_counts.append(len(groups))
        for k in KS:
            by_floor = {}
            for f in FLOORS:
                j, z = data[(reg, f)]
                e = j["results"][list(KS).index(k)]
                by_floor[f"F{f}"] = {
                    "mde": {a: e["metrics"][a]["mde"] for a in ARMS},
                    "p90": {a: e["metrics"][a]["p90"] for a in ARMS},
                    "wknn_archived": e["wknn_archived"],
                    "weights": e["weights"],
                    "ap_count": e["ap_count"],
                    "grid": e["grid"],
                }
            by_k[f"K{k}"] = {
                "mde": {a: float(np.mean([by_floor[f"F{f}"]["mde"][a] for f in FLOORS])) for a in ARMS},
                "p90": {a: float(np.mean([by_floor[f"F{f}"]["p90"][a] for f in FLOORS])) for a in ARMS},
                "wknn": {a: float(np.mean([by_floor[f"F{f}"]["wknn_archived"][a] for f in FLOORS]))
                         for a in by_floor["F0"]["wknn_archived"]},
                "per_floor": by_floor
            }
        boot = np.empty((args.bootstraps, len(keys)))
        for b in range(args.bootstraps):
            per_floor = []
            for block_sums, block_n in block_matrices:
                ix = rng.integers(len(block_n), size=len(block_n))
                per_floor.append(block_sums[ix].sum(axis=0) / block_n[ix].sum())
            boot[b] = np.mean(per_floor, axis=0)
        def delta(k1, a1, k2, a2):
            x = boot[:, keys.index((k1, a1))] - boot[:, keys.index((k2, a2))]
            return [float(v) for v in np.quantile(x, [.025, .975])]
        contrasts = {}
        for k in KS:
            for a, baseline in (("seven","gp"), ("nonlinear","gp"), ("nonlinear","seven")):
                contrasts[f"K{k}_{a}_minus_{baseline}"] = delta(k,a,k,baseline)
        for a in ARMS:
            for high,low in ((3,1),(5,1),(5,3)):
                contrasts[f"{a}_K{high}_minus_K{low}"] = delta(high,a,low,a)
        result["regimes"][reg] = {
            "results":by_k, "position_counts":point_counts, "block_counts":block_counts,
            "paired_block_CI95":contrasts
        }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))

if __name__ == "__main__":
    main()
