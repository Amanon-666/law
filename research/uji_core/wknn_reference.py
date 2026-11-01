"""Compare raw- and position-prototype WKNN on the same study.py episodes.

Run from the repository root: python -m research.uji_core.wknn_reference DATA OUT
No source/target priority or residual transport is applied to either reference.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import HASHES, position_weights, read_floors
from .study import NEIGHBORS, episodes


def pool_floor(floor):
    """Average filled RSS within each floor and exact position; preserve equal-position ties."""
    xy, group = np.unique(floor["xy"], axis=0, return_inverse=True)
    group = group.ravel()
    values = np.where(floor["rss"] == 100, -110., floor["rss"])
    total = np.zeros((len(xy), values.shape[1]), dtype=np.float64)
    np.add.at(total, group, values)
    ids = np.full(len(xy), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(ids, group, floor["ids"])
    return dict(xy=xy, ids=ids, rss=total / np.bincount(group)[:, None])


def measure(train, prototypes, official, choices):
    records = []
    for (b, f), target in train.items():
        for representation, historical in (("raw", train), ("prototype", prototypes)):
            # The target always retains its real, single-scan supports and internal source queries.
            inputs = dict(historical)
            inputs[b, f] = target
            if choices is None:
                selections = [(str(k), k) for k in NEIGHBORS]
            else:
                selections = [(method, c["neighbors"]) for method, c in choices[str(b)].items()
                              if c["reference"] == representation]
            for seed, budget, ep, truth, _, _ in episodes(inputs, official, b, f):
                weights = position_weights(truth)
                by_k = {}
                for method, k in selections:
                    if k not in by_k:
                        xy, _, _ = ep.predict_parts(k, 1., ep.anchors(k))
                        errors = np.linalg.norm(xy - truth, axis=1)
                        if not np.isfinite(errors).all():
                            raise FloatingPointError((b, f, seed, budget, method))
                        by_k[k] = float(weights @ errors)
                    records.append((b, f, seed, budget, method, representation, k, by_k[k]))
    return pd.DataFrame(records, columns=("building", "floor", "draw", "K", "method",
                                          "reference", "neighbors", "mde"))


def select(source):
    grouped = source.groupby(["building", "reference", "neighbors"]).mde.mean()
    result = {}
    for held in sorted(source.building.unique()):
        risk = grouped.drop(index=held, level="building").groupby(["reference", "neighbors"]).mean()
        # Declared tie preference: raw scans, then fewer neighbors.
        methods = {name+"_fixed3": dict(reference=name, neighbors=3)
                   for name in ("raw", "prototype")}
        options = [(name, k) for name in ("raw", "prototype") for k in NEIGHBORS]
        for name in ("raw", "prototype", "joint"):
            permitted = [op for op in options if name == "joint" or op[0] == name]
            ref, k = min(permitted, key=lambda op: risk.loc[op])
            methods[name+"_tuned"] = dict(reference=ref, neighbors=k)
        result[str(held)] = methods
    return result


def main(directory, output):
    output.mkdir(exist_ok=True, parents=True)
    train = read_floors(directory, "trainingData.csv")
    prototypes = {task: pool_floor(floor) for task, floor in train.items()}
    source = measure(train, prototypes, None, choices=None)
    source = source.groupby(["building", "floor", "reference", "neighbors"], as_index=False).mde.mean()
    source.to_csv(output/"wknn_source.csv", index=False)
    choices = select(source)
    record = dict(sha256=HASHES, source_objective="Internal UCI training-file MDE. Equal K, draw, floor "
                  "and building weighting. Entire target building excluded from all 3 selections.",
                  reference="raw H scans or per-floor exact XY prototypes after 100->-110 filled mean. "
                  "Append the same R=1 S; RMS over H+current S AP union; inverse distance; "
                  "tie by raw CSV id (minimum member id per prototype).",
                  candidates=list(NEIGHBORS), selected=choices)
    frozen = output/"wknn_parameters.json"
    frozen.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    selected = json.loads(frozen.read_text())["selected"]
    official = read_floors(directory, "validationData.csv")
    result = measure(train, prototypes, official, choices=selected)
    result.to_csv(output/"wknn_events.csv", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.data, args.output)
