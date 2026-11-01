"""One content-conditioned historical map transform; learn directly on post-S MDE.

The two learned coefficients scale the same per-AP historical local contrast
at weak and strong local signal levels. Global contrast is the tied-weight
submodel; zero recovers the unmixed historical map. No changes to S propagation,
candidates, AP catalogue, matcher or coordinate readout.
"""
import argparse
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from external_baselines import tpm_adapter as tpm
from .protocol import BUDGETS, SEEDS, read_floors, history_floors, split_rows, position_weights

COEFFICIENTS = (0., -.25, .25, -.5, .5)
GRID = tuple(product(COEFFICIENTS, repeat=2))


def basis(m, xy, length=10.):
    dist2 = cdist(xy, xy, "sqeuclidean")
    smooth = np.exp(-dist2/(2*length**2))
    smooth /= smooth.sum(1, keepdims=True)
    local = smooth @ m
    contrast = m - local
    strength = np.clip((local+110.) / 50., 0., 1.)
    return (1.-strength)*contrast, strength*contrast


def conditioned(m, xy, weak, strong, sxy, sy):
    """Exploit linearity to propagate each prior basis with exactly the same S."""
    cross = np.exp(-cdist(xy, sxy, "sqeuclidean")/(2*40.**2))
    ss = np.exp(-cdist(sxy, sxy, "sqeuclidean")/(2*40.**2))
    projection = np.linalg.solve(ss + np.eye(len(sxy)), cross.T).T
    base = m + projection @ (sy - tpm.lookup(sxy, xy, m))
    return (base, weak - projection @ tpm.lookup(sxy, xy, weak),
            strong - projection @ tpm.lookup(sxy, xy, strong))


def measure(train, official=None, frozen=None):
    rows = []
    for (b, f), target in train.items():
        hist = history_floors(train, b, f)
        full, xy = tpm.history_map(hist, None, None)
        weak, strong = basis(full, xy)
        visible = np.logical_or.reduce([(h["rss"] != 100).any(0) for h in hist])
        for draw in SEEDS:
            srows, qrows = split_rows(target, b, f, draw, source=official is None)
            q = ({key: target[key][qrows] for key in ("rss", "xy")}
                 if official is None else official[b,f])
            weight = position_weights(q["xy"])
            for k in BUDGETS:
                sr = srows[:k]
                mask = visible | (target["rss"][sr] != 100).any(0)
                sy, sxy = tpm.position_map(tpm.dbm(target["rss"][sr][:, mask]), target["xy"][sr])
                base, w, t = conditioned(full[:,mask], xy, weak[:,mask], strong[:,mask], sxy, sy)
                coords = np.vstack((xy, sxy)) - xy[0]
                query = q["rss"][:, mask]
                arms = {str(pair):pair for pair in GRID} if official is None else frozen[str(b)]
                cache = {}
                for name, params in arms.items():
                    a, c = params
                    if (a, c) not in cache:
                        prototypes = np.vstack((np.clip(base+a*w+c*t,-110.,0.), sy))
                        pred = tpm.tobit_weights(query, prototypes, 12., -85.) @ coords + xy[0]
                        cache[a, c] = weight @ np.linalg.norm(pred-q["xy"], axis=1)
                    rows.append((b,f,draw,k,name,a,c,cache[a,c]))
    return pd.DataFrame(rows, columns=["building","floor","draw","K","method","weak","strong","mde"])


def main(data, out):
    out.mkdir(parents=True, exist_ok=True)
    train = read_floors(data, "trainingData.csv")
    losses = measure(train)
    floor = losses.groupby(["building","floor","K","weak","strong"], as_index=False).mde.mean()
    floor.to_csv(out/"shape_source.csv", index=False)
    by_building = floor.groupby(["building","weak","strong"]).mde.mean()
    selections = {}
    for b in sorted({b for b,_ in train}):
        risk = by_building.drop(index=b, level="building").groupby(["weak","strong"]).mean()
        selections[str(b)] = dict(identity=(0.,0.), smooth=(-.5,-.5), sharpen=(.5,.5),
            global_fit=min(((a,a) for a in COEFFICIENTS), key=lambda p: risk.loc[p]),
            content_fit=min(GRID, key=lambda p: risk.loc[p]))
    (out/"shape_parameters.json").write_text(json.dumps(dict(choices=selections,
        grid=COEFFICIENTS, source_objective="post-S location MDE, equal buildings/floors/K/draws",
        weak_strong="local mean strength interpolates between two gains; 10 U smoothing, 40 U propagation",
        scope="post hoc development hypothesis on repeatedly examined UJI buildings; no official fitting"), indent=2)+"\n")
    test = read_floors(data, "validationData.csv")
    measure(train, test, selections).to_csv(out/"shape_events.csv", index=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    args=p.parse_args()
    main(args.data, args.out)
