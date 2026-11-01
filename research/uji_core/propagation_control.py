"""On matched S, keep real anchors fixed and vary only the update of H cells."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from external_baselines import tpm_adapter as tpm
from .forward import FIXED_RADIUS, RIDGE, MATCH
from .protocol import BUDGETS, SEEDS, read_floors, history_floors, split_rows, position_weights


def run(data, out):
    train = read_floors(data, "trainingData.csv")
    valid = read_floors(data, "validationData.csv")
    rows = []
    for (b, f), target in train.items():
        hist = history_floors(train, b, f)
        hp, hx = tpm.history_map(hist, None, None)
        hseen = np.logical_or.reduce([(h["rss"] != 100).any(0) for h in hist])
        q = valid[b, f]
        w = position_weights(q["xy"])
        for draw in SEEDS:
            srows, _ = split_rows(target, b, f, draw)
            for k in BUDGETS:
                sr = srows[:k]
                raw = target["rss"][sr]
                visible = hseen | (raw != 100).any(0)
                # Preserve random support draw order: a cyclic reassignment has no
                # fixed point for K > 1 and does not preferentially keep nearby XY.
                sxy = target["xy"][sr]
                anchors = tpm.dbm(raw[:, visible])
                prior = hp[:, visible], hx
                residual = anchors - tpm.lookup(sxy, hx, prior[0])
                output_xy = np.vstack((hx, sxy))
                for arm in ("aligned", "permuted", "constant"):
                    if arm == "constant":
                        # K_SS = 11' and observation ridge=1.
                        h_adj = np.clip(prior[0] + residual.sum(0) / (k + RIDGE), -110., 0.)
                        prototypes = np.vstack((h_adj, anchors))
                    else:
                        r = residual if arm == "aligned" else np.roll(residual, 1, axis=0)
                        prototypes, _ = tpm.propagate(prior, anchors, sxy, r, (FIXED_RADIUS, RIDGE))
                    weights = tpm.tobit_weights(q["rss"][:, visible], prototypes, *MATCH)
                    pred = weights @ (output_xy - hx[0]) + hx[0]
                    error = np.linalg.norm(pred - q["xy"], axis=1)
                    rows.append((b, f, draw, k, arm, w @ error))
    pd.DataFrame(rows, columns=["building", "floor", "draw", "K", "method", "mde"]).to_csv(out, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    run(args.data, args.out)
