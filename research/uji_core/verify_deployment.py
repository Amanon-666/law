"""Replay the public deployment interface on each official episode."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from refi import fit_map
from .protocol import BUDGETS, SEEDS, read_floors, split_rows, history_floors, position_weights


def main(data, out):
    train = read_floors(data, "trainingData.csv")
    valid = read_floors(data, "validationData.csv")
    rows = []
    for (b, f), target in train.items():
        hist = history_floors(train, b, f)
        q = valid[b, f]
        for draw in SEEDS:
            sr, _ = split_rows(target, b, f, draw)
            for k in BUDGETS:
                support = {key:target[key][sr[:k]] for key in ("rss","xy")}
                p = fit_map(hist, support).predict(q["rss"])
                errors = np.linalg.norm(p-q["xy"], axis=1)
                rows.append(dict(building=b, floor=f, draw=draw, K=k,
                    mde=position_weights(q["xy"]) @ errors, scan_mde=errors.mean()))
    result = pd.DataFrame(rows)
    keys = ["building","floor","draw","K"]
    expected = pd.read_csv(out/"shape_events.csv").query('method=="smooth"').set_index(keys)
    actual = result.set_index(keys)
    if not actual.index.is_unique or not actual.index.sort_values().equals(expected.index.sort_values()):
        raise ValueError("Deployment and evaluation episode sets differ")
    np.testing.assert_allclose(actual.mde, expected.mde.reindex(actual.index), atol=1e-8, rtol=0)
    result.to_csv(out/"deployment_check.csv", index=False)
    print("Verified", len(result), "complete deployment episodes")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    a=p.parse_args(); main(a.data,a.out)
