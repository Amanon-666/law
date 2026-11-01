"""Post hoc B1F1 check: replace R1 by every same-phone repeat at the identical XY."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from refi import fit_map
from .protocol import BUDGETS, SEEDS, history_floors, position_weights, read_floors, split_rows


def run(data, out):
    train = read_floors(data, "trainingData.csv")
    q = read_floors(data, "validationData.csv")[1, 1]
    floor = train[1, 1]
    phone = pd.read_csv(data/"trainingData.csv", usecols=["PHONEID"]).PHONEID.to_numpy()[floor["ids"]]
    histories = history_floors(train, 1, 1)
    hseen = np.logical_or.reduce([(h["rss"] != 100).any(0) for h in histories])
    w = position_weights(q["xy"])
    records = []
    for draw in SEEDS:
        order, _ = split_rows(floor, 1, 1, draw)
        # Compare the same sequence of updates across two query collections.
        # Exclude the whole largest support prefix before evaluating any K.
        reserved = set(map(tuple, floor["xy"][order]))
        iq = np.asarray([tuple(point) not in reserved for point in floor["xy"]])
        internal_w = position_weights(floor["xy"][iq])
        groups = [np.flatnonzero((floor["xy"] == floor["xy"][i]).all(1) & (phone == phone[i])) for i in order]
        for k in BUDGETS:
            r1 = order[:k]
            ap = hseen | (floor["rss"][r1] != 100).any(0)
            h = [dict(rss=x["rss"][:,ap], xy=x["xy"]) for x in histories]
            for arm, ids in [("R1", r1), ("same_phone_repeats", np.concatenate(groups[:k]))]:
                s = dict(rss=floor["rss"][ids][:,ap], xy=floor["xy"][ids])
                model = fit_map(h,s)
                if not model.visible.all() or len(np.unique(s["xy"],axis=0)) != k:
                    raise AssertionError("Query permission or support coordinates changed")
                err = np.linalg.norm(model.predict(q["rss"][:,ap]) - q["xy"], axis=1)
                internal = model.predict(floor["rss"][iq][:,ap])
                ierr = np.linalg.norm(internal - floor["xy"][iq], axis=1)
                records.append(dict(draw=draw,K=k,arm=arm,mde=float(w@err),n_scans=len(ids),
                                    internal_mde=float(internal_w @ ierr),
                                    internal_positions=len(np.unique(floor["xy"][iq], axis=0)),
                                    min_repeats=min(map(len,groups[:k])) if arm!="R1" else 1,
                                    max_repeats=max(map(len,groups[:k])) if arm!="R1" else 1,
                                    n_query=len(err),aps=int(ap.sum())))
    pd.DataFrame(records).to_csv(out,index=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data",type=Path); p.add_argument("out",type=Path)
    a=p.parse_args(); run(a.data,a.out)
