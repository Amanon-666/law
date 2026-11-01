"""Replay the one public entry point against the 1,300 paired round-two cases."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from refi import fit_map
from research.uji_core.protocol import read_floors, split_rows, position_weights, SEEDS, BUDGETS


def run(data, phone_design, receipts):
    train = read_floors(data, "trainingData.csv")
    later = read_floors(data, "validationData.csv")
    phones = pd.read_csv(data/"trainingData.csv", usecols=["PHONEID"]).PHONEID.to_numpy()
    enroll = {(r["building"], r["floor"]): r for r in json.loads(phone_design.read_text())["events"]}
    records = []
    for name in ("official", "phone"):
        choices = pd.read_csv(receipts/(name + ".csv"))
        expected = choices.query('method=="gap_only"').set_index(["building","floor","draw","K"])
        for (b, f), target in train.items():
            floors = [(hf,h) for (hb,hf), h in train.items() if hb==b and hf!=f]
            if name == "phone":
                entry = enroll[b,f]
                blocked = entry["phone"]
                floors = [(hf,{k:v[phones[h["ids"]]!=blocked] for k,v in h.items()}) for hf,h in floors]
                rows = {int(r):i for i,r in enumerate(target["ids"])}
                qrow = [rows[r] for r in entry["query_ids"]]
                q = {k:v[qrow] for k,v in target.items()}
                orders = [[rows[r] for r in order] for order in entry["support_orders"]]
            else:
                q = later[b,f]
                orders = [split_rows(target,b,f,i)[0] for i in SEEDS]
            for draw, order in enumerate(orders):
                for k in BUDGETS:
                    s = {field:values[order[:k]] for field,values in target.items()}
                    model = fit_map([h for _,h in floors], s, floor_gaps=[hf-f for hf,_ in floors])
                    dist = np.linalg.norm(model.predict(q["rss"])-q["xy"],axis=1)
                    mde = position_weights(q["xy"]) @ dist
                    ref = expected.at[(b,f,draw,k),"mde"]
                    if abs(ref-mde) > 1e-8: raise AssertionError((name,b,f,draw,k,ref,mde))
                    records.append((name,b,f,draw,k,mde,abs(ref-mde)))
    return pd.DataFrame(records,columns=["regime","building","floor","draw","K","mde","abs_difference"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data","phone_design","receipts","out"):p.add_argument(key,type=Path)
    a=p.parse_args()
    result=run(a.data,a.phone_design,a.receipts)
    result.to_csv(a.out,index=False)
    print(len(result),result.abs_difference.max())
