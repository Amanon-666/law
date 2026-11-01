"""Run from repo root: python -m research.uji_core.study DATA_DIR OUT_DIR."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .inverse import HistoryMap, Episode, signal
from .protocol import BUDGETS, SEEDS, HASHES, history_floors, read_floors, split_rows, position_weights

NEIGHBORS = (1, 3, 7, 15)
SCALES = (1., .7, .5, .35, 1.4)
TRANSPORT = (0., .25, .5, .75, 1.)


def episodes(train, official, b, f):
    target = train[b, f]
    bank = HistoryMap(history_floors(train, b, f))
    source = official is None
    draws = [split_rows(target, b, f, seed, source) for seed in SEEDS]
    # Compute H distances once for just the target scans that will be requested.
    needed = np.unique(np.concatenate([np.r_[s, q] for s, q in draws]))
    inverse = {r: i for i, r in enumerate(needed)}
    features = signal(target["rss"][needed])
    hm = bank.match(features, max(NEIGHBORS))
    if not source:
        valid = official[b, f]
        q = signal(valid["rss"])
        qmatch = bank.match(q, max(NEIGHBORS))
        truth, qids = valid["xy"], valid["ids"]
    for seed, (srows, qrows) in zip(SEEDS, draws):
        if source:
            qi = [inverse[i] for i in qrows]
            q, qmatch = features[qi], (hm[0][qi], hm[1][qi])
            truth, qids = target["xy"][qrows], target["ids"][qrows]
        for budget in BUDGETS:
            sr = srows[:budget]
            si = [inverse[i] for i in sr]
            ep = Episode(bank, q, features[si], target["xy"][sr], target["ids"][sr],
                         qmatch, (hm[0][si], hm[1][si]))
            yield seed, budget, ep, truth, qids, target["ids"][sr]


def source_losses(train):
    records = []
    for b, f in sorted(train):
        for _, _, ep, truth, _, _ in episodes(train, None, b, f):
            for k in NEIGHBORS:
                anchor = ep.anchors(k)
                for scale in SCALES:
                    direct, shift, _ = ep.predict_parts(k, scale, anchor)
                    errors = direct[None] + np.asarray(TRANSPORT)[:, None, None]*shift[None] - truth
                    losses = np.linalg.norm(errors, axis=2).mean(1)
                    records.extend((b, f, k, scale, g, v) for g, v in zip(TRANSPORT, losses))
    table = pd.DataFrame(records, columns=["building", "floor", "neighbors", "scale", "transport", "source_mde"])
    return table.groupby(list(table.columns[:-1]), as_index=False).mean()


def select(losses, held):
    keys = ["neighbors", "scale", "transport"]
    source = losses[losses.building != held].groupby(["building", *keys]).source_mde.mean()
    score = source.groupby(keys).mean()
    def best(options):
        return min(options, key=lambda opt: score.loc[opt])
    # Select WKNN k first on sources, then hold it fixed for all correction ablations.
    k, _, _ = best((k, 1., 0.) for k in NEIGHBORS)
    direct = best((k, s, 0.) for s in SCALES)
    joint = best((k, s, g) for s in SCALES for g in TRANSPORT)
    return dict(pooled3=(3, 1., 0.), pooled_tuned=(k, 1., 0.),
                fixed_rule=(k, .7, 1.), target_priority=direct, transported=joint)


def evaluate(train, official, configs):
    results, predictions, design = [], [], []
    for b, f in sorted(train):
        conf = configs[str(b)]
        for seed, budget, ep, truth, qids, sids in episodes(train, official, b, f):
            w = position_weights(truth)
            if budget == max(BUDGETS):
                design.append(dict(building=b, floor=f, draw=seed, support_order=sids.tolist(), query_ids=qids.tolist()))
            k = conf["pooled_tuned"][0]
            a = ep.anchors(k)
            values = {"H_only": (a[0], None), "S_only3": (ep.local(3), None)}
            for name, (n, scale, gamma) in conf.items():
                direct, shift, sw = ep.predict_parts(n, scale, a if k == n else ep.anchors(n))
                values[name] = (direct + gamma*shift, sw)
            for name, (pred, mass) in values.items():
                errors = np.linalg.norm(pred-truth, axis=1)
                if not np.isfinite(errors).all():
                    raise ValueError((b, f, seed, budget, name))
                results.append((b, f, seed, budget, name, w @ errors,
                                np.nan if mass is None else w @ mass))
                predictions.append(pd.DataFrame(dict(building=b, floor=f, draw=seed, K=budget,
                    method=name, query_id=qids, x=pred[:, 0], y=pred[:, 1], error=errors)))
    cols = ["building", "floor", "draw", "K", "method", "mde", "support_mass"]
    return pd.DataFrame(results, columns=cols), pd.concat(predictions, ignore_index=True), design


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    train = read_floors(args.data, "trainingData.csv")
    loss = source_losses(train)
    loss.to_csv(args.output/"inverse_source.csv", index=False)
    configs = {str(b): select(loss, b) for b in sorted({b for b, _ in train})}
    (args.output/"inverse_parameters.json").write_text(json.dumps(configs, indent=2) + "\n")
    # The official table is first read only after all three outer fits are saved.
    official = read_floors(args.data, "validationData.csv")
    events, predictions, design = evaluate(train, official, configs)
    events.to_csv(args.output/"inverse_events.csv", index=False)
    predictions.to_csv(args.output/"inverse_queries.csv.gz", index=False)
    (args.output/"manifest.json").write_text(json.dumps(dict(checksums=HASHES, index="zero-based CSV data row", episodes=design)) + "\n")


if __name__ == "__main__":
    main()
