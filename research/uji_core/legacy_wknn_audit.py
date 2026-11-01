"""Reconstruct the two UJI training-file episodes from run_k_curve.py and audit raw WKNN.

Run from any directory: python legacy_wknn_audit.py path/to/trainingData.csv --output results.json
Coordinates and MDE are in original UJI units. No fitted weights or query-driven selection.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def load_floors(path):
    columns = [f"WAP{i:03}" for i in range(1, 521)]
    floors = {}
    with path.open(newline="") as f:
        for row_id, row in enumerate(csv.DictReader(f), start=1):
            b, floor = int(row["BUILDINGID"]), int(row["FLOOR"])
            if b not in (0, 2):
                continue
            rows = floors.setdefault((b, floor), [])
            rows.append((row_id, (float(row["LONGITUDE"]), float(row["LATITUDE"])),
                         [int(row[ap]) for ap in columns]))
    return {key: tuple(map(np.asarray, zip(*rows))) for key, rows in floors.items()}


def split_rows(xy, seed):
    # Match public_corpus/k_curve_episodes.split_rows, including the order of scan draws.
    positions, group = np.unique(xy, axis=0, return_inverse=True)
    group = group.ravel()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(positions))

    def draw(ids):
        return [rng.choice(np.flatnonzero(group == i)) for i in ids]

    first = draw(order[:5])
    query = draw(order[5:17])
    extra = draw(order[17:32])
    return np.array(first + extra), np.array(query)


def neighbors(rss, q):
    # full_ap_study.wknn(raw): ties use the same NumPy default argsort.
    d = np.sqrt(np.mean((rss - q) ** 2, axis=1))
    ids = np.argsort(d)[:3]
    weights = 1 / np.maximum(d[ids], 1e-8)
    return ids, weights / weights.sum(), d


def audit_event(hxy, hrss, sxy, srss, qxy, qrss):
    active = (hrss != 100).any(0) | (srss != 100).any(0)
    known = np.vstack((hrss, srss))[:, active]
    known = np.where(known == 100, -110, known)
    queries = qrss[:, active]
    queries = np.where(queries == 100, -110, queries)
    xy = np.vstack((hxy, sxy))
    # Use the historical scan-weighted origin and low/high rounding from the old protocol.
    origin = hxy.mean(0)
    lower = 2 * np.floor((xy - origin).min(0) / 2)
    upper = 2 * np.ceil((xy - origin).max(0) / 2)
    errors, local_errors, history_errors = [], [], []
    n_support, repeated_xy, repeated_all_xy = [], [], []
    cutoff_ties = []
    for qi, (q, point) in enumerate(zip(queries, qxy)):
        ids, w, distances = neighbors(known, q)
        sl, sw, _ = neighbors(known[len(hxy):], q)
        hl, hw, _ = neighbors(known[:len(hxy)], q)
        # Only ties across different coordinates with a partially filled boundary
        # can move the estimate. Record the exact range for a single vacant slot.
        boundary = np.flatnonzero(distances == distances[ids[-1]])
        strict = ids[distances[ids] < distances[ids[-1]]]
        slots = len(ids) - len(strict)
        if len(boundary) > slots and len(np.unique(xy[boundary], axis=0)) > 1:
            tie = dict(query_offset=qi, open_slots=slots, tied_scans=len(boundary))
            if slots == 1:
                fixed = w[:len(strict)] @ (xy[strict] - point)
                choices = fixed + w[-1] * (xy[boundary] - point)
                choices_error = np.linalg.norm(choices, axis=1)
                tie["possible_errors"] = np.unique(choices_error).tolist()
            cutoff_ties.append(tie)
        errors.append(float(np.linalg.norm(w @ (xy[ids] - point))))
        local_errors.append(float(np.linalg.norm(sw @ (sxy[sl] - point))))
        history_errors.append(float(np.linalg.norm(hw @ (hxy[hl] - point))))
        n_support.append(int((ids >= len(hxy)).sum()))
        n_distinct_xy = len(np.unique(xy[ids], axis=0))
        repeated_xy.append(n_distinct_xy < len(ids))
        repeated_all_xy.append(n_distinct_xy == 1)
    return dict(aps=int(active.sum()), mde_hs=float(np.mean(errors)),
                mde_s=float(np.mean(local_errors)), mde_h=float(np.mean(history_errors)),
                q_with_s_in_top3=int(np.count_nonzero(n_support)), s_slots=sum(n_support),
                q_top3_repeated_xy=sum(repeated_xy), q_top3_one_xy=sum(repeated_all_xy),
                q_outside_grid=int((((qxy-origin) < lower) | ((qxy-origin) > upper)).any(1).sum()),
                q_to_nearest_s=float(np.linalg.norm(qxy[:, None, :] - sxy, axis=2).min(1).mean()),
                q_to_nearest_h=float(np.linalg.norm(qxy[:, None, :] - np.unique(hxy, axis=0),
                                                     axis=2).min(1).mean()),
                q_all_missing=int((qrss[:, active] == 100).all(1).sum()),
                errors_hs=errors, ambiguous_cutoff_ties=cutoff_ties)


def run(path):
    floors = load_floors(path)
    events = []
    for b, target, histories in [(0, 1, [0, 2, 3]), (2, 2, [0, 1, 3, 4])]:
        row_ids, xy, rss = floors[b, target]
        hxy = np.concatenate([floors[b, f][1] for f in histories])
        hrss = np.concatenate([floors[b, f][2] for f in histories])
        for seed in (261026, 261027, 261028):
            s, q = split_rows(xy, seed)
            for k in (1, 3, 5, 10, 20):
                result = audit_event(hxy, hrss, xy[s[:k]], rss[s[:k]], xy[q], rss[q])
                events.append(dict(building=b, target=target, history_floors=histories, seed=seed,
                                   K=k, support_csv_rows=row_ids[s[:k]].tolist(),
                                   query_csv_rows=row_ids[q].tolist(), **result))
    return dict(protocol="Old K-curve internal queries only, R=1, 12 Q per seed; "
                "H-only WKNN uses the current H+S AP mask as a matched diagnostic. "
                "Rows are 1-based data records, excluding the header.",
                training_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                numpy_version=np.__version__, coordinate_unit="original UJI coordinate distance",
                events=events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(run(args.training_csv), indent=2, allow_nan=False) + "\n")
