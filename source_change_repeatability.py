"""Source-only split-half description of observable temporal change.

No neural training or localization scoring. Cross-half products describe
repeatability; independence of the acquisition halves is not established.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data import load_period


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "Law-Continuous-Change-20261023/data/OfficeP2.npz"
PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]


def split_statistics(floor):
    groups = [np.flatnonzero(floor["group"] == i) for i in range(len(floor["positions"]))]
    repeat_fraction = [1 - len(np.unique(floor["rss"][g], axis=0)) / len(g) for g in groups]
    halves = []
    for side in (0, 1):
        counts, detections, sums = [], [], []
        for g in groups:
            part = g[:len(g) // 2] if side == 0 else g[len(g) // 2:]
            values = floor["rss"][part]
            detected = values != 100
            counts.append(len(part))
            detections.append(detected.sum(0))
            sums.append(np.where(detected, values, 0).sum(0))
        n, k, total = np.array(counts), np.array(detections), np.array(sums)
        rate = k / np.maximum(n[:, None], 1)
        mean = np.divide(total, k, out=np.full(total.shape, np.nan), where=k > 0)
        halves.append(dict(n=n, k=k, rate=rate, mean=mean))
    return halves, dict(scans=len(floor["rss"]), positions=len(groups),
                       median_scans=float(np.median(floor["stats"]["n"])),
                       mean_within_position_duplicate_fraction=float(np.mean(repeat_fraction)))


def components(a, b):
    cross = np.mean(a * b)
    pooled = np.mean(((a + b) / 2) ** 2)
    disagreement = np.mean((a - b) ** 2 / 4)
    constant = np.mean(a) * np.mean(b)
    return dict(cross=float(cross), pooled=float(pooled), disagreement=float(disagreement),
                constant=float(constant), shape=float(cross - constant))


def describe(a, b, mask, rss=False):
    records = []
    for j in range(a.shape[1]):
        use = mask[:, j]
        if use.sum() < 4:
            continue
        u, v = a[use, j], b[use, j]
        row = dict(ap_column=j, positions=int(use.sum()), **components(u, v))
        row["both_raw_signs_repeated"] = bool(((u > 0) & (v > 0)).any() and
                                               ((u < 0) & (v < 0)).any())
        if rss:
            row["both_raw_signs_repeated_3db"] = bool(((u >= 3) & (v >= 3)).any() and
                                                       ((u <= -3) & (v <= -3)).any())
        records.append(row)
    fields = ("cross", "pooled", "disagreement", "constant", "shape")
    means = {key: float(np.mean([r[key] for r in records])) for key in fields} if records else {}
    return dict(eligible_cells=int(mask.sum()), total_cells=mask.size, aps=len(records),
                ap_equal_components=means,
                raw_opposite_sign_ap_count=sum(r["both_raw_signs_repeated"] for r in records),
                raw_opposite_sign_3db_ap_count=sum(r.get("both_raw_signs_repeated_3db", False) for r in records),
                per_ap=records)


def main():
    # These identities catch artificial sign creation or silently clipped energy.
    assert components(np.ones(4) * 2, np.ones(4) * 2)["shape"] == 0
    a = np.array([-2., 2., -2., 2.])
    assert components(a, a)["shape"] == 4
    c = components(a, -a)
    assert c["cross"] == -4 and c["cross"] == c["pooled"] - c["disagreement"]
    out = ROOT / "results/source_change_repeatability_20261024.json"
    if out.exists():
        raise FileExistsError(out)
    result = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  scope="source observable change only; no independent-half or environmental-cause claim",
                  pairs=PAIRS, data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  numpy_version=np.__version__, records=[])
    for old_period, new_period in PAIRS:
        old = load_period(DATA, old_period)
        new = load_period(DATA, new_period)
        np.testing.assert_array_equal(old["positions"], new["positions"])
        h, hm = split_statistics(old)
        s, sm = split_statistics(new)
        enough = np.logical_and.reduce([half["n"] >= 2 for half in h + s])
        observed = np.logical_and.reduce([half["k"] >= 2 for half in h + s]) & enough[:, None]
        detection_mask = np.broadcast_to(enough[:, None], h[0]["k"].shape)
        # All-zero AP columns are kept in the all-catalog description. The active
        # view avoids treating never-observed APs as evidence of stable detection.
        active = np.logical_or.reduce([half["k"].sum(0) > 0 for half in h + s])
        da, db = s[0]["rate"] - h[0]["rate"], s[1]["rate"] - h[1]["rate"]
        record = dict(periods=[old_period, new_period], old_metadata=hm, current_metadata=sm,
                      detection=describe(da, db, detection_mask),
                      detection_active=describe(da, db, detection_mask & active[None]),
                      detected_rss=describe(s[0]["mean"] - h[0]["mean"],
                                            s[1]["mean"] - h[1]["mean"], observed, rss=True))
        result["records"].append(record)
        print(json.dumps({"periods": record["periods"], "old_metadata": hm, "current_metadata": sm,
                          **{key: {k: v for k, v in record[key].items() if k != "per_ap"}
                             for key in ("detection_active", "detected_rss")}}, ensure_ascii=False))
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
