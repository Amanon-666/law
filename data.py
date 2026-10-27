"""Period-scoped MTLoc scans; missing acquisition is not AP non-detection.

Each scan set has positions [P,2], rss [N,A], xy [N,2], group [N], and
stats={n [P], k/s1/s2 [P,A]}.  n counts every original scan, k counts
detections, and s1/s2 sum detected RSS/detected RSS squared in dBm units.
RSS=100 denotes non-detection. No scan deduplication or MAC output occurs.
"""

import numpy as np


NO_DETECT = 100
SOURCE_TRAIN_PERIODS = tuple(range(1, 12))
SOURCE_VALIDATION_PERIODS = (12,)


def group_scans(rss, xy):
    """Aggregate actual acquisitions, including identical repeated scans."""
    rss, xy = np.asarray(rss), np.asarray(xy, dtype=np.float64)
    positions, group = np.unique(xy, axis=0, return_inverse=True)
    group = group.ravel()
    detected = rss != NO_DETECT
    values = np.where(detected, rss, 0).astype(np.float64)
    k, s1, s2 = (np.zeros((len(positions), rss.shape[1])) for _ in range(3))
    np.add.at(k, group, detected)
    np.add.at(s1, group, values)
    np.add.at(s2, group, values * values)
    n = np.bincount(group, minlength=len(positions)).astype(np.float64)
    return dict(positions=positions, rss=rss, xy=xy, group=group,
                stats=dict(n=n, k=k, s1=s1, s2=s2))


def load_period(path, period, split="training"):
    """Read only the requested NPZ period/split, never the other splits.

    The archive's AP column order is retained without loading its BSSIDs.
    Callers choose periods explicitly; the source constants load nothing.
    """
    if split not in ("training", "testing"):
        raise ValueError("split must be training or testing")
    with np.load(path, allow_pickle=False) as arrays:
        rss = arrays[f"{split}_{period}_rss"]
        xy = arrays[f"{split}_{period}_xy"]
    if rss.ndim != 2 or xy.shape != (len(rss), 2):
        raise ValueError("expected RSS [N,A] and coordinates [N,2]")
    if not np.isfinite(rss).all() or not np.isfinite(xy).all():
        raise ValueError("RSS and coordinates must be finite")
    return group_scans(rss, xy)


def _subset(floor, position_indices, ap_indices):
    rows = np.isin(floor["group"], position_indices)
    return group_scans(floor["rss"][rows][:, ap_indices], floor["xy"][rows])


def draw_episode(sourcefloor, currentfloor, H_fraction, K, Q_count, rng):
    """Draw H and disjoint current S/Q positions; retain every scan there.

    sourcefloor/currentfloor must already share the same local AP catalog
    (e.g. two periods of one NPZ), not arbitrary same-width buildings.
    H_fraction selects ceil(fraction * historical positions), or zero at 0.
    K and Q_count count positions, not scans. All requested counts are exact.
    Q.scan_weights averages scans within each position, then positions equally.
    No Q signal or hidden H signal participates in AP selection.
    """
    if not 0 <= H_fraction <= 1:
        raise ValueError("H_fraction must be between 0 and 1")
    if K < 0 or Q_count < 1 or K + Q_count > len(currentfloor["positions"]):
        raise ValueError("need K >= 0 and enough disjoint positions for Q_count >= 1")
    if sourcefloor["rss"].shape[1] != currentfloor["rss"].shape[1]:
        raise ValueError("historical and current AP axes must be aligned")

    h_count = int(np.ceil(H_fraction * len(sourcefloor["positions"])))
    h_ids = np.sort(rng.choice(len(sourcefloor["positions"]), h_count, replace=False))
    current_ids = rng.permutation(len(currentfloor["positions"]))
    s_ids, q_ids = np.sort(current_ids[:K]), np.sort(current_ids[K:K + Q_count])
    h_seen = (sourcefloor["stats"]["k"][h_ids] > 0).any(axis=0)
    s_seen = (currentfloor["stats"]["k"][s_ids] > 0).any(axis=0)
    ap_indices = np.flatnonzero(h_seen | s_seen)
    history = _subset(sourcefloor, h_ids, ap_indices)
    support = _subset(currentfloor, s_ids, ap_indices)
    query = _subset(currentfloor, q_ids, ap_indices)
    query["scan_weights"] = 1.0 / (Q_count * query["stats"]["n"][query["group"]])
    return dict(H=history, S=support, Q=query, ap_indices=ap_indices,
                h_position_indices=h_ids, s_position_indices=s_ids,
                q_position_indices=q_ids,
                historical_positive=h_seen[ap_indices])


def _check():
    # Identical physical acquisitions retain their weight; a query-only AP
    # must not enter a model whose inputs are H and S.
    rss = np.array([[-60, 100], [-60, 100], [100, -70], [-80, 100]])
    xy = np.array([[0, 0], [0, 0], [1, 0], [2, 0]])
    scans = group_scans(rss, xy)
    assert scans["stats"]["n"].tolist() == [2, 1, 1]
    assert scans["stats"]["k"][0].tolist() == [2, 0]
    assert scans["stats"]["s1"][0, 0] == -120
    assert scans["stats"]["s2"][0, 0] == 7200
    history = group_scans(np.array([[-60, 100]]), np.array([[0, 0]]))
    episode = draw_episode(history, scans, 1, 0, 3, np.random.default_rng(0))
    assert episode["ap_indices"].tolist() == [0]
    query = episode["Q"]
    assert len(query["rss"]) == 4
    np.testing.assert_allclose(np.bincount(query["group"], weights=query["scan_weights"]), 1 / 3)
    split = draw_episode(history, scans, 0, 1, 2, np.random.default_rng(1))
    assert len(split["H"]["rss"]) == 0
    assert not np.intersect1d(split["s_position_indices"], split["q_position_indices"]).size


if __name__ == "__main__":
    _check()
    print("scan counts, input-only AP selection, position weights and split isolation passed")
