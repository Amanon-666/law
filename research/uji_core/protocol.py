"""UJI data and nested, position-sampled support. CSV row IDs are zero based."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from residual_probe.episode_rows import nested_rows

BUDGETS = (1, 3, 5, 10, 20)
SEEDS = tuple(range(10))
HASHES = {
    "trainingData.csv": "45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168",
    "validationData.csv": "5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0",
}


def read_floors(directory, filename):
    path = Path(directory) / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[filename]:
        raise ValueError(f"Unexpected UCI CSV contents: {filename}")
    frame = pd.read_csv(path)
    wap = [f"WAP{i:03}" for i in range(1, 521)]
    return {
        (int(b), int(f)): dict(
            rss=rows[wap].to_numpy(np.int16),
            xy=rows[["LONGITUDE", "LATITUDE"]].to_numpy(np.float64),
            ids=rows.index.to_numpy(),
        )
        for (b, f), rows in frame.groupby(["BUILDINGID", "FLOOR"], sort=True)
    }


def history_floors(train, building, floor):
    return [v for (b, f), v in sorted(train.items()) if b == building and f != floor]


def split_rows(target, building, floor, seed, source=False):
    """Local row indices. Source episodes have disjoint Q; official Q is a separate CSV."""
    manifest = nested_rows(target["xy"], range(len(target["xy"])), BUDGETS,
                           24 if source else 0,
                           f"uji-core-v1/{'source' if source else 'official'}/{building}/{floor}/{seed}")
    return np.asarray(manifest["support_rows"][max(BUDGETS)]), np.asarray(manifest["query_rows"], dtype=int)


def position_weights(xy):
    _, group, count = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
    return 1.0 / (len(count) * count[group.ravel()])
