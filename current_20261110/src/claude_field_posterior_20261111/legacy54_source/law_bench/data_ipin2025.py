"""IPIN2025 Training receipts for the new static bench protocol.

This is separate from the completed official causal-stream confirmation.
Keep raw receipts and metric labels; a spatial split bin is not a repeat
measurement. The task generator must use R=1 for this trajectory corpus.
"""
from pathlib import Path
import sys

import numpy as np

from ipin2025_observation_data import (
    BUILDING, load_training, metric_xy, rss_matrix,
)

DATA = Path('/home/panyushuo/datasets/ipin_t3_2025/2025_IPIN_Competition_Track03')
OLD = Path('/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/Law-Probabilistic-Transfer-20261023')


def load(dataset_root=DATA, *, parse_log=None):
    """Return (buildings, ap_ids_by_building), using only Training logs.

    All floors share the stored AP axis. Methods select detected H/current-S
    columns before fitting; unused target and Query columns are not evidence.
    Extra floor metadata retains receipt identities and label provenance for
    the task generator. No history subsampling or spatial pooling occurs here.
    """
    if parse_log is None:
        sys.path.append(str(OLD))
        from public_corpus.adapter import parse_log
    training = load_training(dataset_root, parse_log=parse_log)
    floors = {f: training['floors'][f] for f in training['eligible_floors']}
    aps = sorted({ap for rows in floors.values() for r in rows for ap in r['rss']})
    acquisitions = {path: i for i, path in enumerate(training['training_files'])}
    packed = {}
    for floor, rows in floors.items():
        packed[floor] = dict(
            rss=rss_matrix(rows, aps), xy=metric_xy(rows),
            acq=np.array([acquisitions[r['file']] for r in rows]),
            time=np.array([r['app_timestamp'] for r in rows]),
            device=np.full(len(rows), -1, dtype=int),
            scan_ids=np.array([r['scan_id'] for r in rows]),
            label_provenance=[dict(file=r['relative_file'],
                landmark_lines=r['landmark_lines'], landmark_times=r['landmark_times'],
                interpolation_fraction=r['fraction']) for r in rows],
            meta=dict(dataset='ipin2025_training_resample', max_supported_R=1, split_grid_m=2.0,
                position_kind='time_interpolated_POSI_landmarks',
                reference=training['reference'], evaluation_status='exposed_development',
                unsupported_floors=[f for f, m in training['floor_metadata'].items()
                                    if not m['eligible']]))
    building = str(BUILDING)
    return {building: packed}, {building: aps}


def describe(buildings):
    """Training inventory; same-label counts are not independent repeats."""
    from scipy.spatial import cKDTree

    summary = {}
    for building, floors in buildings.items():
        positions, seen, stats = {}, {}, {}
        for floor, data in sorted(floors.items()):
            positions[floor], counts = np.unique(data['xy'], axis=0, return_counts=True)
            seen[floor] = (data['rss'] != 100).any(0)
            stats[floor] = dict(scans=len(data['rss']), positions=len(counts),
                scans_per_label_min=int(counts.min()),
                scans_per_label_median=float(np.median(counts)),
                scans_per_label_max=int(counts.max()),
                detected_aps=int(seen[floor].sum()), max_supported_R=1)
        adjacent = []
        ordered = sorted(floors)
        for lower, upper in zip(ordered, ordered[1:]):
            union = (seen[lower] | seen[upper]).sum()
            adjacent.append(dict(lower=lower, upper=upper,
                ap_jaccard=float((seen[lower] & seen[upper]).sum() / union) if union else None,
                lower_within_2m=float(np.mean(cKDTree(positions[upper]).query(positions[lower])[0] <= 2)),
                upper_within_2m=float(np.mean(cKDTree(positions[lower]).query(positions[upper])[0] <= 2))))
        summary[building] = dict(floors=stats, adjacent=adjacent)
    return summary
