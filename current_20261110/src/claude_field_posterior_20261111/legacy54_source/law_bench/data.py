"""Training-side raw scans for the shared bench; no pooling or AP filtering.

load(name, root=None) returns (buildings, ap_ids_by_building). root is the
dataset directory (or training CSV for UJI/UTS/SOD). Server defaults point to
the already-present originals. No download or test/validation-file access.
Acquisition IDs are unknown (-1) unless supported by source metadata. UJI
uses the original (USERID, PHONEID, UTC date) grouping, retaining its mapping.
Device/time/user values are preserved; SOD SampleTimes is NOT a timestamp.
Anonymous AP columns are namespaced by corpus/building, not invented MACs.

Schemas and existing coordinate checks inspected read-only on 2026-11-11.
Loaders have been statically reviewed only, not run over the full datasets.
"""
from pathlib import Path
import os

import numpy as np


DATA = Path('/home/panyushuo/datasets')
PROJECTS = Path('/home/panyushuo/projects/panyushuo')
TAMPERE_DIR = 'WiFi RSS measurements in Tampere University multi-building campus 2017 - Zenodo 5174851'


def _csv_path(root, default):
    if root is None:
        return default
    root = Path(root)
    return root if root.suffix.lower() == '.csv' else root/default.name


def _floors(rss, xy, floor, meta, **columns):
    """Slice arrays without changing order, coordinates or repeated scans."""
    n = len(rss)
    if len(xy) != n or len(floor) != n:
        raise ValueError('RSS and coordinate files have different row counts')
    rows = dict(acq=np.full(n, -1, dtype=int), device=np.full(n, -1, dtype=int),
                time=np.full(n, -1, dtype=int))
    rows.update(columns)
    out = {}
    for f in sorted(np.unique(floor)):
        mask = floor == f
        out[int(f)] = dict(rss=rss[mask], xy=xy[mask], meta=dict(meta),
                          **{k: np.asarray(v)[mask] for k, v in rows.items()})
    return out


def _uji(root):
    import pandas as pd

    default = Path(os.environ.get('LAW_UJI_CSV',
        str(PROJECTS/'UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv')))
    path = _csv_path(root, default)
    data = pd.read_csv(path)
    aps = [c for c in data.columns if c.startswith('WAP')]
    dates = pd.to_datetime(data.TIMESTAMP, unit='s', utc=True).dt.strftime('%Y-%m-%d')
    acquisition_keys = list(zip(data.USERID.astype(int), data.PHONEID.astype(int), dates))
    identities = sorted(set(acquisition_keys))
    index = {key: i for i, key in enumerate(identities)}
    acquisitions = np.array([index[k] for k in acquisition_keys], dtype=int)
    buildings = {}
    for b in sorted(data.BUILDINGID.unique()):
        mask = (data.BUILDINGID == b).to_numpy()
        d = data.loc[mask]
        buildings[f'B{int(b)}'] = _floors(
            d[aps].to_numpy(float), d[['LONGITUDE', 'LATITUDE']].to_numpy(float),
            d.FLOOR.to_numpy(int), dict(dataset='uji', source=str(path),
                position_kind='static_reference_point', max_supported_R=3,
                evaluation_status='exposed_development',
                coordinate_frame='original_shared_metric_xy',
                acquisition_definition='USERID,PHONEID,UTC date',
                acquisition_identities=[list(k) for k in identities]),
            acq=acquisitions[mask], device=d.PHONEID.to_numpy(), time=d.TIMESTAMP.to_numpy(),
            user=d.USERID.to_numpy(), row_ids=np.flatnonzero(mask)+1)
    return buildings, {b: list(aps) for b in buildings}


def _tampere(root):
    import pandas as pd

    root = DATA/'tampere_5174851'/TAMPERE_DIR if root is None else Path(root)
    if root.name == 'tampere_5174851':
        root = root/TAMPERE_DIR
    buildings, axes = {}, {}
    for b in ('TIE1', 'SAH1'):
        coordinates = root/f'{b}_training_coordinates.csv'
        signals = root/f'{b}_training_rss.csv'
        c = pd.read_csv(coordinates, header=None, names=['x', 'y', 'z', 'floor'])
        rss = pd.read_csv(signals, header=None).to_numpy(float)
        buildings[b] = _floors(rss, c[['x', 'y']].to_numpy(float), c.floor.to_numpy(int),
            dict(dataset='tampere', source=[str(coordinates), str(signals)],
                position_kind='static_reference_point', max_supported_R=3,
                coordinate_frame='original_shared_local_metres',
                coordinate_evidence=str(root/'README.txt'),
                frame_check=str(PROJECTS/'multifloor-check-20261026/results/frame_check.log'),
                acquisition_definition='unknown; no device/time/session columns',
                ap_identity='anonymous training RSS column, shared across building floors'),
            z=c.z.to_numpy(), row_ids=np.arange(1, len(c)+1))
        axes[b] = [f'tampere/{b}/column{i+1}' for i in range(rss.shape[1])]
    return buildings, axes


def _uts(root):
    import pandas as pd

    path = _csv_path(root, DATA/'utsindoorloc/UTS_training.csv')
    data = pd.read_csv(path)
    aps = [c for c in data.columns if c.startswith('WAP')]
    # Existing check_mf2.load('utsT'): only tower floors have shared XY evidence.
    data = data.loc[data.Floor_ID.between(4, 12)]
    buildings, axes = {}, {}
    for b, d in data.groupby('Building_ID', sort=True):
        b = str(b)
        buildings[b] = _floors(d[aps].to_numpy(float), d[['Pos_x', 'Pos_y']].to_numpy(float),
            d.Floor_ID.to_numpy(int), dict(dataset='uts', source=str(path),
                position_kind='static_reference_point', max_supported_R=3,
                coordinate_frame='original_tower_xy_metres_floors_4_to_12',
                coordinate_evidence=str(PROJECTS/'multifloor-check2-20261026/RESULTS.md'),
                excluded_floors=list(range(-3, 4)),
                acquisition_definition='unknown; raw User_ID/Phone_type/Time retained'),
            device=d.Phone_type.to_numpy(), time=d.Time.to_numpy(), user=d.User_ID.to_numpy(),
            row_ids=d.index.to_numpy()+1)
        axes[b] = [f'uts/{b}/{ap}' for ap in aps]
    return buildings, axes


def _sod(root):
    import pandas as pd

    if root is not None and Path(root).name == 'SODIndoorLoc':
        root = Path(root)/'CETC331'
    path = _csv_path(root, DATA/'SODIndoorLoc/CETC331/Training_CETC331.csv')
    data = pd.read_csv(path)
    aps = [c for c in data.columns if c.startswith('MAC')]
    floors = _floors(data[aps].to_numpy(float), data[['ECoord', 'NCoord']].to_numpy(float),
        data.FloorID.to_numpy(int), dict(dataset='sod', source=str(path),
            position_kind='static_reference_point', max_supported_R=3,
            coordinate_frame='published_local_metre_columns',
            registration_status='cross-scene alignment not yet checked',
            coordinate_evidence=str(path.parent.parent/'README.md'),
            acquisition_definition='unknown; UserID/PhoneID retained, SampleTimes is a repeat index'),
        device=data.PhoneID.to_numpy(), user=data.UserID.to_numpy(),
        sample_index=data.SampleTimes.to_numpy(), scene=data.SceneID.to_numpy(),
        row_ids=np.arange(1, len(data)+1))
    return {'CETC331': floors}, {'CETC331': [f'sod/CETC331/{ap}' for ap in aps]}


def load(name, root=None):
    """Names: uji, tampere, uts (tower 4–12 only), sod (CETC331), ipin2025."""
    if name == 'ipin2025':
        from .data_ipin2025 import load as load_ipin
        buildings, axes = load_ipin() if root is None else load_ipin(root)
        for floors in buildings.values():
            for data in floors.values():
                data['meta']['split_grid_m'] = 2.
        return buildings, axes
    loaders = dict(uji=_uji, tampere=_tampere, uts=_uts, sod=_sod)
    if name not in loaders:
        raise ValueError(f'Unknown dataset {name!r}; choose {list(loaders) + ["ipin2025"]}')
    return loaders[name](root)


def describe(buildings):
    """Raw label counts plus adjacent-floor AP overlap and 2 m proximity."""
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
                scans_per_label_max=int(counts.max()), ap_columns=data['rss'].shape[1],
                detected_aps=int(seen[floor].sum()), **data['meta'])
        adjacent = []
        ordered = sorted(floors)
        for lower, upper in zip(ordered, ordered[1:]):
            if upper != lower + 1:
                continue
            union = (seen[lower] | seen[upper]).sum()
            adjacent.append(dict(lower=lower, upper=upper,
                ap_jaccard=float((seen[lower] & seen[upper]).sum()/union) if union else None,
                lower_within_2m=float(np.mean(cKDTree(positions[upper]).query(positions[lower])[0] <= 2)),
                upper_within_2m=float(np.mean(cKDTree(positions[lower]).query(positions[upper])[0] <= 2))))
        summary[building] = dict(floors=stats, adjacent=adjacent)
    return summary
