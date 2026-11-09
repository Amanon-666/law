"""UJI-only adapter with original floor-local indexing and GP cache coordinates."""
import csv
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from fields import History
NO_DETECT = 100

def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)

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

def ap_stats(scans, j):
    return {key: tensor(value if key == "n" else value[:, j])
            for key, value in scans["stats"].items()}

@lru_cache(maxsize=2048)
def load_history(path, shift, base_items):
    with np.load(path, allow_pickle=False) as a:
        if bool(a['empty']):
            return None
        return History(tensor(a['z']) + tensor(shift),
                       dict(mean=tensor(a['mean']), cov=tensor(a['cov'])), dict(base_items))

def support_cases(meta, task, base):
    """No Q fields enter this interface or either support fit."""
    cases = {}
    for column, ap in enumerate(meta['AP_indices']):
        histories = []
        for floor in meta['histories']:
            info = meta['history_metadata'][str(floor)]
            shift = tuple(np.asarray(info['cache_origin']) - np.asarray(meta['origin']))
            histories.append(load_history(str(Path(info['cache']) / f'ap{ap}.npz'),
                                          shift, tuple(base.items())))
        cases[ap] = dict(histories=histories, dz=task['dz'],
                         sx=tensor(task['S']['positions']), stats=ap_stats(task['S'], column))
    return cases

def grid(meta):
    g = meta['grid']
    axes = [np.linspace(g['lower'][i], g['upper'][i], g['shape_xy'][i]) for i in range(2)]
    return tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1, 2))


def load_uji(path):
    floors = {}
    with Path(path).open(newline='') as stream:
        for rowid, row in enumerate(csv.DictReader(stream), 1):
            name = f"B{row['BUILDINGID']}F{row['FLOOR']}"
            floors.setdefault(name, []).append((rowid,
                [int(row[f'WAP{i:03}']) for i in range(1,521)],
                [float(row['LONGITUDE']),float(row['LATITUDE'])]))
    return {name:dict(row_ids=np.array([r[0] for r in rows]),
                     rss=np.array([r[1] for r in rows],dtype=np.int16),
                     xy=np.array([r[2] for r in rows])) for name,rows in floors.items()}


def event_data(meta, k, floors):
    name = f"{meta['building'][4:]}F{meta['target']}"
    raw=floors[name];aps=meta['AP_indices'];origin=np.array(meta['origin'])
    sr=meta['support_rows'][str(k)];qr=meta['query_rows']
    support=group_scans(raw['rss'][sr][:,aps],raw['xy'][sr]-origin)
    query=group_scans(raw['rss'][qr][:,aps],raw['xy'][qr]-origin)
    actual=[f'trainingData.csv@row{r}' for r in raw['row_ids'][qr]]
    assert actual==meta['query_scan_ids']
    assert not (support['positions'][:,None]==query['positions'][None]).all(-1).any()
    return dict(S=support,Q=query,dz=[f-meta['target'] for f in meta['histories']])
