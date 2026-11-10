"""Replicates the project's 54-event nine-floor protocol (building_information_protocol) and scoring."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol9 as P
CSV = os.environ.get('LAW_UJI_CSV', '/home/claude/data/trainingData.csv')
_d = pd.read_csv(CSV)
WAPS = [c for c in _d.columns if c.startswith('WAP')]
RSS = _d[WAPS].to_numpy(np.float64)            # 100 = not detected
XY = _d[['LONGITUDE', 'LATITUDE']].to_numpy(np.float64)
BLD = _d.BUILDINGID.to_numpy(); FLR = _d.FLOOR.to_numpy()
USER = _d.USERID.to_numpy(); PHONE = _d.PHONEID.to_numpy(); TS = _d.TIMESTAMP.to_numpy()
DET = RSS != 100
MAN = P.build_manifest(CSV)
rows = lambda ids: np.asarray(ids, int) - 1   # manifest ids are one-based

def event_arrays(e):
    H = np.concatenate([rows(MAN['history_rows'][h]) for h in e['histories']])
    return H, rows(e['support_rows']), rows(e['query_rows'])

def nearest3(rss, xy, qrss, aps, bounds):
    a = np.where(rss[:, aps] == 100, -110, rss[:, aps]); b = np.where(qrss[:, aps] == 100, -110, qrss[:, aps])
    out = []
    for row in b:
        d = np.sqrt(((row - a) ** 2).mean(-1)); s = np.argsort(d, kind='stable')[:3]
        w = 1 / np.maximum(d[s], 1e-8); w /= w.sum(); out.append(w @ xy[s])
    return np.clip(np.asarray(out), bounds[0], bounds[1])

def bounds_of(xy):
    return np.stack((np.floor(xy.min(0) / 2) * 2, np.ceil(xy.max(0) / 2) * 2))

def event_error(e, pred):
    """position-balanced mean error of one event"""
    _, _, Q = event_arrays(e)
    err = np.linalg.norm(pred - XY[Q], axis=1)
    _, g, c = np.unique(XY[Q], axis=0, return_inverse=True, return_counts=True)
    return float((err / (len(c) * c[g.ravel()])).sum())

def summarize(errs):
    """errs: {event_id: error}. Project aggregation: per K equal buildings, floors, directions."""
    ev = {e['id']: e for e in MAN['events']}
    df = pd.DataFrame([dict(floor=ev[i]['floor'], b=ev[i]['building'], K=ev[i]['K'], err=v) for i, v in errs.items()])
    byK = df.groupby(['K', 'b', 'floor']).err.mean().groupby(['K', 'b']).mean().groupby('K').mean()
    byF = df.groupby('floor').err.mean()
    return dict(mean=float(byK.mean()), by_K=byK.to_dict(), by_floor=byF.to_dict())


def dev_manifest(floors=('B0F0', 'B0F1', 'B0F2', 'B0F3')):
    """Same pairing/position/scan rules as the project protocol, applied to another building (development fold)."""
    import csv, hashlib, random
    from collections import defaultdict
    from datetime import datetime, timezone
    rws = {}
    with open(CSV, newline='') as stream:
        for row_id, rec in enumerate(csv.DictReader(stream), 1):
            fl = f"B{rec['BUILDINGID']}F{rec['FLOOR']}"
            if fl in floors:
                ts = int(rec['TIMESTAMP'])
                rws[row_id] = dict(building=fl.split('F')[0], floor=fl, xy=(float(rec['LONGITUDE']), float(rec['LATITUDE'])),
                    timestamp=ts, acquisition=(int(rec['USERID']), int(rec['PHONEID']),
                    datetime.fromtimestamp(ts, timezone.utc).date().isoformat()))
    hist = {f: sorted(i for i, r in rws.items() if r['floor'] == f) for f in floors}
    events = []
    for floor in floors:
        groups = P._groups(rws, floor)
        try:
            left, right, common = P._pair(groups)
        except ValueError:
            continue
        sxy, qxy = P._positions(floor, common)
        b = floor.split('F')[0]
        for direction, src, qry in (('AB', left, right), ('BA', right, left)):
            for k in P.KS:
                events.append(dict(id=f'{floor}_{direction}_K{k}', building=b, floor=floor, direction=direction, K=k,
                    histories=[f for f in floors if f != floor], support_acquisition=list(src), query_acquisition=list(qry),
                    support_rows=[i for xy in sxy[:k] for i in groups[src][xy][:P.R]],
                    query_rows=[i for xy in qxy for i in groups[qry][xy][:P.R]]))
    return dict(history_rows=hist, events=events)

DEV = dev_manifest()

def event_arrays_m(man, e):
    H = np.concatenate([rows(man['history_rows'][h]) for h in e['histories']])
    return H, rows(e['support_rows']), rows(e['query_rows'])

def summarize_m(man, errs):
    ev = {e['id']: e for e in man['events']}
    df = pd.DataFrame([dict(floor=ev[i]['floor'], b=ev[i]['building'], K=ev[i]['K'], err=v) for i, v in errs.items()])
    byK = df.groupby(['K', 'b', 'floor']).err.mean().groupby(['K', 'b']).mean().groupby('K').mean()
    return dict(mean=float(byK.mean()), by_K=byK.to_dict(), by_floor=df.groupby('floor').err.mean().to_dict())

def event_error_m(man, e, pred):
    _, _, Q = event_arrays_m(man, e)
    err = np.linalg.norm(pred - XY[Q], axis=1)
    _, g, c = np.unique(XY[Q], axis=0, return_inverse=True, return_counts=True)
    return float((err / (len(c) * c[g.ravel()])).sum())
