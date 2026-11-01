"""UM DSI lite: paired true vs nearest-foil discrimination on eight early surveys.

Usage: python check_order.py /path/UM_DSI_DB_v1.0.0_lite.zip /path/out
Requires Python 3 and NumPy. The four reserved dates and all monitors are excluded.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

VISITS = [('2019-02-19','2019-03-25'), ('2019-06-11','2019-07-03'),
          ('2019-07-29','2019-09-11'), ('2019-10-09','2019-11-13')]
PREFIX = 'UM_DSI_DB_v1.0.0_lite/data/site_surveys/'
SENSORS = ('timestamps.csv','coordinates.csv','rssis.csv')
ALLOW = {f'{PREFIX}{visit}/{name}' for pair in VISITS for visit in pair for name in SENSORS}
MISS = -110.
METRICS = ('R1_correct','R1_tie','R1_margin','mean_correct','mean_tie','mean_margin',
           'R1_wrong_to_mean_correct','R1_correct_to_mean_wrong','any_order_change')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def visit(z, day, manifest):
    def lines(name):
        path = PREFIX + day + '/' + name
        if path not in ALLOW:
            raise ValueError('Forbidden survey or sensor')
        content = z.read(path)
        manifest[path] = sha(content)
        return content.decode('ascii').splitlines()
    # Integer micrometres preserve exact published decimals & geometric distance ties.
    coords = np.array([list(map(float, row.split(','))) for row in lines('coordinates.csv')])
    pts = np.rint(coords * 1e6).astype(np.int64)
    np.testing.assert_allclose(coords, pts/1e6, rtol=0, atol=2e-12)
    if not np.all(pts[:,2] == 1_000_000):
        raise ValueError('Unexpected survey plane')
    times = lines('timestamps.csv')
    if not all(x.startswith(day.replace('-','')) for x in times):
        raise ValueError('Dates do not agree with survey IDs')
    scans = []
    positions = {}
    for i, row in enumerate(lines('rssis.csv')):
        tokens = [v.split(':') for v in row.split(',') if v]
        observed = {name:int(val) for name,val in tokens}
        if not observed or len(observed) != len(tokens) or min(observed.values()) <= MISS or max(observed.values()) >= 0:
            raise ValueError('Missing sentinel, repeated key, empty scan or nonnegative signal')
        scans.append(observed)
        positions.setdefault(tuple(map(int, pts[i,:2])), []).append(i)
    if len(scans) != len(times) or len(times) != len(pts):
        raise ValueError('Misaligned rows')
    axes = sorted({a for row in scans for a in row})
    counts = list(map(len,positions.values()))
    return scans, positions, {'rows':len(scans), 'positions':len(positions), 'APs':len(axes),
        'AP_axis_sha256':sha(('\n'.join(axes)+'\n').encode('ascii')),
        'rows_per_position_range':[min(counts),max(counts)], 'rssi_range':[min(min(d.values()) for d in scans),
        max(max(d.values()) for d in scans)], 'first_time':min(times), 'last_time':max(times)}


def matrix(scans, axes):
    idx = {a:i for i,a in enumerate(axes)}
    raw = np.full((len(scans),len(axes)), MISS)
    for i, row in enumerate(scans):
        for ap, rssi in row.items():
            if ap in idx:
                raw[i,idx[ap]] = rssi
    return raw


def sample_rows(day, seed, points, early_ids):
    chosen = []
    for x,y in points:
        key = f'minho-order-v1/{day}/{seed}/{x}/{y}'.encode('ascii')
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:16], 'big'))
        # First row of a per-site random permutation; all-repeat uses the full set.
        chosen.append(int(rng.permutation(early_ids[x,y])[0]))
    return chosen


def distances(q, template, foil):
    dt = np.mean((q-template)**2, axis=1)
    df = np.mean((q-foil)**2, axis=1)
    sign = np.sign(df-dt)
    sign[np.abs(df-dt)<=1e-12] = 0
    return sign, np.sqrt(df)-np.sqrt(dt)


def measure(z, pair, manifest):
    earlier,later = pair
    erows, emap, einfo = visit(z, earlier, manifest)
    lrows, lmap, linfo = visit(z, later, manifest)
    if einfo['last_time'] >= linfo['first_time']:
        raise ValueError('Reused or out-of-order survey')
    points = sorted(set(emap).intersection(lmap))
    axes = sorted({a for rss in erows for a in rss})  # earlier survey only
    early, late = matrix(erows, axes), matrix(lrows, axes)
    mu = np.stack([early[emap[p]].mean(0) for p in points])
    coords = np.array(points, dtype=np.int64)
    sq = np.sum((coords[:,None]-coords[None,:])**2, axis=2)
    np.fill_diagonal(sq, np.iinfo(np.int64).max)
    foils = sq.argmin(1)  # lexicographically sorted common positions break exact ties
    spacing = np.sqrt(sq[np.arange(len(coords)),foils])/1e6
    seeds = [sample_rows(earlier, i, points, emap) for i in range(10)]
    single = early[np.asarray(seeds)]
    position_summaries = []
    for i, (pos, foil) in enumerate(zip(points, foils)):
        queries = late[lmap[pos]]
        mean_sign, mean_margin = distances(queries,mu[i],mu[foil])
        draws = []
        for draw in single:
            sign, margin = distances(queries,draw[i],draw[foil])
            draws.append([(sign>0).mean(),(sign==0).mean(),margin.mean(),
                          (mean_sign>0).mean(),(mean_sign==0).mean(),mean_margin.mean(),
                          ((sign<0)&(mean_sign>0)).mean(),((sign>0)&(mean_sign<0)).mean(),
                          (sign!=mean_sign).mean()])
        position_summaries.append(np.mean(draws,axis=0))
    data = dict(early=earlier,late=later,early_info=einfo,late_info=linfo,common_positions=len(points),
                early_rows_at_common=sum(len(emap[p]) for p in points),query_rows=sum(len(lmap[p]) for p in points),
                median_foil_metres=float(np.median(spacing)),earlier_axis_width=len(axes),
                all_early_indices_sha256=sha(np.asarray(seeds,dtype='<i8').tobytes()),
                stats=dict(zip(METRICS,np.mean(position_summaries,axis=0))))
    return data, spacing


def run(archive, out):
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    assert md5 == '4c15884cb7edcd2c602ec5d567ddd8a8'
    manifest = {}
    with ZipFile(archive) as z:
        pairs, spacing = zip(*(measure(z,pair,manifest) for pair in VISITS))
    assert set(manifest)==ALLOW
    out.mkdir(parents=True,exist_ok=True)
    macro = {name:float(np.mean([r['stats'][name] for r in pairs])) for name in METRICS}
    # Weighted median over positions, each of the four date pairs receiving 1/4.
    x = np.concatenate(spacing)
    w = np.concatenate([np.full(len(s),1/(len(s)*len(spacing))) for s in spacing])
    i = np.argsort(x)
    global_median = float(x[i][np.searchsorted(np.cumsum(w[i]),.5)])
    spec = dict(dataset='University of Minho UM DSI v1.0.0 lite DOI 10.5281/zenodo.6646008; Silva, Pendao and Moreira.',
                archive_md5=md5,archive_sha256=sha(archive.read_bytes()),opened_payload_sha256=manifest,
                publisher_format='One comma-separated anonymized AP ID:integer dBm pair for each detection. Absent pairs are missing. No explicit -110, 0 or +100 in the eight surveys.',
                procedure='Use all earlier AP IDs at every K and in every method, fill missing at -110 then take arithmetic means. Rows not at shared XY supply only vocabulary. Later-only APs never enter the axis. Foil is nearest distinct common XY by exact 2D squared integer micrometre distance, ties lexical (x,y).',
                statistics='Strictly correct when true earlier prototype has lower RMS dBm distance than its foil for that future scan. Sign ties use squared-distance tolerance 1e-12 dBm^2. Positive margin=RMS foil distance minus RMS true distance. Equal future scans per position, positions within pair, and four disjoint pairs. 10 deterministic independent draws across sites and seeds; single R1 first of data-independent early-row permutation, full mean over the superset of rows. A foil that is another site always uses its globally shared drawn row. Raw row indices are zero-based in archive order.',
                scope='Same acquisition system and one height in a single non-UJI building. Full-repeat averages use more measurements than R1. No parameter fitting or testing of a localization map, K gain or held UJI location.',
                pairs=pairs,macro=macro,median_foil_metres_equal_pair_position=global_median)
    (out/'source_receipt.json').write_text(json.dumps(spec,indent=2,allow_nan=False)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(pairs=[{k:p[k] for k in ('early','late','common_positions','query_rows','median_foil_metres','earlier_axis_width','stats')} for p in pairs],macro=macro,median_foil_m=global_median),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    run(args.archive,args.out)
