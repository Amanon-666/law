"""B0-only real-scan tasks for the unchanged original HGB fitting routine."""
import csv
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

import claude_hurdle_assets as original_assets
from .features import T, WIDTH, build_context


MODES = ('Hfull_Sfull', 'Hsingle_Sfull', 'Hfull_Ssingle', 'Hsingle_Ssingle')
EXPOSURE_RNG_TAG = 20261111
CSV_SHA = '45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168'
UPSTREAM_FILES = ('translator.py', 'vt_lib.py', 'hurdle.py')


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode()+a.tobytes()).hexdigest()


def load_source(csv_path):
    """Parse RSS only for B0, preserving original one-based CSV row IDs.

Load verbatim vt_lib/hurdle against that source-only core. Never execute the
upstream core.py, its target manifests, or any B1/B2 RSS/statistics loader.
"""
    assert original_assets.digest(csv_path) == CSV_SHA
    for name in UPSTREAM_FILES:
        expected = original_assets.BLOBS[name]
        data = (original_assets.PACKAGE/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == expected
    assert (Path(T.__file__).read_bytes() == (original_assets.PACKAGE/'translator.py').read_bytes())
    C = ModuleType('finite_observation_source_core')
    ids, rss, xy, floor, user, phone, timestamp = [], [], [], [], [], [], []
    with Path(csv_path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        C.WAPS = [name for name in reader.fieldnames if name.startswith('WAP')]
        for row_id, row in enumerate(reader, 1):
            if int(row['BUILDINGID']) != 0:
                continue
            ids.append(row_id)
            rss.append([float(row[name]) for name in C.WAPS])
            xy.append([float(row['LONGITUDE']), float(row['LATITUDE'])])
            floor.append(int(row['FLOOR']))
            user.append(int(row['USERID']))
            phone.append(int(row['PHONEID']))
            timestamp.append(int(row['TIMESTAMP']))
    C.ROW_IDS = np.asarray(ids, dtype=np.int64)
    C.RSS, C.XY = np.asarray(rss, dtype=np.float64), np.asarray(xy, dtype=np.float64)
    C.FLR, C.BLD = np.asarray(floor), np.zeros(len(ids), dtype=int)
    C.USER, C.PHONE, C.TS = np.asarray(user), np.asarray(phone), np.asarray(timestamp)
    C.DET = C.RSS != 100
    assert set(C.FLR) == {0, 1, 2, 3}

    def module(name, filename):
        spec = importlib.util.spec_from_file_location(name, original_assets.PACKAGE/filename)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value

    names = ('core', 'translator', 'vt_lib')
    previous = {name: sys.modules[name] for name in names if name in sys.modules}
    try:
        sys.modules.update(core=C, translator=T)
        V = module('finite_observation_original_vt_lib', 'vt_lib.py')
        V.FLOORS = {0: V.FLOORS[0]}
        V.MAPS = {key: value for key, value in V.MAPS.items() if key[0] == 0}
        sys.modules['vt_lib'] = V
        H = module('finite_observation_original_hurdle', 'hurdle.py')
    finally:
        # Restore only injected names; retain imported estimator modules for pickle.
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)
    return C, V, H


def one_per_position(C, rows, rng):
    _, inverse = np.unique(C.XY[rows], axis=0, return_inverse=True)
    inverse = inverse.ravel()
    return np.asarray([rng.choice(rows[inverse == i]) for i in range(inverse.max()+1)])


def task(C, V, H, floor, K, draw, rng, exposure_rng):
    """One original support draw, with its prescribed whole-scan exposure mode."""
    base_s = V.sample_support(0, floor, K, rng, False)
    srows = one_per_position(C, base_s, exposure_rng) if draw in (2, 3) else base_s
    all_h = {f: np.flatnonzero(C.FLR == f) for f in V.FLOORS[0] if f != floor}
    hrows = {f: one_per_position(C, rows, exposure_rng) if draw in (1, 3) else rows
             for f, rows in all_h.items()}
    histories = {f: dict(rss=C.RSS[rows], xy=C.XY[rows]) for f, rows in hrows.items()}
    support = dict(rss=C.RSS[srows], xy=C.XY[srows])
    context = build_context(histories, support, floor)
    all_target = np.flatnonzero(C.FLR == floor)
    qrows = all_target[~np.isin(all_target, base_s)]
    # Choose the original nearest target position FIRST, then test Q availability.
    # A depleted target position never redirects a candidate to a different label.
    Pt = H.STATS[(0, floor)][0]
    D = np.linalg.norm(context['cand_xy'][:, None] - Pt[None], axis=2)
    original_label = D.argmin(1)
    Pq, nq, ndq, mq, vq = H.det_stats(C.RSS[qrows], C.XY[qrows])
    qindex = {tuple(xy): i for i, xy in enumerate(Pq)}
    labels = np.asarray([qindex.get(tuple(Pt[j]), -1) for j in original_label], dtype=int)
    ok = (D.min(1) <= T.R_LABEL) & (labels >= 0)
    aps = context['aps']
    jj = labels[ok]
    X = context['features'][ok].reshape(-1, WIDTH)
    n = np.repeat(nq[jj], len(aps))
    nd, mean, variance = (value[jj][:, aps].ravel() for value in (ndq, mq, vq))
    return dict(context=context, histories=histories, support=support, H_rows=hrows,
                base_S_rows=base_s, S_rows=srows, Q_rows=qrows, original_label=original_label,
                Q_positions=Pq, labels=labels, eligible=ok, table=(X, n, nd, mean, variance),
                floor=floor, K=K, draw=draw, mode=MODES[draw])


def provenance(C, item):
    ctx = item['context']
    def stats_binding(st):
        return {key: array_sha(value) for key, value in st.items()}
    return dict(floor=item['floor'], K=item['K'], draw=item['draw'], mode=item['mode'],
        H_rows={str(f): (dict(all_source_floor=f, count=len(rows)) if item['draw'] in (0, 2)
                          else dict(csv_row_ids=C.ROW_IDS[rows].tolist(), count=len(rows)))
                for f, rows in item['H_rows'].items()},
        base_S_csv_rows=C.ROW_IDS[item['base_S_rows']].tolist(),
        S_csv_rows=C.ROW_IDS[item['S_rows']].tolist(),
        Q=dict(rule='All source target-floor rows minus every base S row',
               count=len(item['Q_rows']), csv_row_ids_sha256=array_sha(C.ROW_IDS[item['Q_rows']])),
        aps=ctx['aps'].tolist(), candidate_count=len(ctx['cand_xy']),
        eligible_candidates=int(item['eligible'].sum()), candidate_sha256=array_sha(ctx['cand_xy']),
        statistics=dict(H={str(f): stats_binding(st) for f, st in ctx['history_stats'].items()},
                        S=stats_binding(ctx['support_stats'])),
        preselection_features_sha256=array_sha(item['table'][0]))


def make_cell_table(C, V, H, seed, records):
    """Only replace task construction; H.HurdleField owns all original fits."""
    def cell_table(pairs, rng, draws=4, Ks=(1, 3, 5), max_rows=6000):
        assert draws == 4 and Ks == (1, 3, 5) and max_rows == 6000
        parts = []
        group = pairs[0][1] % 2
        for building, floor in pairs:
            assert building == 0
            for K in Ks:
                for draw in range(draws):
                    exposure_rng = np.random.default_rng([EXPOSURE_RNG_TAG, seed, group, building, floor, K, draw])
                    item = task(C, V, H, floor, K, draw, rng, exposure_rng)
                    X, n, nd, mean, variance = item['table']
                    # Same original info rule, negative retention probability and cap.
                    info = (np.nan_to_num(X[:, 0:40:5], nan=T.ND).max(1) > T.ND) | (nd > 0) | \
                           (np.nan_to_num(X[:, 44], nan=T.ND) > T.ND)
                    selected = np.flatnonzero(info | (rng.random(len(X)) < .05))
                    if len(selected) > max_rows:
                        selected = rng.choice(selected, max_rows, replace=False)
                    parts.append(tuple(value[selected] for value in (X, n, nd, mean, variance)))
                    records.append(dict(provenance(C, item), parity_group=group,
                        selected_cells=len(selected), selected_cell_indices_sha256=array_sha(selected)))
        return tuple(np.concatenate([part[i] for part in parts], axis=0) for i in range(5))
    return cell_table
