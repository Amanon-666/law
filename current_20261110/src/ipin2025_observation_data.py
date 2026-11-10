"""Explicit IPIN2025 training inputs and an unexecuted causal Query interface.

No main, writes, model calls, recursive dataset search or GT reader. Training
uses only <dataset_root>/01 Logfiles/01 - Training/*.txt. Inject the existing
public_corpus.adapter.parse_log callable; do not use its convert() (2025 is
not registered), or check_mf2.load_ipin_year (recursive, mixed partitions).
The server dependency is:
/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/
Law-Probabilistic-Transfer-20261023/public_corpus/adapter.py:parse_log

Its same-floor/building <=120s POSI interpolation and last-BSSID record rule
are unchanged. R=1: a receipt snapshot is not certified synchronous, fresh or
an independent static repeat. Label provenance remains in every packed row.
Fixed 9011 WGS84 reference and the existing equirectangular formula are shared
by all floors. Neither unused target positions nor GT choose the origin/APs.

Known-floor confirmation only: prepare and seal all eligible floor streams;
GT may select a floor stream only later in the external scorer. Level1 remains
unsupported and stays in full-site coverage. This module never infers floors.
Query ingestion below is independent of POSI and has NOT been run on scoring
files. The query horizon is the last sensor AppTimestamp in that same log,
never GT or the last Wi-Fi receipt alone. Retaining a last snapshot does not
certify freshness. The outer runner seals all prediction streams and their
input/code hashes before any scoring-stage timestamp or floor selection.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import math

import numpy as np


SEED = 11509159
REF_LON, REF_LAT = 23.7789900899589, 61.4935809264024
BUILDING = 2025
PHYSICAL_FLOORS = (1, 2, 3, 4, 5, 6)
SENSOR_TAGS = frozenset(('ACCE', 'ACCU', 'GYRO', 'GYRU', 'MAGN', 'MAGU',
    'STEP', 'PRES', 'LIGH', 'PROX', 'HUMI', 'TEMP', 'AHRS', 'GNSS', 'GNSR',
    'WIFI', 'BLUE', 'BLE4', 'SOUN', 'RFID', 'IMUX', 'IMUE', 'IMUI'))


def metric_xy(records):
    """Absolute common-frame metres, x east/y north; original adapter formula."""
    return np.asarray([((r['lon'] - REF_LON)*111320*np.cos(np.radians(REF_LAT)),
                        (r['lat'] - REF_LAT)*111320) for r in records], dtype=np.float64)


def _rank(*parts):
    return hashlib.sha256('|'.join((str(SEED), *map(str, parts))).encode()).digest()


def load_training(dataset_root, *, parse_log):
    """Read the sole training whitelist with the unchanged injected parser.

    Return raw per-floor records, eligibility and ordered S identities. Source
    parser access and input hashes are frozen by the outer runner. No RSS-based
    choice or target outcome participates in eligibility or stable hash ranks.
    """
    root = Path(dataset_root).resolve()
    directory = root/'01 Logfiles'/'01 - Training'
    files = sorted(p for p in directory.glob('*.txt') if not p.name.startswith('._'))
    if not files:
        raise FileNotFoundError(f'No training logs in {directory}')
    floors = defaultdict(list)
    for path in files:
        if path.resolve().parent != directory.resolve():
            raise ValueError('Training file resolves outside the allowed directory')
        relative = path.relative_to(root).as_posix()
        for r in parse_log(path):
            assert r['building'] == BUILDING and r['floor'] in PHYSICAL_FLOORS
            record = dict(r, file=str(path), relative_file=relative,
                          scan_id=f"{relative}@{float(r['app_timestamp']).hex()}")
            floors[r['floor']].append(record)
    positions = {}
    for floor, records in floors.items():
        grouped = defaultdict(list)
        for r in records:
            grouped[(r['lat'], r['lon'])].append(r)
        positions[floor] = grouped
    eligible = [f for f in PHYSICAL_FLOORS if len(positions.get(f, {})) >= 20
                and any(h != f and rows for h, rows in floors.items())]
    support_order, metadata = {}, {}
    for f in PHYSICAL_FLOORS:
        groups = positions.get(f, {})
        metadata[f] = dict(scans=len(floors.get(f, [])), positions=len(groups),
                           eligible=f in eligible,
                           status='eligible' if f in eligible else 'unsupported_training_coverage')
        if f in eligible:
            ordered = sorted(groups, key=lambda xy: (
                _rank('position', f, float(xy[0]).hex(), float(xy[1]).hex()), xy))
            support_order[f] = [min(groups[xy], key=lambda r:
                (_rank('scan', f, r['scan_id']), r['scan_id'])) for xy in ordered[:5]]
    return dict(floors=dict(floors), eligible_floors=eligible,
                floor_metadata=metadata, support_order=support_order,
                seed=SEED, reference=dict(id=9011, longitude=REF_LON, latitude=REF_LAT),
                training_files=[str(p) for p in files])


def rss_matrix(records, ap_ids):
    """Align only to the caller's H/S AP axis; Query-only BSSIDs are ignored."""
    column = {ap: i for i, ap in enumerate(ap_ids)}
    rss = np.full((len(records), len(ap_ids)), 100., dtype=np.float64)
    for row, record in enumerate(records):
        for ap, value in record['rss'].items():
            if ap in column:
                rss[row, column[ap]] = value
    return rss


def _pack(records, ap_ids):
    return dict(rss=rss_matrix(records, ap_ids), xy=metric_xy(records),
                scan_ids=[r['scan_id'] for r in records],
                label_provenance=[dict(file=r['file'], app_timestamp=r['app_timestamp'],
                    landmark_lines=r['landmark_lines'], landmark_times=r['landmark_times'],
                    interpolation_fraction=r['fraction'], raw_latlon=(r['lat'], r['lon']),
                    position_kind='time_interpolated_POSI_landmarks') for r in records])


def farthest_history_rows(xy, floor):
    """Original public_corpus.history_cache.farthest_rows, coordinate-only.

    Same unique-position sorting, random start, farthest-point ties and final
    one-real-row draw; fixed limit 128 and seed 261026+2025*100+floor. xy is
    metric_xy(training['floors'][floor]) in its unchanged row order. No RSS,
    target S or Query is read. The caller freezes these indices and scan IDs
    once per floor and supplies the same selected H to Hurdle and GP.
    This is the existing public-history resource budget, not a model change.
    """
    positions, group = np.unique(xy, axis=0, return_inverse=True)
    group = group.ravel()
    rng = np.random.default_rng(261026 + 2025*100 + floor)
    keep, nearest = [], np.full(len(positions), np.inf)
    j = int(rng.integers(len(positions)))
    for _ in range(min(128, len(positions))):
        keep.append(j)
        nearest = np.minimum(nearest, ((positions - positions[j])**2).sum(1))
        nearest[keep] = -1
        j = int(nearest.argmax())
    return np.array([rng.choice(np.flatnonzero(group == g)) for g in keep], dtype=int)


def make_context(training, target_floor, k, *, history_rows=None):
    """H=other training floors; S=first K hash-ranked positions, one scan each.

    Pass histories/support/target_floor to FrozenHurdleTranslator.prepare. Reuse ap_ids for
    rss_matrix(Query records, ap_ids). Different K/floors may have different AP
    axes; no common axis may be expanded using unselected target or Query RSS.
    history_rows is an explicit frozen {physical_floor: original_row_indices}
    for every H floor. None retains all H; this function never selects rows.
    Selected H defines the H part of the AP axis and candidate geometry for
    BOTH consumers. S selection, eligibility and site coverage stay unchanged.
    """
    assert target_floor in training['eligible_floors'] and k in (1, 3, 5)
    hrows = {f: rows for f, rows in sorted(training['floors'].items()) if f != target_floor}
    if history_rows is not None:
        hrows = {f: [rows[i] for i in history_rows[f]] for f, rows in hrows.items()}
    srows = training['support_order'][target_floor][:k]
    ap_ids = sorted({ap for rows in [*hrows.values(), srows] for r in rows for ap in r['rss']})
    name = lambda f: f'IPIN2025_B{BUILDING}F{f}'
    return dict(histories={name(f): _pack(rows, ap_ids) for f, rows in hrows.items()},
                support=_pack(srows, ap_ids), target_floor=name(target_floor),
                ap_ids=ap_ids, K=k, R=1,
                floor_metadata=training['floor_metadata'])


def read_query_stream(path):
    """Read one explicit raw log into receipts and a fixed causal 2Hz index.

    path is ONE explicitly supplied log, never a directory or inferred file.
    For documented sensor records, only AppTimestamp is read to establish the
    acquisition horizon; WIFI additionally supplies BSSID/RSS and provenance.
    POSI, GT, sensor GNSS coordinates and floor values are not parsed.
    All records with the same float AppTimestamp form one complete receipt,
    using last-record-per-BSSID in original file order. Sensor timestamps and
    frequency of retained observations, plus conflict counts, are preserved.
    Output starts at ceil(2*first_receipt)/2 (same-time receipt is available),
    ends at floor(2*last_sensor_AppTimestamp)/2 and never uses a future receipt.
    The clock is not reset at the first Wi-Fi receipt. There is no GT-derived
    end_time argument, no future interpolation and no synthetic missing scan.

    Return snapshots (ordered unique receipt keys), prediction_times and
    snapshot_indices (one index per prediction time), used_snapshot_indices,
    and acquisition_start/end. Predict only used_snapshot_indices once per
    prepared floor/K, then expand by snapshot_indices when sealing each stream.
    Equal RSS at different receipt times remain different snapshots. Reading
    the file first groups complete receipts; selection is causal by receipt
    time. The full horizon is offline metadata, not a predictor feature.
    No Wi-Fi or no 2Hz tick yields an empty stream, to be reported as uncovered.
    """
    path = Path(path).resolve()
    snapshots = {}
    start_time = end_time = None
    with path.open(encoding='latin-1') as stream:
        for line_number, line in enumerate(stream, 1):
            tag = line.split(';', 1)[0]
            if tag not in SENSOR_TAGS:
                continue
            stamp = float(line.split(';', 2)[1])
            start_time = stamp if start_time is None else min(start_time, stamp)
            end_time = stamp if end_time is None else max(end_time, stamp)
            if tag != 'WIFI':
                continue
            fields = line.strip().rstrip(';').split(';')
            # Same fields/rstrip/last-record convention as adapter.parse_log.
            ap, value = fields[4].lower(), float(fields[-1])
            record = snapshots.setdefault(stamp, dict(file=str(path), app_timestamp=stamp,
                scan_id=f'{path}@{stamp.hex()}', rss={}, observations={},
                duplicate_bssid_rows=0, conflicting_rss_rows=0))
            if ap in record['rss']:
                record['duplicate_bssid_rows'] += 1
                record['conflicting_rss_rows'] += int(record['rss'][ap] != value)
            record['rss'][ap] = value
            record['observations'][ap] = dict(line=line_number,
                sensor_timestamp=float(fields[2]), frequency=float(fields[5]))
    stamps = np.asarray(sorted(snapshots), dtype=np.float64)
    times = (np.arange(math.ceil(2*stamps[0]), math.floor(2*end_time)+1,
                       dtype=np.int64)/2 if len(stamps) else np.empty(0, dtype=np.float64))
    indices = np.searchsorted(stamps, times, side='right')-1
    return dict(file=str(path), snapshots=[snapshots[t] for t in stamps],
                prediction_times=times, snapshot_indices=indices,
                used_snapshot_indices=np.unique(indices),
                acquisition_start=start_time, acquisition_end=end_time,
                horizon_source='sensor AppTimestamp only; POSI/GT excluded')


def iter_causal_query(path):
    """Yield the same stream as read_query_stream, without a caller-set end."""
    stream = read_query_stream(path)
    for time, index in zip(stream['prediction_times'], stream['snapshot_indices']):
        yield dict(time=float(time), snapshot_index=int(index),
                   snapshot=stream['snapshots'][index])


def match_prediction_times(prediction_times, evaluation_times):
    """Scoring-stage indices matching evaluateCompetitor2025.m lines 90-92.

    Only call AFTER all eligible floor streams are sealed. Inputs are strictly
    increasing unique 2Hz prediction times and scoring timestamps in original
    AppTimestamp seconds. Return the latest prediction <= evaluation_time+.01,
    or -1 when none exists. The scorer must mask -1, never index the last row.
    This function reads no GT file, coordinates or floor, and changes no stream.

    Official GT columns used by that script are timestamp, longitude, latitude,
    floor and evaluation-point indicator. A floor outside the frozen eligible
    list is unsupported regardless of a valid time match; preserve it in the
    full-site coverage denominator. Prediction after log end is not fabricated:
    the official last-past rule may match the last sealed tick, whose age the
    scorer can report. No nearest-future matching or stale-point deletion.
    """
    return np.searchsorted(np.asarray(prediction_times),
                           np.asarray(evaluation_times)+.01, side='right')-1
