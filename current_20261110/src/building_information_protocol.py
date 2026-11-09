"""Frozen B0-trained assets -> B1/B2 information comparison protocol.

Only CSV metadata is interpreted. No WAP RSS values are parsed, no model is
loaded, and official validationData is not used. This is building-level
development evaluation, not three-fold LOBO or a previously unseen test.
"""
import csv
import hashlib
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone


SEED = 11509159
KS = (1, 3, 5)
R = 3
QUERY_POSITIONS = 12
FLOORS = tuple(f'B{b}F{f}' for b, count in ((1, 4), (2, 5)) for f in range(count))


def read_metadata(csv_path):
    """Return B1/B2 metadata keyed by original, one-based CSV data-row ID."""
    rows = {}
    with open(csv_path, newline='') as stream:
        for row_id, record in enumerate(csv.DictReader(stream), 1):
            building = f"B{record['BUILDINGID']}"
            floor = f"{building}F{record['FLOOR']}"
            if floor not in FLOORS:
                continue
            timestamp = int(record['TIMESTAMP'])
            rows[row_id] = dict(
                building=building, floor=floor,
                xy=(float(record['LONGITUDE']), float(record['LATITUDE'])),
                timestamp=timestamp,
                acquisition=(int(record['USERID']), int(record['PHONEID']),
                             datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()))
    return rows


def _groups(rows, floor):
    groups = defaultdict(lambda: defaultdict(list))
    for row_id, record in rows.items():
        if record['floor'] == floor:
            groups[record['acquisition']][record['xy']].append(row_id)
    for positions in groups.values():
        for ids in positions.values():
            ids.sort(key=lambda row_id: (rows[row_id]['timestamp'], row_id))
    return groups


def _pair(groups):
    keys = sorted(groups)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            if left[:2] == right[:2]:
                continue
            common = sorted(xy for xy in groups[left].keys() & groups[right].keys()
                            if len(groups[left][xy]) >= R and len(groups[right][xy]) >= R)
            if len(common) >= max(KS) + QUERY_POSITIONS:
                return left, right, common
    raise ValueError('No different-user-phone acquisition pair with 17 common R3 positions')


def _positions(floor, common):
    seed = int.from_bytes(hashlib.sha256(f'{SEED}:{floor}'.encode()).digest()[:8], 'little')
    chosen = random.Random(seed).sample(common, max(KS) + QUERY_POSITIONS)
    return chosen[:max(KS)], chosen[max(KS):]


def build_manifest(csv_path):
    """Build and verify 54 metadata-only events; do not write any files."""
    rows = read_metadata(csv_path)
    history_rows = {floor: sorted(i for i, row in rows.items() if row['floor'] == floor)
                    for floor in FLOORS}
    events, floor_pairs = [], {}
    for floor in FLOORS:
        groups = _groups(rows, floor)
        left, right, common = _pair(groups)
        sxy, qxy = _positions(floor, common)
        building = floor.split('F')[0]
        histories = [name for name in FLOORS if name.startswith(building) and name != floor]
        floor_pairs[floor] = dict(
            acquisition_a=list(left), acquisition_b=list(right),
            common_R3_positions=len(common),
            support_positions=[list(xy) for xy in sxy],
            query_positions=[list(xy) for xy in qxy])
        for direction, source, query in (('AB', left, right), ('BA', right, left)):
            for k in KS:
                events.append(dict(
                    id=f'{floor}_{direction}_K{k}', building=building, floor=floor,
                    direction=direction, K=k, R=R, histories=histories.copy(),
                    support_acquisition=list(source), query_acquisition=list(query),
                    support_rows=[i for xy in sxy[:k] for i in groups[source][xy][:R]],
                    query_rows=[i for xy in qxy for i in groups[query][xy][:R]]))
    with open(csv_path, 'rb') as stream:
        input_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = dict(
        seed=SEED, csv_path=str(csv_path), csv_sha256=input_sha256,
        buildings=['B1', 'B2'], floors=list(FLOORS), K=list(KS), R=R,
        history_rows=history_rows, floor_pairs=floor_pairs, events=events,
        contract=dict(
            scope='B0-trained assets evaluated on B1/B2; not three-fold LOBO or blind evaluation',
            pair='Lexically first unordered acquisition pair with different (user,phone) and >=17 common R3 positions',
            position_seed='First 8 SHA256 bytes of UTF-8 seed:floor, little endian; stdlib random.sample on sorted common XY',
            positions='Five nested support positions and twelve query positions; identical XY in AB/BA; full S5 excluded from Q for every K',
            scans='First three real rows per acquisition/XY sorted by (timestamp,row_id); no replacement or RSS selection',
            histories='All trainingData rows on all other floors of the same building; global history_rows indexed by event.histories',
            information_permissions={
                'H': 'AP catalogue, domain and fitted context statistics use H only; no S-derived calibration or bounds',
                'S': 'AP catalogue, domain and fitted context statistics use current S only',
                'HS': 'AP catalogue, domain and fitted context statistics use H plus current S only',
                'Q': 'RSS only for forward prediction; query coordinates only for final scoring; no query-fitted transforms'},
            aggregation='For each K: equal buildings, equal floors within building, equal AB/BA, equal query positions and equal three scans per position',
            validation='Official validationData is not opened here and was already exposed in earlier project work',
            limits=['Public foundation-model pretraining has not been audited',
                    'History on other floors may contain the query device or user',
                    'Acquisition tuples are observed groups, not certified independent physical replicates',
                    'Repeated scans, directions, K values and floors do not create independent buildings']))
    verify_manifest(rows, manifest)
    return manifest


def verify_manifest(rows, manifest):
    """Check the actual metadata contract; no RSS, model calls or synthetic tests."""
    assert manifest['seed'] == SEED and manifest['K'] == list(KS) and manifest['R'] == R
    assert manifest['buildings'] == ['B1', 'B2'] and manifest['floors'] == list(FLOORS)
    assert {row['floor'] for row in rows.values()} == set(FLOORS)
    assert set(manifest['history_rows']) == set(FLOORS)
    assert set(manifest['floor_pairs']) == set(FLOORS)
    for floor in FLOORS:
        assert manifest['history_rows'][floor] == sorted(i for i, row in rows.items() if row['floor'] == floor)
    events = manifest['events']
    assert len(events) == 54 and len({event['id'] for event in events}) == 54
    assert Counter(event['floor'] for event in events) == Counter({floor: 6 for floor in FLOORS})
    by_key = {(event['floor'], event['direction'], event['K']): event for event in events}
    assert len(by_key) == 54
    for floor in FLOORS:
        groups = _groups(rows, floor)
        left, right, common = _pair(groups)
        sxy, qxy = _positions(floor, common)
        pair = manifest['floor_pairs'][floor]
        assert pair['acquisition_a'] == list(left) and pair['acquisition_b'] == list(right)
        assert pair['common_R3_positions'] == len(common)
        assert pair['support_positions'] == [list(xy) for xy in sxy]
        assert pair['query_positions'] == [list(xy) for xy in qxy]
        assert len(set(sxy)) == 5 and len(set(qxy)) == 12 and not set(sxy) & set(qxy)
        building = floor.split('F')[0]
        expected_h = [name for name in FLOORS if name.startswith(building) and name != floor]
        for direction, source, query in (('AB', left, right), ('BA', right, left)):
            all_support = by_key[floor, direction, 5]['support_rows']
            expected_query = [i for xy in qxy for i in groups[query][xy][:R]]
            for k in KS:
                event = by_key[floor, direction, k]
                assert event['id'] == f'{floor}_{direction}_K{k}'
                assert event['building'] == building and event['R'] == R
                assert event['histories'] == expected_h
                assert event['support_acquisition'] == list(source) and event['query_acquisition'] == list(query)
                expected_support = [i for xy in sxy[:k] for i in groups[source][xy][:R]]
                assert event['support_rows'] == expected_support == all_support[:k * R]
                assert event['query_rows'] == expected_query
                assert len(set(expected_support)) == k * R and len(set(expected_query)) == QUERY_POSITIONS * R
                assert not set(expected_support) & set(expected_query)
                assert all(rows[i]['floor'] == floor and rows[i]['acquisition'] == source for i in expected_support)
                assert all(rows[i]['floor'] == floor and rows[i]['acquisition'] == query for i in expected_query)
            # Both directions were checked against the same ordered XY lists.
    return True
