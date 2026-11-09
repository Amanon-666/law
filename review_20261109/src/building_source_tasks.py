"""Metadata-only two-building source tasks with a third building excluded.

New development protocol, not a relabeling of previous B2 tests as training.
Original catalogs, fitted fields, checkpoints and results remain unchanged.
"""
import hashlib
import json
import math
import socket
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from paired_acquisition_tasks import make_quartet, position_index, verify_quartet
from source_episode_catalog import build_catalog, read_metadata, verify_catalog

ROOT = Path(__file__).resolve().parent
CSV = ROOT.parent/'UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv'
OUT = ROOT/'results/building_source_tasks_20261106'
SOURCE = {'B0': 4, 'B2': 5}
HELD_OUT = 'B1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def building(floor):
    return floor.split('F')[0]


def paired_families(catalog):
    """Admit pairs by observed coverage only, never a fitted score."""
    candidates = sorted({tuple(sorted((f['support_group'], f['query_group'])))
                         for f in catalog['families']})
    accepted, rejected = [], []
    for left, right in candidates:
        a, b = catalog['groups'][left], catalog['groups'][right]
        common = {phase: sorted(position_index(a, phase).keys() & position_index(b, phase).keys())
                  for phase in ('train', 'validation')}
        record = dict(id=f'{left}__{right}', floor=a['floor'], groups=[left, right],
                      common_positions={p: [list(x) for x in xs] for p, xs in common.items()})
        reasons = []
        if len(common['train']) < 17: reasons.append('Fewer than 5 S + 12 disjoint fit-Q common R3 positions')
        if not common['validation']: reasons.append('No common protected R3 position')
        if reasons:
            rejected.append(dict(record, reasons=reasons))
        else:
            accepted.append(record)
    floors = defaultdict(set)
    for pair in accepted: floors[building(pair['floor'])].add(pair['floor'])
    assert set(floors) == set(SOURCE), 'A whole source building lacks admissible paired tasks'
    counts = Counter(p['floor'] for p in accepted)
    for pair in accepted:
        pair['sampling_probability'] = 1/(len(floors)*len(floors[building(pair['floor'])])*counts[pair['floor']])
    assert math.isclose(sum(p['sampling_probability'] for p in accepted), 1.)
    return accepted, rejected


def run():
    OUT.mkdir(exist_ok=False)
    floors = {f'{b}F{f}' for b, count in SOURCE.items() for f in range(count)}
    code = ['building_source_tasks.py', 'source_episode_catalog.py',
            'paired_acquisition_tasks.py', 'acquisition_tasks.py', 'uji.py']
    frozen = dict(created=datetime.now(timezone.utc).isoformat(), source_buildings=sorted(SOURCE),
        held_out_building=HELD_OUT, CSV_sha256=sha(CSV), code={n: sha(ROOT/n) for n in code},
        question='Construct actual cross-building source diversity without changing the model or opening target RSS',
        permissions='No B1 rows in source H/S/Q, normalization or future source calibration; deployment B1 H/S is separate',
        history='All other floors in the same source building; all protected positions removed',
        sampling='Equal building, then eligible floor, then unordered acquisition pair, then four matched/cross cells',
        partitions='Existing source-catalog seed and 25% position protection, now including B0F3/B2 floors',
        model='Existing source-preserving ReFi operator, no new architecture or checkpoint chosen here',
        limitations=['Only two source buildings, not nine independent buildings.',
            'B1/B2 have previously been used for development; this is not a fresh blind test.',
            'B2 is a source only in this new protocol; old evaluation results keep their original roles.',
            'No promise of transferable signal or gain follows from adding source diversity.',
            'Same-coordinate/AP correspondence is used only within a building, never across buildings.'],
        resources=dict(cpu=1, gpu=0, seconds=60, downloads=0), RSS_parsed=False)
    write(OUT/'freeze.json', frozen)
    for n in code: (OUT/n).write_bytes((ROOT/n).read_bytes())
    rows = read_metadata(CSV, floors)
    assert {building(r['floor']) for r in rows.values()} == set(SOURCE)
    by_floor = {f: sorted(i for i, row in rows.items() if row['floor'] == f) for f in sorted(floors)}
    parents = []
    for floor in sorted(floors):
        names = sorted(f for f in floors if building(f) == building(floor) and f != floor)
        parents.append(dict(role='train', floor=floor, histories=names,
                            history_rows={h: by_floor[h] for h in names}))
    catalog = build_catalog(rows, dict(events=parents))
    counts = Counter(f['floor'] for f in catalog['families'])
    # Keep the unpaired catalog's weights consistent with the building hierarchy too.
    for family in catalog['families']:
        family['sampling_probability'] = 1/(len(SOURCE)*SOURCE[building(family['floor'])]*counts[family['floor']])
    catalog['contract'].update(source_buildings=sorted(SOURCE), held_out_building=HELD_OUT,
        sampling='Equal building/floor/directed family for unpaired tasks; paired_catalog supplies paired probabilities',
        prior_results='New development protocol; no reassignment of previous experiment roles')
    verify_catalog(rows, catalog)
    pairs, excluded = paired_families(catalog)
    checked = 0
    examples = []
    for pair in pairs:
        for phase in ('fit', 'validation'):
            full = make_quartet(catalog, pair, 0, phase=phase)
            verify_quartet(rows, catalog, full, phase)
            for k in (1, 3, 5):
                quartet = make_quartet(catalog, pair, 0, k=k, phase=phase)
                verify_quartet(rows, catalog, quartet, phase)
                for e, reference in zip(quartet, full):
                    assert e['support_rows'] == reference['support_rows'][:3*k]
                    assert e['query_rows'] == reference['query_rows']
                    assert all(building(h) == building(e['floor']) for h in e['histories'])
                    assert all(building(rows[i]['floor']) != HELD_OUT
                               for i in e['support_rows']+e['query_rows'])
                    assert all(building(rows[i]['floor']) != HELD_OUT
                               for ids in e['history_rows'].values() for i in ids)
                checked += 4
        if not any(building(e['floor']) == building(pair['floor']) for e in examples):
            examples.append(full[0])
    mass = {b: sum(p['sampling_probability'] for p in pairs if building(p['floor']) == b) for b in SOURCE}
    assert all(math.isclose(v, .5) for v in mass.values())
    prior = json.loads((ROOT/'results/source_episode_catalog_20261105/catalog.json').read_text())
    assert all(catalog['protected_positions'][f] == prior['protected_positions'][f] for f in prior['source_floors'])
    write(OUT/'catalog.json', catalog)
    write(OUT/'paired_catalog.json', dict(pairs=pairs, excluded_pairs=excluded))
    write(OUT/'example_validation_events.json', examples)
    result = dict(complete=True, source_buildings=sorted(SOURCE), held_out_building=HELD_OUT,
        metadata_rows=len(rows), protected_rows=len(catalog['protected_rows']),
        floors=catalog['source_floors'], paired_counts_by_floor=dict(Counter(p['floor'] for p in pairs)),
        floors_without_paired_tasks=sorted(floors-{p['floor'] for p in pairs}),
        pairs=len(pairs), excluded_common_coverage_pairs=len(excluded),
        source_probability_mass=mass, history_counts={f: len(h) for f, h in catalog['history_rows'].items()},
        verified_event_descriptions=checked, old_protected_positions_preserved=True,
        RSS_parsed=False, model_updates=0, GP_refits=0, localization_scores=0)
    write(OUT/'complete.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    run()
