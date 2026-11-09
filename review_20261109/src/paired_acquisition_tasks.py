"""Same-position 2x2 acquisition tasks from the existing protected source pool.

Metadata is a sampling control, not a predictor input or a device-causal label.
No RSS parsing, fitting, localization, or changes to the original task catalog.
"""
import json
import socket
from collections import Counter

from source_episode_catalog import generator, read_metadata, verify_catalog


def position_index(group, phase):
    return {tuple(p['xy']): p['rows'] for p in group[phase]}


def build_pairs(catalog):
    keys = sorted({tuple(sorted((f['support_group'], f['query_group'])))
                   for f in catalog['families']})
    pairs = []
    for left, right in keys:
        a, b = catalog['groups'][left], catalog['groups'][right]
        assert a['floor'] == b['floor']
        common = {phase: sorted(position_index(a, phase).keys() &
                                position_index(b, phase).keys())
                  for phase in ('train', 'validation')}
        assert len(common['train']) >= 17 and common['validation']
        pairs.append(dict(id=f'{left}__{right}', floor=a['floor'], groups=[left, right],
                          common_positions={p: [list(x) for x in xs] for p, xs in common.items()}))
    counts = Counter(p['floor'] for p in pairs)
    assert set(counts) == set(catalog['source_floors'])
    for pair in pairs:
        pair['sampling_probability'] = 1/(len(counts)*counts[pair['floor']])
    return pairs


def make_quartet(catalog, pair, draw_index, *, k=5, histories=None, phase='fit'):
    """Return AA, AB, BA, BB with exactly the same H and S/Q positions.

    All maximum-K support positions are excluded from fit Q even for K=1.
    Validation uses every paired protected position, fixed across K and draws.
    """
    assert k in (1, 3, 5) and draw_index >= 0 and phase in ('fit', 'validation')
    common = [tuple(x) for x in pair['common_positions']['train']]
    support = generator('paired_support', pair['id'], draw_index).sample(common, 5)
    if phase == 'fit':
        pool = [x for x in common if x not in set(support)]
        query = generator('paired_query', pair['id'], draw_index).sample(pool, 12)
    else:
        query = [tuple(x) for x in pair['common_positions']['validation']]
    allowed = catalog['history_rows'][pair['floor']]
    names = sorted(allowed) if histories is None else sorted(histories)
    assert names and len(names) == len(set(names)) and set(names) <= set(allowed)
    result = []
    for si, sid in enumerate(pair['groups']):
        for qi, qid in enumerate(pair['groups']):
            sg, qg = catalog['groups'][sid], catalog['groups'][qid]
            sp = position_index(sg, 'train')
            qp = position_index(qg, 'train' if phase == 'fit' else 'validation')
            cell = ('A', 'B')[si]+('A', 'B')[qi]
            result.append(dict(id=f"{pair['id']}_draw{draw_index}_K{k}_{cell}_{phase}_H{'-'.join(names)}",
                role='train' if phase == 'fit' else 'source_validation',
                floor=pair['floor'], K=k, R=3, histories=names,
                history_rows={h: allowed[h] for h in names},
                support_rows=[i for x in support[:k] for i in sp[x]],
                query_rows=[i for x in query for i in qp[x]],
                support_acquisition=sg['acquisition'], query_acquisition=qg['acquisition'],
                paired_cell=cell, pair_id=pair['id'],
                sampling_probability=pair['sampling_probability']/4))
    return result


def verify_quartet(rows, catalog, quartet, phase):
    forbidden = set(catalog['protected_rows'])
    first = quartet[0]
    sxy = [rows[i]['xy'] for i in first['support_rows']]
    qxy = [rows[i]['xy'] for i in first['query_rows']]
    assert not set(sxy).intersection(qxy)
    assert [e['paired_cell'] for e in quartet] == ['AA', 'AB', 'BA', 'BB']
    for e in quartet:
        assert e['history_rows'] == first['history_rows']
        assert [rows[i]['xy'] for i in e['support_rows']] == sxy
        assert [rows[i]['xy'] for i in e['query_rows']] == qxy
        assert len(set(e['support_rows'])) == 3*e['K']
        assert len(set(e['query_rows'])) == len(qxy)
        assert sorted(Counter(rows[i]['xy'] for i in e['support_rows']).values()) == [3]*e['K']
        assert set(Counter(rows[i]['xy'] for i in e['query_rows']).values()) == {3}
        assert not forbidden.intersection(e['support_rows'])
        assert all(not forbidden.intersection(ids) for ids in e['history_rows'].values())
        assert all((i in forbidden) == (phase == 'validation') for i in e['query_rows'])
        assert all(list(rows[i]['acquisition']) == e['support_acquisition'] for i in e['support_rows'])
        assert all(list(rows[i]['acquisition']) == e['query_acquisition'] for i in e['query_rows'])
    assert quartet[0]['support_rows'] == quartet[1]['support_rows']
    assert quartet[2]['support_rows'] == quartet[3]['support_rows']
    assert quartet[0]['query_rows'] == quartet[2]['query_rows']
    assert quartet[1]['query_rows'] == quartet[3]['query_rows']


def main():
    from recomposition_trial import ROOT, CSV, digest, write, now
    out = ROOT/'results/paired_acquisition_tasks_20261105'
    out.mkdir(exist_ok=False)
    original = ROOT/'results/source_episode_catalog_20261105'
    previous = json.loads((original/'freeze.json').read_text())
    assert previous['inputs'][str(CSV)] == digest(CSV)
    names = ['paired_acquisition_tasks.py', 'source_episode_catalog.py',
             'acquisition_tasks.py', 'uji.py', 'recomposition_trial.py']
    assert all(digest(ROOT/n) == h for n, h in previous['code'].items())
    write(out/'freeze.json', dict(created=now(),
        inputs={str(p): digest(p) for p in (CSV, original/'catalog.json', original/'freeze.json')},
        code={n: digest(ROOT/n) for n in names},
        scope='Metadata-only same-position acquisition quartet; not independent environments or causal device effects',
        limits=dict(cpu=1, gpu=0, seconds=120, downloads=0),
        inference='Acquisition labels select rows only; no new predictor privileges or query-assisted fit',
        learning_object='Existing source-preserving spatial operator; no training in this task'))
    for n in names: (out/n).write_bytes((ROOT/n).read_bytes())
    catalog = json.loads((original/'catalog.json').read_text())
    rows = read_metadata(CSV, set(catalog['source_floors']))
    verify_catalog(rows, catalog)
    pairs = build_pairs(catalog)
    checked = 0
    for pair in pairs:
        validation_reference = None
        for draw in (0, 1):
            for phase in ('fit', 'validation'):
                full = make_quartet(catalog, pair, draw, phase=phase)
                assert full == make_quartet(catalog, pair, draw, phase=phase)
                verify_quartet(rows, catalog, full, phase)
                if phase == 'validation':
                    queries = [e['query_rows'] for e in full]
                    if validation_reference is not None: assert queries == validation_reference
                    validation_reference = queries
                for k in (1, 3, 5):
                    quartet = make_quartet(catalog, pair, draw, k=k, phase=phase)
                    verify_quartet(rows, catalog, quartet, phase)
                    assert all(e['support_rows'] == f['support_rows'][:3*k] and
                               e['query_rows'] == f['query_rows'] for e, f in zip(quartet, full))
                    checked += len(quartet)
                for h in full[0]['histories']:
                    single = make_quartet(catalog, pair, draw, histories=[h], phase=phase)
                    assert all(e['support_rows'] == f['support_rows'] and
                               e['query_rows'] == f['query_rows'] for e, f in zip(single, full))
    # A different query location can look like an acquisition effect: reject it.
    corrupted = json.loads(json.dumps(make_quartet(catalog, pairs[0], 0)))
    corrupted[1]['query_rows'][:3] = corrupted[1]['query_rows'][3:6]
    try:
        verify_quartet(rows, catalog, corrupted, 'fit')
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatched query positions were accepted')
    write(out/'paired_catalog.json', dict(base_catalog=str(original/'catalog.json'), pairs=pairs))
    write(out/'example_quartet.json', make_quartet(catalog, pairs[0], 0))
    result = dict(complete=True, created=now(), source_floors=catalog['source_floors'],
        pairs=len(pairs), pairs_by_floor=dict(Counter(p['floor'] for p in pairs)),
        paired_validation_positions={p['id']: len(p['common_positions']['validation']) for p in pairs},
        verified_events=checked, protected_rows=len(catalog['protected_rows']),
        mismatched_positions_rejected=True, RSS_parsed=False, model_updates=0, localization_scores=0)
    write(out/'complete.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    main()
