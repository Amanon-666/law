"""Compose historical source tasks for the existing full Tab learner.

This uses the original three-floor protected catalog. It changes training
context composition, not the network or target inference. One context has
matched-position same/cross-acquisition queries; their losses have equal mass.
"""
from collections import Counter
from paired_acquisition_tasks import build_pairs, make_quartet


def training_contexts(catalog):
    pairs = build_pairs(catalog)
    floors = sorted(catalog['source_floors'])
    assert len(floors) == 3 and all(f.startswith('B0F') for f in floors)
    by_floor = {f: [p for p in pairs if p['floor'] == f] for f in floors}
    schedule = []
    for step in range(24):
        batch = []
        k = (1, 3, 5)[step % 3]
        cycle = step // 3
        for floor in floors:
            names = sorted(catalog['history_rows'][floor])
            assert len(names) == 2
            # At each K: four complete contexts and two of each singleton.
            history = names if cycle % 2 == 0 else [names[(cycle // 2) % 2]]
            pair = by_floor[floor][(cycle // 4 + cycle // 2 + step % 3) % len(by_floor[floor])]
            side = (cycle // 4 + step % 3) % 2
            draw = step if step < 2 else step + 1
            quartet = make_quartet(catalog, pair, draw, k=k, histories=history, phase='fit')
            events = quartet[2*side:2*side+2]
            batch.append(dict(floor=floor, K=k, histories=history,
                              pair_id=pair['id'], support_side=('A', 'B')[side],
                              draw_index=draw, events=events))
        schedule.append(batch)
    for floor in floors:
        for k in (1, 3, 5):
            records = [r for batch in schedule for r in batch if r['floor']==floor and r['K']==k]
            assert Counter(len(r['histories']) for r in records) == {1:4, 2:4}
            singles = Counter(r['histories'][0] for r in records if len(r['histories'])==1)
            assert set(singles.values()) == {2} and len(singles) == 2
    return schedule
