"""Prepare source row plans only; no fitting, AP selection or query scoring."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from residual_probe.episode_rows import nested_rows, check


def prepare(root):
    # The current frozen loader already owns raw corpus/coordinate semantics.
    # Import it only in this explicit preparation entry, not in the row helper.
    from residual_probe.prepare import data_module
    old = data_module()
    root = Path(root)
    out = root / 'results/history_learning_stage2_20261027'
    prior = root / 'results/k_curve_v1_20261027/manifest.json'
    old_events = json.loads(prior.read_text())['events']
    budgets = (1, 3, 5, 10, 20)
    events, pools = [], []
    for name in sorted(old.CONFIGS):
        building = old.building_data(name)
        for target, raw in sorted(building['floors'].items()):
            excluded = {tuple(raw['xy'][row]) for e in old_events
                        if e['building'] == name and e['target'] == target
                        for row in e['query_rows']}
            allowed = [i for i, point in enumerate(raw['xy'])
                       if tuple(point) not in excluded]
            histories = sorted(f for f in building['floors'] if f != target)
            unique_positions = len({tuple(raw['xy'][i]) for i in allowed})
            pools.append(dict(building=name, target=target,
                              allowed_scans=len(allowed),
                              allowed_positions=unique_positions,
                              excluded_old_query_positions=len(excluded)))
            for draw in range(8):
                seed = f'stage2-v1/{name}/F{target}/{draw}'
                rows = nested_rows(raw['xy'], allowed, budgets, 12, seed)
                query = rows['query_rows']
                for k in budgets:
                    support = rows['support_rows'][k]
                    events.append(dict(
                        id=f'{name}_F{target}_draw{draw}_K{k}', building=name,
                        target=target, histories=histories, K=k, R=1,
                        seed=seed, draw=draw, support_rows=support,
                        query_rows=query,
                        support_scan_ids=raw['scan_ids'][support].tolist(),
                        query_scan_ids=raw['scan_ids'][query].tolist()))
    old_groups = []
    original = old.PREPARATION / 'manifest.json'
    originals = json.loads(original.read_text())['events']
    for name in sorted(old.CONFIGS):
        rows = [e for e in originals if e['building'] == name]
        old_groups.append(dict(building=name, events=len(rows),
            target_floors=sorted({e['target'] for e in rows}),
            unique_support_sets=len({tuple(e['support_scan_ids']) for e in rows}),
            unique_query_sets=len({tuple(e['query_scan_ids']) for e in rows}),
            unique_query_scans=len({s for e in rows for s in e['query_scan_ids']})))
    files = [Path(__file__), Path(nested_rows.__code__.co_filename),
             Path(old.__file__), prior, original]
    record = dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        status='rows_prepared_not_training_ready', events=events, pools=pools,
        budgets=budgets, draws_per_target=8, query_count=12,
        old_task_coverage=old_groups, row_checks=check(),
        input_and_code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in files},
        permissions='Only four development buildings. Held-building fitting must exclude every event of that building. Official IPIN2023/2025 are excluded.',
        historical_query_exclusion='All old K-curve query positions excluded from new source S and Q pools on their target floor.',
        limits='New layouts/floors do not create independent buildings or acquisition sessions. H fields, S adaptation, numerical checks and training freeze remain pending.',
        training_updates=0, query_scored=False)
    out.mkdir(exist_ok=True)
    with (out / 'source_row_plan.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    return dict(events=len(events), target_floors=len(pools), pools=pools,
                old_task_coverage=old_groups, training_updates=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    print(json.dumps(prepare(parser.parse_args().root)))
