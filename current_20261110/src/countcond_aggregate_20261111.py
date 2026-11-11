"""Summarize the fixed static and legacy54 loaded-model count ablations."""
import json
from pathlib import Path

from law_bench import legacy54
from law_bench.score import aggregate
import numpy as np

root = Path('results')
methods = ('ours_loaded_same', 'ours_loaded_countcond')
fields = ('task_id', 'floor', 'K', 'repeat', 'meta', 'status', 'scores',
          'prediction', 'query_row_indices', 'query_groups')
def stats(records, protocol):
    if protocol == 'legacy54':
        overall = legacy54.overall(records)
        by_k = legacy54.aggregate(records)
    else:
        errors = [e for r in records for e in r['scores']['position_errors']]
        overall = {'tasks': len(records), 'completed_tasks': len(records),
                   'query_positions': len(errors), 'mean': float(np.mean(errors)),
                   'p90': float(np.quantile(errors, .9))}
        by_k = aggregate(records)
    return {'overall': overall, 'by_K': by_k}

result = {}
for protocol, buildings in (('bench_uji', ('B0', 'B1', 'B2')),
                            ('legacy54', ('B1', 'B2'))):
    item = {'buildings': {}, 'all': {}}
    by_method = {m: [] for m in methods}
    for building in buildings:
        directory = root / f'{protocol}_{building}_countcond_20261111'
        original = root / f'{protocol}_{building}_{"A" if protocol == "bench_uji" else "claude"}_20261111'
        item['buildings'][building] = {}
        for method in methods:
            summary = json.loads((directory / f'{method}.json').read_text())
            records = [json.loads((directory / relative).read_text()) for relative in summary['task_records']]
            if any(r['status'] != 'completed' for r in records):
                raise RuntimeError(f'incomplete {directory} {method}')
            if method == 'ours_loaded_same':
                for relative, record in zip(summary['task_records'], records):
                    ref = json.loads((original / relative.replace(method, 'ours_A')).read_text())
                    if any(record[k] != ref[k] for k in fields):
                        raise RuntimeError(f'same differs {relative}')
            by_method[method].extend(records)
            item['buildings'][building][method] = stats(records, protocol)
    for method in methods:
        item['all'][method] = stats(by_method[method], protocol)
    result[protocol] = item

(root / 'countcond_aggregate_20261111.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
