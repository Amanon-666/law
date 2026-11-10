"""Summarize saved JSON only; position pooling matches law_bench.score.aggregate."""
import json
import sys
from pathlib import Path
import numpy as np

roots = [Path(p) for p in sys.argv[1:]]
summaries = [json.loads((p / 'summary.json').read_text()) for p in roots]
print('|method|mean|p50|p90|K1 mean|K3 mean|K5 mean|source_fit.seconds|mean task seconds|completed/tasks|')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
for name in summaries[0]:
    records = []
    fit = 0.0
    for root, summary in zip(roots, summaries):
        method = summary[name]
        records.extend(json.loads((root / p).read_text()) for p in method['task_records'])
        records.extend(method['unavailable'])
        fit += method['source_fit']['seconds']
    done = [r for r in records if r['status'] == 'completed']
    errors = [e for r in done for e in r['scores']['position_errors']]
    values = [np.mean(errors), *np.quantile(errors, [.5, .9])]
    values += [np.mean([e for r in done if r['K'] == k for e in r['scores']['position_errors']]) for k in (1, 3, 5)]
    values += [fit, np.mean([r['seconds'] for r in records if 'seconds' in r])]
    options = json.loads((roots[0] / 'freeze.json').read_text())['configuration']['method_options'].get(name, {})
    label = name + ('_map' if options.get('readout') == 'map' else '')
    print('|' + '|'.join([label, *map(str, values), f'{len(done)}/{len(records)}']) + '|')
