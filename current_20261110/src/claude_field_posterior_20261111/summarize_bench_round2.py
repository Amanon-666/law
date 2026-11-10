"""Read saved benchmark receipts only; pool position errors as score.aggregate does."""
import json
import sys
from pathlib import Path
import numpy as np

roots = [Path(p) for p in sys.argv[1:]]
summaries = [json.loads((p / 'summary.json').read_text()) for p in roots]
print('|method|mean|p50|p90|K1 mean|K3 mean|K5 mean|fit seconds (sum)|task seconds (mean)|completed|failed/unavailable|')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for name in summaries[0]:
    records, fits = [], []
    for root, summary in zip(roots, summaries):
        item = summary[name]
        records.extend(json.loads((root / f).read_text()) for f in item['task_records'])
        records.extend(item['unavailable'])
        fits.append(item['source_fit'].get('seconds', 0))
    done = [r for r in records if r['status'] == 'completed']
    errors = [e for r in done for e in r['scores']['position_errors']]
    values = [float(np.mean(errors)), *map(float, np.quantile(errors, [.5, .9]))] if errors else [None]*3
    for k in (1, 3, 5):
        es = [e for r in done if r['K'] == k for e in r['scores']['position_errors']]
        values.append(float(np.mean(es)) if es else None)
    seconds = [r['seconds'] for r in records if 'seconds' in r]
    values += [sum(fits), float(np.mean(seconds)) if seconds else None, len(done), len(records)-len(done)]
    print('|' + '|'.join([name, *('—' if v is None else str(v) for v in values)]) + '|')
    failures = sorted({r.get('error', r['status']) for r in records if r['status'] != 'completed'})
    for failure in failures:
        print(f'{name} failure: {failure}')
