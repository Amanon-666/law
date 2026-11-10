"""Summarize saved benchmark events only. No model/data loading or inference."""
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/panyushuo/projects/panyushuo/claude_fp_wt/current_20261110/src/results')


def stats(records):
    done = [r for r in records if r['status'] == 'completed']
    errors = np.concatenate([r['scores']['position_errors'] for r in done]) if done else np.array([])
    return dict(completed=len(done), failed=len(records)-len(done), positions=len(errors),
        mean=float(errors.mean()) if len(errors) else None,
        p50=float(np.percentile(errors, 50)) if len(errors) else None,
        p90=float(np.percentile(errors, 90)) if len(errors) else None,
        task_seconds_mean=float(np.mean([r['seconds'] for r in done])) if done else None)


summary = {}
for stage in ('readerablation', 'kcurve'):
    combined = {}; complete_records = {}; completed_folds = []; folds = {}; unavailable = []
    for building in ('B0', 'B1', 'B2'):
        output = ROOT / f'bench_uji_{building}_{stage}_20261111'
        inventory = output / 'task_inventory.json'
        if not inventory.exists():
            continue
        missing = json.loads(inventory.read_text())['unavailable']
        unavailable.extend(missing)
        is_complete = (output / 'summary.json').exists()
        if is_complete:
            completed_folds.append(building)
        fold = {'complete': is_complete, 'protocol_unavailable': missing, 'methods': {}}
        for method_dir in sorted((output / 'events').glob('*')):
            records = [json.loads(p.read_text()) for p in method_dir.rglob('*.json')]
            combined.setdefault(method_dir.name, []).extend(records)
            if is_complete:
                complete_records.setdefault(method_dir.name, []).extend(records)
            fold['methods'][method_dir.name] = dict(overall=stats(records), by_K={str(k):stats([r for r in records if r['K']==k]) for k in sorted({r['K'] for r in records})}, failures=[r for r in records if r['status']!='completed'])
        folds[building] = fold
    summary[stage] = dict(completed_folds_only=dict(buildings=completed_folds, methods={m:dict(overall=stats(rs), by_K={str(k):stats([r for r in rs if r['K']==k]) for k in sorted({r['K'] for r in rs})}) for m,rs in complete_records.items()}), folds=folds, protocol_unavailable=unavailable, methods={m:dict(overall=stats(rs), by_K={str(k):stats([r for r in rs if r['K']==k]) for k in sorted({r['K'] for r in rs})}) for m,rs in combined.items()})
state = ROOT / 'bench_round3_execution_20261111.json'
summary['execution'] = json.loads(state.read_text())
(ROOT / 'bench_round3_summary_20261111.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({stage: summary[stage]['methods'] for stage in ('readerablation','kcurve')}, indent=2))
