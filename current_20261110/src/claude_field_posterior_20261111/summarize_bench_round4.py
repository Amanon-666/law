"""Summarize completed round-four stages without mixing evaluation protocols."""
import json
from pathlib import Path
from law_bench import score, legacy54

results = Path('results')
summary = {}
runtime = {}
for stage in ('smoke', 'dl', 'hfloors', 'legacy54_dl'):
    grouped = {}
    buildings = ('B1',) if stage == 'smoke' else (('B1', 'B2') if stage == 'legacy54_dl' else ('B0', 'B1', 'B2'))
    completed = []
    for building in buildings:
        name = f'legacy54_{building}_dl_20261111' if stage == 'legacy54_dl' else ('bench_round4_smoke_20261111' if stage == 'smoke' else f'bench_uji_{building}_{stage}_20261111')
        directory = results/name
        if not (directory/'summary.json').exists():
            continue
        completed.append(building)
        runtime[name] = {'source_fit': {p.stem: json.loads(p.read_text()) for p in (directory/'source_models').glob('*.json')}, 'selected_hp': json.loads((directory/'selected_hp.json').read_text()) if (directory/'selected_hp.json').exists() else None, 'prediction_seconds': {m.name: sum(json.loads(p.read_text()).get('seconds', 0) for p in m.rglob('*.json')) for m in (directory/'events').iterdir()}}
        for method in (directory/'events').iterdir():
            records = [json.loads(p.read_text()) for p in method.rglob('*.json')]
            grouped.setdefault(method.name, []).extend(records)
        if stage == 'hfloors':
            originals = results/f'bench_uji_{building}_A_20261111/events/ours_A'
            comparison = [json.loads(p.read_text()) for p in originals.rglob('*.json')]
            candidates = [json.loads(p.read_text()) for p in (directory/'events/ours_loaded_h1').rglob('*.json')]
            old = {r['task_id']: r for r in comparison}
            for record in candidates:
                base = old[record['task_id']]
                assert base['query_row_indices'] == record['query_row_indices']
                assert base['query_groups'] == record['query_groups']
            grouped.setdefault('ours_A_all_H', []).extend([old[r['task_id']] for r in candidates])
    stage_summary = {'completed_buildings': completed, 'expected_buildings': list(buildings), 'methods': {}}
    for method, records in grouped.items():
        if stage == 'legacy54_dl':
            overall, by_k = legacy54.overall(records), legacy54.aggregate(records)
        else:
            by_k = score.aggregate(records)
            overall = score.aggregate([dict(r, K=0) for r in records])['0']
        stage_summary['methods'][method] = {'overall': overall, 'by_K': by_k}
    stage_summary['by_building'] = {}
    for building in completed:
        stage_summary['by_building'][building] = {}
        for method, records in grouped.items():
            selected = [r for r in records if '/'+building+'/' in r['task_id']]
            if stage == 'legacy54_dl':
                overall, by_k = legacy54.overall(selected), legacy54.aggregate(selected)
            else:
                overall = score.aggregate([dict(r, K=0) for r in selected])['0']
                by_k = score.aggregate(selected)
            stage_summary['by_building'][building][method] = {'overall': overall, 'by_K': by_k}
    summary[stage] = stage_summary
import torch
summary['runtime'] = {'torch_version': torch.__version__, 'runs': runtime}
(results/'bench_round4_summary_20261111.json').write_text(json.dumps(summary, indent=2)+'\n')
