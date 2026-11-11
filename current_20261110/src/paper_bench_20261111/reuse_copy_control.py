"""Pair existing copied-field predictions with the completed final paper ReFi.

No inference, fitting, or task resampling. The copied baseline is independent
of the number of learned-field ensemble members; its source-calibrated constant
variance and original scan reader are retained from the completed experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(values):
    ordered = sorted(values)
    p = .9*(len(ordered)-1)
    lo = int(p)
    p90 = ordered[lo] + (p-lo)*(ordered[min(lo+1, len(ordered)-1)]-ordered[lo])
    return dict(mean=mean(values), p90=p90, position_repeat_budget_count=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-results', type=Path, required=True)
    parser.add_argument('--archived-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pairs, bindings = [], []
    for building in ['B0', 'B1', 'B2']:
        paper = args.paper_results/f'paper_uji_{building}_20261111'
        archived = args.archived_results/f'bench_uji_{building}_extra_20261111'
        pf, af = (json.loads((p/'freeze.json').read_text()) for p in [paper, archived])
        for key in ['target_data', 'source_data']:
            assert pf[key] == af[key], (building, key)
        for key in ['protocol.py', 'data.py']:
            assert pf['code'][key] == af['code'][key], (building, key)
        for key in ['dataset', 'building', 'root', 'source_dataset', 'source_root',
                    'source_buildings', 'floors', 'budgets', 'seed', 'R', 'query_positions']:
            assert pf['configuration'][key] == af['configuration'][key], (building, key)
        assert set(pf['configuration']['repeats']) <= set(af['configuration']['repeats'])
        assert af['code']['ours.py'] == '07733101709d234416dd7d95759094884ce9316c641b4dba3332c83a49a4a777'
        assert af['code']['methods_claude.py'] == '660048e66c9e2cd4ee6ee25193bd6c41aaefd6aa68109fa05b722949afd0dfd1'
        bindings.append(dict(building=building,
            paper_freeze=dict(path=str(paper/'freeze.json'), sha256=sha(paper/'freeze.json')),
            copied_freeze=dict(path=str(archived/'freeze.json'), sha256=sha(archived/'freeze.json'))))
        for path in sorted((paper/'events/refi').rglob('*.json')):
            r = json.loads(path.read_text())
            old = archived/'events/ours_naive_noM2'/(r['task_id']+'.json')
            c = json.loads(old.read_text())
            assert r['status'] == c['status'] == 'completed'
            for key in ['task_id', 'floor', 'K', 'repeat', 'meta', 'query_row_indices', 'query_groups']:
                assert r[key] == c[key], (path, key)
            assert len(r['scores']['position_errors']) == len(c['scores']['position_errors'])
            pairs.append(dict(task_id=r['task_id'], building=building, floor=r['floor'],
                K=r['K'], repeat=r['repeat'],
                refi_errors=r['scores']['position_errors'], copied_errors=c['scores']['position_errors'],
                refi=dict(path=str(path), sha256=sha(path)), copied=dict(path=str(old), sha256=sha(old))))
    assert len(pairs) == 117

    def summarize(rows):
        r = [x for row in rows for x in row['refi_errors']]
        c = [x for row in rows for x in row['copied_errors']]
        return dict(tasks=len(rows), refi=metrics(r), copied=metrics(c),
            mean_reduction_fraction=1-mean(r)/mean(c),
            positions_improved=sum(x<y for x,y in zip(r,c)),
            positions_worse=sum(x>y for x,y in zip(r,c)),
            positions_equal=sum(x==y for x,y in zip(r,c)))

    groups = {b: {str(k): summarize([r for r in pairs if r['building']==b and r['K']==k])
                    for k in [1,3,5]} for b in ['B0','B1','B2']}
    summary = dict(status='completed_existing_prediction_pairing', new_fits=0, new_predictions=0,
        comparison='Final original five-model ReFi vs copied nearest-side history field with source-calibrated variance',
        interpretation='Complete learned field-construction mechanism; shared H/S candidates, empirical S atoms and original scan reader',
        protocol='static-resample-v2; UJI leave-one-building; K1/3/5; r0-2; seed20261111',
        metric='Pooled position errors within each task; repeats and K retained, not independent people or buildings',
        evaluation_status='exposed_development; reader selected on UJI B0',
        code_identity=dict(copied_ours_sha256='07733101709d234416dd7d95759094884ce9316c641b4dba3332c83a49a4a777',
            copied_ours_git_commit='5b74abcfe3e8603b0528028c6be207fa2f7fe3c0'),
        bindings=bindings, overall=summarize(pairs), by_building_K=groups,
        by_building={b:summarize([r for r in pairs if r['building']==b]) for b in groups})
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in [('summary.json',summary),('paired_tasks.json',pairs)]:
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary['overall'],ensure_ascii=False))


if __name__ == '__main__':
    main()
