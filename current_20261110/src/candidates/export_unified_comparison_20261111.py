"""Append the completed unified recipe to preserved comparison table inputs.

Only aggregate saved errors; do not fit, predict, select models or combine
different dataset metrics. Paired-K counts are descriptive, not independent
replications: IPIN points belong to two recorded trajectories.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/unified_scan_field_20261111'


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def paired(data):
    by_k = {}
    for k in (1,3,5):
        by_k[k] = {(p['log_id'],p['gt_row']):p for p in data['points']
                   if p['K'] == k and p['covered']}
    assert by_k[1].keys() == by_k[3].keys() == by_k[5].keys()
    result = {}
    for method in ('hurdle','probability_half','GP'):
        transitions = {}
        for lo,hi in ((1,3),(3,5),(1,5)):
            differences = [by_k[hi][key]['errors'][method]-p['errors'][method]
                           for key,p in by_k[lo].items()]
            transitions[f'{lo}_to_{hi}'] = dict(
                improved=sum(d < -1e-8 for d in differences),
                worsened=sum(d > 1e-8 for d in differences),
                tied=sum(abs(d) <= 1e-8 for d in differences),
                improved_over5m=sum(d < -5 for d in differences),
                worsened_over5m=sum(d > 5 for d in differences))
        result[method] = dict(transitions=transitions,
            over20m_by_K={str(k):sum(p['errors'][method] > 20 for p in points.values())
                         for k,points in by_k.items()})
    return dict(points=len(by_k[1]),methods=result)


def export():
    previous = ROOT/'results/acquisition_tasks_20261111/comparison_rows.json'
    table = read(previous)
    assert table['complete'] and len(table['rows']) == 42
    for name,sha in table['inputs'].items():
        assert binding(ROOT/name)['sha256'] == sha
    rows,inputs = list(table['rows']),dict(table['inputs'])
    inputs[str(previous.relative_to(ROOT))] = binding(previous)['sha256']
    positions = read(ROOT/'results/uji54_unified_scan_20261111/paired_K.json')
    count_uji = positions['versions']['unified']['positions']
    assert count_uji == positions['versions']['original']['positions'] == 216
    for dataset,directory,filename in (
            ('UJI_strict54','uji54_unified_scan_20261111','complete.json'),
            ('IPIN2025','ipin2025_unified_scan_20261111','scores.json')):
        path = ROOT/'results'/directory/filename
        data = read(path)
        assert data['complete']
        if dataset == 'UJI_strict54':
            assert len(data['records']) == 54
        source = str(path.relative_to(ROOT))
        inputs[source] = binding(path)['sha256']
        for k in ('1','3','5'):
            for method in ('hurdle','probability_half','GP'):
                if dataset == 'UJI_strict54':
                    metric = data['summary'][method]['by_K'][k]
                    mean,p90,count = metric['mean'],metric['position_mean_p90'],count_uji
                    aggregation = 'equal building/floor/direction/position hierarchy; within-position scan error means'
                else:
                    metric = data['by_K'][k]['overall']['methods'][method]
                    mean,p90,count = metric['mean'],metric['p90'],metric['count']
                    aggregation = 'official covered GT points; unsupported-floor coverage retained in source result'
                if method == 'GP':
                    control, = [r for r in rows if r['dataset']==dataset and r['K']==int(k) and r['method']=='GP']
                    assert (control['mean_m'],control['p90_m'],control['scoring_units']) == (mean,p90,count)
                    continue
                rows.append(dict(dataset=dataset,K=int(k),recipe='unified_scan_field_77_v1',
                    method=method,mean_m=mean,p90_m=p90,scoring_units=count,aggregation=aggregation,
                    description='Same five B0 models and raw reader on unique H/S union; learned fields at all candidates; no empirical-S append',
                    source=source,evaluation_role='exposed; no independent-test claim'))
    assert len(rows) == 54
    with (OUT/'comparison_rows.json').open('x') as f:
        json.dump(dict(complete=True,scope='Four complete frozen recipes; not the final full-project table',
            rows=rows,inputs=inputs,adopted=False,
            selection='No dataset-specific winner stitching; same unified checkpoint and recipe in both datasets'),f,indent=2)
    versions,paired_inputs = {},{}
    for name,directory in (
            ('original','ipin2025_confirmation_20261110'),
            ('finite_count','ipin2025_finite_observation_20261111'),
            ('acquisition','ipin2025_acquisition_20261111'),
            ('unified','ipin2025_unified_scan_20261111')):
        path = ROOT/'results'/directory/'scores.json'
        versions[name] = paired(read(path))
        paired_inputs[name] = binding(path)
    with (OUT/'paired_K_ipin.json').open('x') as f:
        json.dump(dict(complete=True,kind='saved_error_reaggregation',inputs=paired_inputs,versions=versions,
            definition='Same log and GT-row pairs across K; high K minus low K,1e-8m tie tolerance; descriptive counts from two trajectories'),f,indent=2)
    print(json.dumps(dict(rows=len(rows),output=str(OUT/'comparison_rows.json'))))


if __name__ == '__main__':
    export()
