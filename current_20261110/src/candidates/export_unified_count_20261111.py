"""Preserve complete recipe comparisons and paired-K error distributions.

Saved-error aggregation only, run on the server. UJI position means and IPIN
official trajectory points remain separate; pairs are not independent trials.
"""
from collections import defaultdict
import json
import numpy as np

from candidates.export_unified_comparison_20261111 import ROOT,read,binding,paired

OUT = ROOT/'results/unified_count_20261111'


def uji_pairs(directory):
    data = read(directory/'complete.json');points = []
    for record in data['records']:
        path = ROOT/record['scored_path']
        assert binding(path)['sha256'] == record['scored_sha256']
        with np.load(path) as saved:
            groups = defaultdict(list)
            for i,xy in enumerate(saved['truth']):
                groups[tuple(xy)].append(i)
            for xy,indices in groups.items():
                points.append(dict(K=record['K'],log_id=record['event'].rsplit('_K',1)[0],
                    gt_row=xy,covered=True,
                    errors={m:float(saved[m][indices].mean()) for m in ('hurdle','GP','probability_half')}))
    return paired(dict(points=points))


def export():
    parent = ROOT/'results/unified_scan_field_20261111/comparison_rows.json'
    table = read(parent);assert len(table['rows']) == 54
    for name,sha in table['inputs'].items():
        assert binding(ROOT/name)['sha256'] == sha
    rows,inputs = list(table['rows']),dict(table['inputs'])
    inputs[str(parent.relative_to(ROOT))] = binding(parent)['sha256']
    pairs = {}
    for recipe,uji,ipin in (
            ('original_count','uji54_count_conditioned_20261111','ipin2025_count_conditioned_20261111'),
            ('unified_scan_field_77_v1_count','uji54_unified_count_20261111','ipin2025_unified_count_20261111')):
        pairs[recipe] = {}
        for dataset,directory,filename in (('UJI_strict54',uji,'complete.json'),('IPIN2025',ipin,'scores.json')):
            path = ROOT/'results'/directory/filename;data = read(path);assert data['complete']
            if dataset == 'UJI_strict54':
                assert len(data['records']) == 54
            source = str(path.relative_to(ROOT));inputs[source] = binding(path)['sha256']
            pairs[recipe][dataset] = uji_pairs(path.parent) if dataset == 'UJI_strict54' else paired(data)
            for k in ('1','3','5'):
                for method in ('hurdle','probability_half','GP'):
                    if dataset == 'UJI_strict54':
                        metric = data['summary'][method]['by_K'][k]
                        mean,p90,count = metric['mean'],metric['position_mean_p90'],216
                        aggregation = 'equal building/floor/direction/position hierarchy; within-position scan error means'
                    else:
                        metric = data['by_K'][k]['overall']['methods'][method]
                        mean,p90,count = metric['mean'],metric['p90'],metric['count']
                        aggregation = 'official covered GT points; original coverage retained'
                    if method == 'GP':
                        control, = [r for r in rows if r['dataset']==dataset and r['K']==int(k) and r['method']=='GP']
                        assert (control['mean_m'],control['p90_m'],control['scoring_units']) == (mean,p90,count)
                        continue
                    rows.append(dict(dataset=dataset,K=int(k),recipe=recipe,method=method,
                        mean_m=mean,p90_m=p90,scoring_units=count,aggregation=aggregation,
                        description='Fixed five-model field plus the identical exact count-conditioned reader in both datasets',
                        source=source,evaluation_role='exposed; no independent-test claim'))
    assert len(rows) == 78
    with (OUT/'comparison_rows.json').open('x') as f:
        json.dump(dict(complete=True,scope='Six complete recipes; not the final full-project table',
            rows=rows,inputs=inputs,adopted=False,selection='No dataset-specific winner stitching'),f,indent=2)
    with (OUT/'paired_K.json').open('x') as f:
        json.dump(dict(complete=True,kind='saved_error_reaggregation',versions=pairs,
            definition='Same event/position or log/GT-row pairs; descriptive improvements and deteriorations, not independent trials'),f,indent=2)
    print(json.dumps(dict(rows=len(rows),output=str(OUT))))


if __name__ == '__main__':
    export()
