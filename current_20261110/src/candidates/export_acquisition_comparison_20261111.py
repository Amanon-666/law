"""Export saved complete comparisons; do not pool datasets, K or protocols.

This is table input for the current three frozen recipes, not the final paper
table. UJI uses its original hierarchy-weighted position errors; IPIN uses its
official scored-point aggregation. Raw per-point evidence remains separately
archived, including unsupported-floor coverage and negative results.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    'original_raw': ('hurdle_gp_pool_20261110', 'ipin2025_confirmation_20261110',
        'Original55 features and five B0 models; original raw reader'),
    'finite_count': ('uji54_finite_observation_20261111', 'ipin2025_finite_observation_20261111',
        'Finite-observation75 features and five B0 models; exact count-conditioned reader'),
    'acquisition_raw': ('uji54_acquisition_20261111', 'ipin2025_acquisition_20261111',
        'Original55 features and coherent source acquisition tasks; original raw reader'),
}


def export():
    rows, inputs, controls = [], {}, {}
    paired_path = ROOT/'results/uji54_acquisition_20261111/paired_K.json'
    paired = json.loads(paired_path.read_text())
    count_uji = paired['versions']['acquisition']['positions']
    assert count_uji == paired['versions']['original']['positions']
    inputs[str(paired_path.relative_to(ROOT))] = hashlib.sha256(paired_path.read_bytes()).hexdigest()
    for recipe, (uji, ipin, description) in RECIPES.items():
        for dataset, directory, filename in [('UJI_strict54',uji,'complete.json'),
                                             ('IPIN2025',ipin,'scores.json')]:
            path = ROOT/'results'/directory/filename
            data = json.loads(path.read_text())
            relative = str(path.relative_to(ROOT))
            inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            if dataset == 'UJI_strict54':
                assert data['complete'] and len(data['records']) == 54
            for k in ('1','3','5'):
                for method in ('hurdle','probability_half','GP'):
                    if dataset == 'UJI_strict54':
                        metric = data['summary'][method]['by_K'][k]
                        mean, p90 = metric['mean'], metric['position_mean_p90']
                        aggregation = 'equal building/floor/direction/position hierarchy; within-position scan error means'
                        count = count_uji
                    else:
                        metric = data['by_K'][k]['overall']['methods'][method]
                        mean, p90, count = metric['mean'], metric['p90'], metric['count']
                        aggregation = 'official covered GT points; unsupported-floor coverage retained in source result'
                    if method == 'GP':
                        key = (dataset,k)
                        if key in controls:
                            assert controls[key] == (mean,p90,count)
                            continue
                        controls[key] = (mean,p90,count)
                    rows.append(dict(dataset=dataset,K=int(k),recipe=recipe if method != 'GP' else 'retained_GP',
                        method=method,mean_m=mean,p90_m=p90,scoring_units=count,aggregation=aggregation,
                        description=description if method != 'GP' else 'Same unchanged retained-GP outputs in all three recipes',
                        source=relative,evaluation_role='exposed; no independent-test claim'))
    output = ROOT/'results/acquisition_tasks_20261111/comparison_rows.json'
    output.write_text(json.dumps(dict(complete=True,scope='Current three-recipe comparison only',
        rows=rows,inputs=inputs,adopted=False,
        selection='No per-dataset winner stitching; retain all frozen recipes and protocols separately'),indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),output=str(output))))


if __name__ == '__main__':
    export()
