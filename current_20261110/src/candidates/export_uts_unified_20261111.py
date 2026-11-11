"""Append complete UTS evidence to the preserved comparison-table input.

No experiments or pooling across protocols. Keep fixed B0 transfer distinct
from the older two-source-building model and the existing GP/simple references.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/uts_unified_count_20261111'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export():
    parent=ROOT/'results/unified_count_20261111/comparison_rows.json'
    table=read(parent);assert table['complete'] and len(table['rows'])==78
    for name,sha in table['inputs'].items():
        assert digest(ROOT/name)==sha
    result=read(OUT/'scores.json');assert result['complete'] and result['full_system_complete']
    assert len(result['records'])==48
    rows,inputs=list(table['rows']),dict(table['inputs'])
    inputs[str(parent.relative_to(ROOT))]=digest(parent)
    source=str((OUT/'scores.json').relative_to(ROOT));inputs[source]=digest(OUT/'scores.json')
    identity={
        'original_A02':('original_A02_raw','hurdle',[0,2]),
        'original_A02_GP':('original_A02_raw','probability_half',[0,2]),
        'retained_GP':('retained_GP','GP',[0]),
        'wknn_S':('wknn_S','wknn_S',[]),
        'wknn_HS':('wknn_HS','wknn_HS',[]),
        'unified_count':('unified_scan_field_77_v1_count','hurdle',[0]),
        'unified_count_GP':('unified_scan_field_77_v1_count','probability_half',[0]),
    }
    for name,(recipe,method,buildings) in identity.items():
        for k in ('1','3','5'):
            metric=result['summary'][name][k]['positions']
            rows.append(dict(dataset='UTS_official_all16',K=int(k),recipe=recipe,method=method,
                source_method=name,mean_m=metric['mean'],p90_m=metric['p90'],scoring_units=metric['n'],
                aggregation='Mean scan error within each unique XY on its floor, then pooled positions across all16 floors',
                source_buildings=buildings,source=source,
                description='Fixed existing official split, nested R3 support and original coordinates; no UTS-specific changes',
                evaluation_role='exposed fixed-checkpoint transfer; older A02 has different source corpus'))
    assert len(rows)==99
    with (OUT/'comparison_rows.json').open('x') as f:
        json.dump(dict(complete=True,scope='Current three-dataset complete recipes and references; not all historical project outputs',
            rows=rows,inputs=inputs,adopted=False,
            selection='No per-dataset winner stitching, no cross-protocol average; model source differences explicit'),f,indent=2)
    print(json.dumps(dict(rows=len(rows),output=str(OUT/'comparison_rows.json'))))


if __name__=='__main__':
    export()
