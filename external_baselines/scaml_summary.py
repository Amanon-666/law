"""Paired post-processing of frozen ScaML development outputs, without refitting."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import numpy as np

REFERENCES=('H_base','WKNN_HS','GP_S','H_support')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(root, reference):
    manifest=read(root/'manifest.json')
    old={e['id']:e for e in read(reference)['events']}
    events=[];pending=[];failures=[]
    for meta in manifest['events']:
        baseline=old[meta['id']]
        assert baseline['query_scan_ids']==meta['query_scan_ids']
        assert baseline['unit']==meta['coordinate_unit']
        methods={name:baseline['results'][name] for name in REFERENCES if name in baseline['results']}
        path=root/'events'/meta['id']/'result.json'
        if not path.exists():pending.append(meta['id'])
        else:
            result=read(path)
            if result['success']:
                assert result['query_scan_ids']==meta['query_scan_ids']
                assert all(s['optimizer_success'] for s in result['selected'])
                with np.load(path.parent/'predictions.npz',allow_pickle=False) as p:
                    assert p['query_scan_ids'].tolist()==meta['query_scan_ids']
                    errors=np.linalg.norm(p['predictions']-p['query_xy'],axis=1)
                    # Frozen R=1 and one scan per query location.
                    np.testing.assert_allclose(errors.mean(),result['metrics']['mde'],rtol=1e-12,atol=1e-12)
                methods['ScaML']=result['metrics']
            else:failures.append(dict(id=meta['id'],error=result['error']))
        events.append(dict(id=meta['id'],building=meta['building'],K=meta['K'],seed=meta['seed'],
                           unit=meta['coordinate_unit'],results=methods))
    groups=[]
    for b in sorted({e['building'] for e in events}):
        for k in manifest['K']:
            group=[e for e in events if e['building']==b and e['K']==k]
            assert len(group)==3
            methods={}
            for method in ('ScaML',*REFERENCES):
                values=[e['results'][method]['mde'] for e in group if method in e['results']]
                methods[method]=dict(n=len(values),mean_mde=mean(values) if len(values)==3 else None)
            pairs={}
            for ref in REFERENCES:
                paired=[(e['results']['ScaML']['mde'],e['results'][ref]['mde']) for e in group
                        if 'ScaML' in e['results'] and ref in e['results']]
                pairs[ref]=dict(n=len(paired),mean_gain=mean(y-x for x,y in paired) if len(paired)==3 else None)
            groups.append(dict(building=b,K=k,unit=group[0]['unit'],methods=methods,paired_gain=pairs))
    sources=[read(p) for p in (root/'sources').glob('*/result.json')]
    completed_events=[read(p) for p in (root/'events').glob('*/result.json')]
    attempts=[a for r in sources+completed_events for a in r['attempts']]
    def flatten_selected(value):
        if isinstance(value,dict):yield value
        else:
            for item in value:yield from flatten_selected(item)
    selection={}
    for label, records in [('source',sources),('target',completed_events)]:
        selected=[x for record in records for x in flatten_selected(record.get('selected',[]))]
        gaps=[x['selected_optimizer_objective']-x['best_successful_objective'] for x in selected
              if x.get('selected_optimizer_objective') is not None and x.get('best_successful_objective') is not None]
        selection[label]=dict(models=len(selected),failed_selected=sum(not x['optimizer_success'] for x in selected),
            objective_gap_over_1e_7=sum(g>1e-7 for g in gaps),max_objective_gap=max(gaps) if gaps else None)
    result=dict(finished=not pending and all((root/f'{b}_finished.json').exists() for b in {e['building'] for e in events}),
        expected=len(events),successful=sum('ScaML' in e['results'] for e in events),pending=pending,failures=failures,
        groups=groups,events=events,
        optimizer=dict(attempts=len(attempts),successful_attempts=sum(a.get('success',False) for a in attempts),
                       failed_source_caches=sum(not r['success'] for r in sources),selection=selection),
        fitting=read(root/'freeze.json')['fitting'],
        evidence=dict(manifest_sha256=digest(root/'manifest.json'),freeze_sha256=digest(root/'freeze.json'),
                      reference_summary_sha256=digest(reference)),
        interpretation='Development protocol adaptation of the official ScaML core. Three paired draws required for each mean/gain; missing reference results remain gaps. Different coordinate units are never pooled. Not a single-factor weight constraint ablation or independent generalization proof.')
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('finished','expected','successful','pending','failures','optimizer')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('reference',type=Path)
    a=p.parse_args();summarize(a.root,a.reference)
