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


def summarize(root, reference, original=None):
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
    label_efficiency=[]
    for building in sorted({g['building'] for g in groups}):
        curve=[g for g in groups if g['building']==building]
        anchor=next(g for g in curve if g['K']==5)
        for ref in REFERENCES:
            threshold=anchor['methods'][ref]['mean_mde']
            if threshold is None:continue
            reached={}
            for method in ('ScaML',ref):
                ks=[g['K'] for g in curve if g['methods'][method]['mean_mde'] is not None
                    and g['methods'][method]['mean_mde']<=threshold]
                reached[method]=min(ks) if ks else None
            complete=all(g['methods']['ScaML']['n']==3 for g in curve)
            label_efficiency.append(dict(building=building,reference=ref,threshold_at_reference_K5=threshold,
                minimum_measured_K=reached,scaml_curve_complete=complete,
                saved_positions=reached[ref]-reached['ScaML'] if complete and reached['ScaML'] is not None else None,
                interpretation='Measured grid only; no interpolation or theoretical sample-complexity claim. Null is not zero savings.'))
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
    log_lines=[line for path in root.glob('*.log') for line in path.read_text().splitlines()]
    skipped_restarts=sum('This restart will be skipped.' in line for line in log_lines)
    result=dict(finished=not pending and all((root/f'{b}_finished.json').exists() for b in {e['building'] for e in events}),
        expected=len(events),successful=sum('ScaML' in e['results'] for e in events),pending=pending,failures=failures,
        groups=groups,events=events,label_efficiency=label_efficiency,
        optimizer=dict(attempts=len(attempts),skipped_restart_log_entries=skipped_restarts,
                       requested_attempts_for_returned_models=sum(s['models'] for s in selection.values())*(read(root/'freeze.json')['restarts']+1),
                       source_fit_seconds_sum=sum(r['seconds'] for r in sources),
                       max_source_cache_fit_seconds=max((r['seconds'] for r in sources),default=None),successful_attempts=sum(a.get('success',False) for a in attempts),
                       failed_source_caches=sum(not r['success'] for r in sources),selection=selection),
        fitting=read(root/'freeze.json')['fitting'],
        evidence=dict(manifest_sha256=digest(root/'manifest.json'),freeze_sha256=digest(root/'freeze.json'),
                      reference_summary_sha256=digest(reference)),
        interpretation='Development protocol adaptation of the official ScaML core. Three paired draws required for each mean/gain; missing reference results remain gaps. Different coordinate units are never pooled. Not a single-factor weight constraint ablation or independent generalization proof.')
    if original is not None:
        assert read(original/'manifest.json')==manifest
        paired=[]
        for meta in manifest['events']:
            a_path=original/'events'/meta['id']/'result.json'
            b_path=root/'events'/meta['id']/'result.json'
            if not a_path.exists() or not b_path.exists():continue
            a,b=read(a_path),read(b_path)
            row=dict(id=meta['id'],original_success=a['success'],corrected_success=b['success'])
            if a['success'] and b['success']:
                a_source=read(original/'sources'/a['source_key']/'result.json')
                b_source=read(root/'sources'/b['source_key']/'result.json')
                row.update(mde_corrected_minus_original=b['metrics']['mde']-a['metrics']['mde'],
                    original_selected_optimizer_call_indices=[x['matching_attempts'] for x in a['selected']],
                    corrected_selected_optimizer_call_indices=[x['matching_attempts'] for x in b['selected']],
                    source_parameter_bytes_changed=a_source['parameter_hashes']!=b_source['parameter_hashes'],
                    original_seconds=a['seconds'],corrected_seconds=b['seconds'])
            paired.append(row)
        result['selection_comparison']=dict(original=str(original),events=paired,
            caveat='Indices refer to recorded optimizer calls, not scheduled restarts skipped before optimization. Source parameter byte changes can be numerically tiny. Event time includes source fitting on first cache use. Neither equal nor different target indices alone attribute the MDE difference.')
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('finished','expected','successful','pending','failures','optimizer')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('reference',type=Path);p.add_argument('--original',type=Path)
    a=p.parse_args();summarize(a.root,a.reference,a.original)
