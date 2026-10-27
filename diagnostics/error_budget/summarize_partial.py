"""Post-hoc stopped-run accounting; never starts or resumes inference."""
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/uji_error_budget_reference_v1_20261026'


def main():
    assert socket.gethostname()=='ray-web'
    pid=int((OUT/'pid.txt').read_text())
    assert not Path(f'/proc/{pid}').exists(),'Runner must actually be stopped'
    prep=json.loads((ROOT/'results/uji_error_budget_preparation_v1_20261026/manifest.json').read_text())['events']
    original=json.loads((ROOT/'results/uji_history_bank_v1_20261026/manifest.json').read_text())['events']
    base=json.loads((ROOT/'results/uji_history_bank_v1_20261026/parameters.json').read_text())['base']
    status=[];rows=[];info=[];coverage=[];invalid=[];checked=0
    for e in prep:
        dest=OUT/e['event'];complete=(dest/'completed.json').exists()
        fields=list(dest.glob('fold*/ap*.npz'));reference=list(dest.glob('fold*/reference_ap*.npz'))
        state='complete' if complete else ('partial' if fields or reference else 'not_started')
        status.append(dict(event=e['event'],state=state,completed_AP_fields=len(fields),fitted_reference_APs=len(reference),
            expected_AP_fields=len(e['ap_columns'])*len(e['folds']),folds={str(f['fold']):len(list((dest/f"fold{f['fold']}").glob('ap*.npz'))) for f in e['folds']}))
        for p in fields+reference+list(dest.glob('predictions.npz')):
            try:
                with np.load(p) as arrays:
                    for key in arrays.files:arrays[key]
                checked+=1
            except Exception as exc:
                invalid.append(dict(path=str(p),error=str(exc)))
        info += [json.loads(p.read_text()) for p in dest.glob('fold*/ap*.json') if not p.name.endswith('_failure.json')]
        if not complete:continue
        rows.extend(json.loads((dest/'completed.json').read_text())['records'])
        meta=next(m for m in original if m['id']==e['event']);ni=len(meta['internal_rows'])
        with np.load(dest/'predictions.npz') as p:
            for split,sl in [('internal',slice(0,ni)),('official',slice(ni,None))]:
                xy=p['query_xy'][sl];distance=p['reference_distance'][sl]
                _,ix=np.unique(xy,axis=0,return_index=True);d=distance[ix]
                coverage.append(dict(event=e['event'],floor=e['floor'],split=split,positions=len(ix),
                    mean=float(d.mean()),median=float(np.median(d)),maximum=float(d.max()),
                    fraction_gt_base_length=float((d>base['length']).mean())))
    aggregate=[]
    for floor in sorted({r['floor'] for r in rows}):
        for split in ['internal','official']:
            for stratum in sorted({r['stratum'] for r in rows}):
                subset=[r for r in rows if r['floor']==floor and r['split']==split and r['stratum']==stratum]
                if not subset:continue
                arms={a:{k:float(np.mean([r['arms'][a][k] for r in subset])) for k in ['mde','p90','weighted_p90','true_scan_nll','detected_RSS_RMSE']} for a in ['a','b','d']}
                aa,bb,dd=[arms[a]['mde'] for a in ['a','b','d']]
                aggregate.append(dict(floor=floor,split=split,stratum=stratum,events=len(subset),arms=arms,
                    a_minus_b=aa-bb,a_minus_d=aa-dd,b_closure_ratio=(aa-bb)/(aa-dd) if aa-dd>.5 else None))
    record=dict(created_unix=time.time(),stage='stopped_early_posthoc_geometry_mismatch',original_plan_events=len(prep),
        termination=dict(pid=pid,signal='SIGTERM',identity_checked='Exact Python executable, run_reference.py run arguments, and project cwd; /proc PID absent after exit',
            reason='Leader stopped at three completed B1F1 events and partial B1F2_s0: macro-block gaps test spatial extrapolation, not sufficiently calibrated reference. Stop follows coordinate-only coverage diagnosis; not performance tuning.'),
        original_freeze_preserved=True,original_runner_sha256=hashlib.sha256((ROOT/'diagnostics/error_budget/run_reference.py').read_bytes()).hexdigest(),
        summary_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status=status,completed_events=sum(s['state']=='complete' for s in status),partial_events=sum(s['state']=='partial' for s in status),
        records=rows,aggregate=aggregate,base_length=base['length'],completed_query_coverage=coverage,
        all_planned_query_coverage=json.loads((OUT/'coverage.json').read_text())['summary'],
        minimum_ESS=min(r[a]['ess'] for r in info for a in ['b','d']),completed_AP_fields=len(info),
        cache_integrity=dict(readable_npz_files=checked,invalid_files=invalid),missing_arms=['c','E_naive'],
        limits='Only B1F1 three events complete. No B1F2 localization result. Extra-label cross-block reference, not K-only, full map, theoretical lower bound, or mean-bottleneck rejection. Negative a-d yields no closure percentage. Any replacement folding needs a separate post-hoc revised freeze; no second prediction round started.')
    (OUT/'summary.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:record[k] for k in ['stage','status','minimum_ESS','cache_integrity']},indent=2))
    assert not invalid,'Interrupted NPZ files retained and marked invalid; never reuse them'


if __name__=='__main__':main()
