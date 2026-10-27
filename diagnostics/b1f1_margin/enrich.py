"""Aggregate existing T1 outputs; approximate spatial match is descriptive only."""
import csv
import hashlib
import json
from pathlib import Path
import socket
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(sys.argv[1]).resolve();sys.path.insert(0,str(ROOT))
from uji_probability_study import load_raw
from diagnose import position_weights,selected


def main():
    assert socket.gethostname()=='ray-web'
    out=ROOT/'results/b1f1_official_diag_20261026'
    summary=json.loads((out/'summary.json').read_text())
    detection=json.loads((out/'detection.json').read_text())['records']
    events=json.loads((ROOT/'results/uji_history_bank_v1_20261026/manifest.json').read_text())['events']
    training=load_raw('trainingData.csv',{'B1F1'})['B1F1'];official=load_raw('validationData.csv',{'B1F1'})['B1F1']
    coverage=[];subsets=[];matches=[]
    for event in events:
        if event['floor']!='B1F1':continue
        parts=[selected(training,event['internal_rows']),selected(official,event['official_rows'])]
        positions=[np.unique(p[1],axis=0) for p in parts]
        distance,nearest=cKDTree(positions[0]).query(positions[1])
        common=sum(np.any(np.all(positions[0]==p,axis=1)) for p in positions[1])
        coverage.append(dict(event=event['id'],internal_positions=len(positions[0]),official_positions=len(positions[1]),
            internal_bbox=[positions[0].min(0).tolist(),positions[0].max(0).tolist()],
            official_bbox=[positions[1].min(0).tolist(),positions[1].max(0).tolist()],exact_shared_positions=int(common),
            official_to_internal_nearest_distance_quantiles=np.quantile(distance,[0,.25,.5,.75,1]).tolist(),
            official_positions_within2_of_internal=int((distance<=2).sum())))
        for record in [r for r in detection if r['event']==event['id']]:
            a=record['ap_column'];peak=np.array(record['peak_raw'])
            near=(distance<=2)&(np.linalg.norm(positions[1]-peak,axis=1)<=10)&(np.linalg.norm(positions[0][nearest]-peak,axis=1)<=10)
            if not near.any():continue
            rates=[]
            for k in np.flatnonzero(near):
                ri=parts[0][0][np.all(parts[0][1]==positions[0][nearest[k]],axis=1),a]
                ro=parts[1][0][np.all(parts[1][1]==positions[1][k],axis=1),a]
                rates.append([np.mean(ri!=100),np.mean(ro!=100)])
            matches.append(dict(event=event['id'],ap_column=a,group=record['group'],position_pairs=int(near.sum()),
                internal_rate=float(np.mean(np.array(rates)[:,0])),official_rate=float(np.mean(np.array(rates)[:,1])),
                mean_pair_distance=float(distance[near].mean())))
        with (out/(event['id']+'_queries.csv')).open() as f: rows=list(csv.DictReader(f))
        with np.load(out/(event['id']+'_ap_margins.npz')) as f:
            delta_ap=f['operator_bank']-f['adapted_bank'];detected=f['detected']
            for split in ['internal','official']:
                ix=np.array([i for i,r in enumerate(rows) if r['split']==split]);w=position_weights(np.array([[float(rows[i]['x']),float(rows[i]['y'])] for i in ix]))
                def delta(key):return np.array([float(rows[i]['operator_bank_'+key])-float(rows[i]['adapted_bank_'+key]) for i in ix])
                flags=dict(all=np.ones(len(ix),bool),fixed_margin_worse=delta('fixed_operator_far_margin')<0,
                    posterior_mean_error_worse=delta('mde')>0,
                    truth_better_fixed_margin_worse=(delta('true_loglik')>0)&(delta('fixed_operator_far_margin')<0))
                for label,mask in flags.items():
                    fraction=float(w@mask)
                    if not fraction:continue
                    ws=w*mask/fraction
                    part=delta_ap[ix];d=detected[ix]
                    missing_ap=ws@(part*~d);detected_ap=ws@(part*d)
                    total=missing_ap+detected_ap
                    worst=np.argsort(total)[:5]
                    subsets.append(dict(event=event['id'],split=split,subset=label,position_weight_fraction=fraction,
                        true_loglik_change=float(ws@delta('true_loglik')),fixed_margin_change=float(ws@delta('fixed_operator_far_margin')),
                        missing_change=float(missing_ap.sum()),detected_change=float(detected_ap.sum()),
                        posterior_mean_error_change=float(ws@delta('mde')),
                        strongest_negative_AP_changes=[dict(ap_column=int(f['ap_columns'][j]),missing=float(missing_ap[j]),detected=float(detected_ap[j])) for j in worst]))
    matching={}
    for group in ['positive_correction_ge5','other_H_observed']:
        rr=[r for r in matches if r['group']==group]
        matching[group]=dict(ap_event_pairs=len(rr),position_pairs=sum(r['position_pairs'] for r in rr),
            internal_rate_equal_AP_event=float(np.mean([r['internal_rate'] for r in rr])) if rr else None,
            official_rate_equal_AP_event=float(np.mean([r['official_rate'] for r in rr])) if rr else None)
    summary['additional_diagnostics']=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        coverage=coverage,query_subsets=subsets,near_matched_detection_summary=matching,
        matching_definition='Official position matched to nearest internal Query position if distance <=2 original units; both within 10 of same H peak. Equal official positions per AP, then equal AP-event pairs. This is approximate matching, not identical location or devices. AP group fixed from cached field correction, never called target-owned.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    (out/'near_matched_detection.json').write_text(json.dumps(matches,indent=2,allow_nan=False))
    print(json.dumps(dict(coverage=coverage,matching=matching,subsets=[r for r in subsets if r['split']=='official']),indent=2))


if __name__=='__main__':main()
