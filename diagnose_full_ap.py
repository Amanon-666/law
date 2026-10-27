"""Read saved full-AP predictions to choose (or decline) one mechanism change.

Server only; no refitting, source selection, or alteration of primary rankings.
Coverage and local disagreement strata are explanatory, not causal estimators.
"""
import json
from pathlib import Path
import socket

import numpy as np
import torch

from full_ap_study import DATA,event_data,METHODS
from inference import observation_log_prob
from run import tensor


ROOT=Path(__file__).resolve().parent


def main():
    if socket.gethostname()!='ray-web':raise RuntimeError('diagnostics execute on server only')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    out=ROOT/'results/full_ap_v1';setting=json.loads((out/'freeze.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text());summary=json.loads((out/'summary.json').read_text())
    events=[];extremes=[];readout_rows=[]
    for meta in manifest['events']:
        d=out/meta['id'];histories,support,query=event_data(meta);aps=meta['ap_columns']
        arrays=np.load(d/'predictions.npz');xy=arrays['query_xy']
        errors={name:np.linalg.norm(arrays[name+'_xy']-xy,axis=1) for name in [*METHODS,'wknn_raw','wknn_position_mean']}
        nearest=[];local_means=[];local_rates=[]
        for h in histories:
            scans=h['scans'];distance=np.linalg.norm(xy[:,None]-scans['positions'][None],axis=-1)
            ix=distance.argmin(1);nearest.append(distance.min(1));stats=scans['stats']
            count=stats['k'][ix][:,aps]
            local_means.append(np.divide(stats['s1'][ix][:,aps],count,out=np.full_like(count,np.nan),where=count>0))
            local_rates.append(count/stats['n'][ix,None])
        distances=np.stack(nearest);means=np.stack(local_means);rates=np.stack(local_rates)
        closer_old=distances[:-1].min(0)+1e-6<distances[-1]
        nearby=distances<=2*setting['base']['length']
        visible=(rates>=.5)&nearby[:,:,None];missing=(rates==0)&nearby[:,:,None]
        disagreement=(visible.any(0)&missing.any(0)).sum(1)
        strata=[]
        masks={'older_history_closer':closer_old,'latest_not_farther':~closer_old,
               'local_detection_disagreement':disagreement>0,'no_local_detection_disagreement':disagreement==0}
        for label,mask in masks.items():
            if mask.any():strata.append(dict(stratum=label,queries=int(mask.sum()),errors={k:float(v[mask].mean()) for k,v in errors.items()}))
        for before,after in [('single','multi'),('single','pooled'),('pooled','multi')]:
            gain=errors[before]-errors[after]
            for tag,i in [('improved',int(gain.argmax())),('degraded',int(gain.argmin()))]:
                extremes.append(dict(event=meta['id'],comparison=f'{before}->{after}',case=tag,query_index=i,
                    xy=xy[i].tolist(),before_xy=arrays[before+'_xy'][i].tolist(),after_xy=arrays[after+'_xy'][i].tolist(),
                    before_error=float(errors[before][i]),after_error=float(errors[after][i]),
                    history_nearest_distances=distances[:,i].tolist(),locally_disagreeing_APs=int(disagreement[i]),
                    detected_APs=int((arrays['query_rss'][i]!=100).sum())))
        all_diffs={name:[] for name in METHODS};scan_diffs={name:np.zeros(len(xy)) for name in METHODS}
        per_ap=[]
        for j,ap in enumerate(aps):
            with np.load(d/f'ap{ap}.npz') as pred:
                for name in METHODS:
                    g=query['group'];mu=tensor(pred[name+'_query_mean'])[g];var=tensor(pred[name+'_query_variance'])[g]
                    gauss=-observation_log_prob(tensor(query['rss'][:,ap]),mu,var,setting['obs']).numpy()
                    diff=gauss-pred[name+'_scan_nll'];all_diffs[name].extend(diff.tolist());scan_diffs[name]+=diff
                    per_ap.append(dict(method=name,ap=ap,mean_abs_logp_difference=float(np.abs(diff).mean()),
                                       max_abs_logp_difference=float(np.abs(diff).max()),mean_signed_difference=float(diff.mean())))
        for name in METHODS:
            values=np.array(all_diffs[name]);records=[r for r in per_ap if r['method']==name]
            row=dict(event=meta['id'],method=name,mean_abs_cell_logp_difference=float(np.abs(values).mean()),
                p95_abs_cell_logp_difference=float(np.quantile(np.abs(values),.95)),max_abs_cell_logp_difference=float(np.abs(values).max()),
                mean_signed_scan_logp_difference=float(scan_diffs[name].mean()),
                worst_APs=sorted(records,key=lambda x:x['max_abs_logp_difference'],reverse=True)[:3])
            readout_rows.append(row)
        events.append(dict(event=meta['id'],ap_count=len(aps),query_count=len(xy),strata=strata,
             gains={f'{a}->{b}':dict(mean=float((errors[a]-errors[b]).mean()),improved_queries=int((errors[b]<errors[a]).sum()))
                    for a,b in [('single','multi'),('single','pooled'),('pooled','multi')]}))
    result=dict(scope='post-hoc source-development diagnosis; no parameter modification',events=events,
                examples=extremes,Gaussian_vs_importance_readout=readout_rows,
                caveats=['nearest historical positions differ; local disagreement is not isolated temporal drift',
                         'true-position logp differences do not alone establish localization-readout causality',
                         'all query coordinates used here are diagnostic only, never deployment input'])
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(completed=True,event_gains=[dict(event=e['event'],gains=e['gains']) for e in events],
                         readout=[{k:v for k,v in r.items() if k!='worst_APs'} for r in readout_rows]),allow_nan=False))


if __name__=='__main__':main()
