"""Completed six-event summary and declared post-hoc diagnostics, no tuning.

Uses Query labels only to measure errors and probabilities near their locations.
Detected RSS is compared to E[R | detected], not unconditionally to latent mean.
This does not identify a device effect or establish an independent-building gain.
"""
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from inference import observation_log_prob
from run import tensor
from run_multi_history import digest,write
from uji_probability_study import load_raw

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/uji_operator_paired_v2_20261026'
OLD=ROOT/'results/uji_history_bank_v1_20261026'
METHODS=['adapted_bank','operator_bank']


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    manifest=json.loads((OLD/'manifest.json').read_text())['events']
    missing=[e['id'] for e in manifest if not (OUT/e['id']/'completed.json').exists()]
    if missing:raise RuntimeError(f'Incomplete evaluation: {missing}')
    if (OUT/'summary.json').exists():raise FileExistsError(OUT/'summary.json')
    obs=json.loads((OLD/'parameters.json').read_text())['obs']
    tr=load_raw('trainingData.csv',{'B1F1','B1F2'});va=load_raw('validationData.csv',{'B1F1','B1F2'})
    diagnostics=[];results=[]
    for event in manifest:
        dest=OUT/event['id'];completed=json.loads((dest/'completed.json').read_text())
        results.extend([dict(event=event['id'],floor=event['floor'],**r) for r in completed['results']])
        with np.load(dest/'predictions.npz') as p:
            xy=p['query_xy'];origin=p['origin'];predictions={m:p[m] for m in METHODS}
        def rows(f,ids):
            lookup={int(v):i for i,v in enumerate(f['row_ids'])};ix=[lookup[i] for i in ids]
            return f['rss'][ix],f['xy'][ix]-origin
        internal,ixy=rows(tr[event['floor']],event['internal_rows'])
        official,oxy=rows(va[event['floor']],event['official_rows'])
        np.testing.assert_array_equal(xy,np.concatenate([ixy,oxy]));rss=tensor(np.concatenate([internal,official])[:,event['ap_columns']])
        bounds=np.array(event['grid_bounds'])-origin
        axes=[np.linspace(bounds[0,d],bounds[1,d],int(np.ceil((bounds[1,d]-bounds[0,d])/2))+1) for d in range(2)]
        grid=np.stack(np.meshgrid(*axes),-1).reshape(-1,2)
        distance=cKDTree(np.array(event['support_positions'])-origin).query(xy)[0]
        truth_index=cKDTree(grid).query(xy)[1]
        for method in METHODS:
            means=[];variances=[];ess=[]
            for a in event['ap_columns']:
                with np.load(dest/f'ap{a}.npz') as data:
                    means.append(data[method+'_mean'][truth_index]);variances.append(data[method+'_variance'][truth_index])
                ess.append(json.loads((dest/f'ap{a}.json').read_text())[method]['ess'])
            mean=tensor(np.stack(means,1));variance=tensor(np.stack(variances,1));det=rss!=100
            scale=torch.sqrt(variance+obs['detect_scale']**2);z=(mean-obs['beta'])/scale
            probability=obs['alpha']*torch.special.ndtr(z)
            ratio=torch.exp(-.5*z.square()-.5*np.log(2*np.pi)-torch.special.log_ndtr(z))
            conditional_mean=mean+variance/scale*ratio
            error=rss-conditional_mean;lp=observation_log_prob(rss,mean,variance,obs)
            for split,sl in [('internal',slice(0,len(internal))),('official',slice(len(internal),None))]:
                for region in ['all','near','far']:
                    local=np.arange(len(xy))[sl]
                    if region!='all':local=local[(distance[local]>30) if region=='far' else (distance[local]<=30)]
                    if not len(local):continue
                    _,group,count=np.unique(xy[local],axis=0,return_inverse=True,return_counts=True)
                    w=tensor(1/(len(count)*count[group.ravel()]))
                    detected_weight=w[:,None]*det[local];denom=detected_weight.sum()
                    record=dict(event=event['id'],floor=event['floor'],method=method,split=split,region=region,
                        positions=len(count),scans=len(local),
                        mde=float(w@tensor(np.linalg.norm(predictions[method][local]-xy[local],axis=1))),
                        nll_per_ap=float(-(w*lp[local].mean(1)).sum()),
                        detection_brier=float((w*(probability[local]-det[local].double()).square().mean(1)).sum()),
                        detected_conditional_bias=float((detected_weight*error[local]).sum()/denom) if denom else None,
                        detected_conditional_rmse=float(torch.sqrt((detected_weight*error[local].square()).sum()/denom)) if denom else None,
                        mean_latent_variance=float((w*variance[local].mean(1)).sum()),min_ess=min(ess))
                    if region=='all':
                        expected=next(r['mde'] for r in completed['results'] if r['method']==method and r['split']==split)
                        np.testing.assert_allclose(record['mde'],expected,atol=1e-10,rtol=0)
                    diagnostics.append(record)
    summary={}
    for split in ['internal','official']:
        values={m:[next(r['mde'] for r in results if r['event']==e['id'] and r['split']==split and r['method']==m) for e in manifest] for m in METHODS}
        delta=np.array(values['operator_bank'])-values['adapted_bank']
        summary[split]=dict(per_event=values,mean={m:float(np.mean(v)) for m,v in values.items()},
            mean_change=float(delta.mean()),improved_events=int((delta<0).sum()),
            by_floor={floor:float(np.mean([d for d,e in zip(delta,manifest) if e['floor']==floor])) for floor in ['B1F1','B1F2']})
    outcome=all(s['mean_change']<0 and s['improved_events']>=4 for s in summary.values())
    write(OUT/'posthoc_diagnostics.json',dict(scope=__doc__,created_unix=time.time(),script_sha256=digest(Path(__file__)),
        near_far_boundary_m=30,records=diagnostics))
    write(OUT/'summary.json',dict(summary=summary,development_criterion_met=outcome,
        limits='Two previously viewed floors in one building; more source data and mean/variance both changed. No causal device attribution or independent-domain confirmation.',
        completed_sha256={e['id']:digest(OUT/e['id']/'completed.json') for e in manifest}))
    print(json.dumps(dict(summary=summary,development_criterion_met=outcome)))


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    main()
