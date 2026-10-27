"""Full-scan source prior: reuse target fields and unchanged control outputs.

Query batches limit GPU memory. No target fitting or new GP inference.
"""
import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

from full_ap_study import metrics
from full_scan_pooling_source import OUT,SPARSE
from inference import observation_log_prob
from partial_source_pooling import conditional_pooling,partial_ap_log_prob
from run import tensor
from run_multi_history import digest,write
from run_source_weight_mixture import OUT as FIELDS,load_raw
from source_weight_mixture import mixture_log_prob


def event(event_id):
    assert (OUT/'source_numerical_check/completed.json').exists()
    freeze=json.loads((OUT/'freeze.json').read_text())
    for group in ['code_sha256','inputs']:
        for path,expected in freeze[group].items():assert digest(Path(path))==expected,path
    dest=OUT/event_id;dest.mkdir(exist_ok=True)
    if (dest/'completed.json').exists():return
    fit=json.loads((OUT/'source_fit.json').read_text());lam=fit['unpooled_probability']
    original=FIELDS/event_id;controls=SPARSE/event_id
    meta=next(e for e in json.loads((OUT/'manifest.json').read_text())['events'] if e['id']==event_id)
    field_paths=[original/f'ap{ap}.npz' for ap in meta['ap_columns']]
    write(dest/'evaluation_freeze.json',dict(created_unix=time.time(),source_fit_sha256=digest(OUT/'source_fit.json'),
        source_numerical_check_sha256=digest(OUT/'source_numerical_check/completed.json'),
        code_sha256=digest(Path(__file__)),fields_sha256={str(p):digest(p) for p in field_paths},
        reused_controls_sha256=digest(controls/'predictions.npz'),unpooled_probability=lam,
        query_batch=16,new_target_parameter_updates=0,new_GP_inference=0))
    f=load_raw('trainingData.csv',{meta['floor']})[meta['floor']];q=load_raw('validationData.csv',{meta['floor']})[meta['floor']]
    fi={int(r):i for i,r in enumerate(f['row_ids'])};qi={int(r):i for i,r in enumerate(q['row_ids'])}
    ir=np.array([fi[r] for r in meta['internal_rows']]);orr=np.array([qi[r] for r in meta['official_rows']])
    rss=tensor(np.concatenate((f['rss'][ir],q['rss'][orr])));truth=np.concatenate((f['xy'][ir],q['xy'][orr]))
    with np.load(original/'predictions.npz') as a:grid=tensor(a['grid']);np.testing.assert_array_equal(a['query_xy'],truth)
    with np.load(controls/'predictions.npz') as a:
        predictions={m:a[m].copy() for m in ['fixed_half','target_eb']};np.testing.assert_array_equal(a['query_xy'],truth)
    evidence=[];fields=[]
    for p in field_paths:
        with np.load(p) as a:evidence.append(a['log_evidence']);fields.append((tensor(a['mean']),tensor(a['variance'])))
    z=tensor(np.array(evidence));mass=tensor(freeze['prior_mass']);device=torch.device('cuda:0')
    post=[tuple(v.to(device) for v in conditional_pooling(z,mass,x)) for x in [lam,.5]]
    obs=json.loads((FIELDS/'parameters.json').read_text())['obs'];g=grid.to(device);center=g.mean(0);parts=[]
    with torch.no_grad():
        for start in range(0,len(rss),16):
            end=min(start+16,len(rss));scores=[torch.zeros((len(mass),end-start,len(grid)),device=device) for _ in range(2 if start==0 else 1)]
            for i,ap in enumerate(meta['ap_columns']):
                m,v=(t.to(device) for t in fields[i]);r=rss[start:end,ap].to(device)
                lp=observation_log_prob(r[None,:,None],m[:,None,:],v[:,None,:],obs)
                for j,score in enumerate(scores):
                    _,shared,local,independent,_=post[j];score+=partial_ap_log_prob(lp,shared[i],local[i],independent[i])
            for j,score in enumerate(scores):
                lp=mixture_log_prob(score,post[j][0]);xy=(torch.softmax(lp,1)@(g-center)+center).cpu().numpy()
                if j==0:parts.append(xy)
                else:np.testing.assert_allclose(xy,predictions['fixed_half'][:end],rtol=0,atol=1e-7)
    predictions['source_learned']=np.concatenate(parts);rows=[]
    for split,sl in [('internal',slice(0,len(ir))),('official',slice(len(ir),None))]:
        xy=truth[sl];_,group,count=np.unique(xy,axis=0,return_inverse=True,return_counts=True);sw=1/(len(count)*count[group.ravel()])
        for method,pred in predictions.items():rows.append(dict(method=method,split=split,**metrics(pred[sl],dict(xy=xy,scan_weights=sw))))
    np.savez_compressed(dest/'predictions.npz',**predictions,query_xy=truth,internal_count=len(ir))
    values=json.loads((controls/'adaptation.json').read_text())['values'];values['source_learned']=lam
    write(dest/'completed.json',dict(completed=True,event=event_id,values=values,results=rows,
        unchanged_controls_reused=True,batched_readout_replay_passed=True,predictions_sha256=digest(dest/'predictions.npz')))
    print(json.dumps(dict(event=event_id,unpooled_probability=lam,official=[r for r in rows if r['split']=='official'])),flush=True)


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Server only')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--event',required=True);event(p.parse_args().event)
