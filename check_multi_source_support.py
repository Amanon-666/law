"""Real source-only three-history adaptation audit; no Query read/evaluation."""
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from data import group_scans
from multi_source_adaptation import fit_source_weights
from run_operator_prior import load_history
from run import ap_stats,tensor
from run_multi_history import digest,write
from transfer import Prior,Transfer,rbf
from uji_probability_study import load_raw,DATA


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).resolve().parent;out=root/'results/multi_source_support_audit_20261026';out.mkdir(exist_ok=False)
    bank=root/'results/uji_operator_prior_v1_20261026/train_histories'
    source_manifest=root/'results/uji_partial_pooling_full_scan_v1_20261026/source_manifest.json'
    event=json.loads(source_manifest.read_text())['events'][0]
    assert event['floor']=='B0F0'
    names=['B0F1','B0F2','B0F3']
    parameters=root/'results/uji_history_bank_v1_20261026/parameters.json'
    setting=json.loads(parameters.read_text());base,obs=setting['base'],setting['obs']
    raw=load_raw('trainingData.csv',{event['floor'],*names});f=raw[event['floor']]
    lookup={int(r):i for i,r in enumerate(f['row_ids'])};sr=[lookup[r] for r in event['support_rows']]
    with np.load(bank/'B0F0/target.npz') as data:origin=data['origin']
    support=group_scans(f['rss'][sr],f['xy'][sr]-origin);sx=tensor(support['positions'])
    aps=np.flatnonzero((np.concatenate([raw[n]['rss'] for n in names])!=100).any(0)|(f['rss'][sr]!=100).any(0))
    inputs=[parameters,source_manifest,DATA/'trainingData.csv',bank/'B0F0/target.npz']
    inputs+=[bank/n/f'ap{a}.npz' for n in names for a in aps]
    files=[Path(__file__),root/'multi_source_adaptation.py',root/'history_bank.py',root/'inference.py',root/'transfer.py']
    write(out/'freeze.json',dict(created_unix=time.time(),scope=__doc__,histories=names,target='B0F0',
        support_rows=event['support_rows'],APs=aps.tolist(),code_sha256={str(p):digest(p) for p in files},inputs={str(p):digest(p) for p in inputs}))
    (out/'code').mkdir()
    for p in files:shutil.copy2(p,out/'code'/p.name)
    means=[];covariances=[];residuals=[]
    for a in aps:
        histories=[load_history(bank/n/f'ap{a}.npz',base) for n in names];mm=[];cc=[]
        for h in histories:
            prior=Prior(Transfer('identity'),h,dict(base,drift=0.));m,f=prior.features(sx)
            mm.append(m);cc.append(prior.covariance(sx,sx,f,f))
        if any(h is not None for h in histories):residual=rbf(sx,sx,base['drift'],base['length'])
        else:
            cc=[torch.zeros_like(c) for c in cc];residual=rbf(sx,sx,base['amplitude'],base['length'])
        means.append(torch.stack(mm));covariances.append(torch.stack(cc));residuals.append(residual)
    stats={k:torch.stack([ap_stats(support,int(a))[k] for a in aps]) for k in ['n','k','s1','s2']}
    started=time.perf_counter()
    weights,trials=fit_source_weights(torch.stack(means),torch.stack(covariances),torch.stack(residuals),stats,obs)
    result=dict(completed=True,history_count=3,APs=len(aps),support_positions=len(sx),support_scans=len(sr),
                weights=weights,trials=trials,seconds=time.perf_counter()-started,query_rows_used=0)
    write(out/'audit.json',result);print(json.dumps(result))
