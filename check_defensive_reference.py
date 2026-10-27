"""Diagnose the stopped AP164 support posterior, no query labels/model tuning."""
import json
import socket
from pathlib import Path
import numpy as np
import torch
from data import group_scans
from history_bank import BankPrior
from inference import gaussian_update
from reference_pmc import setup,sample_reference,log_proposal
from run import tensor,ap_stats
from run_multi_history import digest,write
from uji_history_bank_study import cached_history
from uji_probability_study import load_raw


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).resolve().parent;old=root/'results/uji_history_bank_v1_20261026'
    out=root/'results/uji_operator_paired_v1_20261026/B1F1_s0/ap164_defensive_check.json'
    if out.exists():raise FileExistsError(out)
    meta=next(v for v in json.loads((old/'manifest.json').read_text())['events'] if v['id']=='B1F1_s0')
    setting=json.loads((old/'parameters.json').read_text());base,obs=setting['base'],setting['obs']
    f=load_raw('trainingData.csv',{'B1F1'})['B1F1'];lookup={int(r):i for i,r in enumerate(f['row_ids'])}
    sr=[lookup[r] for r in meta['support_rows']];origin=np.array(json.loads((old/'histories/completed.json').read_text())['origin'])
    s=group_scans(f['rss'][sr],f['xy'][sr]-origin);sx=tensor(s['positions']);stats=ap_stats(s,164)
    weights=json.loads((old/'B1F1_s0/adaptation.json').read_text())['weights']
    prior=BankPrior([cached_history(old,n,164,base) for n in ['B1F0','B1F3']],weights,base)
    sm,sf=prior.features(sx);sk=prior.covariance(sx,sx,sf,sf)
    target,q,c,r,_=setup(sm,sk,stats,obs);rssloss=gaussian_update(sm,sk,stats,obs)['loss'].item();rows=[]
    for seed in [261190,261191,261192]:
        samples,lw,info=sample_reference(target,q,c,r,seed,defensive_fraction=.1)
        w=lw.exp();mean=w@samples;dc=samples-mean;cov=(dc.T*w)@dc
        # Independently reconstruct final proposal; the unnormalized bound is exact.
        u=np.linalg.solve(r,(samples.numpy()-c).T).T
        lq=log_proposal(samples,c,r,info).numpy()+np.log(np.diag(r)).sum()
        bound=float(np.max(target(u)-lq));assert bound<=np.log(10)+1e-10
        row=dict(seed=seed,ess=info['effective_samples'],log_evidence=info['log_detection_evidence']-rssloss,
                 mean=mean.tolist(),variance=cov.diag().tolist(),max_log_weight=bound)
        rows.append(row);print(json.dumps(row),flush=True)
    write(out,dict(scope=__doc__,rows=rows,code_sha256={str(p):digest(p) for p in [Path(__file__),root/'reference_pmc.py']}))
