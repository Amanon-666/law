"""Frozen-checkpoint paired development evaluation with the repaired sampler.

Reuses the already trained operator; does not train or change its parameters.
Both controls are recomputed. Historical random draws were not fully seeded,
so cached 11.7867 is context, never an equality target or the paired control.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from data import group_scans
from full_ap_study import metrics
from history_bank import BankPrior, fit_two_source_weight
from inference import observation_log_prob
from operator_prior import OperatorBankPrior, OperatorNet
from run_operator_prior import load_history
from run import ap_stats, tensor
from run_multi_history import digest, write
from source_weight_mixture import field_from_prior
from transfer import Prior, Transfer, rbf
from uji_probability_study import load_raw, DATA

ROOT=Path(__file__).resolve().parent
TRAIN=ROOT/'results/uji_operator_prior_v1_20261026'
OLD=ROOT/'results/uji_history_bank_v1_20261026'
OUT=ROOT/'results/uji_operator_paired_v2_20261026'
SOURCES=['B1F0','B1F3']


def prepare():
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),*[ROOT/n for n in ['operator_prior.py','run_operator_prior.py',
        'history_bank.py','source_weight_mixture.py','reference_pmc.py','inference.py',
        'transfer.py','data.py','full_ap_study.py','run.py','run_multi_history.py','uji_probability_study.py']]]
    inputs=[TRAIN/'operator.pt',TRAIN/'training.json',TRAIN/'PROTOCOL.md',OLD/'parameters.json',OLD/'manifest.json',
            OLD/'histories/completed.json',DATA/'trainingData.csv',DATA/'validationData.csv']
    inputs+=sorted((OLD/'histories').glob('B*/ap*.npz'))+sorted(OLD.glob('B*/adaptation.json'))
    write(OUT/'freeze.json',dict(created_unix=time.time(),purpose=__doc__,
        prior_hypothesis='Cross-building learned history-profile mean/variance improves paired K5 localization versus mature continuous shared EB at identical H/S/Q',
        training_scope='42 pairs from B0/B2; target marginal prior likelihood, not S/Q conditional meta-training; only J=2 was trained',
        information_difference='Operator training additionally uses B2; no claim of equal source training information with the old B0 base fit',
        methods=['adapted_bank','operator_bank'],checkpoint=digest(TRAIN/'operator.pt'),
        seed_rule='261026 + 10000*event_seed + AP index; same local RNG seed in both methods',
        criterion='Both internal and official six-event mean lower, each at least 4/6 events improved; otherwise partial/unsupported, no B1 retuning',
        exceptions='M-PMC ESS<1000 stops the event for numerical diagnosis, not silently accepted performance',
        numerical_revision='After v1 stopped at baseline AP164, add fixed 0.1 original RSS-Gaussian mass to final proposals for BOTH methods and ALL APs; no model parameters changed or localization scores inspected',
        resource='CPU only, two threads; GPUs currently occupied; no downloads',
        code_sha256={str(p):digest(p) for p in files},inputs={str(p):digest(p) for p in inputs}))
    (OUT/'code').mkdir()
    for p in files: shutil.copy2(p,OUT/'code'/p.name)
    print(json.dumps(dict(stage='prepared',checkpoint=digest(TRAIN/'operator.pt'))),flush=True)


def event(event_id, baseline_directory=None):
    freeze=json.loads((OUT/'freeze.json').read_text())
    for path,value in freeze['code_sha256'].items(): assert digest(Path(path))==value,path
    assert digest(TRAIN/'operator.pt')==freeze['checkpoint']
    dest=OUT/event_id;dest.mkdir(exist_ok=True)
    if (dest/'completed.json').exists(): return
    config=json.loads((OLD/'parameters.json').read_text());base,obs=config['base'],config['obs']
    meta=next(e for e in json.loads((OLD/'manifest.json').read_text())['events'] if e['id']==event_id)
    floor=load_raw('trainingData.csv',{meta['floor']})[meta['floor']]
    official=load_raw('validationData.csv',{meta['floor']})[meta['floor']]
    def indices(f,rows):
        lookup={int(r):i for i,r in enumerate(f['row_ids'])}
        return np.array([lookup[r] for r in rows])
    sr=indices(floor,meta['support_rows']);qi=indices(floor,meta['internal_rows']);qo=indices(official,meta['official_rows'])
    origin=np.array(json.loads((OLD/'histories/completed.json').read_text())['origin'])
    support=group_scans(floor['rss'][sr],floor['xy'][sr]-origin);sx=tensor(support['positions'])
    assert len(sx)==5 and len(sr)==15
    query=group_scans(np.concatenate([floor['rss'][qi],official['rss'][qo]]),np.concatenate([floor['xy'][qi],official['xy'][qo]])-origin)
    assert not (query['xy'][:,None]==support['positions'][None]).all(-1).any()
    bounds=np.array(meta['grid_bounds'])-origin
    axes=[np.linspace(bounds[0,d],bounds[1,d],int(np.ceil((bounds[1,d]-bounds[0,d])/2))+1) for d in range(2)]
    grid=tensor(np.stack(np.meshgrid(*axes),-1).reshape(-1,2))
    net=OperatorNet().double();net.load_state_dict(torch.load(TRAIN/'operator.pt',weights_only=True));net.eval()
    dz=[int(n.split('F')[1])-int(meta['floor'].split('F')[1]) for n in SOURCES]
    histories={a:[load_history(OLD/'histories'/n/f'ap{a}.npz',base) for n in SOURCES] for a in meta['ap_columns']}
    old_w=json.loads((OLD/event_id/'adaptation.json').read_text())['weights']
    if not (dest/'adaptation.json').exists():
        means,covs,residuals=[],[],[]
        with torch.no_grad():
            for a,hs in histories.items():
                op=OperatorBankPrior(hs,[.5,.5],base,net,dz)
                g,s=op.operator(sx) if op.active else (sx.new_zeros(len(sx)),sx.new_zeros(len(sx)))
                mm,cc=[],[]
                for h in hs:
                    p=Prior(Transfer('identity'),h,dict(base,drift=0.));m,f=p.features(sx)
                    mm.append(m+g);cc.append(p.covariance(sx,sx,f,f))
                if any(h is not None for h in hs):
                    residual=rbf(sx,sx,base['drift'],base['length'])+s[:,None]*s[None]*rbf(sx,sx,1.,base['length'])
                else:
                    cc=[torch.zeros_like(c) for c in cc];residual=rbf(sx,sx,base['amplitude'],base['length'])
                means.append(torch.stack(mm));covs.append(torch.stack(cc));residuals.append(residual)
        stats={k:torch.stack([ap_stats(support,a)[k] for a in histories]) for k in ['n','k','s1','s2']}
        weights,trials=fit_two_source_weight(torch.stack(means),torch.stack(covs),torch.stack(residuals),stats,obs)
        write(dest/'adaptation.json',dict(weights=weights,trials=trials,baseline_weights=old_w,APs=len(histories)))
    weights=json.loads((dest/'adaptation.json').read_text())['weights']
    for i,(a,hs) in enumerate(histories.items()):
        path=dest/f'ap{a}.npz'
        if path.exists(): continue
        priors=dict(adapted_bank=BankPrior(hs,old_w,base),operator_bank=OperatorBankPrior(hs,weights,base,net,dz))
        if all(h is None for h in hs): priors={m:BankPrior([],[],base) for m in priors}
        arrays={};diagnostics={}
        for method,prior in priors.items():
            if method=='adapted_bank' and baseline_directory is not None:
                with np.load(baseline_directory/event_id/f'ap{a}.npz') as previous:
                    arrays[method+'_mean']=previous[method+'_mean']
                    arrays[method+'_variance']=previous[method+'_variance']
                diagnostics[method]=json.loads((baseline_directory/event_id/f'ap{a}.json').read_text())[method]
                continue
            result=field_from_prior(prior,obs,sx,ap_stats(support,a),grid,261026+10000*meta['seed']+a,defensive_fraction=.1)
            diagnostics[method]=dict(ess=result['effective_samples'],log_evidence=result['log_evidence'])
            if result['effective_samples']<1000:
                write(dest/f'ap{a}_numerical_failure.json',diagnostics)
                raise RuntimeError(f'Low ESS: {event_id} AP{a} {method} {result["effective_samples"]}')
            arrays[method+'_mean']=result['mean'].numpy();arrays[method+'_variance']=result['variance'].numpy()
        np.savez_compressed(path,**arrays);write(dest/f'ap{a}.json',diagnostics)
        if i%20==0: print(json.dumps(dict(event=event_id,stage='fields',completed=i+1,total=len(histories))),flush=True)
    # Stream bounded query batches on CPU; no unavailable GPU memory is assumed.
    methods=['operator_bank'] if baseline_directory is not None else ['adapted_bank','operator_bank']
    predictions={m:[] for m in methods}
    with torch.no_grad():
        for start in range(0,len(query['rss']),64):
            scans=tensor(query['rss'][start:start+64])
            for method in predictions:
                scores=scans.new_zeros((len(scans),len(grid)))
                for a in histories:
                    with np.load(dest/f'ap{a}.npz') as data:
                        m,v=tensor(data[method+'_mean']),tensor(data[method+'_variance'])
                    scores+=observation_log_prob(scans[:,a,None],m[None],v[None],obs)
                predictions[method].append((scores.softmax(1)@grid).numpy())
    predictions={m:np.concatenate(v) for m,v in predictions.items()};rows=[]
    if baseline_directory is not None:
        with np.load(baseline_directory/event_id/'predictions.npz') as previous:
            np.testing.assert_array_equal(previous['query_xy'],query['xy'])
            np.testing.assert_array_equal(previous['origin'],origin)
            predictions['adapted_bank']=previous['adapted_bank']
    for split,sl in [('internal',slice(0,len(qi))),('official',slice(len(qi),None))]:
        xy=query['xy'][sl];_,group,count=np.unique(xy,axis=0,return_inverse=True,return_counts=True)
        for method,pred in predictions.items():
            rows.append(dict(method=method,split=split,**metrics(pred[sl],dict(xy=xy,scan_weights=1/(len(count)*count[group.ravel()])))))
    np.savez_compressed(dest/'predictions.npz',**predictions,query_xy=query['xy'],origin=origin)
    write(dest/'completed.json',dict(event=event_id,results=rows,weights=weights))
    print(json.dumps(dict(stage='completed',event=event_id,results=rows)),flush=True)


if __name__=='__main__':
    if socket.gethostname()!='ray-web': raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','event']);parser.add_argument('--event');args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:event(args.event)
