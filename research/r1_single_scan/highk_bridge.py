"""Locked R1 F0 draw0 official K5/10/20 Gaussian-VI backend bridge.
Source H, support IDs, three frozen arms and all GPS are identical to score.py.
This is one diagnostic shard; no learning or B1-based hyperparameter selection.
"""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch

from data import group_scans
from inference import infer,observation_log_prob,predict_from_posterior
from multi_history import single_history
from research.r1_single_scan import score as R
from run import ap_stats,tensor

KS=(5,10,20)
ARMS=('gp','seven','nonlinear')
INFER_STEPS=500

@torch.no_grad()
def vi_field(prior,sx,stats,grid):
    qm,qf=prior.features(grid)
    sm,sf=prior.features(sx)
    sk=prior.covariance(sx,sx,sf,sf)
    update=infer(sm,sk,stats,R.O,steps=INFER_STEPS)
    posterior=dict(mean=update['mean'],cov=update['cov'])
    prediction=predict_from_posterior(qm,prior.diagonal(grid,qf),
             prior.covariance(grid,sx,qf,sf),sm,sk,posterior)
    mu,var=prediction['mean'],prediction['variance']
    if not (torch.isfinite(mu).all() and torch.isfinite(var).all()):
        raise RuntimeError('Nonfinite VI predictive moments')
    if float(var.min()) < -1e-7:
        raise RuntimeError(f'Negative VI predictive variance {float(var.min())}')
    return mu,var.clamp_min(0.),float(update['min_diagonal'])

def output_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def run(args):
    assert (args.floor,args.regime,args.draw)==(0,'official',0),'Only pre-registered F0 official draw0'
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    start=time.perf_counter()
    raw=R.read_raw(args.data)
    case=R.validate_layout(R.EVIDENCE,raw,args.regime,args.floor,args.draw)
    assert len(case['query']['ids'])==30
    beta,net=R.models()
    hist=case['history'];floors=list(hist)
    assert floors==[1,2,3]
    origin=np.concatenate([hist[h]['xy'] for h in floors]).mean(0)
    histories={h:group_scans(hist[h]['rss'],hist[h]['xy']-origin) for h in floors}
    h_ap=sorted(set(int(a) for h in floors for a in np.flatnonzero((hist[h]['rss']!=100).any(0))))
    args.out.mkdir(parents=True,exist_ok=True)
    print('HISTORY_BEGIN',len(h_ap),floors,[len(hist[h]['ids']) for h in floors],flush=True)
    start_h=time.perf_counter();banks={}
    for i,a in enumerate(h_ap):
        banks[a]=[single_history({'scans':histories[h]},a,R.B,R.O,steps=640) for h in floors]
        if i%25==0:print('HISTORY_AP',i,len(h_ap),round(time.perf_counter()-start_h,1),flush=True)
    history_s=time.perf_counter()-start_h
    summary=dict(scope='Pre-registered original B1 F0 official draw0, K5/10/20, VI-only; not 400-event R1',
                 regime=args.regime,building=1,floor=args.floor,draw=args.draw,KS=list(KS),R=1,
                 historical_floor_ids=floors,three_history=True,
                 training_hash=R.TRAIN_HASH,manifest_hash=R.MANIFEST_HASH,
                 device_hash=R.PHONE_HASH,
                 affine_source_sha256=R.AFFINE_HASH,
                 model_source_sha256=R.NONLINEAR_HASH,
                 query_rows_sha256=case['query_rows_sha256'],
                 support_order_sha256=case['support_order_sha256'],
                 query_count=len(case['query']['ids']),
                 history_fit_seconds=history_s,
                 inference='inference.infer full-covariance Gaussian VI, steps=500, all K',
                 events=[],failures=[])
    output_json(args.out/'highk_manifest.json',summary)
    for k in KS:
        tick=time.perf_counter()
        entry=dict(K=k,completed_modes=[],failures=[],backend='VI500',R=1)
        try:
            sub={name:v[:k] for name,v in case['support'].items()}
            s=group_scans(sub['rss'],sub['xy']-origin)
            assert len(s['positions'])==k and np.all(s['stats']['n']==1)
            seen=(s['rss']!=100).any(0)
            for h in floors:seen|=(hist[h]['rss']!=100).any(0)
            aps=np.flatnonzero(seen).tolist()
            for a in aps:
                if a not in banks:banks[a]=[None]*3
            sx=tensor(s['positions'])
            print('WEIGHT_BEGIN',k,'APs',len(aps),flush=True)
            w,trials=R.weight(banks,s,aps)
            print('WEIGHT_END',k,w,'secs',round(time.perf_counter()-tick,1),flush=True)
            full=np.vstack([histories[h]['positions'] for h in floors]+[s['positions']])
            axes=[np.linspace(v.min(),v.max(),int(np.ceil(np.ptp(v)/2))+1) for v in full.T]
            grid=tensor(np.stack(np.meshgrid(*axes),axis=-1).reshape(-1,2))
            q=case['query']['rss'][:,aps]
            entry.update(ap_count=len(aps),grid=len(grid),weights=w,support_ids=sub['ids'].tolist(),
                         support_hash=R.digest(np.asarray(sub['ids'],dtype='<i8').tobytes()),
                         weight_fit_valid=sum(x['stationary'] for x in trials),
                         weight_fit_total=len(trials),
                         wknn_archived=R.cached_wknn(case,args.regime,args.floor,args.draw,k))
            arrays=dict(query_ids=case['query']['ids'],query_xy=case['query']['xy'])
            for mode in ARMS:
                st=time.perf_counter()
                try:
                    log=np.zeros((len(q),len(grid)),dtype=np.float64)
                    min_diagonal=float('inf')
                    for j,a in enumerate(aps):
                        p=R.CorrectedBank(banks[a],w,[h-args.floor for h in floors],mode,beta,net)
                        mu,var,d=vi_field(p,sx,ap_stats(s,a),grid)
                        min_diagonal=min(min_diagonal,d)
                        missing=observation_log_prob(tensor(100.),mu,var,R.O).numpy()
                        log+=missing
                        nz=np.flatnonzero(q[:,j]!=100)
                        if len(nz):
                            v,inv=np.unique(q[nz,j],return_inverse=True)
                            extra=observation_log_prob(tensor(v)[:,None],mu,var,R.O).numpy()-missing
                            log[nz]+=extra[inv]
                        if j%40==0:print('VI_AP',k,mode,j,len(aps),round(time.perf_counter()-st,1),flush=True)
                    pred=(tensor(log).softmax(1)@grid).numpy()+origin
                    assert np.isfinite(pred).all()
                    entry.setdefault('metrics',{})[mode]=R.metric(pred,case['query']['xy'])
                    entry.setdefault('vi_min_diagonal',{})[mode]=min_diagonal
                    entry.setdefault('mode_seconds',{})[mode]=time.perf_counter()-st
                    entry['completed_modes'].append(mode)
                    arrays['K'+str(k)+'_'+mode]=pred
                    print('EVENT_RESULT',k,mode,entry['metrics'][mode],flush=True)
                except Exception as ex:
                    msg=dict(K=k,mode=mode,exception=repr(ex),traceback=traceback.format_exc()[-5000:])
                    entry['failures'].append(msg);summary['failures'].append(msg)
                    print('MODE_FAIL',k,mode,msg['exception'],flush=True)
                # Persist the successful earlier modes even if a later mode fails.
                np.savez_compressed(args.out/f'official_F0_0_K{k}.npz',**arrays)
                entry['elapsed_seconds']=time.perf_counter()-tick
                output_json(args.out/f'official_F0_0_K{k}.json',entry)
        except Exception as ex:
            msg=dict(K=k,exception=repr(ex),traceback=traceback.format_exc()[-5000:])
            entry['failures'].append(msg);summary['failures'].append(msg)
            print('K_FAIL',k,msg['exception'],flush=True)
        entry['elapsed_seconds']=time.perf_counter()-tick
        output_json(args.out/f'official_F0_0_K{k}.json',entry)
        summary['events'].append(entry)
        summary['total_seconds']=time.perf_counter()-start
        output_json(args.out/'highk_manifest.json',summary)
    covered=[(e['K'],e['completed_modes']) for e in summary['events']]
    print('COVERAGE',covered,'FAILURES',len(summary['failures']),flush=True)
    if summary['failures']:raise RuntimeError(f'Incomplete numerical bridge: {covered}, {len(summary["failures"])} failures')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('data',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--floor',type=int,default=0)
    p.add_argument('--regime',choices=['official'],default='official')
    p.add_argument('--draw',type=int,default=0)
    run(p.parse_args())
