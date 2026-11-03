"""Replay draw 1, never select a model from B1. Uses run.score_k byte-for-byte."""
import argparse,time,traceback
from pathlib import Path
from unittest.mock import patch
import torch,numpy as np
from . import run as U

def main(data,out,r,f):
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);t=time.perf_counter();out.mkdir(parents=True,exist_ok=True);R=U.R;draw=1
    case=R.validate_layout(R.EVIDENCE,R.read_raw(data),r,f,draw);fs=list(case['history']);org=np.concatenate([v['xy'] for v in case['history'].values()]).mean(0);hg={ff:U.group_scans(v['rss'],v['xy']-org) for ff,v in case['history'].items()};ap=np.flatnonzero(np.logical_or.reduce([(v['rss']!=100).any(0) for v in case['history'].values()]));beta,net=R.models();ht=U.Trace();banks={}
    with patch.object(U.inf,'minimize',ht):
        for a in ap:banks[a]=[U.single_history({'scans':hg[h]},int(a),R.B,R.O,steps=640) for h in fs]
    U.write(out/'header.json',dict(regime=r,building=1,floor=f,draw=draw,history_floors=fs,H_ids=[U.sha(np.asarray(case['history'][h]['ids'],dtype='<i8').tobytes()) for h in fs],model_hashes=[R.AFFINE_HASH,R.NONLINEAR_HASH],forbidden=case['forbidden'],S_order=case['support']['ids'].tolist(),Q_ids_hash=case['query_rows_sha256'],history_vi=U.trace_stat(ht.data),history_secs=time.perf_counter()-t));bad=[]
    for k in U.KS:
        st=time.perf_counter()
        try:
            arr,rec=U.score_k(case,hg,banks,org,f,k,beta,net);rec.update(K=k,seconds=time.perf_counter()-st,archived_raw=R.cached_wknn(case,r,f,draw,k));np.savez_compressed(out/f'K{k}.npz',**arr);rec['npz_sha256']=U.sha((out/f'K{k}.npz').read_bytes());U.write(out/f'K{k}.json',rec);print('DONE',r,f,k,[rec['metrics'][a]['mde'] for a in U.ARMS],flush=True)
        except Exception as e:
            bad.append(k);U.write(out/f'failure_K{k}.json',dict(error=repr(e),trace=traceback.format_exc()[-9000:]));print('FAILED',r,f,k,repr(e),flush=True)
    if bad:raise RuntimeError(f'Incomplete {bad}')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);p.add_argument('regime',choices=['official','phone']);p.add_argument('floor',type=int);a=p.parse_args();main(a.data,a.out,a.regime,a.floor)
