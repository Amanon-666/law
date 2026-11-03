"""Run one pair of ORIGINAL support orders; share only historical GP fits.
Inputs and score_k numerics identical to full_draw0/run.py; save partial data on failure.
"""
import argparse,time,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np, torch
from . import run as U

def main(data,out,reg,f,first):
    if first not in (2,4,6,8):raise ValueError('Only the four locked remaining draw pairs')
    R=U.R;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);t=time.perf_counter();raw=R.read_raw(data);cases=[R.validate_layout(R.EVIDENCE,raw,reg,f,d) for d in [first,first+1]];c0,c1=cases;fs=list(c0['history']);assert fs==[v for v in range(4) if v!=f] and len(fs)==3
    for v in ('xy','rss','phone','ids'):
        np.testing.assert_array_equal(c0['query'][v],c1['query'][v]);
        for h in fs:np.testing.assert_array_equal(c0['history'][h][v],c1['history'][h][v])
    org=np.concatenate([v['xy'] for v in c0['history'].values()]).mean(0);hg={h:U.group_scans(v['rss'],v['xy']-org) for h,v in c0['history'].items()};ap=np.flatnonzero(np.logical_or.reduce([(v['rss']!=100).any(0) for v in c0['history'].values()]));beta,net=R.models();ht=U.Trace();hbank={};
    with patch.object(U.inf,'minimize',ht):
        for a in ap:hbank[a]=[U.single_history({'scans':hg[h]},int(a),R.B,R.O,steps=640) for h in fs]
    ts=time.perf_counter()-t;bad=[]
    for d,c in zip([first,first+1],cases):
        sub=out/f'draw{d}-{reg}-F{f}';sub.mkdir(parents=True,exist_ok=False);banks=dict(hbank)
        U.write(sub/'header.json',dict(regime=reg,building=1,floor=f,draw=d,history_floors=fs,H_ids=[U.sha(np.asarray(c['history'][h]['ids'],dtype='<i8').tobytes()) for h in fs],model_hashes=[R.AFFINE_HASH,R.NONLINEAR_HASH],forbidden=c['forbidden'],S_order=c['support']['ids'].tolist(),Q_ids_hash=c['query_rows_sha256'],history_vi=U.trace_stat(ht.data),history_master_draw=first,history_secs=ts if d==first else 0.))
        for k in U.KS:
            st=time.perf_counter()
            try:
                arr,rec=U.score_k(c,hg,banks,org,f,k,beta,net);rec.update(K=k,seconds=time.perf_counter()-st,archived_raw=R.cached_wknn(c,reg,f,d,k));np.savez_compressed(sub/f'K{k}.npz',**arr);rec['npz_sha256']=U.sha((sub/f'K{k}.npz').read_bytes());U.write(sub/f'K{k}.json',rec);print('DONE',reg,f,d,k,','.join(f'{rec["metrics"][a]["mde"]:.5f}' for a in U.ARMS),flush=True)
            except Exception as ex:
                bad.append((d,k));U.write(sub/f'failure_K{k}.json',dict(error=repr(ex),trace=traceback.format_exc()[-9000:]));print('FAILED',reg,f,d,k,repr(ex),flush=True)
    if bad:raise RuntimeError('Incomplete locked cells: '+str(bad))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);p.add_argument('regime',choices=['official','phone']);p.add_argument('floor',type=int,choices=range(4));p.add_argument('first',type=int,choices=[2,4,6,8]);a=p.parse_args();main(a.data,a.out,a.regime,a.floor,a.first)
