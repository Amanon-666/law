"""Same observation model as inference.observation_log_prob; no fitted parameters or AP selection.
One position vector is cached per map. Non-detections keep their full evidence. The timing
harness uses the exact six archived official query manifests; no calibration or grid changes.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from inference import observation_log_prob as logp

class MapReadout:
    """Input arrays must have AP columns aligned; mean and variance are [AP,grid]."""
    def __init__(self,mean,variance,grid,obs):
        self.mu,self.v,self.grid=[torch.as_tensor(x,dtype=torch.float64) for x in (mean,variance,grid)]
        if self.mu.ndim!=2 or self.v.shape!=self.mu.shape or self.grid.shape!=(self.mu.shape[1],2):raise ValueError('Unexpected field dimensions')
        self.obs=obs
        self.all_missing=logp(100.,self.mu,self.v,obs).sum(0)

    def predict(self,rss):
        q=torch.as_tensor(rss,dtype=torch.float64)
        if q.shape!=(len(self.mu),):raise ValueError('One aligned scan required')
        keep=q!=100
        score=self.all_missing
        if keep.any():
            m,v=self.mu[keep],self.v[keep]
            score=score+(logp(q[keep,None],m,v,self.obs)-logp(100.,m,v,self.obs)).sum(0)
        return score.softmax(dim=0)@self.grid


def benchmark(data,fields,out):
    from .prior_bridge import load
    from .replay import hashids
    here=Path(__file__).resolve().parent
    obs=json.loads((here.parents[1]/'field_parameters.json').read_text())['0']['obs']
    val=load(data,'validationData.csv'); lookup={e['key']:e for e in json.loads((here/'results/replay/replay.json').read_text())['manifest']}
    torch.set_num_threads(1); reports=[]
    with torch.inference_mode():
        for f in (1,2):
            for seed in (0,1,2):
                key=f'B1F{f}_{seed}';file=fields/(key+'_field.npz');p=np.load(fields/(key+'_pred.npz'),allow_pickle=False);v=np.load(file,allow_pickle=False)
                ids=p['official_ids'];assert hashids(ids)==lookup[key+'_official']['query_ids_sha256'];rs={i:r for i,r in zip(val[f]['ids'],val[f]['rss'])};qs=np.array([rs[i][v['aps']] for i in ids]);grid=v['grid'];origin=v['origin'];m=torch.tensor(v['mu']);vari=torch.tensor(v['variance']);tg=torch.tensor(grid)
                tic=time.perf_counter_ns();model=MapReadout(m,vari,tg,obs); setup=(time.perf_counter_ns()-tic)/1e6
                def dense(q):return logp(q[:,None],m,vari,obs).sum(0).softmax(dim=0)@tg
                eq=max(np.max(np.abs(model.predict(q).numpy()+origin-xy)) for q,xy in zip(qs,p['official_full']));assert eq<1e-6
                # 24 equally spaced scans in original acquisition order, per event; round-robin paired timing.
                selected=np.unique(np.linspace(0,len(qs)-1,min(len(qs),24),dtype=int));times=[[],[]];maxdiff=0.
                q=torch.tensor(qs[selected[0]]);dense(q);model.predict(q)
                for rep in range(3):
                    for i in selected:
                        q=torch.as_tensor(qs[i],dtype=torch.float64);pair={}
                        for b in ((rep+i)%2,1-(rep+i)%2):
                            tic=time.perf_counter_ns();y=dense(q) if b==0 else model.predict(q);times[b].append((time.perf_counter_ns()-tic)/1e6);pair[b]=y.numpy()
                        maxdiff=max(maxdiff,float(np.abs(pair[0]-pair[1]).max()))
                assert maxdiff<1e-6
                reports.append(dict(event=key,official_rows=len(ids),ap=len(m),grid=len(grid),sample_ids=ids[selected].tolist(),setup_ms=setup,cache_bytes=model.all_missing.numel()*8,all_query_max_diff_from_saved=eq,max_dense_sparse_diff=maxdiff,median_detected=float(np.median((qs[selected]!=100).sum(1))),median_ms=[float(np.median(x)) for x in times],p90_ms=[float(np.quantile(x,.9)) for x in times],times_ms=times))
    proc=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].strip(' :\t');r=dict(scope='Single-scan CPU FP64 one thread, field already built, medians per job then across six; numerical identity, not a new estimator. Dense comparison vectorized over ALL APs in a single Torch call; sparse uses the same routine. Setup excludes loading identical maps.',processor=proc,torch=torch.__version__,events=reports,median_of_event_medians_ms=np.median([z['median_ms'] for z in reports],axis=0).tolist(),median_of_paired_event_speedups=float(np.median([z['median_ms'][0]/z['median_ms'][1] for z in reports])),max_coordinate_error=max(z['all_query_max_diff_from_saved'] for z in reports));out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print({k:v for k,v in r.items() if k!='events'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('fields',type=Path);p.add_argument('output',type=Path);a=p.parse_args();benchmark(a.data,a.fields,a.output)
