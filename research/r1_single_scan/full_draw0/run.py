"""One preregistered floor/regime shard. No B1 selection or gradient training.
Values/covariances are shared between arms only before S updating: each arm
still receives its own full VI posterior. Each K is an atomic NPZ+JSON pair.
"""
import argparse, hashlib, json, time, traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import inference as inf
from data import group_scans
from history_bank import BankPrior
from multi_history import single_history
from operator_prior_v1.operator_prior import source_tokens
from research.r1_single_scan import score as R
from research.eta_field_v2.r1_observable_reference import map_moment
from run import ap_stats,tensor

ARMS=('gp','seven','nonlinear'); NEI=(1,3,7,15); KS=(1,3,5,10,20)
def sha(x):return hashlib.sha256(x).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def src_case(path,b,f):
    p=path/'trainingData.csv';assert sha(p.read_bytes())==R.TRAIN_HASH
    frame=pd.read_csv(p);frame=frame[frame.BUILDINGID==b];ap=[f'WAP{v:03}' for v in range(1,521)]
    def take(ff):
        d=frame[frame.FLOOR==ff];return dict(ids=d.index.to_numpy(),rss=d[ap].to_numpy(dtype=np.int16),xy=d[['LONGITUDE','LATITUDE']].to_numpy(dtype=np.float64),phone=d.PHONEID.to_numpy())
    target=take(f);coords,g=np.unique(target['xy'],axis=0,return_inverse=True);g=g.ravel();rng=np.random.default_rng(261103+b);order=rng.permutation(len(coords))[:29];ids=[rng.choice(np.flatnonzero(g==i)) for i in order];assert len(ids)==29
    H=sorted(sorted((int(v) for v in frame.FLOOR.unique() if v!=f),key=lambda v:(abs(v-f),v))[:3]);assert len(H)==3
    S={k:v[ids[:5]] for k,v in target.items()};Q={k:v[ids[5:]] for k,v in target.items()};assert not (S['xy'][:,None,:]==Q['xy'][None,:,:]).all(2).any();return dict(history={v:take(v) for v in H},support=S,query=Q,forbidden=None)

def voters(proto,q,centered,origin):
    v=np.where(q==100,-110.,q).astype(float);dd=np.maximum((v*v).sum(1)[:,None]+(proto*proto).sum(1)[None,:]-2*v@proto.T,0.);order=np.argsort(dd,axis=1,kind='stable')[:,:max(NEI)];d=np.sqrt(np.take_along_axis(dd,order,axis=1)/v.shape[1]);out={}
    for nei in NEI:
        w=1/np.maximum(d[:,:nei],1e-8);out[nei]=origin+(w[:,:,None]*centered[order[:,:nei]]).sum(1)/w.sum(1)[:,None]
    return out

class Trace:
    def __init__(self):self.orig=inf.minimize;self.data=[]
    def __call__(self,*args,**kw):
        r=self.orig(*args,**kw);pg=np.asarray(r.jac).copy();x=np.asarray(r.x)
        for i,(lo,hi) in enumerate(kw['bounds']):
            if lo is not None and x[i]<=lo+1e-12 and pg[i]>0:pg[i]=0.
            if hi is not None and x[i]>=hi-1e-12 and pg[i]<0:pg[i]=0.
        self.data.append((bool(r.success),float(np.max(np.abs(pg))),int(r.nit),str(r.message)));return r

def trace_stat(records):
    assert records
    return dict(total=len(records),success=sum(a[0] for a in records),gradmax=max(a[1] for a in records),maxnit=max(a[2] for a in records),gt1e3=sum(a[1]>1e-3 for a in records),messages=sorted(set(a[3] for a in records)))

@torch.no_grad()
def twin_inputs(p,hs,dz,x,beta,net):
    base,feat=p.features(x)
    if not any(h is not None for h in hs):return [base,base,base],feat
    z=torch.stack([source_tokens(h,d,x,R.B) for h,d in zip(hs,dz)]);return [base,base+20*(beta[0]+z.mean(0)@beta[1:]),base+net(z)[0]],feat

@torch.no_grad()
def score_k(c,hg,banks,origin,f,k,beta,net):
    S={n:v[:k] for n,v in c['support'].items()};s=group_scans(S['rss'],S['xy']-origin);assert len(s['positions'])==k and np.all(s['stats']['n']==1);sx=tensor(s['positions']);fs=list(c['history']);dz=[ff-f for ff in fs]
    seen=(S['rss']!=100).any(0)
    for h in c['history'].values():seen|=(h['rss']!=100).any(0)
    aps=np.flatnonzero(seen).tolist()
    for a in aps:banks.setdefault(a,[None]*3)
    with torch.enable_grad():w,trial=R.weight(banks,s,aps)
    xy=np.vstack([hh['positions'] for hh in hg.values()]+[s['positions']]);axs=[np.linspace(z.min(),z.max(),int(np.ceil(np.ptp(z)/2))+1) for z in xy.T];grid=tensor(np.stack(np.meshgrid(*axs),axis=-1).reshape(-1,2));gn=grid.numpy();q=c['query']['rss'][:,aps];Q=len(q);G=len(grid)
    log=np.zeros((3,Q,G));proto=np.empty((3,G,len(aps)));solves={n:[] for n in ARMS};tr=Trace()
    for j,a in enumerate(aps):
        hs=banks[a];p=BankPrior(hs,w,R.B) if any(h is not None for h in hs) else BankPrior([],[],R.B)
        sm,sf=twin_inputs(p,hs,dz,sx,beta,net);gm,gf=twin_inputs(p,hs,dz,grid,beta,net);sk=p.covariance(sx,sx,sf,sf);cross=p.covariance(grid,sx,gf,sf);diag=p.diagonal(grid,gf);stats=ap_stats(s,a);idx=np.flatnonzero(q[:,j]!=100);values,iv=np.unique(q[idx,j],return_inverse=True)
        for v,n in enumerate(ARMS):
            with patch.object(inf,'minimize',tr): up=inf.infer(sm[v],sk,stats,R.O,steps=500)
            solves[n].append(tr.data[-1]);field=inf.predict_from_posterior(gm[v],diag,cross,sm[v],sk,up);mu,var=field['mean'],field['variance'];assert torch.isfinite(mu).all() and torch.isfinite(var).all() and float(var.min())>=-1e-7;var=var.clamp_min(0.);ms=inf.observation_log_prob(tensor(100.),mu,var,R.O).numpy();log[v]+=ms
            if len(idx):log[v,idx]+=(inf.observation_log_prob(tensor(values)[:,None],mu,var,R.O).numpy()-ms)[iv.ravel()]
            _,proto[v,:,j]=map_moment(mu.numpy(),var.numpy(),R.O['alpha'],R.O['beta'],R.O['detect_scale'])
    arr=dict(query_ids=c['query']['ids'],truth=c['query']['xy'],support_ids=S['ids'],aps=np.array(aps));me={};quality={n:trace_stat(solves[n]) for n in ARMS}
    for v,n in enumerate(ARMS):
        arr[n]=(tensor(log[v]).softmax(1)@grid).numpy()+origin
        for nei,coord in voters(proto[v],q,gn,origin).items():arr[n+'_w'+str(nei)]=coord
    for n in arr:
        if n in ARMS or '_w' in n:
            assert np.isfinite(arr[n]).all();me[n]=R.metric(arr[n],arr['truth'])
    return arr,dict(weights=w,weight_trials=trial,grid=G,grid_hash=sha(np.asarray(gn+origin,dtype='<f8').tobytes()),ap_count=len(aps),quality=quality,metrics=me)

def main(data,out,reg,f):
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out.mkdir(parents=True,exist_ok=True);t0=time.perf_counter();source=reg.startswith('src');b=int(reg[-1]) if source else 1
    c=src_case(data,b,f) if source else R.validate_layout(R.EVIDENCE,R.read_raw(data),reg,f,0);fs=list(c['history']);origin=np.concatenate([v['xy'] for v in c['history'].values()]).mean(0);hg={h:group_scans(v['rss'],v['xy']-origin) for h,v in c['history'].items()};ap=np.flatnonzero(np.logical_or.reduce([(v['rss']!=100).any(0) for v in c['history'].values()]));beta,net=R.models();banks={};ht=Trace()
    with patch.object(inf,'minimize',ht):
        for a in ap:banks[a]=[single_history({'scans':hg[h]},int(a),R.B,R.O,steps=640) for h in fs]
    header=dict(regime=reg,building=b,floor=f,draw=0,history_floors=fs,H_ids=[sha(np.asarray(c['history'][h]['ids'],dtype='<i8').tobytes()) for h in fs],model_hashes=[R.AFFINE_HASH,R.NONLINEAR_HASH],forbidden=c['forbidden'],S_order=c['support']['ids'].tolist(),Q_ids_hash=sha(np.asarray(c['query']['ids'],dtype='<i8').tobytes()),history_vi=trace_stat(ht.data),history_secs=time.perf_counter()-t0);write(out/'header.json',header);bad=[]
    for k in ((5,) if source else KS):
        st=time.perf_counter()
        try:
            arr,rec=score_k(c,hg,banks,origin,f,k,beta,net);rec.update(K=k,seconds=time.perf_counter()-st,archived_raw={} if source else R.cached_wknn(c,reg,f,0,k));np.savez_compressed(out/f'K{k}.npz',**arr);rec['npz_sha256']=sha((out/f'K{k}.npz').read_bytes());write(out/f'K{k}.json',rec);print(reg,f,k,[rec['metrics'][a]['mde'] for a in ARMS],flush=True)
        except Exception as e:
            bad.append(k);write(out/f'failure_K{k}.json',dict(error=repr(e),trace=traceback.format_exc()[-9000:]));print('FAILED',reg,f,k,repr(e),flush=True)
    if bad:raise RuntimeError(f'Incomplete budgets {bad}')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);p.add_argument('regime',choices=['official','phone','src0','src2']);p.add_argument('floor',type=int);a=p.parse_args();main(a.data,a.out,a.regime,a.floor)
