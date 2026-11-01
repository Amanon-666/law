"""Fit anchor blending solely by later-date localization MDE on fixed held-out XY.

Run from the law workbench with PYTHONPATH=. and OPENBLAS_NUM_THREADS=1.
The forecast fit and its draw receipts are immutable inputs from the sibling
anchor-learning workbench; no code or assets outside the historical block are
needed to fit or select positional comparators.
"""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr,logsumexp,softmax
from research.uji_core.next import anchor_measure as am
from research.uji_core.next.paired_map_wknn import group_ids
from research.uji_core.inverse import signal,order,squared,vote

EXPECTED='ed3ae84fd08d2286b7c754937d0d1c50b9304e8c339c3a3a5487689eea481162'
GRID=(0.,.25,.5,.75,1.)
NEIGHBORS=(1,3,7,15)
METHODS=('original_map','forecast_map','raw','prototype')
TOL=1e-7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_training(src):
    spec=importlib.util.spec_from_file_location('library2016',src/'fit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def designs(lib,archive,source):
    if sha(source/'frozen.json')!=EXPECTED:raise ValueError('Original forecast freeze changed')
    original=json.loads((source/'protocol.json').read_text())
    if sha(source/'protocol.json')!=json.loads((source/'frozen.json').read_text())['source_protocol_sha256']:
        raise ValueError('Source protocol changed')
    if sha(source/'fit.py')!=original['script_sha256']:raise ValueError('Source draw builder changed')
    paths=json.loads((source/'support_ids.json').read_text())
    orders={(d['pair'],d['floor'],d['draw']):d['S_order'] for d in paths}
    coords=sorted
    for a,b in original['pairs']:
        earlier,later=lib.load(archive,a),lib.load(archive,b)
        if not a['date']<b['date']<'20170000':raise ValueError('Changed 2016 source block')
        pair=a['date']+'-'+b['date']
        for f,ff in ((3,5),(5,3)):
            h,s,q=earlier[ff],earlier[f],later[f];prior=am.history([h],scale=lib.UNIT)
            rowno={z:i for i,z in enumerate(s['ids'])}
            for draw in lib.SEEDS:
                order_ids=orders[pair,f,draw];sr=[rowno[v] for v in order_ids]
                selected=set(map(tuple,s['xy'][sr]))
                qkeep=np.array([tuple(x) not in selected for x in q['xy']])
                truth={k:v[qkeep] for k,v in q.items()}
                pts,cnt=np.unique(truth['xy'],axis=0,return_counts=True)
                if len(selected)!=20 or len(pts)!=4 or not np.all(cnt==12):
                    raise ValueError('Expected 4 completely held positions with 12 repeats')
                if not max(h['times'].max(),s['times'].max())<truth['times'].min():raise ValueError('Nonchronological')
                key=dict(pair=pair,floor=f,draw=draw,
                    q_ids=truth['ids'].tolist(),q_xy=pts.tolist(),s_ids=order_ids)
                yield key,h,s,prior,sr,truth


def arrays(model,q):
    obs=q['rss'][:,model.visible].astype(float)
    seen=(obs!=100).astype(float);miss=1.-seen
    r=seen*(obs+75.)
    def detect(s):
        u=s+75.
        return (r@u.T-.5*seen@(u*u).T)/144.
    h=model.historical
    lh=detect(h)+miss@log_ndtr((-85.-h)/12.).T
    z=logsumexp(lh,axis=1,keepdims=True)
    origin=model.xy[0]
    # H relative weights stay constant as g changes; combine them exactly.
    hx=softmax(lh,axis=1)@(model.xy[:len(h)]-origin)
    raw=model.raw;d=model.posterior-raw
    c0=detect(raw)-z
    c1=(r@d.T-seen@((raw+75.)*d).T)/144.
    c2=-.5*seen@(d*d).T/144.
    return dict(c0=c0,c1=c1,c2=c2,raw=raw,d=d,miss=miss,hxy=hx,
        sxy=model.xy[len(h):]-origin,target=q['xy']-origin)


class FixedK:
    def __init__(self,obs):
        width=max(j['raw'].shape[1] for j in obs)
        for key in obs[0]:
            samples=[j[key] for j in obs]
            if key in ('raw','d','miss'):
                samples=[np.pad(a,((0,0),(0,width-a.shape[1]))) for a in samples]
            setattr(self,key,np.stack(samples))

    def distances(self,g):
        m=self.raw+g*self.d
        miss=self.miss@np.swapaxes(log_ndtr((-85.-m)/12.),1,2)
        anchor=self.c0+g*self.c1+g*g*self.c2+miss
        top=np.maximum(np.max(anchor,axis=2,keepdims=True),0.)
        ph=np.exp(-top);ps=np.exp(anchor-top)
        pred=(ph*self.hxy+ps@self.sxy)/(ph+np.sum(ps,axis=2,keepdims=True))
        return np.linalg.norm(pred-self.target,axis=2).mean(1)

    @lru_cache(maxsize=400)
    def risk(self,g):
        return float(self.distances(g).mean())


def optimize(risk):
    # Always include all five fixed candidates, including endpoints.
    trials=[dict(g=g,mde=risk(g),source='grid') for g in GRID]
    for a,b in zip(GRID,GRID[1:]):
        r=minimize_scalar(risk,bounds=(a,b),method='bounded',
            options={'xatol':TOL,'maxiter':80})
        if not r.success or not np.isfinite(r.fun):raise ValueError('Quarter minimization failed')
        trials.append(dict(g=float(r.x),mde=float(r.fun),source='bounded',
            interval=[a,b],evaluations=r.nfev))
    best=min(t['mde'] for t in trials)
    win=min((x for x in trials if x['mde']<=best+1e-12),key=lambda t:(abs(t['g']-.5),t['g']))
    return win['g'],trials


def train(lib,archive,src,out):
    prior=json.loads((src/'frozen.json').read_text())
    protocol=dict(source_forecast_sha256=EXPECTED,source_orders_sha256=sha(src/'support_ids.json'),
        archive_sha256=lib.library.SHA256,script_sha256=sha(__file__),anchor_impl_sha256=sha(am.__file__),
        study='Objective-only replication of one public target-anchor mixture, 7 early 2016 date pairs, S3 2 registered floors, 10 prior nested S orders, K=1/3/5/10/20.',
        Q='For each date-pair, floor, draw use the four later-day target-floor sites outside MAX S20, all 12 scans at each, same 48 IDs at every K. Q never in S, and other-floor H may cover identical registered physical XY.',
        readout='Exact original censored Gaussian matching sigma=12 threshold=-85, raw anchors blended toward their calibrated field with common g at each K. Keep H, multiplicity and coordinate weights unchanged; predicts in metres.',
        objective='Average Euclidean location error over all real Q scans; 12 equal repeats per location, equal sites, draws, floors, 7 physical pairs and all budgets for pooled g. Per-K and scalar bounded minimization within each of four quarters plus explicit fixed [0,.25,.5,.75,1], lowest source risk first; differences <=1e-12 m tie toward .5 then smaller g. xatol=1e-7. No monotonic penalty or formal guarantee against multiple minima within a quarter.',
        controls=['raw','quarter','half','three_quarters','field','previous_2016_forecast_by_K','previous_2016_forecast_pooled','location_pooled','location_by_K'],
        wknn='One neighbor count per matcher (original map, fitted location map, raw and prototype direct), minimizing source MDE with the exact same fixed 48 Q rows per draw & K; k in [1,3,7,15], lower k breaks ties. Direct jointly selects representation (raw preferred on ties), no floor penalty. Original RMS distances, inverse-distance votes, tie by source row ID. No external outcomes enter this workbench.')
    save(out/'protocol.json',protocol)
    buckets={k:[] for k in lib.BUDGETS};check={k:[] for k in lib.BUDGETS}
    receipts=[]
    for key,h,s,prior_map,sr,q in designs(lib,archive,src):
        # Compute query membership and hash once, before varying K.
        sig=hashlib.sha256(','.join(q['ids'].tolist()).encode()).hexdigest()
        key['query_sha256_by_K']={}
        for k in lib.BUDGETS:
            ix=sr[:k];supports={a:s[a][ix] for a in ('rss','xy')}
            field=am.build(prior_map,supports,scale=lib.UNIT)
            if set(map(tuple,supports['xy'])) & set(map(tuple,q['xy'])):raise ValueError('S/Q overlap')
            buckets[k].append(arrays(field,q))
            key['query_sha256_by_K'][k]=sig
            if len(receipts)%20==0:
                idx=len(receipts)
                for g in GRID:
                    mse=float(np.linalg.norm(field.predict(q['rss'],g,False)-q['xy'],axis=1).mean())
                    check[k].append((idx,g,mse))
        receipts.append(key)
    if len(receipts)!=140:raise ValueError('Source draw count changed')
    save(out/'draws_and_q.json',receipts)
    items={k:FixedK(bucket) for k,bucket in buckets.items()};del buckets
    parity=0.
    for k,bank in items.items():
        for g in GRID:
            scores=bank.distances(g)
            for i,v,expected in check[k]:
                if g==v:parity=max(parity,abs(scores[i]-expected))
    if parity>1e-8:raise AssertionError(('Location predictor mismatch',parity))
    gains={};log={}
    for k,bank in items.items():
        gains[str(k)],log[str(k)]=optimize(bank.risk)
    pooled,log['pooled']=optimize(lambda g:float(np.mean([bank.risk(g) for bank in items.values()])))
    conf=dict(model='lib-2016-anchor-location-objective-v1',g_by_K=gains,g_scalar=pooled,
        orientation='S_template=(1-g)*raw+g*posterior; H templates and coordinate weights unchanged',
        source_protocol_sha256=sha(out/'protocol.json'),source_query_receipts_sha256=sha(out/'draws_and_q.json'),
        training_archive_sha256=lib.library.SHA256,source_only=True,previous_forecast_sha256=EXPECTED,
        counts=dict(source_pairs=7,floors=2,draws_per_floor=10,episodes=700,Q_sites_per_event=4,
            Q_scans_per_event=48,all_140_paths_have_identical_ordered_Q_ids_across_5_K=True,
            public_readout_max_MDE_parity_gap=parity),
        K_monotone_non_decreasing=bool(np.all(np.diff(list(gains.values()))>=0)),frozen_fixed_controls=lib.FIXED)
    save(out/'frozen.json',conf);save(out/'quarter_optima.json',log)
    source=[]
    for k,bank in items.items():
        arms={**lib.FIXED,'previous_forecast':prior['g_by_K'][str(k)],'previous_forecast_pooled':prior['g_scalar'],
            'location_by_K':gains[str(k)],'location_pooled':pooled}
        source.extend(dict(K=k,arm=n,g=g,mde_m=bank.risk(g)) for n,g in arms.items())
    pd.DataFrame(source).to_csv(out/'source_risks.csv',index=False)
    print(sha(out/'frozen.json'),gains,pooled,'parity',parity,flush=True)


def knn(lib,archive,src,out):
    conf=json.loads((out/'frozen.json').read_text())
    if sha(out/'protocol.json')!=conf['source_protocol_sha256'] or sha(out/'draws_and_q.json')!=conf['source_query_receipts_sha256']:
        raise ValueError('Source design changed after fit')
    locked=json.loads((out/'draws_and_q.json').read_text());data=[]
    for i,(key,h,s,prior,sr,q) in enumerate(designs(lib,archive,src)):
        if key['q_ids']!=locked[i]['q_ids']:raise ValueError('Wrong query ids')
        hids=h['ids'].astype(np.int64);tids=s['ids'].astype(np.int64)
        hxy,hid=group_ids(h['xy'],hids);hx=signal(h['rss'])
        hp=np.stack([hx[np.all(h['xy']==p,axis=1)].mean(0) for p in hxy])
        for k in lib.BUDGETS:
            ix=sr[:k];sp={v:s[v][ix] for v in ('rss','xy')};model=am.build(prior,sp,scale=lib.UNIT)
            xy,ids=group_ids(sp['xy'],tids[ix]);vis=model.visible
            obs=signal(q['rss'])[:,vis];gain=conf['g_by_K'][str(k)]
            def evaluate(method,ref,ref_xy,row):
                dist=squared(obs,ref);ixs=order(dist,row,max(NEIGHBORS));ds=np.take_along_axis(dist,ixs,axis=1)
                loc=ref_xy[ixs]-ref_xy[0]
                for kn in NEIGHBORS:
                    pred=vote(ds[:,:kn],loc[:,:kn],obs.shape[1])[0]+ref_xy[0]
                    mde=float(np.linalg.norm(pred-q['xy'],axis=1).mean())
                    data.append(dict(pair=key['pair'],floor=key['floor'],draw=key['draw'],K=k,reference=method,k=kn,mde_m=mde))
            for name,g in [('original_map',0.),('forecast_map',gain)]:
                ts=model.raw+g*(model.posterior-model.raw)
                evaluate(name,np.vstack((model.historical,ts))+110.,model.xy,np.r_[hid,ids])
            for name,ref,pts,ids in [('raw',hx,h['xy'],hids),('prototype',hp,hxy,hid)]:
                evaluate(name,np.vstack((ref[:,vis],signal(sp['rss'])[:,vis])),np.vstack((pts,sp['xy'])),np.r_[ids,tids[ix]])
    df=pd.DataFrame(data);risk=df.groupby(['reference','k']).mde_m.mean()
    choices={a:min(NEIGHBORS,key=lambda n:(risk.loc[a,n],n)) for a in METHODS}
    direct=min(('raw','prototype'),key=lambda a:(risk.loc[a,choices[a]],a!='raw',choices[a]))
    receipt=dict(anchor_freeze_sha256=sha(out/'frozen.json'),source_protocol_sha256=sha(out/'protocol.json'),
        k_original_map=choices['original_map'],k_forecast_map=choices['forecast_map'],direct_rep=direct,k_direct=choices[direct],
        k_raw_direct=choices['raw'],k_prototype_direct=choices['prototype'],direct_alpha=0.,
        note='forecast_map key is the newly location-trained blended map. Exact same four held-out Q positions and 48 ordered Q scan IDs across all K and all WKNNs. See protocol.json.')
    save(out/'wknn_frozen.json',receipt);risk.to_csv(out/'wknn_source_risks.csv');df.to_csv(out/'wknn_events.csv.gz',index=False)
    print(sha(out/'wknn_frozen.json'),choices,direct,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['train','wknn'])
    for key in ('archive','previous','out'):p.add_argument(key,type=Path)
    args=p.parse_args();args.out.mkdir(exist_ok=True,parents=True);lib=load_training(args.previous)
    if args.action=='train':train(lib,args.archive,args.previous,args.out)
    else:knn(lib,args.archive,args.previous,args.out)
