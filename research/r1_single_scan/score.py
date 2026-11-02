"""Bounded original R1: all 3 B1 history floors, one actual scan per support position.
Fixed 0-based row IDs from evidence.zip; no target model fitting or hyperparameter tuning.
Deployment arms: GP, frozen seven-coefficient mean, frozen V1 nonlinear mean.
One shared three-source Support-only EB weight per episode; M-PMC for K<=10.
Existing source-selected WKNN archived scores are read, never retrained or rerun.
"""
import argparse, csv, hashlib, io, json, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import group_scans
from history_bank import BankPrior
from inference import observation_log_prob
from multi_history import single_history
from multi_source_adaptation import fit_source_weights
from operator_prior_v1.operator_prior import OperatorNet, source_tokens
from research.uji_core.protocol import HASHES, position_weights
from run import tensor, ap_stats
from source_weight_mixture import field_from_prior
from transfer import Prior, Transfer, rbf

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'research/uji_core/evidence.zip'
RECEIPTS=ROOT/'research/uji_core/round2/paper_replay/limit_study/receipts'
P=json.loads((ROOT/'research/uji_core/field_parameters.json').read_text())['0']
B,O=P['base'],P['obs']
TRAIN_HASH='45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168'
MANIFEST_HASH='5f4fb0ceccc5a64aa7411cc287715e6fa9a39d593da6713910920a3fda67c613'
PHONE_HASH='76763a87569d885705dccc54cbd72e878f7a711cbb5343fbfee4517caf2482e6'
AFFINE_HASH='cd5c4972f4ad418a58c9c73f88d9c65c245f24ecf5c35842bcdc8535114caf58'
NONLINEAR_HASH='53061b01164f78def406e7b1395c5e62a295adb9a9a3570a2b91efed7b71e7c7'
EXPECTED_OFFICIAL=(30,143,87,47)
EXPECTED_PHONE=(14,7,23,10)
KS=(1,3,5)

def digest(b):return hashlib.sha256(b).hexdigest()
def read_zip_json(z,name,sha):
    raw=z.read(name)
    assert digest(raw)==sha,(name,digest(raw))
    return json.loads(raw)
def read_raw(path):
    data={}
    for file in ('trainingData.csv','validationData.csv'):
        p=path/file
        assert digest(p.read_bytes())==HASHES[file],file
        t=pd.read_csv(p)
        waps=[f'WAP{j:03}' for j in range(1,521)]
        for f in range(4):
            d=t[(t.BUILDINGID==1)&(t.FLOOR==f)]
            assert len(d)>0
            data[(file,f)]={'ids':d.index.to_numpy(dtype=np.int64),'rss':d[waps].to_numpy(dtype=np.int16),
                'xy':d[['LONGITUDE','LATITUDE']].to_numpy(dtype=np.float64),
                'phone':d.PHONEID.to_numpy(dtype=np.int64)}
    return data
def select(d,ids):
    lu={int(r):i for i,r in enumerate(d['ids'])}
    ix=np.asarray([lu[int(r)] for r in ids],dtype=int)
    return {k:v[ix] for k,v in d.items()}
def validate_layout(evidence,raw,regime,f,draw):
    with zipfile.ZipFile(EVIDENCE) as z:
        manifest=read_zip_json(z,'results/manifest.json',MANIFEST_HASH)
        phones=read_zip_json(z,'results/device_stress_design.json',PHONE_HASH)
        m=[e for e in manifest['episodes'] if (e['building'],e['floor'],e['draw'])==(1,f,draw)]
        assert len(m)==1 and len(m[0]['support_order'])==20
        ev=m[0]
        if regime=='official':
            query=select(raw[('validationData.csv',f)],ev['query_ids'])
            assert len(query['ids'])==EXPECTED_OFFICIAL[f]
            order=ev['support_order'];forbidden=None
        else:
            pe=[e for e in phones['events'] if (e['building'],e['floor'])==(1,f)]
            assert len(pe)==1
            d=pe[0];assert len(d['support_orders'])==10 and len(d['query_ids'])==12
            assert d['phone']==EXPECTED_PHONE[f]
            order=d['support_orders'][draw];forbidden=int(d['phone'])
            query=select(raw[('trainingData.csv',f)],d['query_ids'])
            assert len(query['ids'])==12
            assert np.all(query['phone']==forbidden)
        support=select(raw[('trainingData.csv',f)],order)
        assert len(support['ids'])==20 and len(np.unique(support['xy'],axis=0))==20
        if forbidden is not None:
            assert np.all(support['phone']!=forbidden)
            assert not (support['xy'][:,None,:]==query['xy'][None,:,:]).all(-1).any()
        history={}
        for h in range(4):
            if h==f:continue
            d=raw[('trainingData.csv',h)]
            keep=np.ones(len(d['ids']),bool) if forbidden is None else d['phone']!=forbidden
            history[h]={k:v[keep] for k,v in d.items()}
            assert len(history[h]['rss'])>0
        if forbidden is not None:
            assert all(np.all(x['phone']!=forbidden) for x in history.values())
        wk=list(csv.DictReader(io.StringIO(z.read('results/wknn_events.csv').decode())))
        dev=list(csv.DictReader(io.StringIO(z.read('results/device_stress_events.csv').decode())))
    return dict(support=support,query=query,history=history,official_wknn=wk,phone_wknn=dev,forbidden=forbidden,
        query_rows_sha256=digest(np.asarray(query['ids'],dtype='<i8').tobytes()),
        support_order_sha256=digest(np.asarray(order,dtype='<i8').tobytes()))

def models():
    with zipfile.ZipFile(RECEIPTS/'affine-six-with-source.zip') as z:
        fr=json.loads(z.read('source/freeze.json'))
        assert fr['operator_sha256']==AFFINE_HASH and digest(z.read('source/operator.pt'))==AFFINE_HASH
        beta=tensor(fr['mean_only_affine_beta'])
        assert beta.shape==(7,)
    with zipfile.ZipFile(RECEIPTS/'source-v1.zip') as z:
        names=z.namelist()
        ckpt=next((n for n in names if n.endswith('source/operator.pt')),None)
        assert ckpt is not None, ('missing source checkpoint',names[:10])
        b=z.read(ckpt);assert digest(b)==NONLINEAR_HASH
        net=OperatorNet().double()
        net.load_state_dict(torch.load(io.BytesIO(b),map_location='cpu',weights_only=True))
        net.eval()
    return beta,net

class CorrectedBank:
    def __init__(self,hs,w,dz,mode,beta,net):
        self.base=BankPrior(hs,w,B) if any(h is not None for h in hs) else BankPrior([],[],B)
        self.hs=hs
        self.dz=dz
        self.mode=mode
        self.beta=beta
        self.net=net
        self.active=any(h is not None for h in hs)
    def features(self,x):
        mean,feat=self.base.features(x)
        if self.active and self.mode!='gp':
            tokens=torch.stack([source_tokens(h,d,x,B) for h,d in zip(self.hs,self.dz)])
            if self.mode=='seven':
                mu=tokens.mean(0)
                correction=20*(self.beta[0]+mu@self.beta[1:])
            else:correction=self.net(tokens)[0]
            mean=mean+correction
        return mean,feat
    def covariance(self,x,y,fx,fy):return self.base.covariance(x,y,fx,fy)
    def diagonal(self,x,fx):return self.base.diagonal(x,fx)

def weight(banks,s,aps):
    sx=tensor(s['positions']);mm,cc,res=[],[],[]
    for a in aps:
        hs=banks[int(a)]
        means=[];cov=[]
        for h in hs:
            prior=Prior(Transfer('identity'),h,dict(B,drift=0.))
            m,f=prior.features(sx)
            means.append(m);cov.append(prior.covariance(sx,sx,f,f))
        active=any(h is not None for h in hs)
        if not active:cov=[torch.zeros_like(c) for c in cov]
        mm.append(torch.stack(means))
        cc.append(torch.stack(cov))
        res.append(rbf(sx,sx,B['drift'] if active else B['amplitude'],B['length']))
    stats=[ap_stats(s,int(a)) for a in aps]
    st={k:torch.stack([v[k] for v in stats]) for k in stats[0]}
    w,trials=fit_source_weights(torch.stack(mm),torch.stack(cc),torch.stack(res),st,O)
    assert len(w)==3 and min(w)>=0 and abs(sum(w)-1)<1e-8
    return w,trials

def metric(pred,true):
    ww=position_weights(true);dist=np.linalg.norm(pred-true,axis=1)
    order=np.argsort(dist)
    return {'mde':float(ww@dist),'p90':float(dist[order][np.searchsorted(np.cumsum(ww[order]),.9)])}

def cached_wknn(case,regime,f,draw,k):
    rows=case['official_wknn'] if regime=='official' else case['phone_wknn']
    candidates=[]
    for x in rows:
        try:
            xf=int(x['floor']); xb=int(x['building'])
            xd=int(x.get('draw',x.get('seed',-1)));xk=int(x['K'])
        except (KeyError,ValueError):continue
        if (xb,xf,xd,xk)!=(1,f,draw,k):continue
        candidates.append(x)
    assert candidates,('Missing matched archived WKNN',regime,f,draw,k)
    return {r['method']:float(r['mde']) for r in candidates}

def run(args):
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    start=time.perf_counter();raw=read_raw(args.data)
    c=validate_layout(EVIDENCE,raw,args.regime,args.floor,args.draw)
    beta,net=models()
    hist=c['history'];floors=list(hist)
    origin=np.concatenate([hist[h]['xy'] for h in floors]).mean(0)
    histories={h:group_scans(hist[h]['rss'],hist[h]['xy']-origin) for h in floors}
    h_ap=sorted(set(int(a) for h in floors for a in np.flatnonzero((hist[h]['rss']!=100).any(0))))
    print('HISTORY_BEGIN',args.regime,args.floor,args.draw,'APs',len(h_ap),'sources',floors,'hist_rows',[len(hist[h]['ids']) for h in floors],flush=True)
    t0=time.perf_counter();banks={}
    for ix,a in enumerate(h_ap):
        banks[a]=[single_history({'scans':histories[h]},a,B,O,steps=640) for h in floors]
        if ix%25==0:print('HISTORY_AP',ix,'/',len(h_ap),'elapsed',time.perf_counter()-t0,flush=True)
    history_seconds=time.perf_counter()-t0
    ans=[];predictions={}
    for k in KS:
        sub={name:v[:k] for name,v in c['support'].items()}
        s=group_scans(sub['rss'],sub['xy']-origin)
        assert len(s['positions'])==k and np.all(s['stats']['n']==1)
        seen=(s['rss']!=100).any(0)
        for h in floors:seen|=(hist[h]['rss']!=100).any(0)
        aps=np.flatnonzero(seen).tolist()
        for a in aps:
            if a not in banks:banks[a]=[None]*3
        sx=tensor(s['positions'])
        w,trials=weight(banks,s,aps)
        apids=tensor(aps).long()
        full=np.vstack([histories[h]['positions'] for h in floors]+[s['positions']])
        grid_axes=[np.linspace(v.min(),v.max(),int(np.ceil(np.ptp(v)/2))+1) for v in full.T]
        grid=tensor(np.stack(np.meshgrid(*grid_axes),axis=-1).reshape(-1,2))
        Q=c['query']['rss'][:,aps]
        metrics={};ess={}
        for mode in ('gp','seven','nonlinear'):
            log=np.zeros((len(Q),len(grid)),dtype=np.float64)
            min_ess=float('inf')
            for j,a in enumerate(aps):
                p=CorrectedBank(banks[a],w,[h-args.floor for h in floors],mode,beta,net)
                f=field_from_prior(p,O,sx,ap_stats(s,a),grid,261026+10000*args.draw+int(a),defensive_fraction=.1)
                if f['effective_samples'] is not None:min_ess=min(min_ess,float(f['effective_samples']))
                mu,var=f['mean'],f['variance']
                missing=observation_log_prob(tensor(100.),mu,var,O).numpy()
                log+=missing
                nz=np.flatnonzero(Q[:,j]!=100)
                if len(nz):
                    v,indices=np.unique(Q[nz,j],return_inverse=True)
                    lv=observation_log_prob(tensor(v)[:,None],mu,var,O).numpy()-missing
                    log[nz]+=lv[indices]
                if j%40==0:print('SCORE_AP',k,mode,j,'/',len(aps),flush=True)
            assert min_ess>=1000,('low ESS',mode,k,min_ess)
            pred=(tensor(log).softmax(1)@grid).numpy()+origin
            assert np.isfinite(pred).all()
            metrics[mode]=metric(pred,c['query']['xy'])
            ess[mode]=min_ess
            predictions[f'K{k}_{mode}']=pred
            print('EVENT_RESULT',args.regime,args.floor,args.draw,k,mode,metrics[mode],flush=True)
        ans.append({'K':k,'ap_count':len(aps),'weights':w,'support_ids':sub['ids'].tolist(),
            'support_hash':digest(np.asarray(sub['ids'],dtype='<i8').tobytes()),'grid':len(grid),
            'metrics':metrics,'min_ess':ess,
            'wknn_archived':cached_wknn(c,args.regime,args.floor,args.draw,k),
            'weight_fit':{'valid_starts':sum(t['stationary'] for t in trials),'all_starts':len(trials)}})
    args.out.mkdir(parents=True,exist_ok=True)
    key=f'{args.regime}_F{args.floor}_{args.draw}'
    np.savez_compressed(args.out/(key+'.npz'),query_ids=c['query']['ids'],query_xy=c['query']['xy'],**predictions)
    result=dict(regime=args.regime,building=1,floor=args.floor,draw=args.draw,K=list(KS),
        historical_floor_ids=floors,three_history=True,R=1,
        model_source_sha256=NONLINEAR_HASH,affine_source_sha256=AFFINE_HASH,
        manifest_sha256=MANIFEST_HASH,device_design_sha256=PHONE_HASH,
        query_rows_sha256=c['query_rows_sha256'],query_count=len(c['query']['ids']),
        query_positions=len(np.unique(c['query']['xy'],axis=0)),history_fit_seconds=history_seconds,
        total_seconds=time.perf_counter()-start,
        candidate_grid='XY 2 U from visible H and K-prefix S only',
        results=ans,scope='Bounded, one draw per floor/regime, K1/3/5 only. Not a completed R1 curve, not independent generalization.')
    (args.out/(key+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('DONE',key,'seconds',result['total_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('data',type=Path);p.add_argument('out',type=Path)
    p.add_argument('floor',type=int);p.add_argument('regime',choices=['official','phone'])
    p.add_argument('--draw',type=int,default=0)
    a=p.parse_args();assert a.floor in (0,1,2,3) and a.draw==0
    run(a)
