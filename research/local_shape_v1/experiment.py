"""Registered UTS -> Tampere development experiment; see PROTOCOL.md. CPU float64.
Run the train stage without Tampere files; neither stage reads any official test file.
"""
import argparse, hashlib, io, json, os, re, resource, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from data import group_scans
from run import tensor, ap_stats
from multi_history import single_history
from history_bank import BankPrior, fit_two_source_weight
from inference import observation_log_prob, predict_from_posterior
from reference_pmc import setup, sample_reference
from transfer import Prior, Transfer, rbf
from research.uji_core.round2.paper_replay.limit_study.source import ROOT, R

SEED=261102
T=lambda a: torch.as_tensor(a,dtype=torch.float64)
D=ROOT/'research/uji_core/round2/paper_replay/limit_study/receipts'
P=json.loads((ROOT/'research/uji_core/field_parameters.json').read_text())['0']; B,O=P['base'],P['obs']
ARMS=('gp','seven','early','separate'); NETWORK=('early','separate')
SHA={'UTS':'a7cc7d12169be4d6cd9197f5761e463ecdd9e0e798b4f67790ca468d3028785b','TAU':'e2aad829ead7d729c8d5c28ed1b2a8f6d4f60d97cc1827043537addaf93e30ac'}
def digest(b):return hashlib.sha256(b).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rss_mb():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

def raw(file,site):
    b=file.read_bytes();assert digest(b)==SHA['UTS' if site=='UTS' else 'TAU']
    if site=='UTS':
        d=pd.read_csv(io.BytesIO(b)); ww=[s for s in d if re.fullmatch('WAP[0-9]+',s)];assert len(ww)==589
        # Some public CSVs contain a repeated header. Exclude only literal headers, never difficult examples.
        keep=d[ww[0]].astype(str).str.strip().ne(ww[0]);d=d[keep];ids=np.flatnonzero(keep)+1
        r=d[ww].to_numpy(dtype=float);xy=d[['Pos_x','Pos_y']].to_numpy(dtype=float)
        def level(v):
            s=str(v).strip().upper();return -float(s[1:]) if s.startswith('B') else (0. if s in ('G','GROUND') else float(s))
        floor=np.array([level(v) for v in d['Floor_ID']]);assert d['Building_ID'].nunique()==1
    else:
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            def arr(s):
                ns=[n for n in z.namelist() if n.rsplit('/',1)[-1]==site+'_training_'+s+'.csv' and '__MACOSX' not in n];assert len(ns)==1
                return pd.read_csv(io.BytesIO(z.read(ns[0])),header=None).to_numpy(dtype=float)
            r=arr('rss');c=arr('coordinates');xy,floor=c[:,:2],c[:,3];ids=np.arange(1,len(r)+1)
    assert len(r)==len(floor) and np.isfinite(xy).all() and np.isfinite(r).all() and np.all((r==100)|((r>=-130)&(r<=0)))
    return {float(f):dict(r=r[floor==f],xy=xy[floor==f],ids=ids[floor==f]) for f in sorted(set(floor))}

with zipfile.ZipFile(D/'affine-six-with-source.zip') as z:
    fr=json.loads(z.read('source/freeze.json'));assert digest(z.read('source/operator.pt'))==fr['operator_sha256'];BETA=T(fr['mean_only_affine_beta']);assert len(BETA)==7

def groups(x,step):return np.unique(np.floor(x/step).astype('int64'),axis=0,return_inverse=True)
def coords(d,zero,scale):return (d['xy']-zero)*scale

def history_subset(d,zero,scale):
    x=coords(d,zero,scale);cc,g=groups(x,4);g=g.ravel();cells=[np.flatnonzero(g==i) for i in range(len(cc))];cent=np.array([x[ix].mean(0) for ix in cells]);chosen=[int(np.linalg.norm(cent-cent.mean(0),axis=1).argmin())]
    dist=cdist(cent,cent[chosen]).ravel();
    while len(chosen)<min(32,len(cells)):
        dist[chosen]=-1;i=int(np.argmax(dist));chosen.append(i);dist=np.minimum(dist,np.linalg.norm(cent-cent[i],axis=1))
    ix=np.concatenate([v[np.linspace(0,len(v)-1,min(4,len(v)),dtype=int)] for i in chosen for v in [cells[i]]]);return ix,cent[g[ix]]

class Case:
    def __init__(self,floors,target):
        self.f=f=target;ff=sorted(floors);i=ff.index(f);src=ff[i-1],ff[i+1];self.src=src;self.dz=[j-f for j in src];hs=[floors[j] for j in src];hxy=np.vstack([h['xy'] for h in hs]);self.zero=hxy.min(0);self.scale=120/np.linalg.norm(np.ptp(hxy,axis=0));self.hr=[];self.hx=[];self.rawhx=[];self.hids=[]
        for d in hs:
            ix,xx=history_subset(d,self.zero,self.scale);self.hr.append(d['r'][ix]);self.hx.append(xx);self.rawhx.append(coords(d,self.zero,self.scale)[ix]);self.hids.append(d['ids'][ix].tolist())
        count=(np.vstack(self.hr)!=100).sum(0);a=np.lexsort((np.arange(len(count)),-count));self.aps=a[count[a]>0][:32];self.hr=[rr[:,self.aps] for rr in self.hr];self.t=floors[f];self.x=coords(self.t,self.zero,self.scale);self.r=self.t['r'][:,self.aps];h=[group_scans(y,x) for y,x in zip(self.hr,self.hx)];self.banks=[[single_history({'scans':hh},a,B,O,steps=640) for hh in h] for a in range(len(self.aps))]
        assert len(self.aps)>=1
    @torch.no_grad()
    def feat(self,x):
        # history_moments at four positions only to obtain their means; optimize with projection (no variance computation needed there).
        x=T(x);k=B['length']/2;pts=torch.cat([x+u for u in (T([k,0]),T([-k,0]),T([0,k]),T([0,-k]))]);cols=[]
        for hs in self.banks:
            jj=[]
            for h,dz in zip(hs,self.dz):
                z=R.source_tokens(h,dz,x,B);m=20*z[:,0]-75
                if h is None:c=torch.zeros_like(m)
                else:
                    near=B['mean']+h.projection(pts)@(h.m-B['mean']);c=m-near.reshape(4,len(x)).mean(0)
                jj.append(torch.cat((z,c[:,None]/10),1))
            cols.append(torch.stack(jj,1))
        return torch.stack(cols,1) # point,ap,source,feature
    def edges(self):
        cc,g=groups(self.x,4);g=g.ravel();ix=[np.flatnonzero(g==i) for i in range(len(cc))];p=np.array([self.x[v].mean(0) for v in ix]);st=group_scans(self.r,p[g]);# use cell index, not unique order from group_scans
        n=np.stack([(self.r[v]!=100).sum(0) for v in ix]);s=np.stack([np.where(self.r[v]!=100,self.r[v],0).sum(0) for v in ix]);m=s/np.maximum(n,1);d=cdist(p,p);np.fill_diagonal(d,np.inf);pairs=sorted({tuple(sorted((i,int(j)))) for i,row in enumerate(d) for j in np.argsort(row)[:2] if 0<row[j]<=B['length']});assert pairs
        left,right=np.array(pairs).T;ie,ap=np.nonzero((n[left]>=2)&(n[right]>=2));assert len(ie)>0
        z=self.feat(p);a,b=left[ie],right[ie];return (z[a,ap],z[b,ap],T(m[a,ap]-m[b,ap]),len(p))

def seven(z):return 20*(BETA[0]+z[...,:6].mean(-2)@BETA[1:])
class Local(torch.nn.Module):
    def __init__(self):
        super().__init__();self.head=torch.nn.Sequential(torch.nn.Linear(7,32),torch.nn.Tanh(),torch.nn.Linear(32,1));torch.nn.init.zeros_(self.head[-1].weight);torch.nn.init.zeros_(self.head[-1].bias)
    def forward(self,z,mode):
        c=z[...,6:7];c=c if mode=='separate' else c.mean(-2,keepdim=True).expand_as(c);v=torch.cat((z[...,:6],c),-1);return (10*c.squeeze(-1)*torch.tanh(self.head(v).squeeze(-1))).mean(-1)
def adj(z,name,nets):return torch.zeros_like(z[...,0,0]) if name=='gp' else seven(z)+(nets[name](z,name) if name in NETWORK else 0.)
def detected_mean(z,arm,nets):
    mu=(20*z[...,0]-75).mean(-1)+adj(z,arm,nets);v=B['drift']**2+((20*z[...,1])**2).sum(-1)/4;den=(v+O['detect_scale']**2).sqrt();t=(mu-O['beta'])/den;return mu+v/den*torch.exp(-.5*t*t-.5*np.log(2*np.pi)-torch.special.log_ndtr(t))

def process(file,site,out,weights=None):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);t0=time.perf_counter();out.mkdir(parents=True,exist_ok=True);floors=raw(file,site);ff=sorted(floors)[1:-1];
    if site=='UTS' and len(ff)>4:ff=[ff[j] for j in np.linspace(0,len(ff)-1,4,dtype=int)]
    assert ff;cases=[Case(floors,f) for f in ff];edges=[c.edges() for c in cases];t1=time.perf_counter();nets={}
    for name in NETWORK:torch.manual_seed(SEED);nets[name]=Local()
    info={'site':site,'data_sha256':digest(file.read_bytes()),'floors':sorted(floors),'retained_floors':ff,'seven_sha256':fr['operator_sha256'],'caps':[32,32,4],'cache_seconds':t1-t0,'source_cache':[dict(floor=c.f,sources=c.src,scale=c.scale,origin=c.zero.tolist(),H_rows=c.hids,ap_ids=c.aps.tolist(),edges=len(e[2]),cells=e[3]) for c,e in zip(cases,edges)]}
    if weights is None:
        assert site=='UTS';opts={n:torch.optim.Adam(m.parameters(),lr=.001) for n,m in nets.items()};rg=np.random.default_rng(SEED);log=[]
        for step in range(1000):
            aa,bb,yy,_=edges[rg.integers(len(edges))];ix=rg.integers(len(yy),size=256);a,b,y=aa[ix],bb[ix],yy[ix]
            for n in NETWORK:
                err=detected_mean(a,n,nets)-detected_mean(b,n,nets)-y;l=(err/10).square().mean();assert torch.isfinite(l);opts[n].zero_grad();l.backward();opts[n].step()
                if step in (0,99,999):log.append([step,n,l.item()])
            if step==99:info['first100_seconds']=time.perf_counter()-t1
        info['train_seconds']=time.perf_counter()-t1;torch.save({n:m.state_dict() for n,m in nets.items()},out/'frozen.pt');info['checkpoint']=digest((out/'frozen.pt').read_bytes());info['curve']=log;info['parameters_per_arm']=sum(p.numel() for p in nets['early'].parameters())
    else:
        fr1=json.loads((weights/'UTS.json').read_text());assert fr1['checkpoint']==digest((weights/'frozen.pt').read_bytes()) and site!='UTS';info['checkpoint']=fr1['checkpoint'];states=torch.load(weights/'frozen.pt',weights_only=True);[m.load_state_dict(states[n]) for n,m in nets.items()]
        rec=[]
        for c in cases:
            for seed in range(3):rec.append(score(c,seed,nets,out,site));print('scored',site,c.f,seed,flush=True)
        info['events']=rec
    with torch.no_grad():info['contrast_rmse']={n:[float((detected_mean(a,n,nets)-detected_mean(b,n,nets)-y).square().mean().sqrt()) for a,b,y,_ in edges] for n in ARMS}
    info['total_seconds']=time.perf_counter()-t0;info['peak_rss_MiB']=rss_mb();write(out/(site+'.json'),info)

def score(c,seed,nets,out,site):
    from scipy.spatial.distance import cdist
    rg=np.random.default_rng(SEED+seed);_,g=groups(c.x,4);g=g.ravel();select=rg.choice(np.unique(g),5,False);ids=np.array([rg.choice(np.flatnonzero(g==i)) for i in select]);bl=np.floor(c.x/8).astype(int);ban={tuple(p) for p in bl[ids]};qi=np.array([i for i,p in enumerate(bl) if tuple(p) not in ban]);assert len(qi)>0 and len(set(ids)&set(qi))==0
    s=group_scans(c.r[ids],c.x[ids]);sx=T(s['positions']);bounds=np.vstack([*c.hx,sx.numpy()]);ax=[np.linspace(a.min(),a.max(),int(np.ceil(np.ptp(a)/2))+1) for a in bounds.T];grid=T(np.stack(np.meshgrid(*ax),-1).reshape(-1,2));stats=[ap_stats(s,i) for i in range(len(c.aps))];st={k:torch.stack([v[k] for v in stats]) for k in stats[0]};means=[];cov=[];res=[]
    for hs in c.banks:
        mh=[];vh=[]
        for h in hs:
            p=Prior(Transfer('identity'),h,dict(B,drift=0.));m,f=p.features(sx);mh.append(m);vh.append(p.covariance(sx,sx,f,f))
        means.append(torch.stack(mh));cov.append(torch.stack(vh));res.append(rbf(sx,sx,B['drift'],B['length']))
    w,trials=fit_two_source_weight(torch.stack(means),torch.stack(cov),torch.stack(res),st,O);zs=c.feat(sx);zg=c.feat(grid);q=c.r[qi];xy=c.t['xy'][qi];pred={};ess={};records={};t0=time.perf_counter()
    with torch.no_grad():
        for arm in ARMS:
            ads,adg=adj(zs,arm,nets),adj(zg,arm,nets);sc=np.zeros((len(q),len(grid)));miss=np.zeros(len(grid));eff=[]
            for j,hs in enumerate(c.banks):
                p=BankPrior(hs,w,B);sm,sf=p.features(sx);gm,gf=p.features(grid);sm=sm+ads[:,j];gm=gm+adg[:,j];sk=p.covariance(sx,sx,sf,sf);tg,pr,ct,rt,_=setup(sm,sk,stats[j],O);sample,lw,inf=sample_reference(tg,pr,ct,rt,261026+10000*seed+int(c.aps[j]),defensive_fraction=.1);pw=lw.exp();pm=pw@sample;de=sample-pm;f=predict_from_posterior(gm,p.diagonal(grid,gf),p.covariance(grid,sx,gf,sf),sm,sk,dict(mean=pm,cov=(de.T*pw)@de));mu,va=f['mean'],f['variance'];assert torch.isfinite(mu).all() and (va>=0).all();eff.append(float(inf['effective_samples']));ms=observation_log_prob(T(100),mu,va,O).numpy();miss+=ms;seen=np.flatnonzero(q[:,j]!=100)
                if len(seen):vals,inv=np.unique(q[seen,j],return_inverse=True);sc[seen]+=((observation_log_prob(T(vals)[:,None],mu,va,O).numpy()-ms)[inv.ravel()])
            assert min(eff)>=1000;pred[arm]=(T(sc+miss).softmax(1)@grid).numpy()/c.scale+c.zero;ess[arm]=min(eff)
    rr=np.vstack([*c.hr,c.r[ids]]);xx=np.vstack([*c.rawhx,c.x[ids]])/c.scale+c.zero;dd=cdist(np.where(q==100,-110,q),np.where(rr==100,-110,rr));ix=np.argsort(dd,axis=1,kind='stable')[:,:3];ww=1/(np.take_along_axis(dd,ix,axis=1)+1e-8);pred['wknn']=(ww[:,:,None]*xx[ix]).sum(1)/ww.sum(1)[:,None]
    _,siteix,cnt=np.unique(xy,axis=0,return_inverse=True,return_counts=True);siteix=siteix.ravel();blocks,bi=np.unique(bl[qi],axis=0,return_inverse=True);bi=bi.ravel();basew=1/cnt[siteix];roww=basew/np.bincount(bi,weights=basew)[bi]/len(blocks)
    for name,p in pred.items():
        er=np.linalg.norm(p-xy,axis=1);assert np.isfinite(er).all();order=er.argsort();records[name]=dict(mde=float(roww@er),p90=float(er[order][np.searchsorted(roww[order].cumsum(),.9)]),blocks=(np.bincount(bi,weights=er*roww)*len(blocks)).tolist())
    tag=f'{site}_F{c.f:g}_{seed}';np.savez_compressed(out/(tag+'.npz'),qids=c.t['ids'][qi],xy=xy,roww=roww,block=bl[qi],**pred)
    return dict(floor=c.f,seed=seed,sids=c.t['ids'][ids].tolist(),qsha=digest(c.t['ids'][qi].astype('<i8').tobytes()),nq=len(qi),blocks=blocks.tolist(),G=len(grid),w=w,ess=ess,metrics=records,scoring_seconds=time.perf_counter()-t0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('site',choices=['UTS','TIE1','SAH1']);p.add_argument('file',type=Path);p.add_argument('out',type=Path);p.add_argument('--weights',type=Path);a=p.parse_args();process(a.file,a.site,a.out,a.weights)
