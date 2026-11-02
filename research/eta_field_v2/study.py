"""UJI field law study. Run from repo root: python -m research.eta_field_v2.study fit DATA OUT, or score DATA OUT F SEED FROZEN_JSON."""
import sys,json,time,io,zipfile,hashlib
from pathlib import Path
import numpy as np
import torch
from research.uji_core.round2.paper_replay.radio_kernel import load
from research.uji_core.round2.paper_replay.replay import hashids
from research.uji_core.protocol import position_weights,HASHES
from source_weight_mixture import field_from_prior
from history_bank import fit_two_source_weight
from transfer import Prior,Transfer,rbf
from multi_history import single_history
from inference import observation_log_prob, infer, predict_from_posterior
from run import tensor,ap_stats
from data import group_scans
from .operator import Law,selftest
ROOT=Path(__file__).resolve().parents[2]; D=ROOT/'research/uji_core/round2/paper_replay';PAR=json.loads((ROOT/'research/uji_core/field_parameters.json').read_text())['0'];B,O=PAR['base'],PAR['obs'];G=[0.,-.25,.25,-.5,.5]
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def banks(h,aa,b):return [[single_history({'scans':s},int(a),b,O,steps=640) for s in h] for a in aa]
def center(t,hs,sr):
    orig=np.vstack([h['positions'] for h in hs]).mean(0);s=group_scans(t['rss'][sr],t['xy'][sr]-orig);h=[group_scans(j['rss'],j['xy']-orig) for j in hs];return orig,s,h

def weight(bs,s,aa):
    x=tensor(s['positions']);M=[];C=[];E=[];ss=[ap_stats(s,int(a)) for a in aa];stats={k:torch.stack([a[k] for a in ss]) for k in ss[0]}
    for hs in bs:
        mm=[];cc=[];active=any(h is not None for h in hs)
        for h in hs:
            p=Prior(Transfer('identity'),h,dict(B,drift=0.));m,f=p.features(x);mm.append(m);cc.append(p.covariance(x,x,f,f))
        M.append(torch.stack(mm));C.append(torch.stack(cc)*(1. if active else 0.));E.append(rbf(x,x,B['drift'] if active else B['amplitude'],B['length']))
    return fit_two_source_weight(torch.stack(M),torch.stack(C),torch.stack(E),stats,O)[0]

def source(data,out):
    start=time.perf_counter();r=load(data,'trainingData.csv',{0,2});records=[]
    for b in (0,2):
        floors=sorted(f for bb,f in r if bb==b);hs=[r[b,floors[0]],r[b,floors[-1]]];count=(np.vstack([v['rss'] for v in hs])!=100).sum(0);aa=np.lexsort((np.arange(len(count)),-count))[:24];assert (count[aa]>0).all();cached=None
        for f in floors[1:-1]:
            t=r[b,f];rng=np.random.default_rng(261102);ids=np.sort(rng.choice(np.flatnonzero(t['stats']['n']>=3),5,False));sr=np.concatenate([rng.choice(np.flatnonzero(t['group']==j),3,False) for j in ids]);orig,s,h=center(t,hs,sr);sx=tensor(s['positions']);cached=banks(h,aa,B) if cached is None else cached;w=weight(cached,s,aa);ix=np.flatnonzero(~np.isin(t['group'],ids));pos,inv=np.unique(t['xy'][ix],axis=0,return_inverse=True);inv=inv.ravel();gx=tensor(pos-orig);ww=position_weights(t['xy'][ix]);results={}
            for mode in ('full','point'):
                vals=[]
                for e in G:
                    av=[]
                    for a,hpair in zip(aa,cached):
                        p=Law(hpair,w,B,eta=e if mode=='full' else 0.,propagate=mode=='full',point=e if mode=='point' else 0.);sm,sf=p.features(sx);sk=p.covariance(sx,sx,sf,sf);post=infer(sm,sk,ap_stats(s,int(a)),O,steps=800);gm,gf=p.features(gx);v=predict_from_posterior(gm,p.diagonal(gx,gf),p.covariance(gx,sx,gf,sf),sm,sk,post);lp=observation_log_prob(tensor(t['rss'][ix,int(a)]),v['mean'][inv],v['variance'][inv],O);av.append(float(-ww@lp.detach().numpy()))
                    vals.append(float(np.mean(av)))
                results[mode]=vals
            records.append(dict(b=b,f=f,aps=aa.tolist(),S=t['ids'][sr].tolist(),Q=hashids(t['ids'][ix]),w=w,NLL=results));print('source',b,f,results,flush=True)
    risk={m:np.mean([np.mean([v['NLL'][m] for v in records if v['b']==b],axis=0) for b in (0,2)],axis=0).tolist() for m in ('full','point')};chosen={m:G[int(np.argmin(v))] for m,v in risk.items()};write(out/'freeze.json',dict(source_sha=HASHES['trainingData.csv'],grid=G,selected=chosen,risk=risk,episodes=records,source_backend='full Gaussian detection VI, target uses original M-PMC',seconds=time.perf_counter()-start));print('FROZEN',chosen,risk)

def target(data,out,f,seed,freeze):
    start=time.perf_counter();fr=json.loads(freeze.read_text());e=fr['selected']['full'];q=fr['selected']['point'];eid=f'B1F{f}_{seed}';manifest={i['key']:i for i in json.loads((D/'results/replay/replay.json').read_text())['manifest']};M=manifest[eid+'_internal'];tall=load(data,'trainingData.csv',{1});t=tall[1,f];hs=[tall[1,0],tall[1,3]];vd=load(data,'validationData.csv',{1})[1,f];lu=dict(zip(t['ids'],range(len(t['ids']))));sr=np.array([lu[k] for k in M['support_1based']]);orig,s,h=center(t,hs,sr);sx=tensor(s['positions']);aa=np.flatnonzero(np.logical_or.reduce([(v['rss']!=100).any(0) for v in [*hs,s]]));assert len(aa)==M['aps'];ps=np.vstack([k['positions'] for k in h]+[s['positions']]);axes=[np.linspace(v.min(),v.max(),int(np.ceil(np.ptp(v)/2))+1) for v in ps.T];grid=tensor(np.stack(np.meshgrid(*axes),axis=-1).reshape(-1,2));ii=np.flatnonzero(~np.isin(t['group'],t['group'][sr]));oi=np.flatnonzero(~(vd['xy'][:,None]==t['positions'][np.unique(t['group'][sr])][None]).all(-1).any(-1));sets=[('internal',t,ii),('official',vd,oi)];outarrays={}
    for name,ds,ix in sets:
        assert hashids(ds['ids'][ix])==manifest[eid+'_'+name]['query_ids_sha256'];outarrays[name+'_ids']=ds['ids'][ix];outarrays[name+'_xy']=ds['xy'][ix]
    with zipfile.ZipFile(D/'limit_study/receipts/affine-six-with-source.zip') as z:meta=json.loads(z.read(eid+'.json'));w=meta['w'];aff=np.load(io.BytesIO(z.read(eid+'.npz')));oldplain={sp:aff[sp+'_plain'] for sp,_,_ in sets}
    bs=banks(h,aa,B);short=dict(B,length=.7*B['length']);ss=banks(h,aa,short);Q=np.vstack([ds['rss'][ix][:,aa] for _,ds,ix in sets]);arms={'plain':(0,False,0,False),'short':(0,False,0,True),'point':(0,False,q,False),'mean':(e,False,0,False),'full':(e,True,0,False)}
    if e==0:arms.update(sensitivity_mean=(.5,False,0,False),sensitivity_full=(.5,True,0,False))
    metrics={};ess={};predcache={};delta={}
    for arm,(eta,full,point,sh) in arms.items():
        key=(eta,bool(full and eta),point,sh)
        if key not in predcache:
            ll=np.zeros((len(Q),len(grid)));missing=np.zeros(len(grid));eff=[]
            for j,(a,hp) in enumerate(zip(aa,ss if sh else bs)):
                p=Law(hp,w,B,eta,full,point,hb=short if sh else B);field=field_from_prior(p,O,sx,ap_stats(s,int(a)),grid,261026+10000*seed+int(a),defensive_fraction=.1);mu,var=field['mean'],field['variance'];eff.append(field['effective_samples']);ms=observation_log_prob(tensor(100.),mu,var,O).numpy();missing+=ms;seen=np.flatnonzero(Q[:,j]!=100)
                if len(seen):
                    vals,inv=np.unique(Q[seen,j],return_inverse=True);tab=observation_log_prob(tensor(vals)[:,None],mu,var,O).numpy()-ms;ll[seen]+=tab[inv.ravel()]
            pr=(tensor(ll+missing).softmax(1)@grid).numpy()+orig;assert np.isfinite(pr).all() and min(eff)>=1000;predcache[key]=(pr,float(min(eff)))
        pred,ess[arm]=predcache[key];off=0;metrics[arm]={}
        for sp,ds,ix in sets:
            p=pred[off:off+len(ix)];off+=len(ix);xy=outarrays[sp+'_xy'];ww=position_weights(xy);err=np.linalg.norm(p-xy,axis=1);ordr=np.argsort(err);metrics[arm][sp]=dict(mde=float(ww@err),p90=float(err[ordr][np.searchsorted(ww[ordr].cumsum(),.9)]));outarrays[sp+'_'+arm]=p
            if arm=='plain':delta[sp]=float(np.abs(p-oldplain[sp]).max())
        print(eid,arm,metrics[arm],flush=True)
    write(out/(eid+'.json'),dict(event=eid,eta=e,point=q,w=w,aps=aa.tolist(),support=M['support_1based'],metrics=metrics,min_ess=ess,plain_max_coordinate_diff=delta,seconds=time.perf_counter()-start,freeze_sha=hashlib.sha256(freeze.read_bytes()).hexdigest()));np.savez_compressed(out/(eid+'.npz'),**outarrays)
if __name__=='__main__':
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);print('operator self-test',selftest(),flush=True);mode,dest,output=sys.argv[1:4];out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if mode=='fit':source(Path(dest),out)
    else:target(Path(dest),out,int(sys.argv[4]),int(sys.argv[5]),Path(sys.argv[6]))
