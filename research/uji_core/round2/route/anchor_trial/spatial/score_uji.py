"""Out-of-source transfer of one library-fitted anchor interpolation; no fitting.

Use --preflight to verify manifests only. Scoring requires a reachable immutable
Git preregistration commit and both pre-issued SHA-256s. Final draws 0–9 and
10–49 are reported independently. No prospective UJI route/external data read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, softmax

FIXED=(0.,.25,.5,.75,1.)
EPS=1e-9


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(repo):
    sys.path.insert(0,str(repo))
    global BUDGETS,SEEDS,read_floors,position_weights,local_rows,group_ids,GapReference,order,vote,squared,build,historical_prior,fit_map
    from research.uji_core.protocol import BUDGETS,SEEDS,read_floors,position_weights
    from research.uji_core.smooth_scale import local_rows
    from research.uji_core.next.paired_map_wknn import group_ids
    from research.uji_core.next.external_floor_wknn import GapReference
    from research.uji_core.inverse import order,vote,squared
    from research.uji_core.next.anchor_measure import build
    from refi import historical_prior,fit_map


def manifests(a):
    old= json.loads((a.old/'manifest.json').read_text())['episodes']
    old_devices={(v['building'],v['floor']):v for v in json.loads((a.old/'device_stress_design.json').read_text())['events']}
    new=json.loads((a.fresh/'freeze.json').read_text());assert new['new_draws']==list(range(10,50))
    merged=[]
    for job in new['jobs']:
        b,f,mode=job['building'],job['floor'],job['mode']
        if mode=='official':
            matches=sorted((v for v in old if (v['building'],v['floor'])==(b,f)),key=lambda v:v['draw'])
            assert [v['draw'] for v in matches]==list(range(10)) and all(v['query_ids']==job['qids'] for v in matches)
            original=[v['support_order'] for v in matches]
        else:
            entry=old_devices[b,f];assert entry['phone']==job['device'] and entry['query_ids']==job['qids']
            original=entry['support_orders']
        fits=list(original)+list(job['orders']);assert len(fits)==50 and len({tuple(r) for r in fits})==50 and all(len(set(r))==20 for r in fits)
        merged.append(({k:v for k,v in job.items() if k!='orders'},fits))
    return merged,new


def input_guard(a):
    if len(a.freeze_commit)!=40 or any(c not in '0123456789abcdef' for c in a.freeze_commit):raise ValueError('Need permanent Git freeze SHA')
    rev=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','--verify',a.freeze_commit+'^{commit}'],text=True).strip()
    if rev!=a.freeze_commit:raise ValueError('Freeze Git commit unavailable')
    for file,expected in [(a.params,a.params_sha),(a.knn_params,a.knn_sha)]:
        if digest(file)!=expected:raise ValueError(f'Frozen input does not match {file.name}')
    if a.freeze_receipt:
        data=subprocess.check_output(['git','-C',str(a.repo),'show',a.freeze_commit+':'+a.freeze_receipt])
        if not (a.params_sha.encode() in data or hashlib.sha256(data).hexdigest()==a.params_sha):raise ValueError('Provided commit receipt does not cover learned fit')


def log_score(seen,r,mu):
    v=mu+75.
    # Gaussian query-only constant cancels; original missing likelihood retained.
    return (r@v.T-.5*seen@(v*v).T)/144.+(1.-seen)@log_ndtr((-85.-mu)/12.).T


def estimates(amap,q,ids,settings,learned,scalar,oldmapk,libmapk,liborigk,draw,k):
    r=q['rss'][:,amap.visible];seen=(r!=100).astype(float);r75=np.where(r!=100,r.astype(float)+75.,0.);hr=log_score(seen,r75,amap.historical);w=position_weights(q['xy']);x=amap.xy-amap.xy[0]
    predcache={};scores={}
    for name,g in settings.items():
        if g not in predcache:
            anchor=amap.posterior if g==1. else amap.raw+g*(amap.posterior-amap.raw)
            logits=np.hstack([hr,log_score(seen,r75,anchor)]);pred=softmax(logits,axis=1)@x+amap.xy[0]
            predcache[g]=float(w@np.linalg.norm(pred-q['xy'],axis=1))
        scores[name]=predcache[g]
    for tag,g,neighbors in [('same_forecast_map_wknn',learned,oldmapk),('library_same_forecast_wknn',learned,libmapk),('library_raw_anchor_map_wknn',0.,liborigk),('old_same_map_wknn_check',0.,oldmapk)]:
        anchor=amap.posterior if g==1. else amap.raw+g*(amap.posterior-amap.raw)
        m=np.vstack([amap.historical,anchor]);dist=squared(np.where(r==100,0.,r.astype(float)+110.),m+110.)
        ix=order(dist,ids,neighbors);pred=vote(np.take_along_axis(dist,ix,1),x[ix],r.shape[1])[0]+amap.xy[0]
        scores[tag]=float(w@np.linalg.norm(pred-q['xy'],axis=1))
    if draw in (0,10) and k in (1,20):
        for name,g in [('per_K',learned),('scalar',scalar),('fixed0.0',0.),('fixed1.0',1.)]:
            pred=amap.predict(q['rss'],g,False);assert np.isclose(w@np.linalg.norm(pred-q['xy'],axis=1),scores[name],rtol=0.,atol=1e-8)
    return scores


def read_anchors(a,stage,drawset):
    ix=['building','floor','draw','K']
    if drawset=='new':
        d=pd.read_csv(a.fresh/(stage+'.csv')).set_index(ix).rename(columns={'public':'parent','floor_wknn':'strong_wknn','mapped_wknn':'old_same_map_wknn'})
    else:
        d=pd.read_csv(a.paired/f'paired_map_{stage}.csv').set_index(ix).rename(columns={'field_tobit':'parent','old_wknn_gap_fit':'strong_wknn','field_wknn':'old_same_map_wknn'})
    return d[['parent','strong_wknn','old_same_map_wknn']].sort_index()


def run(a):
    setup(a.repo);a.out.mkdir(parents=True,exist_ok=True);jobs,design=manifests(a);oldmapk={int(k):v for k,v in design['map_k'].items() if k.isdigit()}
    if a.preflight:
        print(json.dumps({'jobs':len(jobs),'draws':50,'old_count':13*10*5,'new_count':13*40*5,'fresh_design_sha256':digest(a.fresh/'freeze.json'),'map_k':oldmapk,'scored':False}));return
    input_guard(a);cfg=json.loads(a.params.read_text());kf=json.loads(a.knn_params.read_text())
    gains={int(k):float(g) for k,g in cfg['g_by_K'].items()};scalar=float(cfg['g_scalar'])
    if set(gains)!=set(BUDGETS) or min([*gains.values(),scalar])<0 or max([*gains.values(),scalar])>1:raise ValueError('Invalid weights')
    libmap=int(kf['k_forecast_map']);liborig=int(kf['k_original_map']);lrep=kf['direct_rep'];lk=int(kf['k_direct']);la=float(kf['direct_alpha'])
    if max(libmap,liborig,lk)>15 or min(libmap,liborig,lk)<1 or lrep not in ('raw','prototype'):raise ValueError('Unrecognized source WKNN setting')
    train=read_floors(a.data,'trainingData.csv');official=read_floors(a.data,'validationData.csv');ph=pd.read_csv(a.data/'trainingData.csv',usecols=['PHONEID']).PHONEID.to_numpy();rows=[];checks=[]
    for j,sg in jobs:
        b,f,mode,dev=j['building'],j['floor'],j['mode'],j['device'];t=train[b,f];hs=[];gaps=[]
        for (bb,ff),h in sorted(train.items()):
            if bb!=b or ff==f:continue
            keep=np.ones(len(h['ids']),bool) if dev is None else ph[h['ids']]!=dev
            if keep.any():hs.append({key:v[keep] for key,v in h.items()});gaps.append(abs(ff-f))
        src=official[b,f] if mode=='official' else t;q={key:v[local_rows(src,j['qids'])] for key,v in src.items()};qweights=position_weights(q['xy']);sr=local_rows(t,np.array(sg))
        if dev is not None and ((ph[t['ids'][sr]]==dev).any() or not (ph[q['ids']]==dev).all()):raise ValueError('Device isolation failed')
        bank=GapReference(hs,gaps,lrep);matched=bank.query(q['rss'])
        hp,hxy=historical_prior(hs,10.,.5,gaps);hm=np.logical_or.reduce([(h['rss']!=100).any(0) for h in hs]);xh,hids=group_ids(np.vstack([h['xy'] for h in hs]),np.concatenate([h['ids'] for h in hs]));assert np.array_equal(xh,hxy)
        for draw,sd in enumerate(sr):
            for k in BUDGETS:
                s={key:v[sd[:k]] for key,v in t.items()};am=build((hp,hxy,hm),s);sx,sids=group_ids(s['xy'],s['ids']);assert np.array_equal(am.xy[len(hxy):],sx)
                ids=np.r_[hids,sids];g=gains[k];settings={**{f'fixed{v}':v for v in FIXED},'per_K':g,'scalar':scalar}
                model=estimates(am,q,ids,settings,g,scalar,oldmapk[b],libmap,liborig,draw,k)
                direct=bank.event(matched,s,alphas=(la,))[la,lk];model['library_direct_wknn']=float(qweights@np.linalg.norm(direct-q['xy'],axis=1))
                rows.append(dict(mode=mode,building=b,floor=f,draw=draw,K=k,**model))
        print('scored',mode,b,f,flush=True)
    r=pd.DataFrame(rows);summary={};full_tiers=[];builds=[];floors=[]
    for mode in ['official','phone']:
        for group,draws in [('old',list(range(10))),('new',list(range(10,50)))]:
            d=r[(r['mode']==mode)&r.draw.isin(draws)].drop(columns='mode').set_index(['building','floor','draw','K']).sort_index();old=read_anchors(a,mode,group)
            if not d.index.equals(old.index):raise ValueError('Archived paired IDs differ')
            d=d.join(old);err=float(abs(d['fixed0.0']-d.parent).max());kerr=float(abs(d.old_same_map_wknn-d.old_same_map_wknn_check).max());assert max(err,kerr)<1e-8
            checks.append(dict(stage=mode,orders=group,max_error=err,max_knn_error=kerr));d.to_csv(a.out/f'{mode}_{group}.csv');rec,tiers,br,fl=analyze(d,mode,group,draws);summary[mode+'_'+group]=rec;full_tiers+=tiers;builds+=br;floors+=fl
    pd.DataFrame(full_tiers).to_csv(a.out/'contrasts.csv',index=False);pd.DataFrame(builds).to_csv(a.out/'building_K.csv',index=False);pd.DataFrame(floors).to_csv(a.out/'floor_K.csv',index=False);pd.DataFrame(checks).to_csv(a.out/'parity.csv',index=False)
    summary['preregistered_commit']=a.freeze_commit;summary['source_gain_sha256']=a.params_sha;summary['source_knn_sha256']=a.knn_sha;summary['orders_sha256']=digest(a.fresh/'freeze.json');summary['scope']='Split 0–9 vs 10–49 reused fixed Q/histories; paired support-order uncertainty only. No Q-drawn tuning or target hyperparameter selection.'
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print('replay complete',flush=True)


def analyze(d,mode,group,draws):
    m='per_K';cols=[m,'scalar',*[f'fixed{v}' for v in FIXED],'parent','strong_wknn','old_same_map_wknn','same_forecast_map_wknn','library_same_forecast_wknn','library_raw_anchor_map_wknn','library_direct_wknn']
    pred=d[cols];n=len(draws);R=5000 if n==10 else 10000;rng=np.random.default_rng(20261101 if n==10 else 261101)
    ordered=pred.to_numpy().reshape(13,n,5,len(cols));b=np.repeat(np.arange(3),[4,4,5]);floor=ordered.mean(1);B=np.stack([floor[b==i].mean(0) for i in range(3)]);midx=cols.index(m)
    reps=[]
    if n==10:allweights=rng.multinomial(n,np.ones(n)/n,size=(R,13)).astype(float)/n
    for i in range(13):
        weights=allweights[:,i] if n==10 else rng.multinomial(n,np.ones(n)/n,size=R).astype(float)/n
        reps.append((weights@ordered[i].reshape(n,-1)).reshape(R,5,len(cols)))
    boot=np.stack([np.mean([reps[k] for k in np.flatnonzero(b==i)],axis=0) for i in range(3)],axis=1);macro=B.mean((0,1));tiers=[];brows=[];frows=[];crit={}
    def ci(a):return np.quantile(a,[.025,.975]).tolist()
    for j,name in enumerate(cols):
        if name==m:continue
        diff=B[:,:,midx]-B[:,:,j];sample=boot[:,:,:,midx]-boot[:,:,:,j];lo,hi=ci(sample.mean((1,2)));crit[name]=(float(diff.mean()),hi)
        tiers.append(dict(mode=mode,draw_set=group,reference=name,model_MDE=macro[midx],MDE=macro[j],delta=diff.mean(),conditional_lo=lo,conditional_hi=hi,
              building_K_wins=(diff < -EPS).sum(),building_K_ties=(abs(diff)<=EPS).sum(),floor_K_wins=(floor[:,:,midx]-floor[:,:,j]<-EPS).sum(),floor_K_ties=(abs(floor[:,:,midx]-floor[:,:,j])<=EPS).sum(),draw_K_wins=(ordered[:,:,:,midx]-ordered[:,:,:,j]<-EPS).sum()))
    offsets=np.r_[0,4,8];
    for i in range(13):
        for ki,k in enumerate(BUDGETS):frows.append(dict(mode=mode,draw_set=group,building=int(b[i]),floor=int(i-offsets[b[i]]),K=k,**dict(zip(cols,floor[i,ki]))))
    for i in range(3):
        for ki,k in enumerate(BUDGETS):
            row=dict(mode=mode,draw_set=group,building=i,K=k,**dict(zip(cols,B[i,ki])))
            for j,name in enumerate(cols):
                if name==m:continue
                row[name+'_delta']=B[i,ki,midx]-B[i,ki,j];row[name+'_lo'],row[name+'_hi']=ci(boot[:,i,ki,midx]-boot[:,i,ki,j])
            if ki:
                row['adjacent']=B[i,ki,midx]-B[i,ki-1,midx];row['adjacent_lo'],row['adjacent_hi']=ci(boot[:,i,ki,midx]-boot[:,i,ki-1,midx])
            brows.append(row)
    control=['scalar',*[f'fixed{v}' for v in FIXED]];knn=['strong_wknn','old_same_map_wknn','same_forecast_map_wknn','library_same_forecast_wknn','library_raw_anchor_map_wknn','library_direct_wknn']
    envelope=B[:,:,midx]-np.min(np.stack([B[:,:,cols.index(k)] for k in knn]),axis=0)
    res=dict(n=len(d),macro=dict(zip(cols,macro)),A_building_decreases=int((np.diff(B[:,:,midx],axis=1)<-EPS).sum()),A_floor_decreases=int((np.diff(floor[:,:,midx],axis=1)<-EPS).sum()),A_draw_decreases=int((np.diff(ordered[:,:,:,midx],axis=2)<-EPS).sum()),
        B_all_available=int((envelope<=EPS).sum()),B_worst=float(envelope.max()),C_all_point=all(crit[x][0]<-EPS for x in control),C_all_upper=all(crit[x][1]<0 for x in control),limiter=max(control,key=lambda x:crit[x][0]))
    return res,tiers,brows,frows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','data','old','fresh','paired','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    for k in ('params','knn-params'):p.add_argument('--'+k,type=Path)
    for k in ('params-sha','knn-sha','freeze-commit','freeze-receipt'):p.add_argument('--'+k,default='')
    run(p.parse_args())
