"""Exposed F2 post-hoc diagnostic. No fitting, model loading or cache writes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, json, hashlib, datetime, socket
from pathlib import Path
ROOT=Path('/home/panyushuo/projects/panyushuo/ReFi-UJI-Core-20261031')
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(OUT.parent/'xacq_selfcal_claude_20261111')]
import numpy as np
from scipy.spatial import ConvexHull
from scipy.special import logsumexp, softmax
from scipy.integrate import quad_vec
from frozen_hurdle_translator import PreparedFrozenHurdle, BETA, EPS, OFFSETS, U_DET, U_MISS
from selfcal import apply_calibration
from ipin2025_observation_data import load_training, metric_xy, read_query_stream, rss_matrix, REF_LAT, REF_LON, match_prediction_times
sys.path.append('/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/Law-Probabilistic-Transfer-20261023')
from public_corpus.adapter import parse_log
assert socket.gethostname()=='ray-web'
inputs={}
def bind(path,sha=None):
    path=Path(path); digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha: assert digest==sha, str(path)
    inputs[str(path)]=digest
    return path
def read(path,sha=None): return json.loads(bind(path,sha).read_text())
def load(record):
    with np.load(bind(record['path'],record['sha256'])) as z: return {k:z[k].copy() for k in z.files}
BASE=ROOT/'results/ipin2025_confirmation_20261110'
SELF=OUT.parent/'xacq_selfcal_claude_20261111/results'
assets=read(ROOT/'results/ipin2025_history_assets_20261110/manifest.json')
freeze=read(ROOT/'results/ipin2025_history_assets_20261110/freeze.json')
for p,sha in freeze['inputs'].items(): bind(p,sha)
training=load_training('/home/panyushuo/datasets/ipin_t3_2025/2025_IPIN_Competition_Track03',parse_log=parse_log)
tx=metric_xy(training['floors'][2]); hull=ConvexHull(tx); vertices=tx[hull.vertices]
def geometry(points):
    points=np.atleast_2d(points)
    inside=(points@hull.equations[:,:2].T+hull.equations[:,2]<=1e-8).all(1)
    a=vertices; b=np.roll(a,-1,axis=0); edge=b-a
    t=np.clip(((points[:,None]-a)*edge).sum(2)/(edge**2).sum(1),0,1)
    dist=np.linalg.norm(points[:,None]-(a+t[:,:,None]*edge),axis=2).min(1)
    return inside,np.where(inside,0,dist)
contexts=read(BASE/'contexts_complete.json'); oldmanifest=read(BASE/'predictions.json')
scores=read(SELF/'ipin/all/scores.json'); newmanifest=read(SELF/'ipin/all/predictions.json')
cal=read(SELF/'ipin_calibrations.json')['2']
hist={f:load(assets['histories'][str(f)]) for f in range(3,7)}
support=load(assets['support']['2'])
streams={r['id']:read_query_stream(bind(r['query_path'],r['query_sha256'])) for r in oldmanifest['logs']}
truths={r['id']:np.loadtxt(bind(r['gt_path'],r['gt_sha256']),delimiter=',',ndmin=2) for r in oldmanifest['logs']}
def terms(h,q):
    d=q!=100; scale=BETA/len(q)
    per=np.stack([np.logaddexp(np.log(1-EPS)+np.log(h.p[:,d])-.5*np.log(2*np.pi*h.s2[:,d])-.5*(q[d][None]-o-h.m[:,d])**2/h.s2[:,d],np.log(EPS*U_DET)) for o in OFFSETS])
    miss=np.log((1-EPS)*(1-h.p)+EPS*U_MISS)*scale
    dt=(logsumexp(per.sum(2),axis=0)-np.log(len(OFFSETS)))*scale
    mt=miss[:,~d].sum(1)
    return d,per,miss,dt,mt
rows=[]; summaries=[]; tops=[]; candidate_tables=[]
for k in (1,3,5):
    record=next(r for r in contexts['contexts'] if r['floor']==2 and r['K']==k)
    context=read(record['path'],record['sha256']); h0=PreparedFrozenHurdle(**load(context['hurdle']))
    h=apply_calibration(h0,cal,k,'all'); aps=np.array(context['ap_ids'])[h.aps]
    sxy=support['xy'][:k]; nh=len(h.cand_xy)-k
    expected=np.unique(np.concatenate([v['xy'] for v in hist.values()]),axis=0)
    assert np.array_equal(expected,h.cand_xy[:nh])
    assert np.array_equal(np.unique(sxy,axis=0),h.cand_xy[nh:])
    sources=[]
    for i,xy in enumerate(h.cand_xy):
        source=[f'F{f}' for f,v in hist.items() if np.any(np.all(np.isclose(v['xy'],xy,rtol=0,atol=1e-8),axis=1))] if i<nh else ['S']
        sources.append(source)
    ci,cd=geometry(h.cand_xy); si,sd=geometry(sxy)
    masks={'all':np.ones(len(ci),bool),'S30':np.linalg.norm(h.cand_xy[:,None]-sxy,axis=2).min(1)<=30,'S60':np.linalg.norm(h.cand_xy[:,None]-sxy,axis=2).min(1)<=60,'oracle_hull':ci}
    selected=[p for p in scores['points'] if p['floor']==2 and p['K']==k and p['covered']]
    assert len(selected)==22
    apraw=[]; kr=[]
    for point in selected:
        log=point['log_id']; idx=point['prediction_index']; stream=streams[log]
        assert idx==match_prediction_times(stream['prediction_times'],[point['time']])[0]
        qi=int(stream['snapshot_indices'][idx]); q=rss_matrix([stream['snapshots'][qi]],context['ap_ids'])
        gt=truths[log][point['gt_row']]; assert gt[0]==point['time'] and gt[3]==2
        xy=np.array([(gt[1]-REF_LON)*111320*np.cos(np.radians(REF_LAT)),(gt[2]-REF_LAT)*111320])
        pred=h.predict(q); l=pred['log_position_mass'][0]; mode=int(l.argmax())
        dist=np.linalg.norm(h.cand_xy-xy,axis=1); nearest=np.flatnonzero(np.isclose(dist,dist.min(),atol=1e-10,rtol=0)); correct=int(nearest[l[nearest].argmax()])
        saved=load(next(r for r in newmanifest['predictions'] if r['floor']==2 and r['K']==k and r['log_id']==log))
        assert np.allclose(pred['projected'][0],saved['hurdle'][idx],atol=1e-8,rtol=0)
        r=dict(K=k,log=log,gt_row=point['gt_row'],time=point['time'],prediction_index=idx,snapshot_index=qi,truth=xy.tolist(),nearest_index=correct,nearest_distance=float(dist.min()),MAP_index=mode,MAP_sources=sources[mode],decomposition={})
        for label,pos in [('truth',xy),('MAP',h.cand_xy[mode]),('mean',pred['projected'][0]),('GP',saved['GP'][idx]),('GP_candidates',saved['GP_candidates'][idx])]:
            ins,ds=geometry(pos); r[label+'_xy']=pos.tolist(); r[label+'_inside']=bool(ins[0]); r[label+'_hull_distance']=float(ds[0]); r[label+'_error']=float(np.linalg.norm(pos-xy))
        for label,field in [('before',h0),('after',h)]:
            pp=field.predict(q); lm=pp['log_position_mass'][0]; mm=int(lm.argmax()); cc=int(nearest[lm[nearest].argmax()])
            d,per,miss,dt,mt=terms(field,q[0,field.aps]); delta=dt[cc]-dt[mm]+mt[cc]-mt[mm]
            assert abs(delta-(lm[cc]-lm[mm]))<1e-9
            r['decomposition'][label]=dict(correct=cc,MAP=mm,detected_correct=float(dt[cc]),detected_MAP=float(dt[mm]),missing_correct=float(mt[cc]),missing_MAP=float(mt[mm]),detected_delta=float(dt[cc]-dt[mm]),missing_delta=float(mt[cc]-mt[mm]),total_delta=float(delta),nearest_rank=int(1+(lm>lm[cc]).sum()))
            if label=='after':
                # Aumann-Shapley path attribution of the shared offset logsumexp.
                diff=per[:,cc]-per[:,mm]; start=per[:,mm].sum(1); slope=diff.sum(1)
                weights,_=quad_vec(lambda t:softmax(start+t*slope),0,1,epsabs=1e-11,epsrel=1e-11)
                ad=np.zeros(len(aps)); ad[d]=weights@diff*BETA/len(aps)
                am=np.zeros(len(aps)); am[~d]=miss[cc,~d]-miss[mm,~d]
                assert abs(ad.sum()-(dt[cc]-dt[mm]))<1e-8
                assert abs(am.sum()-(mt[cc]-mt[mm]))<1e-8
                apraw.append(dict(detected=ad,missing=am,p_truth=h.p[cc],p_MAP=h.p[mm],query_detected=d))
        for name,mask in masks.items():
            mass=softmax(l[mask]); pos=mass@h.cand_xy[mask]; mi=np.flatnonzero(mask)[l[mask].argmax()]
            r[name+'_restricted_mean_xy']=pos.tolist(); r[name+'_restricted_mean_error']=float(np.linalg.norm(pos-xy)); r[name+'_restricted_MAP_error']=float(dist[mi]); r[name+'_retained_mass']=float(np.exp(l[mask]).sum())
        rows.append(r); kr.append(r)
    srss=np.full((k,len(aps)),100.)
    scol={ap:i for i,ap in enumerate(support['ap_ids'])}
    for a,ap in enumerate(aps):
        if ap in scol: srss[:,a]=support['rss'][:k,scol[ap]]
    rates={}
    for f in range(2,7): rates[f]=(rss_matrix(training['floors'][f],aps)!=100).mean(0)
    hrates={f:np.array([(v['rss'][:,list(v['ap_ids']).index(ap)]!=100).mean() if ap in v['ap_ids'] else 0. for ap in aps]) for f,v in hist.items()}
    np.savez_compressed(OUT/f'AP_raw_K{k}.npz',ap_ids=aps,**{name:np.stack([r[name] for r in apraw]) for name in apraw[0]},S_rss=srss,**{f'F{f}_all_training_detection_rate':v for f,v in rates.items()},**{f'F{f}_H_cache_detection_rate':v for f,v in hrates.items()})
    for term in ('missing','detected'):
        values=np.stack([r[term] for r in apraw]); mean=values.mean(0)
        # Largest absolute mean contribution; retain sign, plus adverse rank.
        order=np.argsort(-np.abs(mean),kind='stable')[:20]
        for rank,a in enumerate(order,1):
            tops.append(dict(K=k,term=term,rank=rank,ap=str(aps[a]),delta=float(mean[a]),absolute_mean=float(np.abs(values[:,a]).mean()),training_rates={str(f):float(rates[f][a]) for f in rates},H_rates={str(f):float(hrates[f][a]) for f in hrates},S_detected=(srss[:,a]!=100).tolist(),p_truth=float(np.mean([r['p_truth'][a] for r in apraw])),p_MAP=float(np.mean([r['p_MAP'][a] for r in apraw])),query_detected_count=int(sum(r['query_detected'][a] for r in apraw))))
    summaries.append(dict(K=k,n=len(kr),candidates=len(ci),candidate_inside=int(ci.sum()),S_xy=sxy.tolist(),S_inside=si.tolist(),S_hull_distance=sd.tolist(),candidate_mask_counts={n:int(m.sum()) for n,m in masks.items()},means={name:float(np.mean([r[name] for r in kr])) for name in kr[0] if name.endswith('_error') or name.endswith('_hull_distance')},inside_counts={name:sum(r[name+'_inside'] for r in kr) for name in ('truth','MAP','mean','GP','GP_candidates')},decomposition={phase:{term:float(np.mean([r['decomposition'][phase][term] for r in kr])) for term in ('detected_delta','missing_delta','total_delta','nearest_rank')} for phase in ('before','after')},missing_overrides={phase:sum(r['decomposition'][phase]['detected_delta']>0 and r['decomposition'][phase]['total_delta']<0 for r in kr) for phase in ('before','after')}))
    candidate_tables.append(dict(K=k,xy=h.cand_xy.tolist(),sources=sources,inside=ci.tolist(),distance=cd.tolist()))
    print('completed K',k, summaries[-1]['means']['mean_error'],flush=True)
for name in ('frozen_hurdle_translator.py','ipin2025_observation_data.py'): bind(ROOT/name)
bind(OUT.parent/'xacq_selfcal_claude_20261111/selfcal.py'); bind(__file__)
result=dict(status='exposed post-hoc diagnostic only',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),calibration=cal,hull_vertices=vertices.tolist(),training_counts=training['floor_metadata'],summaries=summaries,top_APs=tops,rows=rows,candidates=candidate_tables,inputs=inputs)
(OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
