"""Internal Table-II diagnostic only. Shuffle AP columns independently WITHIN exact location and same phone or within exact location pooled across phones. Preserve every observed RSS marginal including misses, every row count, all maps and legal output grid. Eligibility set held constant (only location-phone groups with >=5 rows); same position weights in all arms, compare real and 12 fixed shuffles. Also capture per-site, detected-count variance vs sum of marginal Bernoulli variances. This is privileged diagnostic use of evaluation labels, not a proposed locator and not an estimate on official singleton positions."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from ..prior_bridge import load
from research.uji_core.protocol import position_weights
from .model_ceiling import score,BASE

def run(d,fi,out):
    out.mkdir(exist_ok=True,parents=True);pp=json.loads((BASE/'field_parameters.json').read_text())['0'];tr=load(d,'trainingData.csv');va=load(d,'validationData.csv');
    with (d/'trainingData.csv').open(newline='') as s: phones=np.array([int(x['PHONEID']) for x in csv.DictReader(s)])
    rows=[]
    for f in (1,2):
        for sd in range(3):
            key=f'B1F{f}_{sd}';ref=np.load(next(fi.rglob(key+'_pred.npz')));z=np.load(next(fi.rglob(key+'_field.npz')));grid=z['grid'];org=z['origin'];mu=z['mu'];v=z['variance'];aps=z['aps'];t=tr[f];ids=ref['internal_ids'];idx=np.searchsorted(t['ids'],ids);assert np.array_equal(t['ids'][idx],ids);xy=t['xy'][idx];rss=t['rss'][idx][:,aps];ph=phones[ids-1];_,grp,counts=np.unique(np.column_stack((xy,ph)),axis=0,return_inverse=True,return_counts=True);keep=counts[grp.ravel()]>=5
            xy=xy[keep];rss=rss[keep];ph=ph[keep];w=position_weights(xy);_,a=np.unique(xy,axis=0,return_inverse=True);_,b=np.unique(np.column_stack((xy,ph)),axis=0,return_inverse=True);a=a.ravel();b=b.ravel();
            def metric(Q):
                result=softmax(score(Q,mu,v,pp['obs']),axis=1)@grid+org;return float(w@np.linalg.norm(result-xy,axis=1))
            real=metric(rss);assert abs(real-float(w@np.linalg.norm(ref['internal_full'][keep]-xy,axis=1)))<1e-7
            shuffled={};rat={}
            for tag,grp in [('same_site_phone',b),('same_site',a)]:
                groups=[np.flatnonzero(grp==i) for i in np.unique(grp)];observed=indep=0.
                for ii in groups:
                    seen=(rss[ii]!=100).astype(float);p=seen.mean(0);observed+=w[ii].sum()*seen.sum(1).var();indep+=w[ii].sum()*np.sum(p*(1-p))
                rat[tag]=dict(observed=observed,independent=indep,ratio=observed/indep if indep else None);ss=[]
                for draw in range(12):
                    rng=np.random.default_rng(457153+1000*f+100*sd+draw);Q=rss.copy()
                    for ii in groups:
                        zq=rss[ii];order=np.argsort(rng.random(zq.shape),axis=0);Q[ii]=np.take_along_axis(zq,order,axis=0)
                    ss.append(metric(Q))
                shuffled[tag]=ss
            _,on=np.unique(ref['official_xy'],axis=0,return_counts=True);rows.append(dict(event=key,nrows=len(xy),nsites=len(np.unique(a)),ncells=len(np.unique(b)),official_sites_repeats5=int((on>=5).sum()),real=real,shuffled=shuffled,detection_variance=rat));print(key,real,{k:np.mean(v) for k,v in shuffled.items()},rat,flush=True)
    summary={'real':float(np.mean([r['real'] for r in rows]))};
    for tag in ['same_site_phone','same_site']:
        summary[tag]=float(np.mean([np.mean(r['shuffled'][tag]) for r in rows])); summary[tag+'_detection_variance_ratio']=float(np.mean([r['detection_variance'][tag]['ratio'] for r in rows]))
    (out/'joint_check.json').write_text(json.dumps(dict(summary=summary,events=rows),indent=2)+'\n');print('JOINT',summary)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data',type=Path);a.add_argument('fields',type=Path);a.add_argument('out',type=Path);p=a.parse_args();run(p.data,p.fields,p.out)
