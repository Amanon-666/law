"""Can actual single scans distinguish reference sites when given their target signatures?
Only original UJI training-file B1, no validation RSS opened. For each frozen event exclude S coordinates, use its identical AP mask. Per target phone split ALL eligible original scans at that phone-floor's median timestamp (independent of RSS). Keep every exact XY with >=3 distinct pre-cut timestamps and >=1 post-cut timestamp. Require >=8 such sites. Three real first distinct-time scans per site, all strictly before all corresponding test scans. Deduplicate repeated rows per timestamp/site within the phone; log repeat/effective Q counts. Knowledge of target site-list, reference surveys and phone is privileged. Score exact same later phone scans/candidate sites with (1) historic GP at nearest grid site, (2) actual three-scan target templates simple 3-WKNN. No model selection and no fallback to a friendlier split when coverage is inadequate. Not B1 manuscript, not a universal Bayes bound."
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
from .model_ceiling import score,BASE
from ..prior_bridge import load
from research.uji_core.protocol import position_weights

def run(d,fi,out):
    out.mkdir(exist_ok=True,parents=True); p=BASE/'round2/paper_replay/results/replay/replay.json';man={r['key']:r for r in json.loads(p.read_text())['manifest']};obs=json.loads((BASE/'field_parameters.json').read_text())['0']['obs'];train=load(d,'trainingData.csv')
    with (d/'trainingData.csv').open(newline='') as ss:meta=np.array([(int(r['PHONEID']),int(r['TIMESTAMP'])) for r in csv.DictReader(ss)])
    records=[];elig=[]
    for f in [1,2]:
        t=train[f];phone,tm=meta[t['ids']-1].T
        for seed in range(3):
            eid=f'B1F{f}_{seed}';F=np.load(next(fi.rglob(eid+'_field.npz')));ap=F['aps'];lu=dict(zip(t['ids'],range(len(tm))));sr=[lu[i] for i in man[eid+'_internal']['support_1based']];allowed=~np.isin(t['group'],t['group'][sr]);
            for ph in np.unique(phone):
                ids=np.flatnonzero((phone==ph)&allowed);cut=np.median(tm[ids]);refs=[];later=[];sites=[]
                for loc in np.unique(t['group'][ids]):
                    here=ids[t['group'][ids]==loc];_,u=np.unique(tm[here],return_index=True);here=here[u];early=here[tm[here]<cut];late=here[tm[here]>cut]
                    if len(early)>=3 and len(late):
                        sites.append(loc);refs.extend(early[:3]);later.extend(late)
                elig.append(dict(event=eid,phone=int(ph),common_sites=len(sites),n_later=len(later)))
                if len(sites)<8:continue
                refs=np.array(refs);ix=np.array(later);assert tm[refs].max()<tm[ix].min() and not np.intersect1d(t['ids'][refs],t['ids'][ix]).size; X=t['positions'][sites];org=F['origin'];nearest=cdist(X-org,F['grid']).argmin(1);rss=t['rss'][ix][:,ap];r=t['rss'][refs][:,ap];yd=t['xy'][ix];weights=position_weights(yd);lp=score(rss,F['mu'][:,nearest],F['variance'][:,nearest],obs);p=softmax(lp,1)@(X-org)+org
                dr=cdist(np.where(rss==100,-110,rss),np.where(r==100,-110,r),'euclidean');nn=np.argsort(dr,axis=1,kind='stable')[:,:3];w=1/np.maximum(np.take_along_axis(dr,nn,axis=1),1e-8);w/=w.sum(1,keepdims=True);pred=(w[:,:,None]*(t['xy'][refs][nn]-org)).sum(1)+org
                records.append(dict(event=eid,floor=f,seed=seed,phone=int(ph),cut=float(cut),sites=len(sites),reference_rows=len(refs),future_rows=len(ix),history_gp_known_sites=float(weights@np.linalg.norm(p-yd,axis=1)),target_repeat_wknn=float(weights@np.linalg.norm(pred-yd,axis=1)),last_reference=int(tm[refs].max()),first_query=int(tm[ix].min())))
    result={'min_common_sites':8,'n_qualified_event_phones':len(records),'coverage':elig,'records':records,'interpretation':'Only same-phone temporal subset, source list privileged; counts across seeds are repeated design cells.'};(out/'revisit.json').write_text(json.dumps(result,indent=2)+'\n');print('REVISIT',json.dumps(records))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('fields',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.data,a.fields,a.out)
