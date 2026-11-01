"""Learn field fractions using only independent 2016 UJI library visits.

At fixed K, T=raw + g(K)*(posterior-raw). Minimize squared dBm error to
later same-coordinate *observed-space* RSS means (-110 for a missed AP).
No position prediction or learned dependence on local measurements.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_baselines.tpm_adapter import dbm
from residual_probe.episode_rows import nested_rows
from research.uji_core.next import anchor_measure as anchors
from research.uji_core.next import longitudinal as library
from research.uji_core.next.external import METRES_PER_UJI_UNIT as UNIT
from research.uji_core.protocol import BUDGETS, SEEDS


FIXED = {'raw': 0., 'quarter': .25, 'half': .5, 'three_quarters': .75, 'field': 1.}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, record):
    path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def old_pairs(archive):
    """One first trn file per physical day, adjacent pairs without date reuse."""
    visits={}
    for month in range(1,8):
        for num in range(1,16 if month==1 else 2):
            data=library.campaign(archive,month,'trn',num)
            days=np.unique(data['times']//1000000000)
            if len(days)!=1 or not str(days[0]).startswith('2016'):
                raise ValueError('Training visit not on one date in 2016')
            day=str(days[0])
            visits.setdefault(day,dict(date=day,month=month,number=num,
                n=len(data['rss']),first=str(data['times'].min()),last=str(data['times'].max())))
    chronology=sorted(visits)
    pairs=[(visits[a],visits[b]) for a,b in zip(chronology[::2],chronology[1::2])]
    used=[v['date'] for pair in pairs for v in pair]
    if len(pairs)!=7 or len(set(used))!=14 or max(used)>='20170000':
        raise ValueError('Changed seven source-only disjoint date pairs')
    return pairs,sorted(set(visits)-set(used))


def load(archive,visit):
    return {f:library.on_floor(archive,visit['month'],'trn',visit['number'],f) for f in (3,5)}


def sufficient(early,late,pair_id,stage):
    records=[];draws=[]
    for floor,other in ((3,5),(5,3)):
        s=early[floor]
        prior=anchors.history([early[other]],scale=UNIT)
        positions,counts=np.unique(s['xy'],axis=0,return_counts=True)
        if len(positions)!=24 or not np.all(counts==12):
            raise ValueError('Changed training survey grid')
        means={}
        for phone,future in late.items():
            v=future[floor]
            pts,inv,qcount=np.unique(v['xy'],axis=0,return_inverse=True,return_counts=True)
            np.testing.assert_array_equal(positions,pts)
            if not np.all(qcount==12) or not max(x['times'].max() for x in early.values())<v['times'].min():
                raise ValueError('Noncausal or incomplete repeat measurements')
            if set(s['ids'])&set(v['ids']):raise ValueError('Overlapping samples')
            filled=dbm(v['rss'])
            means[phone]={tuple(x):filled[inv.ravel()==j].mean(0) for j,x in enumerate(pts)}
        for seed in SEEDS:
            key=(f'lib-anchor-repeat-v1/{floor}/{seed}' if stage=='repeat' else
                 f'lib-anchor-learning-2016-v1/{pair_id}/{floor}/{seed}')
            order=nested_rows(s['xy'],range(len(s['rss'])),BUDGETS,0,key)['support_rows'][20]
            draws.append(dict(pair=pair_id,floor=floor,draw=seed,S_order=s['ids'][order].tolist()))
            for k in BUDGETS:
                support={a:s[a][order[:k]] for a in ('rss','xy')}
                field=anchors.build(prior,support,scale=UNIT)
                xy,idx=np.unique(support['xy'],axis=0,return_index=True)
                np.testing.assert_allclose(xy,field.xy[-k:],atol=1e-11,rtol=0)
                np.testing.assert_array_equal(field.raw,dbm(support['rss'][idx][:,field.visible]))
                d=field.posterior-field.raw
                for phone,positions_mean in means.items():
                    y=np.stack([positions_mean[tuple(x)] for x in xy])[:,field.visible]-field.raw
                    # Per-site observations; divide by the number of allowed APs before any pooling.
                    for i,pt in enumerate(xy):
                        records.append(dict(pair=pair_id,phone=phone,floor=floor,draw=seed,K=k,x=pt[0],y=pt[1],
                           n_ap=int(field.visible.sum()),a=float(np.mean(d[i]*y[i])),b=float(np.mean(d[i]**2)),
                           c=float(np.mean(y[i]**2))))
    return pd.DataFrame(records),draws


def score(frame,fit):
    rows=[]
    for (phone,k),group in frame.groupby(['phone','K']):
        ev=group.groupby(['pair','floor','draw'])[['a','b','c']].mean()
        a,b,c=ev.mean()
        for arm,g in {**FIXED,'pooled':fit['g_scalar'],'by_K':fit['g_by_K'][str(k)]}.items():
            rows.append(dict(phone=phone,K=k,arm=arm,field_fraction=g,
                template_MSE=c-2*g*a+g*g*b))
    return pd.DataFrame(rows)


def train(archive,out):
    pairs,dropped=old_pairs(archive)
    design=dict(archive_sha256=library.SHA256,source_doi='10.5281/zenodo.3748719',script_sha256=digest(__file__),
        implementation_sha256=digest(anchors.__file__),no_outcome_selection=True,training_site='UJI Library, Spain; two aligned floors 3 and 5; Galaxy S3 only.',
        inclusion='Only pre-existing designated early library source block months 1 to 7, June–December 2016. For each distinct physical date choose first publisher trn campaign number. In sorted chronology use successive non-overlapping pairs and drop the odd last singleton. One later day per earlier survey; no records from 2017, 2018, UJIIndoorLoc or any other corpus.',
        K=list(BUDGETS),seeds=list(SEEDS),floor_units_per_uji=UNIT,
        design='H = all S3 earlier-day scans on opposite floor; S = one earlier-day raw S3 scan at each nested selected target-floor position. Label = same target-floor XY mean of all 12 later-day real S3 scans after coding each miss as -110. Use only APs seen in H or current S. Field = unchanged half10/40 ridge1 posterior at S (including exact historical XY lookup).',
        objective='Per K, equal AP weights within a site, equal selected sites per episode, then 10 equal nested draws per floor, 2 equal floors per distinct date pair, then 7 equal disjoint physical date pairs. Scalar gives 5 budgets equal weight. For y=future-raw and d=posterior-raw, g=clip(mean(d*y)/mean(d*d),0,1) with the same weights; no monotonic constraint. Mix only the appended S template; leave H, candidate weights and readout unchanged.',
        declared_controls=FIXED,pairs=pairs,excluded_last_singleton=dropped)
    dump(out/'protocol.json',design)
    tables=[];drawlist=[]
    for start,end in pairs:
        early=load(archive,start);late=load(archive,end);key=start['date']+'-'+end['date']
        tbl,ds=sufficient(early,{'S3':late},key,'train');tables.append(tbl);drawlist+=ds
    rows=pd.concat(tables,ignore_index=True)
    ev=rows.groupby(['pair','floor','draw','K'])[['a','b','c']].mean()
    by_k=ev.groupby('K').mean();all_k=by_k.mean()
    if len(ev)!=700 or len(rows)!=5460 or min(by_k.b)<=0:raise ValueError('Incomplete source objective')
    def g(a,b):return float(np.clip(a/b,0.,1.))
    values={str(k):g(v.a,v.b) for k,v in by_k.iterrows()}
    fit=dict(model='lib-2016-anchor-fraction-v1',g_by_K=values,g_scalar=g(all_k.a,all_k.b),
        orientation='S_template=(1-g)*raw+g*posterior; H templates and coordinate weights unchanged',
        source_protocol_sha256=digest(out/'protocol.json'),training_archive_sha256=library.SHA256,
        counts=dict(date_pairs=len(pairs),unique_physical_visits=2*len(pairs),source_s3_scans=576*2*len(pairs),
            unique_source_floors=2,pair_floor_instances=14,distinct_physical_floor_XY=48,draws_per_floor=10,
            exposures_across_nested_k=len(rows),episodes_across_nested_k=len(ev),actual_AP_range=[int(rows.n_ap.min()),int(rows.n_ap.max())]),
        K_monotone_non_decreasing=bool(np.all(np.diff(list(values.values()))>=0)),frozen_fixed_controls=FIXED,
        limits='One historical site and receiver, repeated registered grid. Visits 1-36 days apart. Training examples reuse grid positions, nested draws and months; 7 date pairs do not represent independent buildings. Forecast loss in squared dBm is not positional MDE.')
    dump(out/'frozen.json',fit)
    dump(out/'support_ids.json',drawlist)
    by_k.to_csv(out/'source_normal_equations.csv')
    rows.to_csv(out/'source_anchor_stats.csv.gz',index=False)
    score(rows,fit).to_csv(out/'source_MSE.csv',index=False)
    print(digest(out/'frozen.json'),fit['g_by_K'],fit['g_scalar'],flush=True)


def repeat(archive,out,expected):
    fitfile=out/'frozen.json'
    if digest(fitfile)!=expected:raise ValueError('Freeze failed prior to repeat audit')
    fit=json.loads(fitfile.read_text())
    h={f:library.on_floor(archive,25,'tst',1,f) for f in (3,5)}
    target={p:{f:library.on_floor(archive,25,'tst',n,f) for f in (3,5)}
            for p,n in [('S3',5),('A5',10)]}
    rows,_=sufficient(h,target,'20180706-20180713','repeat');sc=score(rows,fit)
    sc.to_csv(out/'july_predict_only.csv',index=False)
    sc.groupby(['phone','arm']).template_MSE.mean().unstack().to_csv(out/'july_summary.csv')
    print(sc.groupby(['phone','arm']).template_MSE.mean().unstack().round(4).to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['train','repeat'])
    p.add_argument('archive',type=Path);p.add_argument('out',type=Path);p.add_argument('--freeze-sha')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.action=='train':train(args.archive,args.out)
    else:repeat(args.archive,args.out,args.freeze_sha)
