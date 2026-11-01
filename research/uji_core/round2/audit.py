"""Independent retrospective UJI gate audit. No external score files are read."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("receipts",type=Path)
parser.add_argument("--previous",type=Path,default=Path(__file__).resolve().parents[1]/"results")
options=parser.parse_args()
ROOT=options.receipts
PRE=options.previous
OUT=ROOT/'decision-audit'
IDX=['building','floor','draw','K']
BUDGETS=[1,3,5,10,20]; EPS=1e-9
# Fixed comparator and simpler matched control(s), selected without choosing a different
# challenger in each regime. Shuffled features and label-informed oracles do not count as deployable controls.
SPECS=[]
def spec(directory,methods,parent,simple,path='{m}.csv',extra=None):
    for m in methods.split():SPECS.append((directory,m,parent.split(),simple.split(),path,extra))
spec('anchor-measure','fitted_multi','raw_multi half_multi post_multi','','anchor_{m}.csv')
spec('anchor-measure','fitted_unique','raw_unique half_unique post_unique','fitted_multi','anchor_{m}.csv')
spec('asym','selected period_fit device_fit','half only_fill only_trim mild_fill mild_trim','tied')
spec('bands','selected time_fit device_fit','half fixed_multiscale smooth_fine_preserve_near','single')
spec('contour','selected','standard fixed_contour fixed_weak','matched_spatial')
spec('count_normalization','learned period_only device_only','default fixed_half fixed_full','')
spec('cov','source_fit','identity empirical','scalar diagonal')
spec('evidence','selected AP_fit','parent weak_shared weak_AP fixed_partial','shared_fit')
spec('history_smooth','full','half physical fixed0 fixed75 fixed1','scalar single_best variance_only contrast_coverage AP_mean',extra='{m}_fixed_controls.csv')
spec('history_smooth','single_best variance_only contrast_coverage W B','half fixed0 fixed75 fixed1','scalar',extra='{m}_fixed_controls.csv')
spec('history_smooth','scalar','half fixed0 fixed75 fixed1','',extra='{m}_fixed_controls.csv')
spec('map_variance','full period_fit device_fit','default physical moderate','constant tied repeat_fit between_fit AP_mean_fit')
spec('map_variance','full','default physical moderate','constant AP_mean_fit','half_{m}.csv')
spec('map_variance','conditioned','fixed12 fixed_half always_local','trained_constant_sigma constant_mix','gate_{m}.csv')
spec('match','joint_fit','default fixed_offset default_no_misses','global_fit fixed_offset_fit remove_offset','{s}_events.csv')
spec('match','offset_only_fit','default fixed_offset','global_fit','{s}_events.csv')
spec('match','global_fit','default default_no_misses','', '{s}_events.csv')
spec('match','default_source','default_zero default_fixed6','','coupled_{s}.csv')
spec('match','global_source','global_zero global_fixed6','','coupled_{s}.csv')
spec('match','bank','fixed0 fixed3 fixed6 fixed9 fixed12','external_source','bank_{s}.csv')
spec('metric','supervised','unit fixed_supervised','pca','metric_{m}.csv')
spec('projection','joint selected','identity fixed_ridge','global diagonal')
spec('reliability','learned','identity fixed_quality','temperature')
spec('reliability-final','learned','identity fixed_quality fixed_rarity','temperature source_single')
spec('robust','risk_average risk_robust','identity','temperature detection')
spec('score_shape','asymmetric period_asymmetric device_asymmetric','default huber half_tail positive_half_tail fixed_sign fixed_weak_core fixed_cool','temperature rss_miss symmetric sign_only')
spec('score_shape','symmetric sign_only','default huber half_tail positive_half_tail fixed_sign fixed_weak_core fixed_cool','temperature rss_miss')
spec('table','learned conservative','identity','temperature detection')
spec('transport','source_both','identity remove half_common boost_remainder tied0.5 tied1.0 tied1.5 tied2.0','source_tied source_common','transport_{m}.csv')
spec('transport','source_loo empirical_bayes one_anchor raw_loo','fixed0.5 fixed0.75 fixed1.0 fixed1.25 fixed1.5 fixed2.0','source_tied','loo_{m}.csv')
spec('transport/ap_gain','learned','unit','scalar','ap_gain_events.csv')
spec('trust','local_fit','identity fixed_gap fixed_agreement fixed_local','gap_only global_fit')
spec('trust','global_fit','identity fixed_gap fixed_agreement fixed_local','gap_only')
spec('trust','gap_only','identity fixed_gap','gap_only')  # all three folds select fixed 2U ordinal decay
spec('vertical','source H_closed','parent nearest ridge1 ridge4 ridge_quarter ols','shared')
spec('temporal','future','unit half no_propagation','best_global future_global same_global same','temporal_uji_transfer.csv')
spec('expanded-table','expanded_full','identity','expanded_temperature expanded_detection original_full')
spec('expanded-table','expanded_detection','identity','expanded_temperature original_detection')
spec('expanded-smooth','ipin_full','fixed50 fixed75','ipin_scalar uji_full')
spec('expanded-smooth','early_library_full','fixed50 fixed75','early_library_scalar uji_full ipin_full')
spec('expanded-smooth','ipin_scalar early_library_scalar','fixed50 fixed75','uji_scalar')
spec('expanded-score','expanded_asymmetric','default huber half_tail positive_half_tail fixed_sign fixed_weak_core fixed_cool','expanded_temperature expanded_symmetric expanded_best_fixed_source uji_asymmetric')
spec('expanded-score','expanded_symmetric','default huber half_tail fixed_weak_core fixed_cool','expanded_temperature expanded_best_fixed_source uji_symmetric')
# Additional simple-family candidates: when learned form contains a closed-form subfamily,
# count a tied score as zero gain; an unmatched single parameter does not become proof.
spec('map_variance','repeat_fit between_fit tied','default moderate physical','constant')
spec('map_variance','constant','default','')
spec('evidence','shared_fit','parent weak_shared','')
spec('table','detection','identity','temperature')
spec('expanded-table','expanded_temperature','identity','original_temperature')
spec('projection','global diagonal','identity','')
spec('transport','source_common','identity remove half_common boost_remainder','source_tied','transport_{m}.csv')
spec('robust','detection','identity','temperature')
spec('score_shape','temperature rss_miss','default fixed_cool','')

def get(path,mode):
    z=pd.read_csv(path)
    for c in ('stage','split','mode'):
        if c in z and set(z[c]).issuperset({'official','phone'}): z=z[z[c]==mode]
    if 'method' in z: z=z.pivot(index=IDX,columns='method',values='mde')
    else:z=z.set_index(IDX)
    z=z.sort_index()
    if len(z)!=650 or not z.index.is_unique:raise ValueError(str(path))
    return z

def cube(v):
    return np.asarray(v,dtype=float).reshape((13,10,5))
building=np.repeat(np.arange(3),[4,4,5])
fw=np.array([1./(3*(4 if b!=2 else 5)) for b in building])
rng=np.random.default_rng(20261101)
bootstrap=rng.multinomial(10,np.ones(10)/10,size=(5000,13))/10*fw[None,:,None]
def delta_summary(x):
    c=cube(x)
    boot=bootstrap.reshape(5000,130)@c.mean(2).ravel()
    return np.average(c.mean((1,2)),weights=fw),np.quantile(boot,.025),np.quantile(boot,.975)
def buildingrisk(c):
    return np.vstack([c[building==b].mean((0,1)) for b in range(3)])
def source(mode):
    path=ROOT/'period-wknn'/('period_wknn_'+mode+'.csv')
    z=get(path,mode)
    pm=get(ROOT/'paired-map'/('paired_map_'+mode+'.csv'),mode)
    z=z.join(pm[['field_wknn']]); z['same_parent']=pm.field_tobit
    plain=get(PRE/'wknn_events.csv',mode) if mode=='official' else get(OUT/'fixed_wknn_phone.csv',mode)
    for n in ['raw_fixed3','prototype_fixed3','raw_tuned','prototype_tuned']:
        z[n]=plain[n]
    return z

rows=[];contrasts=[];negative=[]
for mode in ['official','phone']:
    ref=source(mode)
    for family,method,parents,controls,patt,extra in SPECS:
        sub=dict(m=mode,s='device' if mode=='phone' else 'official')
        table=get(ROOT/family/patt.format(**sub),mode)
        if extra:table=table.join(get(ROOT/family/extra.format(**sub),mode),rsuffix='_dupe')
        name=f'{family}/{patt}/{method}'
        if not np.isfinite(table[method].to_numpy()).all():raise ValueError(name)
        c=cube(table[method]); b=buildingrisk(c);levels=np.diff(b,axis=1)
        # Software R1 reported a site-selected prototype choice; fixed3-phone archived separately
        # only if actually computed. Test all available training-ref and floor-aware WKNN together.
        family_refs=[k for k in ref if k not in ('same_parent','field_wknn')]
        baseline_curves=np.stack([buildingrisk(cube(ref[x])) for x in family_refs])
        worst=b-np.min(baseline_curves,axis=0)
        same=b-buildingrisk(cube(ref['field_wknn']))
        entry=dict(mode=mode,ch=name,pair_count=len(c.ravel()),mde=np.average(c.mean((1,2)),weights=fw),
                   A_improves=int((levels < -EPS).sum()),A_total=12,A_floors=int((np.diff(c.mean(1),axis=-1)<-EPS).sum()),
                   A_draws=int((np.diff(c,axis=2)<-EPS).sum()),B_all_wknn_nonloss=int((worst<=EPS).sum()),B_worst_delta=worst.max(),
                   B_same_map_reference_nonloss=int((same<=EPS).sum()),B_same_map_reference_worst=same.max())
        compare=[]
        for role,names in [('fixed',parents),('simple',controls)]:
            for r in names:
                diff=table[method]-table[r]
                center,lo,hi=delta_summary(diff)
                compare.append((role,r,center,lo,hi))
                contrasts.append(dict(mode=mode,ch=name,role=role,reference=r,delta=center,given_draws_lo=lo,given_draws_hi=hi,
                                      buildingK_wins=int((buildingrisk(cube(diff))<-EPS).sum()),
                                      floorK_wins=int((cube(diff).mean(1)<-EPS).sum())))
            worst_by_role=max((a for a in compare if a[0]==role),key=lambda a:a[2],default=None)
            if worst_by_role:entry.update({role+'_control':worst_by_role[1],role+'_delta':worst_by_role[2],role+'_upper95':worst_by_role[4]})
        limiting=max(compare,key=lambda a:a[2])
        entry.update(limit_control=limiting[1],limit_delta=limiting[2],limit_upper95=max(a[4] for a in compare),
                     C_all_point=all(a[2]<-EPS for a in compare),C_all_upper95=all(a[4]<-EPS for a in compare),
                     nontrivial_control=bool(controls))
        rows.append(entry)
a=pd.DataFrame(rows); t=pd.DataFrame(contrasts); t.to_csv(OUT/'contrasts.csv',index=False)
a.to_csv(OUT/'gates_by_regime.csv',index=False)
summ=[]
for ch,g in a.groupby('ch'):
    assert len(g)==2
    ds=g.set_index('mode')
    row=dict(ch=ch,A_both=bool((g.A_improves==12).all()),B_both=bool((g.B_all_wknn_nonloss==15).all()),
             both_paired_only=bool(g.C_all_point.all()),both_paired_upper95=bool(g.C_all_upper95.all()),
             reference_has_simpler_fit=bool(g.nontrivial_control.all()),
             all_with_same_map=bool((g.B_same_map_reference_nonloss==15).all()),
             official_delta=ds.at['official','limit_delta'],phone_delta=ds.at['phone','limit_delta'])
    row['all_internal']=all(row[x] for x in ['A_both','B_both','both_paired_only','reference_has_simpler_fit'])
    row['all_credible']=row['all_internal'] and row['both_paired_upper95']
    summ.append(row)
checks=pd.DataFrame(summ);checks.to_csv(OUT/'joint.csv',index=False)
print(checks[['A_both','B_both','both_paired_only','both_paired_upper95','all_internal','all_credible']].sum())
print(checks[checks.both_paired_only].to_string(index=False))
