"""Summarize the frozen study only after all four processes are terminal.

Resampling measures random S/Q allocation uncertainty on the fixed corpora,
not population uncertainty over unseen buildings. Failed outcomes stay missing.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'results/fusion_study_20261028'


def read(path):
    return json.loads(path.read_text())


def ratio_interval(values,counts):
    """Ratio of weighted sums, retaining empty draws and paired resamples."""
    if not np.isfinite(values).all():return dict(mean=None,pointwise95=None,reason='missing method outcome'),None
    total=values.sum(0)
    if total[1]==0:return dict(mean=None,pointwise95=None,reason='empty stratum'),None
    boot=counts@values
    valid=boot[:,1]>0
    samples=np.divide(boot[:,0],boot[:,1],out=np.full(len(boot),np.nan),where=valid)
    return dict(mean=float(total[0]/total[1]),
        pointwise95=np.quantile(samples,[.025,.975]).tolist() if valid.all() else None,
        empty_bootstrap_draws=int((~valid).sum()),nonempty_original_draws=int((values[:,1]>0).sum()),
        interval_scope='Exploratory pointwise paired percentile bootstrap, not simultaneous or independent-building inference'),samples


def paired_intervals(groups,replicates=20000,seed=2026102801):
    """Each group is draw x contrast; independent draw indices per building."""
    rng=np.random.default_rng(seed);centers=[];ses=[];statistics=[];valids=[]
    for values in groups:
        n,p=values.shape
        valid=np.isfinite(values).all(0)
        mean=np.full(p,np.nan);se=np.full(p,np.nan)
        mean[valid]=values[:,valid].mean(0)
        se[valid]=values[:,valid].std(0,ddof=1)/np.sqrt(n)
        valid &= se>0
        t=np.full((replicates,p),np.nan)
        for start in range(0,replicates,256):
            stop=min(start+256,replicates)
            indices=rng.integers(n,size=(stop-start,n))
            sample=values[:,valid][indices]
            delta=sample.mean(1)-mean[valid]
            boot_se=sample.std(1,ddof=1)/np.sqrt(n)
            z=np.divide(delta,boot_se,out=np.full_like(delta,np.inf),where=boot_se>0)
            z[(boot_se==0)&(delta==0)]=0
            t[start:stop,valid]=z
        centers.extend(mean);ses.extend(se);valids.extend(valid);statistics.append(t)
    center=np.asarray(centers);se=np.asarray(ses);valid=np.asarray(valids)
    t=np.concatenate(statistics,axis=1)
    intervals=[None]*len(center);critical=None
    if valid.any():
        critical=float(np.quantile(np.max(np.abs(t[:,valid]),axis=1),.95))
        for j in np.flatnonzero(valid):
            lo,hi=np.quantile(t[:,j],[.025,.975])
            intervals[j]=dict(mean=float(center[j]),standard_error=float(se[j]),
                pointwise95=[float(center[j]-hi*se[j]),float(center[j]-lo*se[j])],
                simultaneous95=[float(center[j]-critical*se[j]),float(center[j]+critical*se[j])])
    return intervals,dict(critical=critical,valid_contrasts=int(valid.sum()),
        total_contrasts=len(valid),invalid_indices=np.flatnonzero(~valid).tolist(),
        explanation='Missing draws or zero sample variance are not silently dropped or assigned a passing interval.')


def summarize():
    freeze=read(ROOT/'freeze.json');manifest=read(ROOT/'manifest.json')
    digest=hashlib.sha256((ROOT/'freeze.json').read_bytes()).hexdigest()
    buildings=sorted({m['building'] for m in manifest['events']})
    for building in buildings:
        terminal=read(ROOT/f'{building}_finished.json')
        assert terminal['complete'] and terminal['freeze_sha256']==digest
    methods=freeze['methods'];ks=freeze['K'];n=freeze['draws']
    tables={};failures=[];strata={}
    for meta in manifest['events']:
        key=(meta['building'],meta['scenario'],meta['target'])
        table=tables.setdefault(key,np.full((n,len(ks),len(methods)),np.nan))
        for ki,k in enumerate(ks):
            row=read(ROOT/meta['building']/meta['id']/f'K{k}.json')
            assert row['freeze_sha256']==digest
            assert set(row['results'])|set(row['failures'])==set(methods)
            assert not set(row['results'])&set(row['failures'])
            for name,reason in row['failures'].items():
                failures.append(dict(id=meta['id'],K=k,method=name,**reason))
            for mi,name in enumerate(methods):
                if name not in row['results']:continue
                errors=np.asarray(row['results'][name]['errors'])
                assert len(errors)==meta['query_positions'] and np.isfinite(errors).all()
                assert np.isclose(errors.mean(),row['results'][name]['mde'])
                table[meta['draw'],ki,mi]=errors.mean()
                for label in ('support_hull_inside','complementary_geometry','no_geometric_H_coverage'):
                    mask=np.asarray(row['strata'][label],dtype=bool)
                    for flag in (False,True):
                        skey=(*key,ki,mi,label,flag)
                        sums=strata.setdefault(skey,np.full((n,2),np.nan))
                        chosen=mask==flag
                        sums[meta['draw']]=[errors[chosen].sum()/len(errors),chosen.mean()]
    groups=[];labels=[];curves=[]
    for building in buildings:
        floors=sorted(key[2] for key in tables if key[:2]==(building,'all'))
        # A missing floor remains NaN; never average only the successful floors.
        mean=np.stack([tables[(building,'all',f)] for f in floors]).mean(0)
        contrasts=[]
        for name in freeze['candidate_arms']:
            mi=methods.index(name)
            for ki,k in enumerate(ks):
                for reference in ('WKNN_HS','GP_S'):
                    contrasts.append(mean[:,ki,mi]-mean[:,ki,methods.index(reference)])
                    labels.append(dict(building=building,method=name,K=k,reference=reference,type='reference'))
            for ki in range(1,len(ks)):
                contrasts.append(mean[:,ki,mi]-mean[:,ki-1,mi])
                labels.append(dict(building=building,method=name,K=ks[ki],previous_K=ks[ki-1],type='adjacent_K'))
        groups.append(np.stack(contrasts,axis=1))
        for scenario in ('all','far'):
            selected=[table for key,table in tables.items() if key[:2]==(building,scenario)]
            if not selected:continue
            values=np.stack(selected).mean(0)
            for ki,k in enumerate(ks):
                for mi,name in enumerate(methods):
                    v=values[:,ki,mi];ok=np.isfinite(v)
                    curves.append(dict(building=building,scenario=scenario,K=k,method=name,
                        complete_draws=int(ok.sum()),planned_draws=n,
                        mean=float(v.mean()) if ok.all() else None,
                        conditional_complete_case_mean=float(v[ok].mean()) if ok.any() else None,
                        draw_values=[float(x) if np.isfinite(x) else None for x in v]))
    assert len(labels)==224
    ci,diagnostic=paired_intervals(groups)
    for label,interval in zip(labels,ci):label['interval']=interval
    gates={}
    for name in freeze['candidate_arms']:
        required=[x for x in labels if x['method']==name]
        gates[name]=dict(passed=all(x['interval'] is not None and np.isfinite(x['interval']['simultaneous95']).all()
            and x['interval']['simultaneous95'][1]<=0 for x in required),required_contrasts=len(required),
            missing_or_zero_variance=sum(x['interval'] is None for x in required))
    # Each floor has equal population weight before conditioning on a stratum.
    # Resample whole draw groups; never condition by deleting empty draws.
    secondary=[];mechanism=[];rng=np.random.default_rng(2026102802)
    pairs=[('local_identity','global_identity'),('global_affine','global_identity'),
           ('local_affine','global_affine'),('local_affine','local_identity'),
           ('global_identity','legacy_global_identity'),('TPM_pool','SCMT_pool'),('WKNN_HS','WKNN_S')]
    for building in buildings:
        counts=rng.multinomial(n,np.full(n,1/n),size=20000).astype(float)
        for scenario in ('all','far'):
            floors=sorted(key[2] for key in tables if key[:2]==(building,scenario))
            if not floors:continue
            for floor in [None,*floors]:
                selected=floors if floor is None else [floor]
                for ki,k in enumerate(ks):
                    conditions=[('all_queries',None)]+[(label,flag) for label in
                        ('support_hull_inside','complementary_geometry','no_geometric_H_coverage') for flag in (False,True)]
                    for label,flag in conditions:
                        summaries={};samples={}
                        for mi,name in enumerate(methods):
                            if label=='all_queries':
                                error=np.stack([tables[(building,scenario,f)][:,ki,mi] for f in selected]).mean(0)
                                values=np.column_stack([error,np.ones(n)])
                            else:
                                values=np.stack([strata.get((building,scenario,f,ki,mi,label,flag),
                                    np.full((n,2),np.nan)) for f in selected]).mean(0)
                            summary,sample=ratio_interval(values,counts)
                            summaries[name]=summary;samples[name]=sample
                            secondary.append(dict(building=building,scenario=scenario,floor=floor,K=k,
                                method=name,stratum=label,value=flag,**summary))
                        for candidate,reference in pairs:
                            a,b=samples[candidate],samples[reference]
                            valid=a is not None and b is not None and np.isfinite(a).all() and np.isfinite(b).all()
                            mechanism.append(dict(building=building,scenario=scenario,floor=floor,K=k,
                                stratum=label,value=flag,method=candidate,reference=reference,
                                difference=summaries[candidate]['mean']-summaries[reference]['mean'] if valid else None,
                                pointwise95=np.quantile(a-b,[.025,.975]).tolist() if valid else None,
                                scope='Exploratory paired mechanism contrast; does not override primary G1 gate'))
    result=dict(freeze_sha256=digest,complete=True,primary_contrasts=labels,bootstrap=diagnostic,
        G1_gates=gates,curves=curves,failures=failures,secondary=secondary,mechanism_contrasts=mechanism,
        scope='Fixed development corpora and frozen source parameters. Not new-building generalization, new learning benefit, or theoretical performance bound.')
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(G1_gates=gates,failed_method_events=len(failures),bootstrap=diagnostic)))


if __name__=='__main__':
    import sys
    if '--check' in sys.argv:
        rng=np.random.default_rng(9);a=rng.normal(size=(128,3));a[:,1]=a[:,0]+4;a[:,2]=1
        b=a.copy();b[0,0]=np.nan
        intervals,info=paired_intervals([a,b],replicates=1024)
        assert intervals[2] is None and intervals[3] is None and intervals[5] is None
        assert np.allclose(np.subtract(intervals[1]['simultaneous95'],intervals[0]['simultaneous95']),4)
        assert info['valid_contrasts']==3
        counts=rng.multinomial(4,np.full(4,.25),size=1024)
        values=np.array([[0.,0.],[2.,.5],[4.,1.],[0.,0.]])
        r,s=ratio_interval(values,counts)
        assert r['mean']==4 and np.all(s[np.isfinite(s)]==4)
        assert r['empty_bootstrap_draws']>0 and r['pointwise95'] is None
        assert ratio_interval(np.zeros((4,2)),counts)[0]['reason']=='empty stratum'
        assert ratio_interval(np.full((4,2),np.nan),counts)[0]['reason']=='missing method outcome'
        print('Paired intervals and weighted-stratum empty/missing handling passed; no study outcomes read.')
    else:summarize()
