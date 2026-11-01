"""Summarize the paired field and matching experiments without pooling sites."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

INDEX=['building','floor','draw','K']

def arm(path,name):
    df=pd.read_csv(path).rename(columns={'seed':'draw'})
    if 'method' in df:
        return df[df.method==name].set_index(INDEX).mde.sort_index()
    return df.set_index(INDEX)[name].sort_index()

def panel(data):
    x=data.groupby(level=['building','floor','K']).mean()
    return x,x.groupby(level=['building','K']).mean()

def intervals(x):
    # Conditional on these buildings, floors, queries and fixed fitted models;
    # resample whole draw paths independently within each floor.
    rng=np.random.default_rng(32401);cells=[];centers=[]
    for b,df in x.groupby(level='building'):
        u=[];v=[]
        for f,h in df.groupby(level='floor'):
            a=h.droplevel(['building','floor']).unstack('K').sort_index().to_numpy()
            if a.shape!=(10,5):raise ValueError('Unpaired/incomplete draw')
            u.append(a.mean(0));v.append(a[rng.integers(0,10,(5000,10))].mean(1))
        cells.append(np.mean(v,0));centers.append(np.mean(u,0))
    boot=np.stack(cells,axis=1);center=np.array(centers)
    critical=np.quantile((center-boot).max(axis=(1,2)),.95)
    return np.quantile(boot.mean(axis=(1,2)),[.025,.975]).tolist(),int((center+critical<0).sum())

def summary(primary,control):
    if len(primary)!=650 or not primary.index.equals(control.index):raise ValueError('Episode mismatch')
    d=primary-control;floor,building=panel(d)
    y,z=panel(primary)
    bounds,nupper=intervals(d)
    return dict(mde=z.values.mean(),control_mde=panel(control)[1].values.mean(),
                delta=building.values.mean(),delta_macro_lo=bounds[0],delta_macro_hi=bounds[1],
                building_wins=int((building<0).sum()),simultaneous_wins=nupper,floor_wins=int((floor<0).sum()),
                draw_wins=int((d<0).sum()),adj_building=int((z.unstack().diff(axis=1).iloc[:,1:]<0).sum().sum()),
                adj_floor=int((y.unstack().diff(axis=1).iloc[:,1:]<0).sum().sum()))

def make(archive,out):
    out.mkdir(exist_ok=True,parents=True)
    records=[];curves=[]
    for mode in ('official','phone'):
        path=lambda family: archive/family/(mode+'.csv')
        gap=arm(path('trust'),'gap_only')
        parent=arm(path('trust'),'identity')
        period=arm(archive/'period-wknn'/('period_wknn_'+mode+'.csv'),'gap_fit')
        fixed_file=archive/'vertical'/(mode+'.csv')
        comp={'half10':parent,'gap':gap,'gap_nearest':arm(fixed_file,'nearest'),
              'source_evidence_shared':arm(path('evidence'),'shared_fit'),
              'contour':arm(path('contour'),'selected'),
              'spatial_bands':arm(path('bands'),'selected'),
              'learned_match':arm(path('table'),'learned'),
              'robust_match':arm(path('robust'),'risk_robust'),
              'true_period_wknn':period}
        for name,v in comp.items():
            floor,b=panel(v)
            for (bi,k),mean in b.items():curves.append((mode,name,bi,k,mean))
        for name,ref in [('gap','half10'),('gap','true_period_wknn'),('source_evidence_shared','gap'),
                         ('spatial_bands','half10'),('learned_match','gap'),('robust_match','gap')]:
            records.append(dict(regime=mode,method=name,control=ref,**summary(comp[name],comp[ref])))
    pd.DataFrame(records).to_csv(out/'paired.csv',index=False)
    pd.DataFrame(curves,columns=['regime','method','building','K','mde']).to_csv(out/'curves.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();make(a.archive,a.out)
