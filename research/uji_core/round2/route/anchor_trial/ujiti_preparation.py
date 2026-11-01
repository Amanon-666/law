"""One frozen anchor-gain replay on UJITI only. Run with repo on PYTHONPATH.

python replay_gain.py ARCHIVE METADATA_DESIGN FROZEN_FIT ORIGINAL_UJITI OUT --fit-sha ... --freeze-commit ...
The caller must verify the actual remote published freeze before calling.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from external_baselines import tpm_adapter as tpm
from research.uji_core.inverse import order,squared,vote
from research.uji_core.next.paired_map_wknn import group_ids
from research.uji_core.protocol import position_weights
from anchor_readout import fit_anchor_readout


def summary(tab,out):
    cols=['source_K','source_scalar','fixed0','fixed25','fixed50','fixed75','fixed100','floor_wknn','same_mixed_wknn7','previous_mapped_wknn']
    rng=np.random.default_rng(261101);curves=[];contrasts=[];means=[];bootstrap=[]
    def add(f,v,bt):
        for i,k in enumerate([1,3,5,10,20]):
            curves.append(dict(floor=f,K=k,**dict(zip(cols,v[i]))))
        for j,name in enumerate(cols[1:],1):
            delta=v[:,0]-v[:,j];reps=bt[:,:,0]-bt[:,:,j]
            for i,k in enumerate([1,3,5,10,20,'mean_K']):
                estimate=delta[i] if i<5 else delta.mean()
                lo,hi=np.quantile(reps[:,i] if i<5 else reps.mean(1),[.025,.975])
                contrasts.append(dict(floor=f,K=k,reference=name,learned_minus_reference=estimate,lo95=lo,hi95=hi))
    for f,df in tab.groupby('floor'):
        v=df.sort_values(['draw','K'])[cols].to_numpy().reshape(40,5,len(cols))
        boot=v[rng.integers(40,size=(10000,40))].mean(1)
        add(f,v.mean(0),boot);means.append(v.mean(0));bootstrap.append(boot)
    add('all',np.mean(means,0),np.mean(bootstrap,0))
    pd.DataFrame(curves).to_csv(out/'curves.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)


def main(a):
    sys.path.insert(0,str(a.original))
    from replay import ALLOW,R1,R2,KS,UNIT,extract,canonical,axishash,digest
    cfg=json.loads(a.fit.read_text());assert digest(a.fit.read_bytes())==a.fit_sha
    learned={int(k):float(v) for k,v in cfg['g_by_K'].items()}
    scalar=float(cfg['g_scalar'])
    assert set(learned)==set(KS) and all(0<=x<=1 for x in [*learned.values(),scalar])
    design=json.loads((a.design/'protocol.json').read_bytes())
    assert digest((a.design/'protocol.json').read_bytes())=='b7b5483b7c401ecb9ee5061c64021bbf3fb3f293c0f8ac25ffe414295c6c2e1d'
    assert digest(a.archive.read_bytes())==design['input_archive_sha256']
    raw=(a.design/'route_manifest.json').read_bytes();assert digest(raw)==design['manifest_sha256']
    packed=(a.design/'scan_metadata.json.gz').read_bytes();meta_bytes=gzip.decompress(packed)
    assert digest(packed)==design['scan_metadata_sha256'] and digest(meta_bytes)==design['scan_metadata_plain_sha256']
    meta={x['id']:x for x in json.loads(meta_bytes) if x['file'] in (R1,R2)}
    members={x['member']:x for x in design['allowed_members'] if x['member'] in ALLOW}
    with ZipFile(a.archive) as z:
        data={}
        for name in sorted(ALLOW):data.update(extract(z,name,members[name],meta))
    assert data.keys()==meta.keys()
    index=['floor','draw','K']
    original=pd.read_csv(a.original/'events.csv').set_index(index)
    fields=pd.read_csv(a.original/'field_anchor_events.csv').set_index(index)
    rows=[]
    for job in [x for x in json.loads(raw) if x['site']=='UJITI']:
        f=job['floor'];hf={int(n):ids for n,ids in job['H_by_floor'].items()};hids=sum(hf.values(),[])
        qr=job['ordered_Q_ids'];si=job['eligible_target_S_ids']
        xy={i:((meta[i]['lon']-job['lon0'])*job['east_m_per_degree'],(meta[i]['lat']-job['lat0'])*111320.) for i in hids+si+qr}
        assert digest(canonical([(i,xy[i][0].hex(),xy[i][1].hex()) for i in sorted(xy)]))==job['coordinate_frame_sha256']
        hseen=set().union(*(data[i] for i in hids));gaps=[abs(f-n) for n in sorted(hf)]
        for draw in job['draws']:
            d=draw['draw']
            for k in KS:
                ss=draw['ordered_S_scan_ids'][:k];ap=sorted(hseen.union(*(data[i] for i in ss)))
                assert axishash(ap)==draw['permitted_AP_axis'][str(k)]['sha256']
                col={name:j for j,name in enumerate(ap)}
                def pack(ids):
                    r=np.full((len(ids),len(ap)),100.,float)
                    for j,i in enumerate(ids):
                        for name,value in data[i].items():
                            if name in col:r[j,col[name]]=value
                    return dict(rss=r,xy=np.array([xy[i] for i in ids])/UNIT,ids=np.array([meta[i]['first_WIFI_line'] for i in ids]))
                h=[pack(hf[n]) for n in sorted(hf)];s=pack(ss);q=pack(qr);w=position_weights(q['xy'])
                anchors=fit_anchor_readout(h,s,floor_gaps=gaps)
                assert anchors.visible.all()
                scales={'source_K':learned[k],'source_scalar':scalar,'fixed0':0.,'fixed25':.25,'fixed50':.5,'fixed75':.75,'fixed100':1.}
                values={}
                for name,g in scales.items():
                    model=anchors.map_for(g)
                    pred=model.predict(q['rss'])
                    values[name]=float(w@np.linalg.norm(pred-q['xy'],axis=1)*UNIT)
                model=anchors.map_for(learned[k])
                hpos,hid=group_ids(np.vstack([v['xy'] for v in h]),np.concatenate([v['ids'] for v in h]))
                spos,sid=group_ids(s['xy'],s['ids'])
                np.testing.assert_array_equal(np.vstack((hpos,spos)),model.xy)
                dist=squared(tpm.dbm(q['rss'])+110.,model.prototypes+110.)
                ix=order(dist,np.r_[hid,sid],7)
                pred=vote(np.take_along_axis(dist,ix,axis=1),model.xy[ix]-model.xy[0],len(ap))[0]+model.xy[0]
                values['same_mixed_wknn7']=float(w@np.linalg.norm(pred-q['xy'],axis=1)*UNIT)
                prev=original.loc[f,d,k];post=fields.loc[f,d,k]
                np.testing.assert_allclose([values['fixed0'],values['fixed100']],[prev.parent,post.Cfield],atol=1e-9,rtol=0.)
                rows.append(dict(floor=f,draw=d,K=k,**values,floor_wknn=prev.floor_wknn,previous_mapped_wknn=prev.map_wknn))
    result=pd.DataFrame(rows)
    if len(result)!=600 or not np.isfinite(result.to_numpy()).all():raise ValueError('Incomplete paired UJITI replay')
    result.to_csv(a.out/'events.csv',index=False);summary(result,a.out)
    note=dict(remote_freeze=a.freeze_commit,raw_fit_sha256=a.fit_sha,g_by_K=learned,g_scalar=scalar,g_semantics='0 = measured raw S, 1 = field evaluated at S.',
              opened_members=sorted(ALLOW),events=len(result),fit_or_reselection_on_UJITI=False,units='m',
              uncertainty='10,000 paired support draws independently within each fixed floor, preserving K curves and equal floor weights.')
    (a.out/'receipt.json').write_text(json.dumps(note,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('archive','design','fit','original','out'):p.add_argument(n,type=Path)
    p.add_argument('--fit-sha',required=True);p.add_argument('--freeze-commit',required=True)
    a=p.parse_args()
    if len(a.freeze_commit)!=40 or len(a.fit_sha)!=64:raise ValueError('Need explicit published freeze and fit digests')
    main(a)
