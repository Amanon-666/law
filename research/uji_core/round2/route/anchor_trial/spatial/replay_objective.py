"""One frozen positional-objective replay on UJITI only. Use prior replay_gain unchanged as the reference.
Run with repo and prior result directory on PYTHONPATH.

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
from research.uji_core.next.external_floor_wknn import GapReference
from research.uji_core.next.paired_map_wknn import group_ids
from research.uji_core.protocol import position_weights
from anchor_readout import fit_anchor_readout


def summary(tab,out):
    cols=['source_K','source_scalar','fixed0','fixed25','fixed50','fixed75','fixed100','floor_wknn','same_mixed_wknn7','previous_mapped_wknn','src_direct_wknn','src_orig_map_wknn','src_mixed_map_wknn','rss_source_K','rss_scalar','rss_src_direct_wknn','rss_mixed_map_wknn']
    rng=np.random.default_rng(261101);curves=[];contrasts=[];means=[];bootstrap=[];trends=[]
    def add(f,v,bt):
        for i,k in enumerate([1,3,5,10,20]):
            curves.append(dict(floor=f,K=k,**dict(zip(cols,v[i]))))
        ks=[1,3,5,10,20]
        for before,after in [(0,1),(1,2),(2,3),(3,4),(2,4),(0,4)]:
            delta=float(v[after,0]-v[before,0])
            lo,hi=np.quantile(bt[:,after,0]-bt[:,before,0],[.025,.975])
            trends.append(dict(floor=f,K_from=ks[before],K_to=ks[after],
                               learned_error_difference_m=delta,lo95=lo,hi95=hi))
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
    pd.DataFrame(trends).to_csv(out/'trends.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'curves.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)


def main(a):
    sys.path.insert(0,str(a.original))
    from replay import ALLOW,R1,R2,KS,UNIT,extract,canonical,axishash,digest
    cfg=json.loads(a.fit.read_text());assert digest(a.fit.read_bytes())==a.fit_sha
    spec=json.loads(a.wknn_fit.read_text())
    assert digest(a.wknn_fit.read_bytes())==a.wknn_sha and spec['anchor_freeze_sha256']==a.fit_sha
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
    previous=pd.read_csv(a.previous/'events.csv').set_index(index)
    source_receipt=json.loads((a.previous/'receipt.json').read_text())
    assert source_receipt['raw_fit_sha256']=='ed3ae84fd08d2286b7c754937d0d1c50b9304e8c339c3a3a5487689eea481162'
    assert source_receipt['remote_freeze']=='90b24f3b02a916191533b3a3888cb0dc235da1e8'
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
                def map_wknn(fingerprint_map, neighbors):
                    dist=squared(tpm.dbm(q['rss'])+110.,fingerprint_map.prototypes+110.)
                    ix=order(dist,np.r_[hid,sid],neighbors)
                    pred=vote(np.take_along_axis(dist,ix,axis=1),
                              fingerprint_map.xy[ix]-fingerprint_map.xy[0],len(ap))[0]+fingerprint_map.xy[0]
                    return float(w@np.linalg.norm(pred-q['xy'],axis=1)*UNIT)
                values['same_mixed_wknn7']=map_wknn(model,7)
                values['src_mixed_map_wknn']=map_wknn(model,spec['k_forecast_map'])
                values['src_orig_map_wknn']=map_wknn(anchors.map_for(0.),spec['k_original_map'])
                prev=original.loc[f,d,k]
                post=fields.loc[f,d,k]
                old=previous.loc[f,d,k]
                fixed=['fixed0','fixed25','fixed50','fixed75','fixed100']
                np.testing.assert_allclose([values[name] for name in fixed],old[fixed].values,atol=1e-9,rtol=0.)
                np.testing.assert_allclose([values['fixed0'],values['fixed100']],[prev.parent,post.Cfield],atol=1e-9,rtol=0.)
                values['rss_source_K']=old.source_K
                values['rss_scalar']=old.source_scalar
                values['rss_src_direct_wknn']=old.src_direct_wknn
                values['rss_mixed_map_wknn']=old.src_mixed_map_wknn
                bank=GapReference(h,gaps,spec['direct_rep'])
                alpha=spec['direct_alpha']
                matched=bank.event(bank.query(q['rss']),s,[alpha])
                pred=matched[alpha,spec['k_direct']]
                values['src_direct_wknn']=float(w@np.linalg.norm(pred-q['xy'],axis=1)*UNIT)
                rows.append(dict(floor=f,draw=d,K=k,**values,floor_wknn=prev.floor_wknn,previous_mapped_wknn=prev.map_wknn))
    result=pd.DataFrame(rows)
    assert result.set_index(index).index.equals(previous.index)
    if len(result)!=600 or not np.isfinite(result.to_numpy()).all():raise ValueError('Incomplete paired UJITI replay')
    result.to_csv(a.out/'events.csv',index=False);summary(result,a.out)
    note=dict(remote_freeze=a.freeze_commit,raw_fit_sha256=a.fit_sha,wknn_fit_sha256=a.wknn_sha,g_by_K=learned,g_scalar=scalar,g_semantics='0 = measured raw S, 1 = field evaluated at S.',
              opened_members=sorted(ALLOW),events=len(result),fit_or_reselection_on_UJITI=False,units='m',
              uncertainty='10,000 paired support draws independently within each fixed floor, preserving K curves and equal floor weights.')
    note['prior_rss_forecast_fit']=source_receipt['raw_fit_sha256']
    note['prior_rss_event_sha256']=digest((a.previous/'events.csv').read_bytes())
    note['source_WKNN_choices']={key:spec[key] for key in ('k_forecast_map','k_original_map','k_direct','direct_rep','direct_alpha')}
    (a.out/'receipt.json').write_text(json.dumps(note,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('archive','design','fit','original','out'):p.add_argument(n,type=Path)
    p.add_argument('--fit-sha',required=True);p.add_argument('--freeze-commit',required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--wknn-fit',required=True,type=Path);p.add_argument('--wknn-sha',required=True)
    a=p.parse_args()
    if len(a.freeze_commit)!=40 or len(a.fit_sha)!=64:raise ValueError('Need explicit published freeze and fit digests')
    main(a)
