"""Offline verification/analysis, Python + NumPy only. No optimization or model inference.
Usage: python verify.py extracted_run/ evidence.zip old_F0_VI.zip old_lowK.zip summary.json
Pass the unmodified artifacts incl. all strong baselines. If a cell fails, retain its name,
never call the incomplete set a complete macro curve. Bootstrap conditions on draw0/four floors.
"""
import csv,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
A=('gp','seven','nonlinear'); N=(1,3,7,15); KS=(1,3,5,10,20); F=range(4); REG=('official','phone')
D=lambda b:hashlib.sha256(b).hexdigest()
def get(z,n):return json.loads(z.read(n))
def data(z,n):return np.load(io.BytesIO(z.read(n)),allow_pickle=False)
def stats(y,p):
    pos,gr,ct=np.unique(y,axis=0,return_inverse=True,return_counts=True);gr=gr.ravel();w=1/(len(pos)*ct[gr]);e=np.linalg.norm(y-p,axis=1);idx=np.argsort(e);return float(np.dot(w,e)),float(e[idx[np.searchsorted(np.cumsum(w[idx]),.9)]]),np.bincount(gr,e)/ct,pos

def verify(root,ev,high,low):
    z=zipfile.ZipFile(ev);hh=zipfile.ZipFile(high);outer=zipfile.ZipFile(low);ll=zipfile.ZipFile(io.BytesIO(outer.read('r1-draw0-source.zip')))
    assert D(Path(high).read_bytes())=='8fa3b411517e99079b9742347dc063553fd3d1a4252777112a77572322a4b64d' and D(Path(low).read_bytes())=='7b666ad087e15157fd691648e5353d18dde67c2b8e366819d268ea7aa4d59ccc'
    assert D(z.read('results/manifest.json'))=='5f4fb0ceccc5a64aa7411cc287715e6fa9a39d593da6713910920a3fda67c613' and D(z.read('results/device_stress_design.json'))=='76763a87569d885705dccc54cbd72e878f7a711cbb5343fbfee4517caf2482e6'
    man=get(z,'results/manifest.json')['episodes'];dev=get(z,'results/device_stress_design.json')['events'];wks={r:list(csv.DictReader(io.StringIO(z.read('results/'+('wknn_events.csv' if r=='official' else 'device_stress_events.csv')).decode()))) for r in REG}
    def folder(r,f):return root/('gridpilot-'+r+'-F'+str(f))
    src={a:{n:[] for n in N} for a in A}
    for b,f in ((0,1),(2,2)):
        p=folder('src'+str(b),f);h=json.loads((p/'header.json').read_text());rec=json.loads((p/'K5.json').read_text());assert h['building']==b and h['floor']==f and h['regime']==f'src{b}';assert D((p/'K5.npz').read_bytes())==rec['npz_sha256'];z0=np.load(p/'K5.npz');assert len(z0['truth'])==len(np.unique(z0['truth'],axis=0))==24 and len(z0['support_ids'])==5 and not np.intersect1d(z0['query_ids'],z0['support_ids']).size
        for a in A:
            for n in N:
                tag=f'{a}_w{n}';v=stats(z0['truth'],z0[tag])[0];np.testing.assert_allclose(v,rec['metrics'][tag]['mde'],atol=1e-10,rtol=0);src[a][n].append(v)
    picks={a:min(N,key=lambda n:np.mean(src[a][n])) for a in A};out=dict(ks=KS,source_MDE=src,source_selected=picks,missing=[],warnings=[],floor_events=[],curves={},pairs={},k_changes={},backend={},max_F0_bridge_coordinate_gap=0.,receipt_counts={});by={};eboot={};rng=np.random.default_rng(261103);boots={};oldshift={};quality={};wall=0.
    for r in REG:
        for f in F:
            p=folder(r,f);h=json.loads((p/'header.json').read_text()) if (p/'header.json').exists() else {};ee=next(e for e in (man if r=='official' else dev) if e['building']==1 and e['floor']==f and (r!='official' or e['draw']==0));sorder=ee['support_order'] if r=='official' else ee['support_orders'][0];qids=ee['query_ids'];base=data(ll,f'{r}_F{f}_0.npz');ph=None if r=='official' else ee['phone'];np.testing.assert_array_equal(base['query_ids'],qids)
            if h:
                assert h['floor']==f and h['regime']==r and h['forbidden']==ph and h['history_floors']==[x for x in F if x!=f] and h['S_order']==sorder and h['model_hashes']==['cd5c4972f4ad418a58c9c73f88d9c65c245f24ecf5c35842bcdc8535114caf58','53061b01164f78def406e7b1395c5e62a295adb9a9a3570a2b91efed7b71e7c7'];wall+=h['history_secs'];quality[f'{r}_F{f}_H']=h['history_vi']
            for k in KS:
                tag=f'{r}_F{f}_K{k}';npz=p/f'K{k}.npz';js=p/f'K{k}.json'
                if not npz.is_file() or not js.is_file():
                    fp=p/f'failure_K{k}.json';out['missing'].append([tag,json.loads(fp.read_text())['error'][:350] if fp.exists() else 'not archived']);continue
                rec=json.loads(js.read_text());assert D(npz.read_bytes())==rec['npz_sha256'];row=np.load(npz,allow_pickle=False);np.testing.assert_array_equal(row['query_ids'],qids);np.testing.assert_array_equal(row['truth'],base['query_xy']);np.testing.assert_array_equal(row['support_ids'],sorder[:k]);assert len(row['aps'])==rec['ap_count'] and np.all(np.diff(row['aps'])>0);assert any(t['stationary'] for t in rec['weight_trials']);quality[tag]=rec['quality'];wall+=rec['seconds'];pd={};errors={};pm={}
                for a in A:
                    for method in (a,)+tuple(f'{a}_w{n}' for n in N):
                        pred=row[method];assert pred.shape==base['query_xy'].shape and np.isfinite(pred).all();md,tail,er,pos=stats(row['truth'],pred);np.testing.assert_allclose([md,tail],[rec['metrics'][method]['mde'],rec['metrics'][method]['p90']],rtol=0,atol=1e-8);pm[method]=[md,tail];errors[method]=er
                    pm[a+'_map']=pm[f'{a}_w{picks[a]}'];errors[a+'_map']=errors[f'{a}_w{picks[a]}'];
                    if k<=5:oldshift.setdefault((r,k,a),[]).append([pm[a][0]-stats(row['truth'],base[f'K{k}_{a}'])[0],float(np.linalg.norm(row[a]-base[f'K{k}_{a}'],axis=1).mean())]);
                if r=='official' and f==0 and k>=5:
                    old=data(hh,f'official_F0_0_K{k}.npz');np.testing.assert_array_equal(old['query_ids'],row['query_ids']);gap=max(float(np.abs(row[a]-old[f'K{k}_{a}']).max()) for a in A);out['max_F0_bridge_coordinate_gap']=max(out['max_F0_bridge_coordinate_gap'],gap);assert gap<.01,('F0 numerical bridge',k,gap)
                archival={x['method']:float(x['mde']) for x in wks[r] if (int(x['building']),int(x['floor']),int(x.get('draw',x.get('seed',-1))),int(x['K']))==(1,f,0,k)};assert archival==rec['archived_raw'];wm='joint_tuned' if r=='official' else 'strong_wknn';assert wm in archival;pm['empirical_source_tuned']=[archival[wm],None]
                if r=='official':pm['empirical_proto3']=[archival['prototype_fixed3'],None]
                by[r,f,k]=pm;out['floor_events'].append(dict(regime=r,floor=f,K=k,quantities=pm,weights=rec['weights'],positions=len(pos),ap_count=rec['ap_count'],grid=rec['grid'],seconds=rec['seconds']));
                if (r,f) not in boots:
                    _,bi=np.unique(np.floor(pos/8).astype('int64'),axis=0,return_inverse=True);bi=bi.ravel();nb=bi.max()+1;mm=rng.multinomial(nb,np.full(nb,1/nb),size=4000)[:,bi];boots[r,f]=mm/mm.sum(1)[:,None]
                for n,er in errors.items():eboot[r,f,k,n]=boots[r,f]@er
    def ci(v):return [float(x) for x in np.quantile(v,[.025,.975])]
    for r in REG:
        out['curves'][r]={};out['pairs'][r]={};out['k_changes'][r]={}
        for k in KS:
            if not all((r,f,k) in by for f in F):continue
            cc={n:[float(np.mean([by[r,f,k][n][0] for f in F])),None if 'empirical_' in n else float(np.mean([by[r,f,k][n][1] for f in F]))] for n in by[r,0,k]};out['curves'][r][k]=cc;out['pairs'][r][k]={}
            for a,b in [('seven','gp'),('nonlinear','gp'),('nonlinear','seven')]+[(a,a+'_map') for a in A]:
                d=np.mean([eboot[r,f,k,a]-eboot[r,f,k,b] for f in F],0);out['pairs'][r][k][a+'-'+b]=dict(delta=cc[a][0]-cc[b][0],interval=ci(d),floor_wins=sum(by[r,f,k][a][0]<by[r,f,k][b][0] for f in F))
        for left,right in list(zip(KS[:-1],KS[1:]))+[(1,20),(5,20)]:
            if left not in out['curves'][r] or right not in out['curves'][r]:continue
            out['k_changes'][r][f'{left}-{right}']={a:dict(delta=out['curves'][r][right][a][0]-out['curves'][r][left][a][0],interval=ci(np.mean([eboot[r,f,right,a]-eboot[r,f,left,a] for f in F],0))) for a in A}
    for (r,k,a),u in oldshift.items():out['backend'][f'{r}_K{k}_{a}']=np.mean(u,axis=0).tolist()
    out['receipt_counts']=dict(complete_B1=len(by),total_B1=40,source=2,curves_complete=len(by)==40);out['quality']=quality;out['wall_sum_s']=wall
    for e,val in quality.items():
        entries={'H':val} if 'total' in val else val
        for n,v in entries.items():
            if v['total']!=v['success'] or v['gt1e3']:out['warnings'].append([e,n,v])
    return out
if __name__=='__main__':
    root,ev,hi,lo,out=map(Path,sys.argv[1:]);record=verify(root,ev,hi,lo);out.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print('coverage',record['receipt_counts'],'source k',record['source_selected'],'previous bridge',record['max_F0_bridge_coordinate_gap'],'missing',record['missing'])
