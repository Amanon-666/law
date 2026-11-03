"""Offline joint accounting: python verify_two.py draw0/ draw1/ evidence.zip WKNN_SOURCE_LOCK.json joint.json
Draw0 independently bridged to the historically scored backend by verify.py; here recheck both draws.
Only use a K in a two-draw building average if every constituent floor/draw is present.
"""
from pathlib import Path
import hashlib,zipfile,json,io,csv,sys
import numpy as np
KS=(1,3,5,10,20); A=('gp','seven','nonlinear'); N=(1,3,7,15)
def parse(path):return json.loads(path.read_text())
def sha(x):return hashlib.sha256(x).hexdigest()
def main(d0,d1,ev,lock):
    z=zipfile.ZipFile(ev);picks=parse(lock)['selected'];assert picks=={'gp':1,'seven':7,'nonlinear':3};D={};recs=[];warn=[];miss=[];headers=[];xybank={};rs=np.random.default_rng(261103);ws={};ds={};eb={};zboot={};rawtables={};wall=0
    manifest=json.loads(z.read('results/manifest.json'))['episodes'];phone=json.loads(z.read('results/device_stress_design.json'))['events'];files=['wknn_events.csv','device_stress_events.csv'];reads={r:list(csv.DictReader(io.StringIO(z.read('results/'+n).decode()))) for r,n in zip(('official','phone'),files)}
    light={r:{(int(v['floor']),int(v['draw']),int(v['K'])):float(v['mde']) for v in csv.DictReader(io.StringIO(z.read(f'results/smooth_scale_{r}.csv').decode())) if int(v['building'])==1 and v['method']=='half10'} for r in ('official','phone')}
    for r in ('official','phone'):
        for f in range(4):
            refh=None
            for d,root in enumerate((d0,d1)):
                dirname=('gridpilot' if d==0 else 'draw1')+f'-{r}-F{f}';fld=root/dirname;hp=fld/'header.json';
                if not hp.exists():miss.extend([[r,f,d,k,'missing entire shard'] for k in KS]);continue
                h=parse(hp);m=next(x for x in (manifest if r=='official' else phone) if x['building']==1 and x['floor']==f and (r!='official' or x['draw']==d));ids=m['query_ids'];order=m['support_order'] if r=='official' else m['support_orders'][d];assert h['building']==1 and h['floor']==f and h['draw']==d and h['regime']==r and h['S_order']==order and h['forbidden']==(None if r=='official' else m['phone']);assert h['Q_ids_hash']==sha(np.asarray(ids,dtype='<i8').tobytes());assert h['history_floors']==[ff for ff in range(4) if ff!=f] and h['model_hashes']==['cd5c4972f4ad418a58c9c73f88d9c65c245f24ecf5c35842bcdc8535114caf58','53061b01164f78def406e7b1395c5e62a295adb9a9a3570a2b91efed7b71e7c7'];
                if refh is not None:assert h['H_ids']==refh
                refh=h['H_ids'];headers.append([r,f,d,h['history_vi']]);wall+=h['history_secs']
                for k in KS:
                    jp=fld/f'K{k}.json';npz=fld/f'K{k}.npz';
                    if not jp.exists() or not npz.exists():
                        fp=fld/f'failure_K{k}.json';miss.append([r,f,d,k,parse(fp)['error'][:180] if fp.exists() else 'missing record']);continue
                    a=np.load(npz,allow_pickle=False);j=parse(jp);assert sha(npz.read_bytes())==j['npz_sha256'] and j['K']==k;np.testing.assert_array_equal(a['support_ids'],order[:k]);np.testing.assert_array_equal(a['query_ids'],ids);true=a['truth'];
                    if (r,f) in xybank:np.testing.assert_array_equal(xybank[r,f],true)
                    xybank[r,f]=true;pos,g,ct=np.unique(true,axis=0,return_inverse=True,return_counts=True);g=g.ravel();ww=1/(ct[g]*len(ct));assert len(a['aps'])==j['ap_count'] and len(j['weights'])==3 and any(t['stationary'] for t in j['weight_trials']);wall+=j['seconds'];pm={};quality=j['quality'];
                    if (r,f) not in ws:
                        _,bl=np.unique(np.floor(pos/8).astype(int),axis=0,return_inverse=True);bl=bl.ravel();nb=int(bl.max())+1;bc=rs.multinomial(nb,np.full(nb,1/nb),size=4000)[:,bl];ws[r,f]=bc/bc.sum(1)[:,None];ds[r,f]=rs.integers(0,2,(4000,2))
                    for n in A:
                        if quality[n]['total']!=quality[n]['success'] or quality[n]['gt1e3']:warn.append([r,f,d,k,n,quality[n]])
                        for key in (n,)+tuple(f'{n}_w{ne}' for ne in N):
                            pred=a[key];er=np.linalg.norm(pred-true,axis=1);assert pred.shape==true.shape and np.isfinite(er).all();ix=np.argsort(er);mde=float(ww@er);tail=float(er[ix[np.searchsorted(ww[ix].cumsum(),.9)]]);np.testing.assert_allclose([mde,tail],[j['metrics'][key]['mde'],j['metrics'][key]['p90']],rtol=0,atol=1e-8);pm[key]=[mde,tail]
                            if key in A or key==f'{n}_w{picks[n]}':eb[r,f,d,k,(n if key==n else n+'_map')]=ws[r,f]@(np.bincount(g,er)/ct)
                        pm[n+'_map']=pm[f'{n}_w{picks[n]}']
                    raw={x['method']:float(x['mde']) for x in reads[r] if (int(x['building']),int(x['floor']),int(x.get('draw',x.get('seed',-1))),int(x['K']))==(1,f,d,k)};assert raw==j['archived_raw'];pm['empirical']=[raw['joint_tuned' if r=='official' else 'strong_wknn'],None];pm['fixed_half']=[light[r][f,d,k],None];
                    if r=='official':pm['empirical_proto3']=[raw['prototype_fixed3'],None]
                    D[r,f,d,k]=pm;recs.append(dict(regime=r,floor=f,draw=d,K=k,metrics=pm,npos=len(pos),aps=len(a['aps']),grid=j['grid'],weight=j['weights'],quality=quality))
    small=A+tuple(a+'_map' for a in A);cv={};pairs={};sub={};gaps={};wins={};adj={}
    for r in ('official','phone'):
        cv[r]={};pairs[r]={};sub[r]={};gaps[r]={};wins[r]={};adj[r]={}
        for d in (0,1):
            sub[r][d]={}
            for k in KS:
                if all((r,f,d,k) in D for f in range(4)):sub[r][d][k]={n:float(np.mean([D[r,f,d,k][n][0] for f in range(4)])) for n in D[r,0,d,k]}
        for k in KS:
            if not all((r,f,d,k) in D for f in range(4) for d in (0,1)):continue
            cc={n:[float(np.mean([D[r,f,d,k][n][0] for f in range(4) for d in (0,1)])),None if D[r,0,0,k][n][1] is None else float(np.mean([D[r,f,d,k][n][1] for f in range(4) for d in (0,1)]))] for n in D[r,0,0,k]};cv[r][k]=cc
            for n in small:
                vec=[]
                for f in range(4):
                    both=np.stack([eb[r,f,d,k,n] for d in (0,1)],axis=1);v=np.take_along_axis(both,ds[r,f],axis=1).mean(1);vec.append(v)
                zboot[r,k,n]=np.mean(vec,0)
            pp={};wi={}
            for a,b in [('seven','gp'),('nonlinear','gp'),('nonlinear','seven')]+[(a,a+'_map') for a in A]:
                dif=cc[a][0]-cc[b][0];fl=[np.mean([D[r,f,d,k][a][0]-D[r,f,d,k][b][0] for d in (0,1)]) for f in range(4)];pp[a+'-'+b]=dict(delta=dif,relative_pct=100*dif/cc[b][0],ci95=np.quantile(zboot[r,k,a]-zboot[r,k,b],[.025,.975]).tolist(),floor_delta=list(map(float,fl)));wi[a+'-'+b]=int(sum(x<0 for x in fl))
            pairs[r][k]=pp;wins[r][k]=wi
        for l,h in list(zip(KS[:-1],KS[1:]))+[(1,20),(5,20)]:
            if l in cv[r] and h in cv[r]:adj[r][f'{l}-{h}']={n:{'delta':cv[r][h][n][0]-cv[r][l][n][0],'ci95':np.quantile(zboot[r,h,n]-zboot[r,l,n],[.025,.975]).tolist()} for n in A}
    return dict(coverage=len(D),expected=80,missing=miss,source_k=picks,curves=cv,by_draw=sub,pairs=pairs,adjacent=adj,wins=wins,floors=recs,history_diagnostics=headers,numerical_warnings=warn,compute_seconds=sum(float(x) for x in [wall]),scope='Fixed B1 four floors, two support draws only, no new site or confirmation. Curves omit budgets without all eight matched floor/draw cells. P90 is the mean of within-floor/draw position-weighted scan quantiles. Archive-only fixed_half is the established lightweight H-map half smoothing, different grid and reader; not an additional GP arm.')
if __name__=='__main__':
    p=[Path(x) for x in sys.argv[1:]];s=main(*p[:-1]);p[-1].write_text(json.dumps(s,indent=2,allow_nan=False)+'\n');print(s['coverage'],s['missing'])
