"""Pure NumPy offline audit; original B1 R1 draws 0..9, not the extra 40-draw protocol.
python verify_all.py rows draw1 remaining evidence.zip WKNN_SOURCE_LOCK.json ten.json
"""
import csv,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
REG=('official','phone'); K=(1,3,5,10,20); M=('gp','seven','nonlinear'); V=(1,3,7,15); D=range(10); FS=range(4); BS=4000
HASH=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text())
def rd(z,f):return list(csv.DictReader(io.StringIO(z.read('results/'+f).decode())))
def asjson(x):return json.dumps(x,indent=2,allow_nan=False)+'\n'

def calc(d0,d1,tail,evidence,lock):
    known=read(lock)['selected'];assert known=={'gp':1,'seven':7,'nonlinear':3};z=zipfile.ZipFile(evidence);ms=z.read('results/manifest.json');ds=z.read('results/device_stress_design.json');assert HASH(ms)=='5f4fb0ceccc5a64aa7411cc287715e6fa9a39d593da6713910920a3fda67c613' and HASH(ds)=='76763a87569d885705dccc54cbd72e878f7a711cbb5343fbfee4517caf2482e6';man=json.loads(ms)['episodes'];dev=json.loads(ds)['events'];raw={};light={}
    for r,ff in [('official','wknn_events.csv'),('phone','device_stress_events.csv')]:
        for a in rd(z,ff):
            if int(a['building'])==1:raw.setdefault((r,int(a['floor']),int(a.get('draw',a.get('seed',-1))),int(a['K'])),{})[a['method']]=float(a['mde'])
        for a in rd(z,'smooth_scale_'+r+'.csv'):
            if int(a['building'])==1 and a['method']=='half10':light[r,int(a['floor']),int(a['draw']),int(a['K'])]=float(a['mde'])
    assert len(light)==len(raw)==400
    mme={};p90s={};err={};xypos={};visits=[];missing=[];wlog=[];hq={};wall=0.;nbq={};src_hashes=['cd5c4972f4ad418a58c9c73f88d9c65c245f24ecf5c35842bcdc8535114caf58','53061b01164f78def406e7b1395c5e62a295adb9a9a3570a2b91efed7b71e7c7'];baseh={};maxg=0.;absucc=0;solvecount=0
    for r in REG:
        for f in FS:
            for d in D:
                fld=(d0 if d==0 else d1 if d==1 else tail)/(('gridpilot' if d==0 else f'draw{d}')+f'-{r}-F{f}');hp=fld/'header.json'
                if not hp.exists():missing.extend([dict(protocol=r,floor=f,draw=d,K=k,error='missing shard header') for k in K]);continue
                h=read(hp);ev=next(v for v in (man if r=='official' else dev) if int(v['building'])==1 and int(v['floor'])==f and (r!='official' or int(v['draw'])==d));Q=ev['query_ids'];S=ev['support_order'] if r=='official' else ev['support_orders'][d];assert list(h['S_order'])==S and h['regime']==r and (h['building'],h['floor'],h['draw'])==(1,f,d) and h['forbidden']==(None if r=='official' else ev['phone']);assert h['Q_ids_hash']==HASH(np.asarray(Q,dtype='<i8').tobytes()) and len(S)==20 and len(set(S))==20;assert h['model_hashes']==src_hashes and h['history_floors']==[u for u in FS if u!=f]
                if (r,f) in baseh:assert baseh[r,f]==h['H_ids']
                baseh[r,f]=h['H_ids'];wall+=h['history_secs'];nbq[r,f]=len(Q);hq[r+'_'+str(f)+'_'+str(d)]=h['history_vi'];old_ap=set();
                for k in K:
                    key=(r,f,d,k);jp=fld/f'K{k}.json';fp=fld/f'K{k}.npz'
                    if not jp.exists() or not fp.exists():
                        ff=fld/f'failure_K{k}.json';missing.append(dict(protocol=r,floor=f,draw=d,K=k,error=read(ff)['error'][:300] if ff.exists() else 'missing JSON/NPZ'));continue
                    j=read(jp);assert HASH(fp.read_bytes())==j['npz_sha256'];npz=np.load(fp,allow_pickle=False);assert j['K']==k and np.array_equal(npz['support_ids'],S[:k]) and np.array_equal(npz['query_ids'],Q);ap=npz['aps'];assert len(ap)==j['ap_count'] and np.all(np.diff(ap)>0) and old_ap.issubset(set(ap));old_ap=set(ap);assert any(t['stationary'] for t in j['weight_trials']) and abs(sum(j['weights'])-1)<1e-8;assert raw[key]==j['archived_raw'];wall+=j['seconds'];true=npz['truth'];pos,gi,ct=np.unique(true,axis=0,return_inverse=True,return_counts=True);gi=gi.ravel();wt=1/(ct[gi]*len(ct));
                    if (r,f) in xypos:assert np.array_equal(xypos[r,f],pos)
                    xypos[r,f]=pos;mm={};qq={};warns=0
                    for n in M:
                        q=j['quality'][n];assert q['total']==len(ap);solvecount+=q['total'];absucc+=q['total']-q['success'];maxg=max(maxg,q['gradmax']);
                        if q['success']!=q['total'] or q['gt1e3']>0:wlog.append(dict(protocol=r,floor=f,draw=d,K=k,method=n,**q));warns+=1
                        for name in (n,)+tuple(f'{n}_w{v}' for v in V):
                            y=npz[name];assert np.isfinite(y).all() and y.shape==true.shape;loss=np.linalg.norm(y-true,axis=1);order=np.argsort(loss);m=float(wt@loss);q90=float(loss[order[min(len(order)-1,wt[order].cumsum().searchsorted(.9))]]);np.testing.assert_allclose([m,q90],[j['metrics'][name]['mde'],j['metrics'][name]['p90']],rtol=0,atol=1e-8);mm[name]=m;qq[name]=q90
                            if name==n or name==f'{n}_w{known[n]}':err[key+(n if name==n else n+'_map',)]=np.bincount(gi,loss)/ct
                        tag=f'{n}_w{known[n]}';mm[n+'_map']=mm[tag];qq[n+'_map']=qq[tag]
                    for tag in raw[key]:mm['measured_'+tag]=raw[key][tag]
                    mm['empirical']=raw[key]['joint_tuned' if r=='official' else 'strong_wknn'];mm['fixed_half']=light[key];mme[key]=mm;p90s[key]=qq;visits.append(dict(protocol=r,floor=f,draw=d,K=k,mde=mm,p90=qq,npos=len(pos),Q=len(Q),aps=len(ap),grid=j['grid'],weights=j['weights'],warnings=warns,seconds=j['seconds']));npz.close()
    fitted=M+tuple(a+'_map' for a in M);est={};tailest={};pair={};adj={};floor_mean={};bydraw={};seedrep={};eng={};rng=np.random.default_rng(261103);bmeans={};fwin={};calls={}
    for r in REG:
        ok=[k for k in K if all((r,f,d,k) in mme for f in FS for d in D)];est[r]={};tailest[r]={};pair[r]={};adj[r]={};floor_mean[r]={};bydraw[r]={};seedrep[r]={};eng[r]={};fwin[r]={};calls[r]={};
        # Do not exclude incomplete floor/draw cells silently. Separately record complete individual budgets.
        for k in ok:
            ns=list(mme[r,0,0,k]);est[r][k]={n:float(np.mean([mme[r,f,d,k][n] for f in FS for d in D])) for n in ns};tailest[r][k]={n:float(np.mean([p90s[r,f,d,k][n] for f in FS for d in D])) for n in p90s[r,0,0,k]};floor_mean[r][k]={n:[float(np.mean([mme[r,f,d,k][n] for d in D])) for f in FS] for n in ns};bydraw[r][k]={n:[float(np.mean([mme[r,f,d,k][n] for f in FS])) for d in D] for n in ns};seedrep[r][k]={n:float(np.mean(bydraw[r][k][n][2:])) for n in ns};
        for f in FS:
            if not ok:continue
            pos=xypos[r,f];_,bi=np.unique(np.floor(pos/8).astype(int),axis=0,return_inverse=True);bi=bi.ravel();nb=bi.max()+1;mult=rng.multinomial(nb,np.full(nb,1/nb),size=BS)[:,bi];bsw=mult/mult.sum(1)[:,None];dw=rng.multinomial(10,[.1]*10,size=BS)/10.;
            for k in ok:
                for n in fitted:
                    tmp=bsw@np.stack([err[r,f,d,k,n] for d in D],axis=1);boot=(tmp*dw).sum(1)/4;idx=r,k,n;bmeans[idx]=bmeans.get(idx,0.)+boot
        for k in ok:
            pairs=[('seven','gp'),('nonlinear','gp'),('nonlinear','seven')]+[(n,n+'_map') for n in M];pair[r][k]={}
            for a,b in pairs:
                delta=est[r][k][a]-est[r][k][b];ci=np.quantile(bmeans[r,k,a]-bmeans[r,k,b],[.025,.975]);fd=np.subtract(floor_mean[r][k][a],floor_mean[r][k][b]);dd=np.subtract(bydraw[r][k][a],bydraw[r][k][b]);pair[r][k][a+'-'+b]=dict(delta=delta,pct=100*delta/est[r][k][b],ci95=ci.tolist(),floor_delta=fd.tolist(),draw_wins=int((dd<0).sum()),floor_wins=int((fd<0).sum()),new8_delta=seedrep[r][k][a]-seedrep[r][k][b])
        for l,h in list(zip(K[:-1],K[1:]))+[(1,20),(5,20)]:
            if l not in ok or h not in ok:continue
            name=f'{l}-{h}';adj[r][name]={}
            for n in fitted:
                adj[r][name][n]=dict(delta=est[r][h][n]-est[r][l][n],ci95=np.quantile(bmeans[r,h,n]-bmeans[r,l,n],[.025,.975]).tolist(),floor_delta=(np.subtract(floor_mean[r][h][n],floor_mean[r][l][n])).tolist())
    return dict(expected=400,verified=len(mme),missing=missing,source_selected=known,query_rows={r:[nbq.get((r,f),0) for f in FS] for r in REG},curves=est,p90=tailest,paired=pair,adjacent=adj,floor_means=floor_mean,by_draw=bydraw,first_unreported_eight=seedrep,records=visits,optimizer=dict(site_AP_arm_calls=solvecount,unsuccessful_exit=absucc,max_projected_grad=maxg,list=wlog,history=hq),cpu_shard_elapsed_sum_seconds=wall,design='B1, ten pre-existing nested support orders, common Q re-used across budgets and draws. Conditional 8U spatial blocks + support resampling, floors fixed, 4000 bootstraps. Pointwise intervals not multiplicity adjusted. One building, not a new blind confirmation. Prior 40 additional draws and all external holds unopened.')
if __name__=='__main__':
    p=list(map(Path,sys.argv[1:]));s=calc(*p[:-1]);p[-1].write_text(asjson(s));print('all:',s['verified'],'missing:',s['missing'],'VI abnormal:',s['optimizer']['unsuccessful_exit'],'projected max',s['optimizer']['max_projected_grad']);
    for r,cv in s['curves'].items():
        for k,values in cv.items():print(r,k,{n:round(values[n],3) for n in ['gp','seven','nonlinear','fixed_half','empirical','gp_map','seven_map','nonlinear_map']})
