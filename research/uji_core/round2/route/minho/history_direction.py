"""One intervention: shrink noisy between-anchor contrasts toward an independent older map.
CI supplies the original, hash-checked Minho lite zip. No December/later holdouts.
First eligible triple fits on July 3; only September 11 and November 13 score transfer.
A past evaluation visit can later be known history, but never enters the same fit as future input.
"""
import argparse, hashlib, json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from scipy.spatial.distance import cdist
from external_baselines import tpm_adapter as tpm
from refi import smooth_history
from . import check_order as r

SCALES=(0.,.25,1.,4.); BUDGETS=(1,3,5,10,20); FAMILIES=('uniform','geometry','direction'); DRAWS=10

def graph(x):
    n=len(x)
    if n==1:return np.empty((0,n)),np.array([])
    dist=cdist(x,x);np.fill_diagonal(dist,np.inf);near=dist.argmin(1)
    edges=sorted({tuple(sorted([i,int(j)])) for i,j in enumerate(near)})
    a,b=np.asarray(edges).T;D=np.eye(n)[a]-np.eye(n)[b];return D,dist[a,b]

def agreement(ds,dh):
    norm=np.linalg.norm(ds,axis=1)*np.linalg.norm(dh,axis=1)
    return np.clip(np.divide(np.einsum('ij,ij->i',ds,dh),norm,out=np.zeros_like(norm),where=norm>1e-12),-1.,1.)

def auc(s,y):
    a=s[:,None]-s[None,:];w=y[:,None]*(1-y[None,:]);return float(np.sum(w*((a>0)+.5*(a==0)))/w.sum()) if w.sum() else .5

def run_visit(z,j,receipts):
    histday=r.VISITS[j-1][1];sday,qday=r.VISITS[j];assert histday<sday<qday
    hr,hmap,_=r.visit(z,histday,receipts);sr,smap,_=r.visit(z,sday,receipts);qr,qmap,_=r.visit(z,qday,receipts)
    hp=sorted(hmap);sp=sorted(smap);hxy=np.array(hp)*1e-6;sxy=np.array(sp)*1e-6
    # No Q information is used to select or order support or historical positions.
    if len(sp)<=20:raise ValueError('No internal spatial holdout')
    records=[];risk=[]
    for draw in range(DRAWS):
        sid=r.sample_rows(sday,draw,sp,smap)
        S=[sr[i] for i in sid];hk={a for row in hr for a in row}
        seed=int.from_bytes(hashlib.sha256(f'minho-diff-v1/{sday}/{draw}'.encode()).digest()[:8],'big')
        order=np.random.default_rng(seed).permutation(len(sp))[:20]
        Q=sorted(set(qmap)-{sp[i] for i in order});assert Q
        qrows=[i for p in Q for i in qmap[p]];truth=np.array([p for p in Q for _ in qmap[p]])*1e-6
        wt=np.concatenate([np.full(len(qmap[p]),1/(len(Q)*len(qmap[p]))) for p in Q])
        for k in BUDGETS:
            ids=order[:k];ps=sxy[ids];axis=sorted(hk|{a for i in ids for a in S[i]});s=r.matrix([S[i] for i in ids],axis)
            hm=r.matrix(hr,axis);hv=np.stack([hm[hmap[p]].mean(0) for p in hp]);prior=smooth_history((hv,hxy),7.66,.5)
            hist_at=tpm.lookup(ps,hxy,prior[0]);D,d=graph(ps);ds,dh=D@s,D@hist_at
            g={'uniform':np.ones(len(d)), 'geometry':1/np.maximum(d,1e-8), 'direction':(1-agreement(ds,dh))/2}
            # Normalize each criterion's mean edge strength; a scalar magnitude is not a content advantage.
            g={name:w/(w.mean() if len(w) and w.mean()>1e-12 else 1.) for name,w in g.items()}
            residual=s-hist_at;ph,xy=tpm.propagate(prior,s,ps,residual,(30.64,1.));assert np.array_equal(xy[-k:],ps)
            q=r.matrix([qr[i] for i in qrows],axis);q=np.where(q==r.MISS,100.,q);center=xy[0];coords=xy-center
            # Full history bank and single-scan anchors, all fixed across variants. No separate risk threshold.
            for family,w in g.items():
                L=D.T@(w[:,None]*D)
                for lam in SCALES:
                    fitted=hist_at+np.linalg.solve(np.eye(k)+lam*L,residual)
                    np.testing.assert_allclose(fitted.sum(0),s.sum(0),rtol=1e-9,atol=1e-8)
                    proto=np.vstack([ph[:-k],np.clip(fitted,-110.,0.)]);p=tpm.tobit_weights(q,proto,12.,-85.)@coords+center
                    loss=float(wt@np.linalg.norm(p-truth,axis=1));assert np.isfinite(loss)
                    records.append(dict(draw=draw,K=k,family=family,lam=lam,mde=loss,nq=len(Q),ns=len(q),clip=float(((fitted < -110.-1e-8)|(fitted>0.+1e-8)).mean())))
        # Separate diagnostic, not a K-shot localization score. Full earlier S bank, nearest spatial foil.
        axis=sorted(hk|{a for row in S for a in row});s=r.matrix(S,axis);hm=r.matrix(hr,axis)
        hv=np.stack([hm[hmap[p]].mean(0) for p in hp]);h=tpm.lookup(sxy,hxy,smooth_history((hv,hxy),7.66,.5)[0]);d=cdist(sxy,sxy);np.fill_diagonal(d,np.inf);nn=d.argmin(1)
        ids=np.array([i for i,p in enumerate(sp) if p in qmap]);n=nn[ids];ds=s[ids]-s[n];dh=h[ids]-h[n]
        qq=r.matrix(qr,axis);bad=[]
        for i,m in zip(ids,n):
            sign,_=r.distances(qq[qmap[sp[i]]],s[i],s[m]);bad.append((sign<0).mean()+.5*(sign==0).mean())
        y=np.array(bad);score={'direction':-agreement(ds,dh),'geometry':-d[ids,n],'rss':-np.linalg.norm(ds,axis=1),'uniform':np.zeros(len(ids))}
        risk.append({name:auc(v,y) for name,v in score.items()})
    return dict(H=histday,S=sday,Q=qday,history_positions=len(hp),support_pool=len(sp),risk_positions=len(ids),risk_auc={name:float(np.mean([v[name] for v in risk])) for name in risk[0]},rows=records)

def means(rows):return {f:{str(l):float(np.mean([v['mde'] for v in rows if v['family']==f and v['lam']==l])) for l in SCALES} for f in FAMILIES}

def main(path,out):
    with path.open('rb') as f:assert hashlib.file_digest(f,'md5').hexdigest()=='4c15884cb7edcd2c602ec5d567ddd8a8'
    receipts={}
    with ZipFile(path) as z:
        source=run_visit(z,1,receipts);m=means(source['rows']);selected={f:min(SCALES,key=lambda l:m[f][str(l)]) for f in FAMILIES}
        test=[run_visit(z,j,receipts) for j in (2,3)]
    assert set(receipts)<=r.ALLOW
    summary=[];paths=[]
    for t in test:
        mm=means(t['rows']);summary.append(dict(Q=t['Q'],risk_auc=t['risk_auc'],mde=mm,chosen={f:mm[f][str(selected[f])] for f in FAMILIES},history_positions=t['history_positions'],risk_positions=t['risk_positions']))
        losses={(a['draw'],a['K'],a['family'],a['lam']):a['mde'] for a in t['rows']}
        paths.append(np.array([[[np.mean([losses[i,k,'direction',selected['direction']]-losses[i,k,f,l] for k in BUDGETS]) for f,l in [('uniform',0.),('uniform',selected['uniform']),('geometry',selected['geometry']),('direction',1.)]]] for i in range(DRAWS)]).reshape(DRAWS,4))
    rr=np.random.default_rng(17811);boot=np.mean([a[rr.integers(DRAWS,size=(5000,DRAWS))].mean(1) for a in paths],axis=0)
    ci=np.quantile(boot,[.025,.975],axis=0).T.tolist();dr=np.mean(paths,axis=(0,1)).tolist();contrasts={f:dict(delta=d,ci95=c) for f,d,c in zip(('no_change','learn_uniform','learn_geometry','fixed_direction_1'),dr,ci)}
    report=dict(scope='One development site. Conditional support-draw intervals only; not independent site confirmation. AP axis H union current S. Fixed H+S coordinates; H is preceding separate survey. Q never changes graph/candidates. K=1 all identical; actual MDE includes only Q at locations not in S20.',selected=selected,source=source['Q'],source_mde=m,source_auc=source['risk_auc'],test=summary,differences=contrasts,read_members=receipts,events=[source,*test]);out.mkdir(exist_ok=True,parents=True)
    (out/'direction.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('DIRECTION_SUMMARY',json.dumps({k:v for k,v in report.items() if k not in ('read_members','events')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('zip',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.zip,a.out)
