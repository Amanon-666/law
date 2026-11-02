"""Non-deployable empirical references: only the four previously opened Minho dates pairs.
2x2: ten earlier vs ten contemporaneous reference scans per listed site; one query
versus their ten-scan average costs ten acquisitions. No tuning or neural model.
Baseline = earlier map / single query. Unit = metres. ReFi/UJI results are separate.
The same-date reference is intentionally privileged by session and target-site map.
"""
import argparse, hashlib, json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from research.uji_core.round2.route.minho import check_order as ref


def predict(x, templates, xy):
    distances=np.sqrt(((templates-x[None,:])**2).mean(1))
    ix=np.argsort(distances,kind='stable')[:3]; weights=1/np.maximum(distances[ix],1.e-8)
    return (weights/weights.sum())@xy[ix]


def run(path,out):
    with path.open('rb') as f: assert hashlib.file_digest(f,'md5').hexdigest()=='4c15884cb7edcd2c602ec5d567ddd8a8'
    visits=[]; receipts={}
    with ZipFile(path) as z:
        for old, new in ref.VISITS:
            E,em,_=ref.visit(z,old,receipts); L,lm,_=ref.visit(z,new,receipts)
            names=[ref.PREFIX+t+'/timestamps.csv' for t in (old,new)];assert all(n in ref.ALLOW for n in names)
            et,lt=[z.read(n).decode('ascii').splitlines() for n in names]
            pts=[];early=[];current=[];query=[];excluded=0
            for u in sorted(em.keys() & lm.keys()):
                a=sorted(em[u],key=lambda i:(et[i],i));b=sorted(lm[u],key=lambda i:(lt[i],i))
                if len(a)<10 or len(b)<20 or lt[b[9]]>=lt[b[-10]]: excluded+=1;continue
                pts.append(u);early.append(a[:10]);current.append(b[:10]);query.append(b[-10:]);assert not(set(b[:10])&set(b[-10:]))
            assert len(pts)>=8
            af=sorted({ap for gg in early for i in gg for ap in E[i]});xy=np.asarray(pts)/1.e6
            M=[np.stack([ref.matrix([rows[i] for i in gg],af).mean(0) for gg in groups]) for rows,groups in [(E,early),(L,current)]]
            errs=[]
            for h,group in enumerate(query):
                q=ref.matrix([L[i] for i in group],af); v=[]
                for m in M:
                    v.extend([float(np.mean([np.linalg.norm(predict(x,m,xy)-xy[h]) for x in q])),float(np.linalg.norm(predict(q.mean(0),m,xy)-xy[h]))]);
                errs.append(v)
            err=np.array(errs);visits.append(dict(early=old,late=new,sites=len(xy),excluded_sites=excluded,aps=len(af),real_query_scans=10*len(xy),mde=err.mean(0).tolist(),xy=xy.tolist(),site_mde=err.tolist(),selection_sha256=hashlib.sha256(repr((early,current,query)).encode()).hexdigest()))
    assert set(receipts)==ref.ALLOW
    out.mkdir(exist_ok=True,parents=True); names=['earlier_single','earlier_ten','current_single','current_ten']; macro=np.mean([a['mde'] for a in visits],0)
    report=dict(meaning='Same listed sites, fixed old-template AP axis, deterministic 3-WKNN on filled RSS. First ten later rows build the privileged same-session templates; last ten, strictly separated by timestamp, are queries. Early templates also use ten rows. No future RSS used by the earlier-map arm. Same-session serial correlation and the known location list make this an optimistic reference; no universal physical limit, no B1 evidence.',columns=names,macro_mde_m=dict(zip(names,macro.tolist())),visits=visits,read_manifest=receipts)
    (out/'minho_age_and_repeats.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(report['macro_mde_m'],[(x['sites'],x['mde']) for x in visits])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('zip',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.zip,a.out)
