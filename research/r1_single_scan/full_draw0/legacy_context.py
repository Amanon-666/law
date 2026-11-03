"""Secondary fixed-architecture comparison, not a new candidate. No B1 fitting.
python legacy_context.py ten.json 05-followup.zip legacy.json
Old methods supply per-event exact-position MDE only, not per-scan predictions.
Intervals in THIS file condition on all fixed query coordinates; resample draws only.
Do not confuse these with spatial+support intervals on GP and same-map controllers.
"""
import csv,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np

def compute(ten,archive):
    s=json.loads(ten.read_text());b=archive.read_bytes();assert hashlib.sha256(b).hexdigest()=='0f0a81839dd9527aeb740168f40d9ef7c14df58828a7e8b13046bcd60c3f39c3';z=zipfile.ZipFile(io.BytesIO(b));fit=json.loads(z.read('conditional-mean/fit.json'));assert fit['folds']['1']['source_buildings']==[0,2];result={'note':'Retrospective extra fixed/challenger checks. Not a new method. Same B1 original protocol and event keys. These baselines have different field construction and readout. Only MDE per event survives, so intervals below are conditional on the fixed Q and four floors, not spatial uncertainty. Source method training objectives are different. No new data or any combination of the per-protocol winners.', 'source_gains_old_learned':fit['folds']['1']['values']['gate'],'result':{}};raw={(r['protocol'],r['floor'],r['draw'],r['K']):r['mde'] for r in s['records']};rng=np.random.default_rng(26110311)
    for mode in ('official','phone'):
        vals=list(csv.DictReader(io.StringIO(z.read('conditional-mean/'+mode+'.csv').decode())));wp=list(csv.DictReader(io.StringIO(z.read('conditional-map-wknn/'+mode+'.csv').decode())));old={(int(r['floor']),int(r['draw']),int(r['K'])):r for r in vals if int(r['building'])==1};wk={(int(r['floor']),int(r['draw']),int(r['K'])):r for r in wp if int(r['building'])==1};assert old.keys()==wk.keys() and len(old)==200;out=result['result'][mode]={};bw=rng.multinomial(10,[.1]*10,size=(10000,4))/10.
        for K in (1,3,5,10,20):
            if any((mode,f,d,K) not in raw for f in range(4) for d in range(10)):out[K]={'complete':False};continue
            rows=[]
            for f in range(4):
                for d in range(10):
                    a=old[f,d,K];b=wk[f,d,K];assert max(abs(float(a[n])-float(b[n])) for n in ('fixed0.0','gate','old_wknn_gap_fit'))<1e-7;gc=raw[mode,f,d,K];rows.append(dict(fixed_floor_gap=float(a['fixed0.0']),old_conditional=float(a['gate']),old_floor_gap_wknn=float(a['old_wknn_gap_fit']),old_conditional_same_map_wknn=float(b['same_pair_wknn']),**{n:gc[n] for n in ('gp','seven','nonlinear','fixed_half','empirical')}))
            matrix={n:np.array([a[n] for a in rows]).reshape(4,10) for n in rows[0]};pairs={}
            for n in ('gp','seven','nonlinear'):
                for c in ('fixed_floor_gap','old_conditional','old_floor_gap_wknn'):
                    delta=matrix[n]-matrix[c];bt=(bw*delta).sum(2).mean(1);pairs[n+'-'+c]=dict(delta_U=float(delta.mean()),pct=float(100*delta.mean()/matrix[c].mean()),conditional_draw_only_95=np.quantile(bt,[.025,.975]).tolist(),floors_lower=int((delta.mean(1)<0).sum()),floor_delta=delta.mean(1).tolist())
            out[K]=dict(complete=True,macro={n:float(a.mean()) for n,a in matrix.items()},pairs=pairs)
    return result
if __name__=='__main__':
    ten,old,out=map(Path,sys.argv[1:]);out.write_text(json.dumps(compute(ten,old),indent=2,allow_nan=False)+'\n')
