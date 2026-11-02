"""Is the error only that the algorithm trusted the wrong historical floor?
Build original independent-source H GP priors using only F0/F3, H-only visible APs and its own H-only 2-U rectangle. No S measurements, coordinates or Q in an H-only map. Fixed default external floor-gap rule and 0.5 are legal. A 17-point grid w=0..1 and one S5-fitted w per episode without spatial S update diagnose source selection; hindsight oracle is deliberately allowed target labels: restrict to one common w for a whole floor (3 draws), separately per regime. It is neither the population limit nor a deployable choice. Retain the other-regime-at-selected-w value as a development check, never a confirmatory estimate. No new hyperparameter or default change."""
import sys,argparse,json
from pathlib import Path
import numpy as np,torch
from scipy.special import softmax
from data import group_scans
from run import tensor
from multi_history import single_history
from transfer import Prior,Transfer
from .model_ceiling import BASE,score
from ..prior_bridge import load
from ..replay import hashids
from research.uji_core.protocol import position_weights

def run(d,out):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);out.mkdir(exist_ok=True,parents=True);tr=load(d,'trainingData.csv');hs=[tr[k] for k in (0,3)];org=np.vstack([h['positions'] for h in hs]).mean(0);hh=[group_scans(h['rss'],h['xy']-org) for h in hs];aa=np.flatnonzero((hh[0]['rss']!=100).any(0)|(hh[1]['rss']!=100).any(0));pts=np.vstack([h['positions'] for h in hh]);ax=[np.linspace(x.min(),x.max(),int(np.ceil(np.ptp(x)/2.))+1) for x in pts.T];grid=tensor(np.stack(np.meshgrid(*ax),-1).reshape(-1,2));pp=json.loads((BASE/'field_parameters.json').read_text())['0'];base,obs=pp['base'],pp['obs'];ms=[];vs=[]
    for a in aa:
        m=[];v=[]
        for h in hh:
            hist=single_history({'scans':h},int(a),base,obs,steps=640);p=Prior(Transfer('identity'),hist,dict(base,drift=0.));mm,ft=p.features(grid);m.append(mm.detach().numpy());v.append(p.diagonal(grid,ft).detach().numpy())
        ms.append(m);vs.append(v)
    ms=np.asarray(ms);vs=np.asarray(vs);va=load(d,'validationData.csv');mroot=BASE/'round2/paper_replay';record=json.loads((mroot/'results/variance/summary.json').read_text())['events'];events={r['key']:r for r in json.loads((mroot/'results/replay/replay.json').read_text())['manifest']};outrec=[];pred={};wgrid=np.linspace(0,1,17)
    for f in [1,2]:
        t=tr[f];vv=va[f];Q=np.vstack([t['rss'][:,aa],vv['rss'][:,aa]]);Wg=float(softmax(-2*np.abs(np.array([0,3])-f))[0]);cases=[]
        for i in range(3):
            eid=f'B1F{f}_{i}';ws=float(next(e['weights'][0] for e in record if e['event']==eid));msr=events[eid+'_internal']['support_1based'];ix=np.flatnonzero(np.isin(t['ids'],msr));assert len(ix)==15;pos=t['positions'][np.unique(t['group'][ix])];ii=np.flatnonzero(~np.isin(t['group'],t['group'][ix]));oi=np.flatnonzero(~(vv['xy'][:,None]==pos[None]).all(-1).any(-1));sets={'internal':(ii,t['xy'][ii],t['ids'][ii]),'official':(len(t['rss'])+oi,vv['xy'][oi],vv['ids'][oi])};
            for s,(_,xy,ids) in sets.items():assert hashids(ids)==events[eid+'_'+s]['query_ids_sha256']
            cases.append((eid,ws,sets))
        weights=np.unique(np.r_[wgrid,Wg,[w for _,w,_ in cases]]);predByW={}
        for w in weights:
            mu=w*ms[:,0]+(1-w)*ms[:,1];v=w*w*vs[:,0]+(1-w)**2*vs[:,1]+base['drift']**2
            assert np.isfinite(mu).all() and np.isfinite(v).all() and (v>0).all();predByW[w]=softmax(score(Q,mu,v,obs),axis=1)@grid.numpy()+org
        for eid,ww,sets in cases:
            for s,(idx,xy,ids) in sets.items():
                wt=position_weights(xy);mde=lambda w:float(wt@np.linalg.norm(predByW[w][idx]-xy,axis=1));outrec.append(dict(event=eid,split=s,h_only_equal=mde(.5),h_only_gap=mde(Wg),S5_changes_only_weight=mde(ww),w_from_S5=ww,grid_mde=[mde(w) for w in wgrid]));k=eid+'_'+s;pred[k+'_ids']=ids;pred[k+'_xy']=xy;pred[k+'_h_equal']=predByW[.5][idx];pred[k+'_h_gap']=predByW[Wg][idx];pred[k+'_s_weight']=predByW[ww][idx]
    byfloor={};summary={}
    for s in ('internal','official'):
        other='official' if s=='internal' else 'internal';items=[]
        for f in (1,2):
            select=lambda sp:[e for e in outrec if e['split']==sp and e['event'].startswith(f'B1F{f}')];rr=select(s);v=np.mean([e['grid_mde'] for e in rr],axis=0);ov=np.mean([e['grid_mde'] for e in select(other)],axis=0);ix=v.argmin();a={k:float(np.mean([e[k] for e in rr])) for k in ['h_only_equal','h_only_gap','S5_changes_only_weight']};a.update(oracle_discrete=float(v[ix]),oracle_w=float(wgrid[ix]),chosen_on_other_regime=float(v[ov.argmin()]));items.append(a);byfloor[f'{s}_F{f}']=a
        summary[s]={k:float(np.mean([a[k] for a in items])) for k in a if k!='oracle_w'}
    result=dict(summary=summary,per_floor=byfloor,grid=wgrid.tolist(),events=outrec,history_ap_count=len(aa),grid_size=len(grid),scope='Oracle uses Q labels; 17 coefficients with no S update; same w over all three seeds per floor and regime. S5 weight is adapted and not H-only. All use H-only XY and APs.');(out/'envelope.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'envelope_pred.npz',**pred);print('ENVELOPE',summary,byfloor)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.data,a.out)
