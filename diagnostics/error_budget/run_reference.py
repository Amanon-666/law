"""T9 a/b/d cross-block privileged-label diagnostic. c/E_naive absent."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import sys
import time

import numpy as np
import torch
from scipy.spatial import Delaunay, cKDTree

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from data import group_scans
from full_ap_study import metrics
from history_bank import BankPrior
from inference import observation_log_prob
from multi_history import single_history
from run import ap_stats,tensor
from run_multi_history import digest,write
from run_operator_prior import load_history
from source_weight_mixture import field_from_prior
from transfer import History,Prior,Transfer,rbf
from uji_probability_study import DATA,load_raw

PREP=ROOT/'results/uji_error_budget_preparation_v1_20261026'
OLD=ROOT/'results/uji_history_bank_v1_20261026'
BASELINE=ROOT/'results/uji_operator_paired_v2_20261026'
OUT=ROOT/'results/uji_error_budget_reference_v1_20261026'


class MeanReference:
    """Replace mean only; original features remain the covariance arguments."""
    def __init__(self,bank,reference):self.bank,self.reference=bank,reference
    def features(self,x):
        _,features=self.bank.features(x)
        mean,_=self.reference.features(x)
        return mean,features
    def covariance(self,x,y,fx,fy):return self.bank.covariance(x,y,fx,fy)
    def diagonal(self,x,fx):return self.bank.diagonal(x,fx)


def log(**r):print(json.dumps(r,allow_nan=False),flush=True)
def params():return json.loads((OLD/'parameters.json').read_text())
def manifests():return json.loads((PREP/'manifest.json').read_text())['events'],json.loads((OLD/'manifest.json').read_text())['events']
def indices(f,rows):
    lookup={int(r):i for i,r in enumerate(f['row_ids'])}
    return np.array([lookup[r] for r in rows],dtype=int)
def weights(xy):
    _,g,n=np.unique(xy,axis=0,return_inverse=True,return_counts=True)
    return 1/(len(n)*n[g.ravel()])
def grid_for(meta,origin):
    b=np.array(meta['grid_bounds'])-origin
    axes=[np.linspace(b[0,k],b[1,k],int(np.ceil((b[1,k]-b[0,k])/2))+1) for k in range(2)]
    return tensor(np.stack(np.meshgrid(*axes),-1).reshape(-1,2))


def check():
    base,obs=params()['base'],params()['obs']
    z=tensor([[0.,0.],[6.,0.],[0.,8.]])
    h=History(z,dict(mean=tensor([-70.,-85.,-90.]),cov=.2*rbf(z,z,base['amplitude'],base['length'])),base)
    p=Prior(Transfer('identity'),h,dict(base,drift=0.))
    bank=BankPrior([h],[1.],base)
    x=tensor([[1.,2.],[9.,4.]]);m,f=p.features(x)
    projection=h.projection(x)
    torch.testing.assert_close(m,base['mean']+projection@(h.m-base['mean']),rtol=0,atol=1e-12)
    expected=h.kernel(x,x)+projection@(h.c-h.kernel(z,z))@projection.T
    # Cubature/projection use the existing 1e-6 Cholesky jitter.
    torch.testing.assert_close(p.covariance(x,x,f,f),expected,rtol=0,atol=2e-6)
    same=MeanReference(bank,bank)
    bm,bf=bank.features(x);sm,sf=same.features(x)
    torch.testing.assert_close(bm,sm,rtol=0,atol=0)
    torch.testing.assert_close(bank.covariance(x,x,bf,bf),same.covariance(x,x,sf,sf),rtol=0,atol=0)
    counts=dict(n=tensor([3.,3.,3.]),k=tensor([3.,2.,0.]),s1=tensor([-213.,-170.,0.]),s2=tensor([15125.,14452.,0.]))
    a=field_from_prior(bank,obs,z,counts,x,261026,defensive_fraction=.1)
    b=field_from_prior(same,obs,z,counts,x,261026,defensive_fraction=.1)
    for key in ['mean','variance']:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    empty=Prior(Transfer('identity'),None,dict(base,drift=0.));em,ef=empty.features(x)
    assert ef is None
    torch.testing.assert_close(empty.diagonal(x,ef),x.new_full((len(x),),base['amplitude']**2))
    prep,old=manifests();train=load_raw('trainingData.csv',{'B1F1','B1F2'});val=load_raw('validationData.csv',{'B1F1','B1F2'})
    checks=[]
    for e in prep:
        meta=next(m for m in old if m['id']==e['event']);f,o=train[e['floor']],val[e['floor']]
        qi,qo=indices(f,meta['internal_rows']),indices(o,meta['official_rows'])
        xy=np.concatenate([f['xy'][qi],o['xy'][qo]])
        with np.load(BASELINE/meta['id']/'predictions.npz') as pred:
            np.testing.assert_allclose(pred['query_xy']+pred['origin'],xy,rtol=0,atol=1e-8)
            completed=json.loads((BASELINE/meta['id']/'completed.json').read_text())['results']
            for split,sl in [('internal',slice(0,len(qi))),('official',slice(len(qi),None))]:
                value=metrics(pred['adapted_bank'][sl],dict(xy=pred['query_xy'][sl],scan_weights=weights(xy[sl])))['mde']
                expected_mde=next(r['mde'] for r in completed if r['method']=='adapted_bank' and r['split']==split)
                assert abs(value-expected_mde)<1e-10
        for split in ['internal','official']:
            collected=[r for fold in e['folds'] for r in fold['splits'][split]['query_rows']]
            assert sorted(collected)==sorted(meta[split+'_rows']) and len(set(collected))==len(collected)
        checks.append(dict(event=meta['id'],baseline_alignment=True,fold_partition=True))
    OUT.mkdir(exist_ok=True)
    write(OUT/'check.json',dict(created_unix=time.time(),same_mean_field_exact=True,zero_drift_GP_covariance_atol=2e-6,
        missing_reference_fallback=True,baseline_alignment=checks,script_sha256=digest(Path(__file__))))
    log(stage='check',passed=True)


def freeze():
    assert (OUT/'check.json').exists() and not (OUT/'freeze.json').exists()
    assert json.loads((OUT/'check.json').read_text())['script_sha256']==digest(Path(__file__))
    preparation=json.loads((PREP/'freeze.json').read_text())
    for path,sha in preparation['source_paths'].items():assert digest(Path(path))==sha,path
    files=[Path(__file__),ROOT/'diagnostics/error_budget/prepare.py']+[ROOT/(n+'.py') for n in
        ['multi_history','inference','data','transfer','history_bank','source_weight_mixture','reference_pmc','full_ap_study','run','run_multi_history','run_operator_prior','uji_probability_study']]
    inputs=[PREP/n for n in ['freeze.json','manifest.json','summary.json']]+[OLD/'parameters.json',OLD/'manifest.json',OLD/'histories/completed.json',DATA/'trainingData.csv',DATA/'validationData.csv']
    for e in json.loads((OLD/'manifest.json').read_text())['events']:
        inputs += [BASELINE/e['id']/'predictions.npz',BASELINE/e['id']/'completed.json',OLD/e['id']/'adaptation.json']
        inputs += [BASELINE/e['id']/f'ap{a}.npz' for a in e['ap_columns']]
    inputs+=sorted((OLD/'histories').glob('B1F[03]/ap*.npz'))
    write(OUT/'freeze.json',dict(created_unix=time.time(),purpose=__doc__,
        arms=dict(a='exact cached adapted_bank',b='extra-target GP posterior mean only; original Bank covariance and S update',
            d='cross-block sufficient-calibration reference; same target GP posterior covariance, zero drift, then S update'),
        absent=['c: target observation fitting','E_naive: no source-experience hyperparameters'],
        hypothesis='If b provides better true-position RSS and localization than a, the fixed-covariance mean intervention is useful. Negative localization with worse reference RSS cannot reject mean bottleneck. d is a fallible cross-block reference, not a lower bound.',
        training='Only each prepared fold reference_rows; exact original 15 S rows added only by field_from_prior; no Q RSS, AP reselection, domain expansion or obs refit',
        settings=dict(reference_VI_maxiter=640,defensive_fraction=.1,seed='261026+10000*original_event_seed+AP',grid_spacing=2.,threads=2,GPU=False),
        readout='Same summed original-AP observation log likelihood and grid posterior mean; all grid candidates retained for every fold',
        fallback='single_history returns None if reference pool never detects AP; b uses corresponding base mean, d ordinary original base GP',
        aggregation='Reassemble folds in original per-event row order. Equal positions, equal scans within position. Floor means equal events. Existing metrics p90/median scan-quantiles retained; weighted p90 additionally reported.',
        strata='S convex hull inside/outside; nearest reference position <10, 10-20, >=20 raw units; empty strata omitted',
        closure='a-b and a-d raw signed differences. Only a-d>0.5 raw unit permits ratio; ratio may be negative or >1 and is diagnostic, never theoretical remaining error.',
        code_sha256={str(p):digest(p) for p in files},inputs={str(p):digest(p) for p in inputs}))
    (OUT/'code').mkdir()
    for p in files:shutil.copy2(p,OUT/'code'/p.name)
    log(stage='frozen')


def verify():
    frozen=json.loads((OUT/'freeze.json').read_text())
    for key in ['code_sha256','inputs']:
        for path,sha in frozen[key].items():assert digest(Path(path))==sha,path


def fit_and_field(e,meta,floor,origin,grid):
    base,obs=params()['base'],params()['obs']
    sr=indices(floor,meta['support_rows']);support=group_scans(floor['rss'][sr],floor['xy'][sr]-origin);sx=tensor(support['positions'])
    assert len(sr)==15 and len(sx)==5
    w=json.loads((OLD/meta['id']/'adaptation.json').read_text())['weights']
    for fold in e['folds']:
        dest=OUT/e['event']/f"fold{fold['fold']}";dest.mkdir(parents=True,exist_ok=True)
        rr=indices(floor,fold['reference_rows']);reference=group_scans(floor['rss'][rr],floor['xy'][rr]-origin)
        assert fold['support_rows']==meta['support_rows']
        assert not set(fold['reference_rows'])&set(meta['support_rows'])
        held=[]
        for split,file in [('internal','trainingData.csv'),('official','validationData.csv')]:
            q=load_raw(file,{e['floor']})[e['floor']];ix=indices(q,fold['splits'][split]['query_rows']);held.extend(q['xy'][ix])
        assert cKDTree(np.asarray(held)).query(reference['positions']+origin)[0].min()>3.
        assert not (reference['positions'][:,None]==support['positions'][None]).all(-1).any()
        for j,a in enumerate(e['ap_columns']):
            path=dest/f'ap{a}.npz'
            if path.exists():continue
            started=time.time();history_path=dest/f'reference_ap{a}.npz'
            if history_path.exists():h=load_history(history_path,base)
            else:
                h=single_history(dict(scans=reference),a,base,obs,steps=640)
                arrays=dict(empty=h is None,origin=origin)
                if h is not None:arrays.update(z=h.z.numpy(),mean=h.m.numpy(),cov=h.c.numpy())
                np.savez_compressed(history_path,**arrays)
            hs=[load_history(OLD/'histories'/n/f'ap{a}.npz',base) for n in ['B1F0','B1F3']]
            bank=BankPrior(hs,w,base) if any(v is not None for v in hs) else BankPrior([],[],base)
            target=Prior(Transfer('identity'),h,dict(base,drift=0.))
            priors=dict(b=MeanReference(bank,target),d=target);arrays={};info={}
            for name,prior in priors.items():
                result=field_from_prior(prior,obs,sx,ap_stats(support,a),grid,261026+10000*meta['seed']+a,defensive_fraction=.1)
                info[name]=dict(ess=result['effective_samples'],log_evidence=result['log_evidence'])
                if result['effective_samples']<1000:
                    write(dest/f'ap{a}_failure.json',info)
                    raise RuntimeError(f'ESS<1000: {e["event"]} fold{fold["fold"]} AP{a} {name}')
                arrays[name+'_mean']=result['mean'].numpy();arrays[name+'_variance']=result['variance'].numpy()
            np.savez_compressed(path,**arrays)
            write(dest/f'ap{a}.json',dict(info,reference_empty=h is None,elapsed_seconds=time.time()-started))
            if j%10==0:log(event=e['event'],fold=fold['fold'],APs_done=j+1,APs_total=len(e['ap_columns']),last_AP_seconds=time.time()-started)


def score_event(e,meta,f,o,origin,grid,baseline):
    obs=params()['obs'];qi,qo=indices(f,meta['internal_rows']),indices(o,meta['official_rows'])
    rss=np.concatenate([f['rss'][qi],o['rss'][qo]]);xy=np.concatenate([f['xy'][qi],o['xy'][qo]])-origin
    n=len(rss);pred={a:np.full((n,2),np.nan) for a in ['b','d']};pred['a']=baseline
    true_index=cKDTree(grid.numpy()).query(xy)[1];signal={a:np.zeros(n) for a in ['a','b','d']};squared={a:np.zeros(n) for a in signal}
    detected_count=(rss[:,e['ap_columns']]!=100).sum(1);reference_distance=np.full(n,np.nan)
    map_rows={s:{int(r):offset+i for i,r in enumerate(meta[s+'_rows'])} for s,offset in [('internal',0),('official',len(qi))]}
    for fold in e['folds']:
        dest=OUT/e['event']/f"fold{fold['fold']}"
        ix=np.array([map_rows[s][int(r)] for s in ['internal','official'] for r in fold['splits'][s]['query_rows']],dtype=int)
        rr=indices(f,fold['reference_rows']);reference_distance[ix]=cKDTree(f['xy'][rr]-origin).query(xy[ix])[0]
        scores={a:torch.zeros((len(ix),len(grid))) for a in ['b','d']}
        for a in e['ap_columns']:
            with np.load(BASELINE/meta['id']/f'ap{a}.npz') as old,np.load(dest/f'ap{a}.npz') as new:
                for method in signal:
                    m=tensor(old['adapted_bank_mean']) if method=='a' else tensor(new[method+'_mean'])
                    v=tensor(old['adapted_bank_variance']) if method=='a' else tensor(new[method+'_variance'])
                    r=tensor(rss[ix,a]);ii=torch.tensor(true_index[ix]);lp=observation_log_prob(r,m[ii],v[ii],obs)
                    signal[method][ix]+=lp.numpy()
                    sq=(m[ii]-r).square().numpy();squared[method][ix]+=np.where(r.numpy()!=100,sq,0.)
                    if method!='a':
                        unique,inverse=np.unique(rss[ix,a],return_inverse=True)
                        ll=observation_log_prob(tensor(unique[:,None]),m[None],v[None],obs)
                        scores[method]+=ll[torch.tensor(inverse)]
        for a in scores:pred[a][ix]=(scores[a].softmax(1)@grid).numpy()
    assert np.isfinite(reference_distance).all() and all(np.isfinite(v).all() for v in pred.values())
    sx=np.unique(f['xy'][indices(f,meta['support_rows'])],axis=0);center=sx.mean(0)
    inside=Delaunay(sx-center).find_simplex(xy+origin-center)>=0 if np.linalg.matrix_rank(sx-center)==2 else np.zeros(n,bool)
    np.savez_compressed(OUT/e['event']/'predictions.npz',**pred,query_xy=xy,origin=origin,
        reference_distance=reference_distance,inside_support_hull=inside,detected_count=detected_count,
        **{a+'_true_loglik':v for a,v in signal.items()},**{a+'_rss_squared_sum':v for a,v in squared.items()})
    rows=[]
    for split,sl in [('internal',np.arange(len(qi))),('official',np.arange(len(qi),n))]:
        masks=dict(all=np.ones(len(sl),bool),inside_hull=inside[sl],outside_hull=~inside[sl],
            reference_lt10=reference_distance[sl]<10,reference_10to20=(reference_distance[sl]>=10)&(reference_distance[sl]<20),reference_ge20=reference_distance[sl]>=20)
        for stratum,mask in masks.items():
            ix=sl[mask]
            if not len(ix):continue
            w=weights(xy[ix]);record=dict(event=e['event'],floor=e['floor'],split=split,stratum=stratum,positions=len(np.unique(xy[ix],axis=0)),scans=len(ix),
                mean_reference_distance=float(w@reference_distance[ix]),arms={})
            for a in pred:
                error=np.linalg.norm(pred[a][ix]-xy[ix],axis=1);order=np.argsort(error)
                weighted_p90=float(error[order][np.searchsorted(np.cumsum(w[order]),.9)])
                ok=detected_count[ix]>0;signal_weight=w[ok]/w[ok].sum()
                record['arms'][a]=dict(metrics(pred[a][ix],dict(xy=xy[ix],scan_weights=w)),
                    weighted_p90=weighted_p90,true_scan_nll=float(-w@signal[a][ix]),
                    detected_RSS_RMSE=float(np.sqrt(signal_weight@(squared[a][ix][ok]/detected_count[ix][ok]))),
                    detected_RSS_scoring_weight=float(w[ok].sum()))
            aa,bb,dd=[record['arms'][a]['mde'] for a in ['a','b','d']]
            record['differences']=dict(a_minus_b=aa-bb,a_minus_d=aa-dd,b_minus_d=bb-dd,
                b_closure_ratio=(aa-bb)/(aa-dd) if aa-dd>.5 else None,
                reference_quality_b_RSS_improved=record['arms']['b']['detected_RSS_RMSE']<record['arms']['a']['detected_RSS_RMSE'])
            rows.append(record)
    write(OUT/e['event']/'completed.json',dict(records=rows))
    log(event=e['event'],stage='completed',all=[r for r in rows if r['stratum']=='all'])


def run():
    verify();prep,old=manifests();names={e['floor'] for e in prep};train=load_raw('trainingData.csv',names);official=load_raw('validationData.csv',names)
    for e in prep:
        if (OUT/e['event']/'completed.json').exists():continue
        meta=next(m for m in old if m['id']==e['event'])
        with np.load(BASELINE/meta['id']/'predictions.npz') as p:origin=p['origin'];baseline=p['adapted_bank']
        grid=grid_for(meta,origin)
        fit_and_field(e,meta,train[e['floor']],origin,grid)
        with torch.no_grad():score_event(e,meta,train[e['floor']],official[e['floor']],origin,grid,baseline)
    summarize()


def summarize():
    prep,_=manifests();rows=[]
    for e in prep:rows.extend(json.loads((OUT/e['event']/'completed.json').read_text())['records'])
    aggregate=[]
    for floor in sorted({r['floor'] for r in rows}):
        for split in ['internal','official']:
            for stratum in sorted({r['stratum'] for r in rows}):
                subset=[r for r in rows if r['floor']==floor and r['split']==split and r['stratum']==stratum]
                if not subset:continue
                arms={a:{k:float(np.mean([r['arms'][a][k] for r in subset])) for k in ['mde','p90','weighted_p90','true_scan_nll','detected_RSS_RMSE']} for a in ['a','b','d']}
                aa,bb,dd=[arms[a]['mde'] for a in ['a','b','d']]
                aggregate.append(dict(floor=floor,split=split,stratum=stratum,events=len(subset),arms=arms,
                    a_minus_b=aa-bb,a_minus_d=aa-dd,b_closure_ratio=(aa-bb)/(aa-dd) if aa-dd>.5 else None))
    info=[json.loads(p.read_text()) for p in OUT.glob('B*/fold*/ap*.json') if not p.name.endswith('_failure.json')]
    write(OUT/'summary.json',dict(created_unix=time.time(),scope=__doc__,records=rows,aggregate=aggregate,
        minimum_ESS=min(r[a]['ess'] for r in info for a in ['b','d']),AP_fold_count=len(info),
        missing_arms=['c','E_naive'],interpretation='Cross-block extra-target-label references with fixed base/obs; no theoretical lower bound or same-K claim. Negative or poor reference results must retain spatial extrapolation and true-RSS quality limits.'))
    log(stage='summary',aggregate=[r for r in aggregate if r['stratum']=='all'])


if __name__=='__main__':
    assert socket.gethostname()=='ray-web','Numerical work stays on ray-web'
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['check','freeze','run','summarize']);args=parser.parse_args()
    dict(check=check,freeze=freeze,run=run,summarize=summarize)[args.stage]()
