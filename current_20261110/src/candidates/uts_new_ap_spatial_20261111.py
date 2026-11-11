"""Restore spatial conditioning only for APs absent from all H and seen in S.

The retained GP removes K_amp + sigma^2 when every history is absent, so its
S-to-location covariance is identically zero. For these APs only, retain the
existing spatial RBF covariance and still remove the acquisition intercept.
This assumes a shared physical AP field between calibration and localization;
it does not assume identical device offsets. The S posterior, source GP
parameters, chosen intercept variance and sampler are the existing ones.

Keep A's full-K learned fields for H-supported APs. Use each new AP's GP
observation likelihood once in the same robust scan reader, with the original
shared offset marginalization, AP-count tempering, candidates and projection.
This changes an information path, with no new fit, weight, threshold or model.
The original fixed GP pool is also evaluated unchanged. No-H/no-S APs remain
unavailable. A spatial prior cannot identify an arbitrary unseen field from
a few points; uncertainty is propagated, and actual benefit must be measured.

This follow-up is exposed UTS development after the frozen K3->K5 diagnostic.
All original48 tasks remain. Predictions are sealed before scoring.
"""
from candidates import uts_official_evaluation_20261111 as E
from candidates import uts_official_gp_20261111 as G
import argparse
import csv
from pathlib import Path
import socket
import time

import numpy as np
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from frozen_hurdle_translator import EPS,BETA,U_DET,U_MISS,OFFSETS
from observations import observation_log_prob,predict_from_posterior
from position_fusion_gp import _retained_field
from uji import group_scans,tensor,ap_stats
from hurdle_gp_pool import fuse_hurdle_gp

OUTPUT=E.ROOT/'results/uts_new_ap_spatial_20261111'
NAMES=('new_ap_spatial','new_ap_spatial_GP_pool')


def freeze():
    E.check();G.check()
    OUTPUT.mkdir(exist_ok=False)
    E.write(OUTPUT/'freeze.json',dict(created=E.now(),code_sha256=E.digest(__file__),
        A_complete_sha256=E.digest(E.A_OUTPUT/'A_complete.json'),
        GP_freeze_sha256=E.digest(G.OUTPUT/'freeze.json'),
        GP_solver_execution_sha256=E.digest(G.OUTPUT/'solver_continuation_execution.json'),
        diagnostic_scores_sha256=E.digest(E.ROOT/'results/uts_calibration_path_20261111/scores.json'),
        scope='Same48 official UTS tasks; no selected floors, APs selected by H/S only',
        information_rule='H-absent and S-detected AP: shared spatial RBF, separate acquisition intercept',
        changed='Restore only spatial cross covariance for new AP likelihoods in original A reader',
        preserved='Original full-K A for every H-supported AP; GP state, parameters, sampler, reader constants, candidates and fixed pool',
        required_runtime_state='Existing full H/S GP adaptation; its upstream computation cost still belongs to this candidate',
        hypothesis='Use newly calibrated signals without extrapolating an unsupported learned history field',
        timing='Post-score development follow-up; not independent generalization evidence',
        source_fits=0,new_hyperparameters=0,resources=dict(CPU_threads=2,GPU=0,segment_seconds=540)))


def check():
    E.check();G.check()
    frozen=E.read(OUTPUT/'freeze.json')
    assert frozen['code_sha256']==E.digest(__file__)
    for key,path in [('A_complete_sha256',E.A_OUTPUT/'A_complete.json'),
                     ('GP_freeze_sha256',G.OUTPUT/'freeze.json'),
                     ('GP_solver_execution_sha256',G.OUTPUT/'solver_continuation_execution.json'),
                     ('diagnostic_scores_sha256',E.ROOT/'results/uts_calibration_path_20261111/scores.json')]:
        assert frozen[key]==E.digest(path)
    return frozen


def new_fields(task,field,diagnostics,new_aps):
    origin=np.asarray(diagnostics['origin'])
    scans=group_scans(task.S['rss'],task.S['xy']-origin)
    state=diagnostics['state'];parameters=E.read(G.PARAMETERS)
    points=tensor(field['cand_xy']-origin);sx=tensor(scans['positions'])
    context=dict(grid=points,base=parameters['base'],obs=parameters['obs'])
    fields,records={},[]
    for ap in new_aps:
        case=dict(histories=[None]*len(task.H),sx=sx,stats=ap_stats(scans,int(ap)))
        def collect(index,prior,sm,sk,moments):
            qm,qf=prior.features(points);_,sf=prior.features(sx)
            # Keep the physical field shared; acquisition intercept stays separate.
            cross=prior.covariance(points,sx,qf,sf)-state['sigma']['intercept_std']**2
            value=predict_from_posterior(qm,prior.diagonal(points,qf),cross,sm,sk,moments)
            fields[index]={k:v.detach().numpy() for k,v in value.items()}
        _,record=_retained_field(context,state,int(ap),case,task.task_id,task.K,collect)
        records.append(record)
    return fields,records,parameters['obs']


def readout(field,query,fresh,obs):
    aps=field['aps'];old=np.array([int(a) not in fresh for a in aps])
    p,m,v=(field[k][:,old] for k in ('p','m','s2'))
    oldq=query[:,aps[old]]
    lmiss=np.log((1-EPS)*(1-p)+EPS*U_MISS)
    logp=np.log(p);lnorm=-.5*np.log(2*np.pi*v)
    fresh_missing={a:np.logaddexp(np.log(1-EPS)+observation_log_prob(
        100.,f['mean'],f['variance'],obs).numpy(),np.log(EPS*U_MISS)) for a,f in fresh.items()}
    scores=[]
    for i,row in enumerate(query):
        detected=oldq[i]!=100
        missing=lmiss[:,~detected].sum(1)
        for a,ll in fresh_missing.items():
            if row[a]==100:missing=missing+ll
        values=[]
        for offset in OFFSETS:
            ll=np.logaddexp(np.log(1-EPS)+logp[:,detected]+lnorm[:,detected]
                -.5*(oldq[i,detected][None]-offset-m[:,detected])**2/v[:,detected],
                np.log(EPS*U_DET)).sum(1)
            for a,f in fresh.items():
                if row[a]!=100:
                    marginal=observation_log_prob(float(row[a]-offset),f['mean'],f['variance'],obs).numpy()
                    ll=ll+np.logaddexp(np.log(1-EPS)+marginal,np.log(EPS*U_DET))
            values.append(ll)
        scores.append((logsumexp(np.stack(values),axis=0)-np.log(len(OFFSETS))+missing)*BETA/len(aps))
    scores=np.asarray(scores);log_mass=scores-logsumexp(scores,axis=1,keepdims=True)
    return np.clip(np.exp(log_mass)@field['cand_xy'],field['bounds'][0],field['bounds'][1]),log_mass


def predict(seconds):
    check()
    gp_complete=E.read(G.OUTPUT/'predictions_complete.json')
    assert gp_complete['complete'] and len(gp_complete['records'])==48
    assert gp_complete['solver_execution']['sha256']==E.digest(G.OUTPUT/'solver_continuation_execution.json')
    source=E.read(E.A_OUTPUT/'freeze.json')
    train,query=E.table(E.DATA/'UTS_training.csv',history=True),E.table(E.DATA/'UTS_test.csv',history=False)
    records=[];start=time.monotonic()
    for event,gp_binding in zip(source['events'],gp_complete['records']):
        destination=OUTPUT/event['event'];record_path=destination/'complete.json'
        if not record_path.exists():
            if time.monotonic()-start>=seconds:return 2
            assert E.digest(gp_binding['path'])==gp_binding['sha256']
            gc=E.read(gp_binding['path']);ac=E.read(E.A_OUTPUT/event['event']/'complete.json')
            assert gc['event']==ac['event']==event
            for item in gc['files']+ac['files']:
                assert E.digest(item['path'])==item['sha256']
            task=E.inputs_for(event,train,query)
            with np.load(E.A_OUTPUT/event['event']/'field.npz') as f:field={k:f[k].copy() for k in f.files}
            with np.load(E.A_OUTPUT/event['event']/'query.npz') as q:
                original=q['projected'].copy();original_mass=q['log_position_mass'].copy()
                assert q['rows'].tolist()==event['Q_rows_zero_based']
            hseen=np.logical_or.reduce([(h['rss']!=100).any(0) for h in task.H.values()])
            new_aps=field['aps'][~hseen[field['aps']]]
            assert (task.S['rss'][:,new_aps]!=100).any(0).all()
            if len(new_aps):
                diagnostics=E.read(G.OUTPUT/event['event']/'diagnostics.json')
                fresh,solver,obs=new_fields(task,field,diagnostics,new_aps)
                prediction,log_mass=readout(field,task.Q_rss,fresh,obs)
            else:
                fresh,solver={},[];prediction,log_mass=original,original_mass
            with np.load(G.OUTPUT/event['event']/'predictions.npz') as gp:
                assert gp['rows'].tolist()==event['Q_rows_zero_based']
                pooled=fuse_hurdle_gp(field['cand_xy'],log_mass,gp['GP_grid'],gp['GP_scores'],gp['GP_raw'],field['bounds'])
            destination.mkdir(exist_ok=False)
            E.write(destination/'predictions.json',dict(rows=event['Q_rows_zero_based'],
                new_ap_spatial=prediction.tolist(),new_ap_spatial_GP_pool=pooled['predictions']['probability_half'].tolist()))
            if fresh:
                np.savez_compressed(destination/'new_fields.npz',aps=new_aps,candidates=field['cand_xy'],
                    mean=np.stack([fresh[int(a)]['mean'] for a in new_aps]),
                    variance=np.stack([fresh[int(a)]['variance'] for a in new_aps]))
            E.write(record_path,dict(complete=True,created=E.now(),event=event,freeze_sha256=E.digest(OUTPUT/'freeze.json'),
                new_APs=new_aps.tolist(),sampler=solver,query_XY_opened=False,
                predictions_sha256=E.digest(destination/'predictions.json'),
                new_fields_sha256=E.digest(destination/'new_fields.npz') if fresh else None))
        record=E.read(record_path)
        assert record['freeze_sha256']==E.digest(OUTPUT/'freeze.json') and record['event']==event
        assert record['predictions_sha256']==E.digest(destination/'predictions.json')
        records.append(dict(path=str(record_path),sha256=E.digest(record_path)))
        print('Completed new-AP spatial update',event['event'],flush=True)
    E.write(OUTPUT/'predictions_complete.json',dict(complete=True,created=E.now(),records=records,
        freeze_sha256=E.digest(OUTPUT/'freeze.json'),GP_complete_sha256=E.digest(G.OUTPUT/'predictions_complete.json')))
    return 0


def score():
    check();complete=E.read(OUTPUT/'predictions_complete.json')
    assert complete['complete'] and len(complete['records'])==48
    baseline=E.read(E.OUTPUT/'full_scores.json');assert baseline['full_system_complete']
    with (E.DATA/'UTS_test.csv').open(newline='') as stream:
        truth=np.array([[float(r['Pos_x']),float(r['Pos_y'])] for r in csv.DictReader(stream)])
    records=[]
    for binding in complete['records']:
        assert E.digest(binding['path'])==binding['sha256']
        record=E.read(binding['path']);pred_path=Path(binding['path']).parent/'predictions.json'
        assert E.digest(pred_path)==record['predictions_sha256']
        pred=E.read(pred_path);event=record['event'];assert pred['rows']==event['Q_rows_zero_based']
        xy=truth[pred['rows']];_,groups=np.unique(xy,axis=0,return_inverse=True)
        records.append(dict(event=event['event'],floor=event['floor'],K=event['K'],new_APs=record['new_APs'],
            scores={n:E.position_score(pred[n],dict(xy=xy,groups=groups.ravel())) for n in NAMES}))
    summary,paired={},{}
    for method in NAMES:
        summary[method]={str(k):dict(positions=E.metrics([e for r in records if r['K']==k for e in r['scores'][method]['position_errors']]),
            by_floor={str(r['floor']):E.metrics(r['scores'][method]['position_errors']) for r in records if r['K']==k}) for k in (1,3,5)}
        paired[method]={}
        for a,b in ((1,3),(3,5),(1,5)):
            delta=np.concatenate([np.asarray(next(r for r in records if r['floor']==f and r['K']==b)['scores'][method]['position_errors'])
                -next(r for r in records if r['floor']==f and r['K']==a)['scores'][method]['position_errors'] for f in range(-3,13)])
            paired[method][f'{a}_to_{b}']=dict(mean_change=float(delta.mean()),improved=int((delta < -1e-8).sum()),
                worsened=int((delta > 1e-8).sum()),improved_over5m=int((delta < -5).sum()),worsened_over5m=int((delta > 5).sum()))
    E.write(OUTPUT/'scores.json',dict(complete=True,created=E.now(),freeze_sha256=E.digest(OUTPUT/'freeze.json'),
        predictions_complete_sha256=E.digest(OUTPUT/'predictions_complete.json'),baseline_sha256=E.digest(E.OUTPUT/'full_scores.json'),
        summary=summary,paired=paired,records=records,baseline_summary=baseline['summary'],adopted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','predict','score'))
    parser.add_argument('--seconds',type=float,default=540)
    args=parser.parse_args()
    assert socket.gethostname()=='ray-web' and 0<args.seconds<=540
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    with threadpool_limits(limits=2):
        code=predict(args.seconds) if args.phase=='predict' else globals()[args.phase]()
    raise SystemExit(0 if code is None else code)
