"""Frozen Office calibration rules transferred to the existing UJI H-bank.

Four existing kernels, no new source training or target parameter fitting.
Only cross-location correction covariance changes; H/S/means/variances match.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from bank_calibration import CalibratedBankPrior, check, load_relation
from data import group_scans
from full_ap_study import metrics
from inference import gaussian_update, observation_log_prob, predict_from_posterior
from reference_pmc import setup, sample_reference
from run import ap_stats, tensor
from run_multi_history import digest, write
from uji_history_bank_study import OUT as BANK, SOURCES, cached_history, load_raw


ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'results/calibration_learning_v1'
OUT=ROOT/'results/uji_bank_calibration_transfer_v1_20261026'
METHODS={'uniform_initial':('uniform','uniform_initial.pt'),
         'uniform_trained':('uniform','uniform.pt'),
         'strength_initial':('strength_rbf','strength_rbf_initial.pt'),
         'strength_trained':('strength_rbf','strength_rbf.pt')}


def log(**row): print(json.dumps(row,allow_nan=False),flush=True)


def prepare():
    OUT.mkdir(exist_ok=False)
    source=json.loads((SOURCE/'freeze.json').read_text())
    parameters=json.loads((BANK/'parameters.json').read_text());base=parameters['base']
    files=[ROOT/n for n in ['bank_calibration.py','run_bank_calibration_transfer.py','history_bank.py',
        'transfer.py','calibration_relation.py','multi_history.py','inference.py','reference_pmc.py',
        'uji_history_bank_study.py','uji_probability_study.py','data.py','full_ap_study.py']]
    history_files=sorted((BANK/'histories').glob('B*/ap*.npz'))
    freeze=dict(created_unix=time.time(),hypothesis='Source-learned historical-strength calibration improves new-environment localization beyond its identical untrained family and source-trained ordinary global/local correction',
        model='Existing weighted historical Gaussian bank + existing calibration relation + unchanged LAW observation/posterior/localization',
        source='Office periods 1-11, existing 80-update checkpoints; one source environment',
        source_parameter_lineage='Observation/base/mean Office parameters had earlier fits; 80 is additional relation training, not total project data budget',
        field_parameters='Existing B0-only UJI parameters, frozen and common to every method',
        deployment_histories=SOURCES,bank_weights=[.5,.5],kernel_parameters_fit_at_target=0,new_source_updates=0,
        methods=METHODS,source_training_record=source,
        spatial_unit_rule='Use (source learned correction length/source historical length)*UJI historical length',
        strength_unit='dBm difference length unchanged; no target-based rescaling',
        invariants=['same historical means','same historical covariance','same pre-S correction variance',
           'same observation model','same independent calibration positions/scans','same APs and candidate grid'],
        inference='Existing M-PMC 6x4096 adaptation +32768 final draws, Gaussian moment grid readout',
        history_absence='Existing all-H-undetected AP policy: one identical local-GP posterior for all methods',
        query_diagnostics='True-location fields evaluated after S posterior; used only for RSS/NLL diagnostics, never localization or fitting',
        decision='Learning claim requires trained vs initial improvement in localization; RSS/NLL alone insufficient',
        evaluation_role='Same six previously inspected B1 development cases, K5 R3; not untouched test or arbitrary-building alignment',
        counterexamples=dict(no_support='Prior only; learning parameters cannot observe this event without S',
          one_position='One measured change cannot identify arbitrary distant changes',
          zero_drift='Relation disappears and historical covariance remains',
          wrong_history='Correlating corrections cannot invent unconstrained spatial structure',
          all_missing_history='Common local GP, not source-specific confidence manufacture',
          high_historical_uncertainty='Expected strength similarity decreases; residual spatial component remains',
          shift='Joint RSS differences ignore a common strength offset; translation-invariant spatial kernel',
          scale='Length ratio fixes coordinate-unit conversion before any target results',
          source_order='Permutation invariant when histories and weights are permuted together',
          residual_zero='No pseudo-observation is added; finite measurement uncertainty can still update posterior covariance'),
        reverse_audit=['Office checkpoint -> fixed relation hyperparameters with one prescribed unit conversion',
          'H cache -> existing bank moments; no new mean model, grid or S dependent H representation',
          'S prior mean/covariance -> original RSS+detection posterior using S only',
          'Grid prior/cross covariance -> posterior moments; Query RSS -> original likelihood location estimate',
          'Query XY enters scoring/true-location diagnostics after adaptation, never candidate-grid construction'],
        code_sha256={str(p):digest(p) for p in files},
        history_sha256={str(p):digest(p) for p in history_files},
        inputs={str(p):digest(p) for p in [BANK/'manifest.json',BANK/'freeze.json',BANK/'parameters.json',BANK/'histories/completed.json',SOURCE/'freeze.json']},
        source_weights={str(SOURCE/name):digest(SOURCE/name) for _,name in METHODS.values()})
    write(OUT/'freeze.json',freeze);shutil.copy2(BANK/'manifest.json',OUT/'manifest.json')
    shutil.copy2(BANK/'parameters.json',OUT/'parameters.json');shutil.copy2(SOURCE/'freeze.json',OUT/'source_freeze.json')
    (OUT/'code').mkdir();(OUT/'source_weights').mkdir()
    for path in files:shutil.copy2(path,OUT/'code'/path.name)
    summaries={}
    for name,(kind,checkpoint) in METHODS.items():
        shutil.copy2(SOURCE/checkpoint,OUT/'source_weights'/checkpoint)
        relation=load_relation(kind,OUT/'source_weights'/checkpoint,source['base'],base)
        summaries[name]={k:float(v) for k,v in relation.state_dict().items()}
    write(OUT/'transferred_parameters.json',summaries)
    log(stage='prepared',methods=list(METHODS),events=6,new_source_updates=0,target_parameter_updates=0)


def event(event_id):
    destination=OUT/event_id;destination.mkdir(exist_ok=True)
    if (destination/'completed.json').exists():return
    parameters=json.loads((OUT/'parameters.json').read_text());base,obs=parameters['base'],parameters['obs']
    source=json.loads((OUT/'source_freeze.json').read_text())
    meta=next(e for e in json.loads((OUT/'manifest.json').read_text())['events'] if e['id']==event_id)
    f=load_raw('trainingData.csv',{meta['floor']})[meta['floor']]
    q=load_raw('validationData.csv',{meta['floor']})[meta['floor']]
    fl={int(r):i for i,r in enumerate(f['row_ids'])};ql={int(r):i for i,r in enumerate(q['row_ids'])}
    sr=np.array([fl[r] for r in meta['support_rows']]);qi=np.array([fl[r] for r in meta['internal_rows']])
    qo=np.array([ql[r] for r in meta['official_rows']])
    origin=np.array(json.loads((BANK/'histories/completed.json').read_text())['origin'])
    support=group_scans(f['rss'][sr],f['xy'][sr]-origin);sx=tensor(support['positions'])
    query=group_scans(np.concatenate((f['rss'][qi],q['rss'][qo])),np.concatenate((f['xy'][qi],q['xy'][qo]))-origin)
    # This label array is only a diagnostic evaluation input, not an adaptation input.
    qx=tensor(query['positions'])
    bounds=np.array(meta['grid_bounds'])-origin
    axes=[np.linspace(bounds[0,d],bounds[1,d],int(np.ceil((bounds[1,d]-bounds[0,d])/2))+1) for d in range(2)]
    grid=tensor(np.stack(np.meshgrid(*axes),-1).reshape(-1,2))
    points=torch.cat((grid,qx))
    relations={name:load_relation(kind,OUT/'source_weights'/checkpoint,source['base'],base)
               for name,(kind,checkpoint) in METHODS.items()}
    for index,ap in enumerate(meta['ap_columns']):
        path=destination/f'ap{ap}.npz';record=destination/f'ap{ap}.json'
        if path.exists() and record.exists():continue
        histories=[cached_history(BANK,name,ap,base) for name in SOURCES]
        empty=all(h is None for h in histories);saved={};rows=[]
        for method,relation in relations.items():
            if empty and rows:
                first=next(iter(METHODS))
                for key in ['grid_mean','grid_variance','true_mean','true_variance']:
                    saved[f'{method}_{key}']=saved[f'{first}_{key}']
                rows.append(dict(rows[0],method=method,reused_identical_local_posterior=True,seconds=0.))
                continue
            started=time.perf_counter()
            prior=CalibratedBankPrior(histories,[.5,.5],base,relation)
            with torch.no_grad():
                sm,sf=prior.features(sx);sk=prior.covariance(sx,sx,sf,sf)
                stats=ap_stats(support,ap)
                target,proposal,center,root,_=setup(sm,sk,stats,obs)
                samples,logw,info=sample_reference(target,proposal,center,root,261026+meta['seed'])
                w=logw.exp();pm=w@samples;c=samples-pm;post=dict(mean=pm,cov=(c.T*w)@c)
                gm,gf=prior.features(points);gv=prior.diagonal(points,gf)
                cross=prior.covariance(points,sx,gf,sf)
                pred=predict_from_posterior(gm,gv,cross,sm,sk,post)
                if not torch.isfinite(pred['mean']).all() or not torch.isfinite(pred['variance']).all() or (pred['variance']<0).any():
                    raise RuntimeError(f'Invalid transferred field: {event_id}, {method}, {ap}')
                for key,value in dict(grid_mean=pred['mean'][:len(grid)],grid_variance=pred['variance'][:len(grid)],
                                      true_mean=pred['mean'][len(grid):],true_variance=pred['variance'][len(grid):]).items():
                    saved[f'{method}_{key}']=value.numpy()
                evidence=info['log_detection_evidence']-gaussian_update(sm,sk,stats,obs)['loss'].item()
                rows.append(dict(method=method,seconds=time.perf_counter()-started,
                    support_log_evidence=evidence,effective_samples=info['effective_samples'],
                    reused_identical_local_posterior=False))
        np.savez_compressed(path,**saved);write(record,dict(ap=ap,no_history_detection=empty,results=rows))
        if index%10==0:log(stage='field',event=event_id,ap_index=index+1,total=len(meta['ap_columns']))
    predictions={};diagnostics={};device=torch.device('cuda:0')
    with torch.no_grad():
        for method in METHODS:
            scores=torch.zeros((len(query['rss']),len(grid)),device=device)
            nll=np.zeros(len(query['rss']));sse=np.zeros(len(query['rss']));count=np.zeros(len(query['rss']))
            for ap in meta['ap_columns']:
                with np.load(destination/f'ap{ap}.npz') as a:
                    mean=tensor(a[f'{method}_grid_mean']).to(device)
                    var=tensor(a[f'{method}_grid_variance']).to(device)
                    tm=tensor(a[f'{method}_true_mean'])[query['group']]
                    tv=tensor(a[f'{method}_true_variance'])[query['group']]
                rss=tensor(query['rss'][:,ap]);gpu_rss=rss.to(device)
                for start in range(0,len(rss),128):
                    scores[start:start+128]+=observation_log_prob(gpu_rss[start:start+128,None],mean[None],var[None],obs)
                nll-=observation_log_prob(rss,tm,tv,obs).numpy()
                denominator=torch.sqrt(tv+obs['detect_scale']**2);z=(tm-obs['beta'])/denominator
                mills=torch.exp(-.5*(math.log(2*math.pi)+z.square())-torch.special.log_ndtr(z))
                conditional_mean=tm+tv/denominator*mills
                detected=rss!=100
                sse+=torch.where(detected,(rss-conditional_mean).square(),0.).numpy();count+=detected.numpy()
            predictions[method]=(torch.softmax(scores,1)@grid.to(device)).cpu().numpy()
            diagnostics[method]=dict(nll=nll,sse=sse,count=count)
    rows=[]
    for split,sl in [('internal',slice(0,len(qi))),('official',slice(len(qi),None))]:
        xy=query['xy'][sl];_,g,n=np.unique(xy,axis=0,return_inverse=True,return_counts=True);g=g.ravel()
        sw=1/(len(n)*n[g])
        for method,pred in predictions.items():
            d=diagnostics[method]
            rows.append(dict(method=method,split=split,**metrics(pred[sl],dict(xy=xy,scan_weights=sw)),
                true_location_nll_per_ap=float(sw@d['nll'][sl]/len(meta['ap_columns'])),
                detected_rss_rmse=float(np.sqrt((sw@d['sse'][sl])/(sw@d['count'][sl])))))
    np.savez_compressed(destination/'predictions.npz',**{k:v+origin for k,v in predictions.items()},
        query_xy=query['xy']+origin,internal_count=len(qi),support_xy=support['positions']+origin,
        grid=grid.numpy()+origin,ap_columns=meta['ap_columns'])
    write(destination/'completed.json',dict(completed=True,event=event_id,results=rows,
        predictions_sha256=digest(destination/'predictions.npz')))
    log(stage='completed',event=event_id,results=[r for r in rows if r['split']=='official'])


def summarize():
    rows=[]
    for meta in json.loads((OUT/'manifest.json').read_text())['events']:
        d=json.loads((OUT/meta['id']/'completed.json').read_text())
        rows.extend(dict(event=meta['id'],floor=meta['floor'],**r) for r in d['results'])
    summary=[]
    for method in METHODS:
        for split in ['internal','official']:
            selected=[r for r in rows if r['method']==method and r['split']==split]
            summary.append(dict(method=method,split=split,events=len(selected),**{
                k:float(np.mean([r[k] for r in selected])) for k in ['mde','true_location_nll_per_ap','detected_rss_rmse']}))
    write(OUT/'summary.json',dict(completed=True,summary=summary,rows=rows))
    log(stage='summary',summary=summary)


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('All numerical work is server-only')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['check','prepare','event','summarize']);p.add_argument('--event')
    args=p.parse_args()
    if args.stage=='check':log(stage='checks',**check())
    elif args.stage=='prepare':prepare()
    else:
        freeze=json.loads((OUT/'freeze.json').read_text())
        for collection in ['code_sha256','history_sha256','inputs','source_weights']:
            for path,sha in freeze[collection].items():
                if digest(Path(path))!=sha:raise RuntimeError(f'Frozen input changed: {path}')
        if args.stage=='event':event(args.event)
        else:summarize()
