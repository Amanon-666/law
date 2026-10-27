"""Existing ICML2017 method, adapted to the frozen K-position LAW cases."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from data import group_scans
from full_ap_study import metrics
from run_multi_history import digest, write
from tcms_stack_bridge import SourcePair, ORIGINAL, SOLVE, check, fit
from uji_history_bank_study import OUT as BANK, SOURCES, load_raw


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/tcms_history_bank_v1_20261026'
TRAIN = ['B0F0', 'B0F1', 'B0F2']
VARIANTS = {'tcms_h2_group_loo': SOURCES, 'tcms_all5_group_loo': TRAIN + SOURCES}


def log(**item): print(json.dumps(item, allow_nan=False), flush=True)


def prepare():
    OUT.mkdir(exist_ok=False)
    freeze = dict(created_unix=time.time(), purpose='Mature external comparator; no claimed new algorithm',
        paper='https://proceedings.mlr.press/v70/wei17a/wei17a.pdf', equations=[1,15,16],
        existing_implementation=str(ORIGINAL), exact_solve_reuse=str(SOLVE),
        implementation='Algebraically exact source Schur complement, no source subsampling or sparse approximation',
        preprocessing='Non-detection=-110 for this RSS-feature baseline; (RSS+110)/110; same permitted AP columns as LAW',
        coordinate_normalization='Mean/std of the common B1 historical bank only; no held target labels enter LOO normalization',
        target_budget='Frozen 5 independent positions, 3 scans each; exact same six manifests as LAW',
        objective='TC_MSStack weighted location squared error with fixed-hyperparameter leave-one-position-out predictions',
        protocol_adaptation='All scans from held position removed together; paper permits LOO but does not specify grouping',
        variants=VARIANTS,
        information_ledger=dict(tcms_h2_group_loo='Same deployment H/S, but no B0 pretraining; this is not full training-information parity',
          tcms_all5_group_loo='Also receives all three B0 floors used for LAW source fitting as raw extra source domains',
          law='B0 is compressed into fixed shared parameters; B1 history remains in inference',
          ap_columns='All methods use each frozen event AP list; no query-selected AP or target statistics'),
        methods_role='Mature comparators under project protocol, not exact paper figure reproduction or new meta-learning',
        restarts=5, function_evaluations_per_restart=200, optimizer='L-BFGS-B; paper allows alternatives to CG',
        lambda_parameter='Direct [-1,1], equivalent kernel family; removes redundant mu,b coordinates',
        numerical_log_bounds=dict(length=[1e-3,1e3],signal_std=[float(np.exp(-5)),float(np.exp(5))],noise_std=[1e-4,10]),
        target_choice='Only group-LOO source/support objective chooses parameters; Query loaded after fit',
        prediction='Two scalar coordinate GPs; weighted posterior means; no clipping to improve query error',
        expected='Tests whether explicit spatial signal-field transfer outperforms established RSS-to-position transfer',
        falsification='If mature comparator matches or beats LAW, adopt it as strong base; no superiority claim from weak baselines',
        counterexamples=dict(no_support='No adaptation objective; outside this K5 study',
          single_position='Group LOO leaves zero other target labels, so prediction is source-only',
          repeated_scans='All held-location scans removed jointly',
          zero_lambda='Per-source GP reduces to target GP; Eq15 undefined if ALL coefficients zero',
          identical_RSS_different_locations='No deterministic input regression can separate identical inputs',
          missing_AP='Fixed preprocessing, no implicit AP correspondence',
          incorrect_source='Lambda can suppress or reverse covariance; poor S coverage can still select wrong source',
          source_permutation='Stack sum invariant under source/parameter permutation',
          translation='Historical-only centering restores translation with predictions',
          noise_zero='Interpolation possible; group LOO avoids scoring own-label interpolation',
          parameter_bounds='Boundary and budget-limited solutions reported, not called converged'),
        reverse_audit=['Historical H and S RSS/labels -> normalized GP inputs, no Query',
          'Source conditional GP -> target precision -> whole-position LOO prediction',
          'Joint LOO objective -> kernel and source coefficients, selected among five starts',
          'Frozen fitted models + Query RSS -> coordinates; Query XY is used only for metrics'],
        code_sha256={str(p):digest(p) for p in [ROOT/'tcms_stack_bridge.py',ROOT/'run_tcms_history_bank.py',ORIGINAL,SOLVE]},
        bank_manifest_sha256=digest(BANK/'manifest.json'),bank_freeze_sha256=digest(BANK/'freeze.json'))
    write(OUT/'freeze.json',freeze); shutil.copy2(BANK/'manifest.json',OUT/'manifest.json')
    (OUT/'code').mkdir()
    for src,name in [(ORIGINAL,'original_icml17_core.py'),(SOLVE,'existing_precursor_gp.py'),
                      (ROOT/'tcms_stack_bridge.py','tcms_stack_bridge.py'),(ROOT/'run_tcms_history_bank.py','run_tcms_history_bank.py')]:
        shutil.copy2(src,OUT/'code'/name)
    log(stage='prepared', variants=VARIANTS,events=6,restarts=5,evaluations=200)


def event(name, variant):
    destination=OUT/variant/name;destination.mkdir(parents=True,exist_ok=True)
    if (destination/'completed.json').exists(): return
    meta=next(e for e in json.loads((OUT/'manifest.json').read_text())['events'] if e['id']==name)
    sources=VARIANTS[variant]
    floors=load_raw('trainingData.csv',set(sources+[meta['floor']]+SOURCES))
    f=floors[meta['floor']];lookup={int(r):i for i,r in enumerate(f['row_ids'])}
    sr=np.array([lookup[r] for r in meta['support_rows']]);aps=meta['ap_columns']
    support=group_scans(f['rss'][sr],f['xy'][sr])
    known=np.concatenate([floors[s]['xy'] for s in SOURCES])
    center,scale=known.mean(0),known.std(0)
    device=torch.device('cuda:0')
    def features(rss): return torch.tensor((np.where(rss[:,aps]==100,-110,rss[:,aps])+110)/110.,device=device)
    sx=features(support['rss']); hx=[features(floors[s]['rss']) for s in sources]
    coordinate=[]
    for axis in range(2):
        path=destination/f'axis{axis}.json'
        ys=[torch.tensor((floors[s]['xy'][:,axis]-center[axis])/scale[axis],device=device) for s in sources]
        yt=torch.tensor((support['xy'][:,axis]-center[axis])/scale[axis],device=device)
        pairs=[SourcePair(x,sx,y,yt,support['group']) for x,y in zip(hx,ys)]
        if not path.exists():
            started=time.perf_counter()
            p,w,diagnostics=fit(pairs,261026+axis)
            write(path,dict(parameters=p.tolist(),weights=w.tolist(),diagnostics=diagnostics,
                  seconds=time.perf_counter()-started,source_scan_counts=[len(x) for x in hx],target_scans=len(sx)))
            log(stage='fitted',event=name,variant=variant,axis=axis,weights=w.tolist(),seconds=time.perf_counter()-started,
                converged_restarts=sum(r['optimizer_converged'] for r in diagnostics['runs']))
        fitted=json.loads(path.read_text())
        coordinate.append((pairs,torch.tensor(fitted['parameters'],device=device),torch.tensor(fitted['weights'],device=device)))
    # Query is only touched after BOTH coordinate models have been selected.
    q=load_raw('validationData.csv',{meta['floor']})[meta['floor']]
    qlookup={int(r):i for i,r in enumerate(q['row_ids'])}
    qi=np.array([lookup[r] for r in meta['internal_rows']]);qo=np.array([qlookup[r] for r in meta['official_rows']])
    rss=np.concatenate((f['rss'][qi],q['rss'][qo]));xy=np.concatenate((f['xy'][qi],q['xy'][qo]))
    qx=features(rss)
    with torch.no_grad():
        prediction=np.column_stack([(w@torch.stack([g.predict(v,qx) for g,v in zip(pairs,p)])).cpu().numpy()
                                    for pairs,p,w in coordinate])*scale+center
    if not np.isfinite(prediction).all(): raise FloatingPointError('Nonfinite TC_MSStack prediction')
    rows=[]
    for split,sl in [('internal',slice(0,len(qi))),('official',slice(len(qi),None))]:
        _,g,n=np.unique(xy[sl],axis=0,return_inverse=True,return_counts=True);g=g.ravel()
        for weighting,sw in [('scan',np.ones(len(g))/len(g)),('position',1/(len(n)*n[g]))]:
            rows.append(dict(method=variant,split=split,weighting=weighting,**metrics(prediction[sl],dict(xy=xy[sl],scan_weights=sw))))
    np.savez_compressed(destination/'predictions.npz',prediction=prediction,query_xy=xy,internal_count=len(qi),support_xy=support['positions'])
    write(destination/'completed.json',dict(completed=True,event=name,variant=variant,results=rows,
         prediction_sha256=digest(destination/'predictions.npz')))
    log(stage='completed',event=name,variant=variant,results=[r for r in rows if r['weighting']=='position'])


def summarize():
    rows=[]
    for variant in VARIANTS:
        for meta in json.loads((OUT/'manifest.json').read_text())['events']:
            d=json.loads((OUT/variant/meta['id']/'completed.json').read_text())
            rows.extend(dict(event=meta['id'],floor=meta['floor'],**r) for r in d['results'])
    summary=[]
    for method in VARIANTS:
        for split in ('internal','official'):
            v=[r['mde'] for r in rows if r['method']==method and r['split']==split and r['weighting']=='position']
            summary.append(dict(method=method,split=split,mean_error=float(np.mean(v)),events=len(v)))
    write(OUT/'summary.json',dict(completed=True,summary=summary,rows=rows))
    log(stage='summary',summary=summary)


if __name__=='__main__':
    if socket.gethostname()!='ray-web': raise RuntimeError('All numerical work must run on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['check','prepare','event','summarize'])
    p.add_argument('--event');p.add_argument('--variant',choices=list(VARIANTS));args=p.parse_args()
    if args.stage=='check': log(stage='checks',**check())
    elif args.stage=='prepare': prepare()
    else:
        freeze=json.loads((OUT/'freeze.json').read_text())
        for path,sha in freeze['code_sha256'].items():
            if digest(Path(path))!=sha: raise RuntimeError(f'Frozen source changed: {path}')
        if args.stage=='event': event(args.event,args.variant)
        else: summarize()
