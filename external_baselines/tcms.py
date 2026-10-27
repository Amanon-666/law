"""TC-MSStack on the same 24 H/S/Q tasks; reuse the existing paper implementation.

The new optimizer returns actual terminal parameters, never an earlier trial
with a different point's success flag. No old code or frozen result is changed.
"""
import argparse
import json
import math
from pathlib import Path
import socket
import time

import numpy as np
import torch
from scipy.optimize import minimize

from external_baselines.run import data_module, task_for, ROOT, prepared
from full_ap_study import metrics
from run_multi_history import digest, write
from tcms_stack_bridge import SourcePair, weights, ORIGINAL, SOLVE

OUT = ROOT / 'results/source_tcms_v2_20261026'
PAIR_BOUNDS = [(math.log(1e-3), math.log(1e3)), (-5., 5.),
               (math.log(1e-4), math.log(10)), (math.log(1e-4), math.log(10)), (-1., 1.)]


def log(**row):
    print(json.dumps(dict(time=prepared.now(), **row), allow_nan=False), flush=True)


def features(rss):
    return torch.from_numpy((np.where(rss == 100, -110, rss) + 110) / 110.).double()


def pairs_for(task, axis):
    known = np.concatenate([h['xy'] for h in task['H']])
    center, scale = known.mean(0), known.std(0)
    if np.any(scale == 0):
        raise ValueError('Selected H cannot normalize both coordinate axes')
    support = task['S']; sx = features(support['rss'])
    yt = torch.from_numpy((support['xy'][:, axis] - center[axis]) / scale[axis]).double()
    pairs = [SourcePair(features(h['rss']), sx,
                        torch.from_numpy((h['xy'][:, axis] - center[axis]) / scale[axis]).double(),
                        yt, support['group']) for h in task['H']]
    return pairs, center, scale


def objective(pairs, value):
    p = torch.from_numpy(value).double().requires_grad_()
    ps = p.reshape(len(pairs), 5)
    prediction = weights(ps) @ torch.stack([pair.target_prediction(v) for pair, v in zip(pairs, ps)])
    loss = (prediction - pairs[0].yt).square().mean()
    gradient, = torch.autograd.grad(loss, p)
    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite TC-MSStack support objective/gradient')
    return float(loss.detach()), gradient.detach().numpy()


def projected_gradient(value, gradient, bounds):
    result = gradient.copy()
    for i, (x, g, (lo, hi)) in enumerate(zip(value, gradient, bounds)):
        if (x <= lo + 1e-8 and g > 0) or (x >= hi - 1e-8 and g < 0):
            result[i] = 0.
    return float(np.abs(result).max())


def fit(pairs, seed, settings):
    bounds = PAIR_BOUNDS * len(pairs)
    rng = np.random.default_rng(seed); runs = []
    for restart in range(settings['restarts']):
        initial = np.concatenate([pair.initial(rng) for pair in pairs])
        trace = []
        def value_gradient(value):
            loss, gradient = objective(pairs, value)
            trace.append(loss)
            return loss, gradient
        start = time.perf_counter()
        result = minimize(value_gradient, initial, jac=True, method='L-BFGS-B', bounds=bounds,
            options=dict(maxiter=settings['maxiter'], maxfun=settings['maxfun'], maxls=30,
                         ftol=1e-12, gtol=1e-6))
        # Re-evaluate the actual returned point; this also checks finite/SPD state.
        loss, gradient = objective(pairs, result.x)
        pg = projected_gradient(result.x, gradient, bounds)
        runs.append(dict(restart=restart, parameters=result.x.tolist(), objective=loss,
            optimizer_reported_objective=float(result.fun), terminal_objective_difference=loss-float(result.fun),
            optimizer_success=bool(result.success), message=str(result.message),
            iterations=int(result.nit), evaluations=int(result.nfev), trace=trace,
            projected_gradient=pg, raw_gradient_max=float(np.abs(gradient).max()),
            converged=bool(result.success and pg <= settings['projected_gradient_tolerance']),
            near_bounds=[i for i, (x, (lo, hi)) in enumerate(zip(result.x, bounds)) if min(x-lo, hi-x)<1e-6],
            seconds=time.perf_counter()-start))
    converged = [r for r in runs if r['converged']]
    chosen = min(converged or runs, key=lambda r: r['objective'])
    ps = torch.tensor(chosen['parameters']).double().reshape(len(pairs), 5)
    return ps, dict(chosen_restart=chosen['restart'], selected_converged=chosen['converged'],
                    selection='Minimum recomputed S-only loss among converged terminal points; if none, retain an explicitly unfinished result',
                    converged_restarts=len(converged), runs=runs)


def pilot():
    """One full-H support objective/gradient timing; no fit or query score."""
    manifest = data_module().verify_freeze(prepared.PREPARATION)
    # Select by dense-solve size only, never by signal or query performance.
    meta = max(manifest['events'], key=lambda e: sum(h['scans']**3 for h in e['history_metadata'].values()))
    task = task_for(meta); pairs, _, _ = pairs_for(task, 0)
    value = np.concatenate([pair.initial(np.random.default_rng(261026+i)) for i, pair in enumerate(pairs)])
    started = time.perf_counter(); loss, gradient = objective(pairs, value)
    seconds = time.perf_counter()-started
    OUT.mkdir(exist_ok=True)
    if (OUT/'pilot.json').exists():
        raise FileExistsError(OUT/'pilot.json')
    record = dict(created=prepared.now(), event=meta['id'], H_scans=[len(p.xs) for p in pairs],
        S_scans=len(pairs[0].xt), APs=len(meta['AP_indices']), seconds=seconds,
        support_objective=loss, gradient_max=float(np.abs(gradient).max()),
        query_scored=False, optimizer_updates=0, threads=torch.get_num_threads(),
        code_sha256={str(p):digest(p) for p in [Path(__file__),ROOT/'tcms_stack_bridge.py',ORIGINAL,SOLVE]},
        interpretation='Full retained H, one source-development objective only. Select and freeze the complete run budget after this timing.')
    write(OUT/'pilot.json',record);log(stage='pilot',**record)


def freeze(args):
    if (OUT/'freeze.json').exists():
        raise FileExistsError(OUT/'freeze.json')
    pilot_record = json.loads((OUT/'pilot.json').read_text())
    for path, sha in pilot_record['code_sha256'].items():
        assert digest(Path(path)) == sha, path
    if min(args.restarts,args.maxiter,args.maxfun) < 1:
        raise ValueError('Explicit positive fitting budgets are required')
    manifest = data_module().verify_freeze(prepared.PREPARATION)
    for meta in manifest['events']:
        task_for(meta)  # exact H/S/Q raw row, AP and coordinate mapping
    files = [Path(__file__),ROOT/'external_baselines/run.py',ROOT/'tcms_stack_bridge.py',
             ORIGINAL,SOLVE,Path(data_module().__file__),Path(data_module().source_task.__code__.co_filename)]
    record = dict(created=prepared.now(), paper='https://proceedings.mlr.press/v70/wei17a/wei17a.pdf',
        provenance='Project independent implementation of Eqs 1/15/16; not author code or an exact reproduction of the paper benchmark',
        settings=dict(restarts=args.restarts,maxiter=args.maxiter,maxfun=args.maxfun,projected_gradient_tolerance=1e-3),
        seed_rule='261026 + coordinate_axis; same source-only starts across methods/events',
        preprocessing='Frozen AP columns; RSS=100 -> -110; input=(RSS+110)/110. Coordinate center/std from actual retained H only.',
        inference='Two coordinate models; existing SourcePair and Eq15 weights; full retained raw H, no source subsampling; raw coordinates, no clipping',
        source_information='Same deployment H/S/AP as LAW. This method does not additionally use the other buildings used to train g/base/obs; deployment-information parity is not complete training-information parity.',
        objective='Whole-position S-LOO squared coordinate loss; source/target kernels and source coefficients jointly fitted',
        convergence='Record optimizer status and recomputed projected gradient at actual returned parameters. Select among converged restarts; if none, report lowest-loss unfinished result. Report terminal loss discrepancy without treating roundoff as whole-run failure.',
        chronological_revision='v1 stopped before Q scoring on a 1.6e-9 reported/recomputed loss difference at an abnormal line-search termination. Its code, freeze, logs and S-only diagnosis remain in source_tcms_v1_20261026.',
        budget='maxfun/maxiter are SciPy termination budgets per restart; final-point diagnostic is one extra evaluation. No automatic continuation or query-driven reruns.',
        reverse_audit='H/S raw rows -> SourcePair -> S-LOO fitting -> fixed parameters -> Q RSS -> coordinates; Q XY only enters metrics',
        boundaries='K=5/R1; incorrect source and unseen geometry can cause negative transfer. All-zero source coefficients leave Eq15 undefined and fail explicitly. Same RSS at different coordinates remains ambiguous.',
        pilot_sha256=digest(OUT/'pilot.json'),code_sha256={str(p):digest(p) for p in files},
        preparation_manifest_sha256=digest(prepared.PREPARATION/'manifest.json'))
    write(OUT/'manifest.json',manifest);write(OUT/'freeze.json',record);log(stage='frozen',settings=record['settings'])


def verify():
    record=json.loads((OUT/'freeze.json').read_text())
    for path,sha in record['code_sha256'].items():
        assert digest(Path(path))==sha,path
    assert digest(prepared.PREPARATION/'manifest.json')==record['preparation_manifest_sha256']
    data_module().verify_freeze(prepared.PREPARATION)
    return record,json.loads((OUT/'manifest.json').read_text())


def event(name):
    record,manifest=verify();meta=next(e for e in manifest['events'] if e['id']==name)
    task=task_for(meta);dest=OUT/name;dest.mkdir(exist_ok=True)
    if (dest/'completed.json').exists():
        return
    fitted=[]
    for axis in range(2):
        pairs,center,scale=pairs_for(task,axis);path=dest/f'axis{axis}.json'
        if not path.exists():
            ps,diagnostics=fit(pairs,261026+axis,record['settings'])
            write(path,dict(parameters=ps.tolist(),diagnostics=diagnostics))
        saved=json.loads(path.read_text());ps=torch.tensor(saved['parameters']).double()
        fitted.append((pairs,ps,saved['diagnostics']['selected_converged']))
    # No Q RSS enters either coordinate fit; both models are now fixed.
    xq=features(task['Q']['rss'])
    with torch.no_grad():
        pred=np.column_stack([(weights(ps)@torch.stack([pair.predict(p,xq) for pair,p in zip(pairs,ps)])).numpy()
                              for pairs,ps,_ in fitted])*scale+center
    if not np.isfinite(pred).all():
        raise FloatingPointError('Nonfinite TC-MSStack coordinates')
    np.savez_compressed(dest/'predictions.npz',prediction=pred,query_xy=task['Q']['xy'],
                        query_scan_ids=task['Q']['scan_ids'],AP_indices=meta['AP_indices'])
    result=dict(event=name,building=meta['building'],J=len(meta['histories']),coordinate_unit=meta['coordinate_unit'],
                selected_models_converged=[done for _,_,done in fitted],results=metrics(pred,task['Q']))
    write(dest/'completed.json',result);log(stage='event_complete',**result)


if __name__=='__main__':
    if socket.gethostname()!='ray-web':
        raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','freeze','event'])
    parser.add_argument('--event');parser.add_argument('--restarts',type=int,default=0)
    parser.add_argument('--maxiter',type=int,default=0);parser.add_argument('--maxfun',type=int,default=0)
    args=parser.parse_args()
    if args.stage=='pilot':pilot()
    elif args.stage=='freeze':freeze(args)
    else:event(args.event)
