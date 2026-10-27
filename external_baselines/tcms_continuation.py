"""Bounded TCMS continuation: failed numerical restart is explicit, not a fit.

Only the three unfinished UJI_B2 s261027 J2/J3/J4 events are fitted. Kernel,
bounds, starts, settings, row permissions and prediction reuse the frozen v2.
No changed jitter, penalty objective, narrowed bounds or Q-driven selection.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.linalg import eigvalsh
from scipy.optimize import minimize

from external_baselines import tcms as old
from run_multi_history import digest,write
from tcms_stack_bridge import SourcePair

ORIGINAL_OUT=old.OUT
OUT=ROOT/'results/source_tcms_continuation_v1_20261027'
EVENTS=[f'UJI_B2_F2_s261027_J{j}' for j in [2,3,4]]
CURRENT_EVENT=None


class NoValidEndpoint(RuntimeError):pass


@torch.no_grad()
def matrix_diagnostics(pairs,value):
    """Diagnostic arithmetic only; nothing here repairs optimizer matrices."""
    records=[]
    for i,(pair,p) in enumerate(zip(pairs,torch.tensor(value).reshape(len(pairs),5))):
        kernel=pair.kernel(pair.ss,p)
        noise=float(torch.exp(2*p[2])+1e-8)
        a=kernel+noise*pair.isource
        _,info=torch.linalg.cholesky_ex(a)
        aa=a.numpy();symmetric=(aa+aa.T)/2
        minimum=float(eigvalsh(symmetric,subset_by_index=[0,0],check_finite=False)[0])
        item=dict(source_index=i,source_rows=len(pair.xs),parameters=p.tolist(),length=float(p[0].exp()),
            signal_variance=float(torch.exp(2*p[1])),noise_variance=float(torch.exp(2*p[2])),fixed_jitter=1e-8,
            A_symmetry_max=float(np.abs(aa-aa.T).max()),A_symmetric_average_minimum_eigenvalue=minimum,
            A_cholesky_info=int(info),distance_diagonal_max=float(pair.ss.diagonal().abs().max()))
        if int(info):
            lower=np.tril(aa)+np.tril(aa,-1).T
            item['A_lower_triangle_interpretation_minimum_eigenvalue']=float(eigvalsh(lower,subset_by_index=[0,0],check_finite=False)[0])
            item['A_symmetric_average_cholesky_info']=int(torch.linalg.cholesky_ex((a+a.T)/2)[1])
            # Direct pairwise subtraction avoids cancellation in ||x||²+||y||²-2x.y.
            direct=torch.cdist(pair.xs,pair.xs,compute_mode='donot_use_mm_for_euclid_dist').square()
            stable=pair.kernel(direct,p)+noise*pair.isource
            _,stable_info=torch.linalg.cholesky_ex(stable)
            ss=stable.numpy()
            item.update(direct_distance_max_difference=float((direct-pair.ss).abs().max()),
                direct_A_max_difference=float((stable-a).abs().max()),direct_A_cholesky_info=int(stable_info),
                direct_A_minimum_eigenvalue=float(eigvalsh((ss+ss.T)/2,subset_by_index=[0,0],check_finite=False)[0]),
                note='Direct distances are diagnostic only, not substituted into fitting.')
        records.append(item)
    return records


def fit(pairs,seed,settings,diagnostic=False):
    bounds=old.PAIR_BOUNDS*len(pairs);rng=np.random.default_rng(seed);runs=[]
    axis=seed-261026
    dest=(OUT/'diagnosis/restarts' if diagnostic else OUT/CURRENT_EVENT/f'axis{axis}_restarts')
    dest.mkdir(parents=True,exist_ok=True)
    for restart in range(settings['restarts']):
        path=dest/f'restart{restart}.json'
        initial=np.concatenate([pair.initial(rng) for pair in pairs])
        if path.exists() and not diagnostic:
            record=json.loads(path.read_text());np.testing.assert_array_equal(record['initial'],initial)
            runs.append(record);continue
        trace=[];last=None;phase='objective';started=time.perf_counter()
        def objective(value):
            nonlocal last
            last=value.copy()
            loss,gradient=old.objective(pairs,value)
            trace.append(dict(parameters=value.tolist(),objective=loss,gradient=gradient.tolist(),raw_gradient_max=float(np.abs(gradient).max())))
            return loss,gradient
        try:
            result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,
                options=dict(maxiter=settings['maxiter'],maxfun=settings['maxfun'],maxls=30,ftol=1e-12,gtol=1e-6))
            phase='terminal_recheck';last=result.x.copy()
            loss,gradient=old.objective(pairs,result.x)
            pg=old.projected_gradient(result.x,gradient,bounds)
            record=dict(restart=restart,initial=initial.tolist(),status='valid_terminal',parameters=result.x.tolist(),objective=loss,
                optimizer_reported_objective=float(result.fun),terminal_objective_difference=loss-float(result.fun),
                optimizer_success=bool(result.success),message=str(result.message),iterations=int(result.nit),evaluations=int(result.nfev),
                projected_gradient=pg,raw_gradient_max=float(np.abs(gradient).max()),
                converged=bool(result.success and pg<=settings['projected_gradient_tolerance']),
                near_bounds=[i for i,(x,(lo,hi)) in enumerate(zip(result.x,bounds)) if min(x-lo,hi-x)<1e-6])
        except torch.linalg.LinAlgError as exc:
            # Abort this restart. No artificial scalar/gradient is sent to SciPy.
            record=dict(restart=restart,initial=initial.tolist(),status='failed_numerical_restart',converged=False,
                optimizer_success=False,failure_phase=phase,failure_message=str(exc),failed_parameters=last.tolist(),
                successful_evaluations=len(trace),failure_traceback=traceback.format_exc(),matrix_diagnostics=matrix_diagnostics(pairs,last))
        record.update(trace=trace,seconds=time.perf_counter()-started)
        write(path,record);runs.append(record)
        old.log(stage='restart_finished',event=CURRENT_EVENT,axis=axis,restart=restart,status=record['status'],
            converged=record['converged'],seconds=record['seconds'])
        if diagnostic and record['status']=='failed_numerical_restart':
            return None,dict(runs=runs,first_failure_captured=True)
    terminal=[r for r in runs if r['status']=='valid_terminal']
    converged=[r for r in terminal if r['converged']]
    if not terminal:
        raise NoValidEndpoint(f'{CURRENT_EVENT} axis{axis}: all fixed starts failed; no Q prediction')
    chosen=min(converged or terminal,key=lambda r:r['objective'])
    parameters=torch.tensor(chosen['parameters']).reshape(len(pairs),5)
    return parameters,dict(chosen_restart=chosen['restart'],selected_converged=chosen['converged'],
        converged_restarts=len(converged),failed_restarts=sum(r['status']=='failed_numerical_restart' for r in runs),
        selection='Lowest recomputed S loss among converged terminal points; otherwise explicit unfinished valid terminal. Failed restart cannot be selected.',runs=runs)


def diagnose():
    global CURRENT_EVENT
    old.verify();OUT.mkdir(exist_ok=True)
    assert not (OUT/'diagnosis.json').exists()
    shutil.copy2(__file__,OUT/'diagnosis_code.py')
    record=json.loads((ORIGINAL_OUT/'freeze.json').read_text())
    manifest=json.loads((ORIGINAL_OUT/'manifest.json').read_text())
    CURRENT_EVENT=EVENTS[0];meta=next(e for e in manifest['events'] if e['id']==CURRENT_EVENT)
    task=old.task_for(meta);pairs,_,_=old.pairs_for(task,0)
    _,diagnostics=fit(pairs,261026,record['settings'],diagnostic=True)
    write(OUT/'diagnosis.json',dict(created=old.prepared.now(),event=CURRENT_EVENT,axis=0,query_scored=False,
        source_model='Same RBF Gram distance, positive source noise + original 1e-8 jitter. Exact Euclidean RBF+positive diagonal is SPD.',
        settings=record['settings'],script_sha256=digest(Path(__file__)),diagnostics=diagnostics))
    old.log(stage='diagnosis_complete',failure_captured=diagnostics.get('first_failure_captured',False))


def explain():
    """Use the captured S trial, without repeating its optimization."""
    _,manifest=old.verify();meta=next(e for e in manifest['events'] if e['id']==EVENTS[0])
    diagnosis=json.loads((OUT/'diagnosis.json').read_text())
    failed=diagnosis['diagnostics']['runs'][-1]['failed_parameters']
    pairs,_,_=old.pairs_for(old.task_for(meta),0)
    write(OUT/'matrix_explanation.json',dict(created=old.prepared.now(),query_scored=False,optimization_updates=0,
        matrices=matrix_diagnostics(pairs,failed),
        cause='At legal extreme hyperparameters, rounded Gram distances lose symmetry; tiny distance errors are amplified by signal_variance/length^2. The lower triangle consumed by Cholesky can be indefinite even if the symmetric average is positive. Direct Euclidean distances pass with unchanged noise/jitter.',
        handling='Record and reject the failed restart; keep all other original fixed starts and numerical/model settings unchanged. Direct/symmetrized matrices are diagnostic only.'))


def freeze():
    assert not (OUT/'freeze.json').exists()
    old_record,manifest=old.verify()
    diagnosis=json.loads((OUT/'diagnosis.json').read_text())
    assert diagnosis['diagnostics']['first_failure_captured']
    files=[Path(__file__),OUT/'diagnosis_code.py']
    files+=[Path(p) for p in old_record['code_sha256']]
    inputs=[ORIGINAL_OUT/'freeze.json',ORIGINAL_OUT/'manifest.json',ORIGINAL_OUT/'batch_status.json',OUT/'diagnosis.json',OUT/'matrix_explanation.json']
    inputs+=list((OUT/'diagnosis/restarts').glob('*.json'))
    for event in manifest['events']:
        d=ORIGINAL_OUT/event['id']
        if (d/'completed.json').exists():inputs.extend([d/'completed.json',d/'predictions.npz',*d.glob('axis*.json')])
    inputs+=list((ORIGINAL_OUT/'logs').glob('*'))
    assert sum((ORIGINAL_OUT/e['id']/'completed.json').exists() for e in manifest['events'])==21
    write(OUT/'freeze.json',dict(old_record,created=old.prepared.now(),events=EVENTS,
        numerical_revision='A failed Cholesky aborts and records that fixed restart; continue the other fixed starts. No jitter, kernel, distance, bound, S objective or budget change. No penalty/fake gradient. No best-trial/terminal-status mixing.',
        numerical_scope='Failed trial diagnostics retained in full, including direct-distance comparison only for diagnosis. Both actual terminal axes must be scored/convergence-labelled.',
        prior_results='21 original completed v2 events retained byte-for-byte and merged by reference; only the remaining three events fit here',
        diagnosis_extra_budget='One S-only replay until first failure, separately logged; formal per-start budget is identical to v2',
        code_sha256={str(p):digest(p) for p in files},inputs_sha256={str(p):digest(p) for p in inputs},
        preparation_manifest_sha256=old_record['preparation_manifest_sha256']))
    shutil.copy2(ORIGINAL_OUT/'manifest.json',OUT/'manifest.json')
    shutil.copy2(__file__,OUT/'frozen_code.py')
    old.log(stage='frozen_continuation',events=EVENTS,settings=old_record['settings'])


def verify():
    record=json.loads((OUT/'freeze.json').read_text())
    for section in ['code_sha256','inputs_sha256']:
        for path,sha in record[section].items():assert digest(Path(path))==sha,path


def run():
    global CURRENT_EVENT
    verify();old.OUT=OUT;old.fit=fit
    for name in EVENTS:
        CURRENT_EVENT=name
        try:old.event(name)
        except NoValidEndpoint as exc:
            write(OUT/name/'incomplete.json',dict(event=name,reason=str(exc),query_scored=False))
    summarize()


def summarize():
    manifest=json.loads((ORIGINAL_OUT/'manifest.json').read_text());rows=[];incomplete=[]
    for event in manifest['events']:
        source=OUT if event['id'] in EVENTS else ORIGINAL_OUT
        path=source/event['id']/'completed.json'
        if not path.exists():incomplete.append(event['id']);continue
        result=json.loads(path.read_text());result['source_result']=str(path);rows.append(result)
    groups=[]
    for building in sorted({r['building'] for r in rows}):
        br=[r for r in rows if r['building']==building]
        groups.append(dict(building=building,events=len(br),coordinate_unit=br[0]['coordinate_unit'],
            mean_mde=float(np.mean([r['results']['mde'] for r in br])),
            events_both_axes_converged=sum(all(r['selected_models_converged']) for r in br)))
    write(OUT/'summary.json',dict(created=old.prepared.now(),complete=len(rows)==24,events=len(rows),incomplete=incomplete,
        new_events=[r for r in rows if r['event'] in EVENTS],all_records=rows,by_building=groups,
        limits='Source-development only, K5R1/12Q. Project independent TCMS implementation. Budget/line-search-unfinished models are not evidence of beating a trained SOTA. No kernel/jitter or Q-driven change.'))
    old.log(stage='summary',complete=len(rows)==24,new_events=[r for r in rows if r['event'] in EVENTS])


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['diagnose','explain','freeze','run','summarize']);args=parser.parse_args()
    dict(diagnose=diagnose,explain=explain,freeze=freeze,run=run,summarize=summarize)[args.stage]()
