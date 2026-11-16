"""Complete the original long-K UJI tasks using existing ReFi and GP assets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

BASE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110')
sys.path.insert(0, str(BASE/'src'))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from law_bench import run
from law_bench.data import load
from law_bench.method_retained_gp import history_key
from law_bench.protocol import make_tasks
from law_bench.score import aggregate, score
from paper_bench_20261111 import ext_gp
import predict as gp_prediction
import position_fusion_gp
import cached_retained_readout

BUDGETS = (1,3,5,10,20)


def read(path):
    return json.loads(path.read_text())


def matched(event, task, truth):
    assert event['status'] == 'completed' and event['task_id'] == task.task_id
    assert event['query_row_indices'] == truth['row_indices'].tolist()
    assert event['query_groups'] == truth['groups'].tolist() and event['meta'] == task.meta


def context_identity(task, keys, parameter_sha):
    digest = hashlib.sha256(json.dumps(dict(dataset=task.dataset,building=task.building,
        floor=task.floor,K=task.K,APs=task.ap_ids,history_keys=keys,
        parameters_sha256=parameter_sha),sort_keys=True).encode())
    for name in ('rss','xy','acq'):
        array = np.ascontiguousarray(task.S[name])
        digest.update(str((name,array.dtype.str,array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def prepare(output):
    buildings, axes = load('uji')
    tasks, bindings, folds = {}, {}, {}
    for building in ('B0','B1','B2'):
        original = BASE/f'results/paper_uji_{building}_kcurve_20261111'
        main = BASE/f'results/paper_uji_{building}_ext_20261111'
        old, main_freeze = read(original/'freeze.json'), read(main/'freeze.json')
        config = old['configuration']
        assert tuple(config['budgets']) == BUDGETS and config['repeats'] == [0,1,2]
        assert config['protocol'] == 'static-resample-v2'
        for name in ('seed','R','query_positions'):
            assert config[name] == main_freeze['configuration'][name]
        for name,function in [('protocol.py',make_tasks),('score.py',score)]:
            actual = run._sha(sys.modules[function.__module__].__file__)
            assert old['code'][name] == main_freeze['code'][name] == actual
        target_identity = run._data_identity({building:buildings[building]},{building:axes[building]})
        source_names = sorted(old['source_data'])
        source_identity = run._data_identity({b:buildings[b] for b in source_names},{b:axes[b] for b in source_names})
        assert target_identity == old['target_data'] == main_freeze['target_data']
        assert source_identity == old['source_data'] == main_freeze['source_data']
        options = main_freeze['configuration']['method_options']['refi_GP']['gp_options']
        assert options['fit_missing_history'] is False and options.get('uji_assets_dir') is None
        parameter_sha = run._sha(options['parameters_path'])
        assert parameter_sha == options['provenance']['parameters_sha256']
        fit = read(original/'source_models/refi.json')
        assert fit['status'] == 'completed'
        assert run._sha(original/'source_models/refi.joblib') == fit['sha256']
        folds[building] = dict(source_run=str(original),main_GP_run=str(main),
            original_freeze_sha256=run._sha(original/'freeze.json'),
            main_GP_freeze_sha256=run._sha(main/'freeze.json'),
            source_model_sha256=fit['sha256'],source_data=source_identity,target_data=target_identity,
            gp_options=options)
        for floor in sorted(buildings[building]):
            for repeat in (0,1,2):
                long_tasks = make_tasks('uji',building,buildings[building],axes[building],floor,
                    budgets=BUDGETS,repeat=repeat,seed=config['seed'],R=config['R'],query_positions=config['query_positions'])
                short = {task.K:(task,truth) for task,truth in make_tasks('uji',building,buildings[building],axes[building],floor,
                    budgets=(1,3,5),repeat=repeat,seed=config['seed'],R=config['R'],query_positions=config['query_positions'])}
                reference_truth = long_tasks[0][1]
                previous = None
                keys = {str(f):history_key(long_tasks[0][0],f,parameter_sha) for f in long_tasks[0][0].H}
                for task, truth in long_tasks:
                    assert np.array_equal(truth['row_indices'],reference_truth['row_indices'])
                    assert np.array_equal(truth['groups'],reference_truth['groups'])
                    assert len(np.unique(task.S['xy'],axis=0)) == task.K
                    if previous is not None:
                        for name in ('rss','xy','acq'):
                            assert np.array_equal(task.S[name][:len(previous.S[name])],previous.S[name])
                    previous = task
                    old_event_path = original/'events/refi'/(task.task_id+'.json')
                    matched(read(old_event_path),task,truth)
                    identity = context_identity(task,keys,parameter_sha)
                    binding = dict(task_id=task.task_id,building=building,floor=floor,repeat=repeat,K=task.K,
                        H_rows=sum(len(h['rss']) for h in task.H.values()),S_rows=len(task.S['rss']),
                        query_scans=len(truth['xy']),query_positions=len(np.unique(truth['groups'])),
                        original_learning_event_sha256=run._sha(old_event_path),
                        GP_context_sha256=identity,history_keys=keys,state_reused=False)
                    if task.K in short:
                        state_path = main/'events/retained_gp'/(task.task_id+'.json')
                        state_event = read(state_path)
                        matched(state_event,*short[task.K])
                        assert state_event['diagnostics']['parameters_sha256'] == parameter_sha
                        assert state_event['diagnostics']['history_keys'] == keys
                        binding['main_Query_rows_equal'] = state_event['query_row_indices'] == truth['row_indices'].tolist()
                        same = context_identity(short[task.K][0],keys,parameter_sha) == identity
                        if same:
                            binding.update(state_reused=True,state_source=str(state_path),state_event_sha256=run._sha(state_path))
                    tasks[task.task_id] = (task,truth)
                    bindings[task.task_id] = binding
    assert len(tasks) == 195
    pilot = max((b for b in bindings.values() if b['K']==20 and b['repeat']==0),
                key=lambda b:(b['H_rows'],b['task_id']))['task_id']
    review = dict(task_count=195,budgets=list(BUDGETS),repeats=[0,1,2],folds=folds,
        state_reuse_tasks=sum(b['state_reused'] for b in bindings.values()),
        new_adaptation_tasks=sum(not b['state_reused'] for b in bindings.values()),
        main_Query_equal_tasks=sum(b.get('main_Query_rows_equal',False) for b in bindings.values()),
        fixed_timing_task=pilot,timing_rule='K20/r0 task with largest total raw H rows, selected before new predictions.',
        task_bindings=list(bindings.values()))
    freeze = dict(scope='Original 195 long-K tasks and same Query/nested Support within each floor/repeat.',
        folds=folds,budgets=list(BUDGETS),task_count=195,
        learning_model='Original 55-feature five-member ReFi, trained at K1/3/5. K10/20 are frozen-model calibration-budget extrapolation.',
        GP='Existing complete raw-H covariance fields and fold-specific parameters. fit_missing_history=False. Original adaptation/proposal/ESS/solver settings.',
        state_reuse='Only for identical actual H keys, S RSS/XY/acquisition arrays, AP axis, K and parameters. New long-K Query RSS is rescored.',
        reader='Original log_position_mass and fixed probability_half pooling, same candidate atoms and bounds.',
        permissions='ReFi and GP receive only H/S and Query RSS. Truth XY is used after both predictions for scoring.',
        review=dict(input='Old long-K Query rows/groups/meta and complete nested actual S arrays are checked before prediction.',
            sampling='Retained-field proposal seeds retain original task_id and K. Source code and random/ESS policy are unchanged.',
            edge_cases='K1, missing/new APs, zero residuals, bad priors, coordinate frame and clipping follow the original implementations. No new mechanism or fallback is introduced.',
            interpretation='More S may saturate or worsen individual predictions. No monotonicity or K10/20 source-training claim is made.'),
        training_performed=False,new_history_fitting_allowed=False,
        code_sha256={str(p):run._sha(p) for p in [Path(__file__),Path(ext_gp.__file__),Path(run.__file__),
            Path(sys.modules[make_tasks.__module__].__file__),Path(sys.modules[score.__module__].__file__),
            Path(gp_prediction.__file__),Path(position_fusion_gp.__file__),Path(cached_retained_readout.__file__)]})
    if (output/'freeze.json').exists():
        assert read(output/'freeze.json') == freeze
        assert read(output/'source_review.json') == review
    else:
        run._write(output/'freeze.json',freeze)
        run._write(output/'source_review.json',review)
    return buildings,axes,tasks,bindings,folds,pilot


def timed(function, times, name):
    def call(*args,**kwargs):
        tick = time.monotonic()
        try:
            return function(*args,**kwargs)
        finally:
            times[name] = time.monotonic()-tick
    return call


def export_gp(path, source_path, combined, truth):
    common = {name:combined[name] for name in ('task_id','dataset','building','floor','repeat','K',
        'meta','status','query_row_indices','query_groups','input_binding')}
    diagnostics = combined['diagnostics']
    prediction = diagnostics['GP_projected']
    run._write(path,dict(common,method='retained_gp',prediction=prediction,
        scores=score(prediction,truth),seconds=diagnostics['gp_seconds'],
        diagnostics={name:diagnostics['GP'][name] for name in ('state','parameters_sha256')},
        derived_from=dict(path=str(source_path),sha256=run._sha(source_path))))


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    buildings,axes,tasks,bindings,folds,pilot = prepare(args.output)
    print(json.dumps(dict(stage='source_review_complete',tasks=195,
        reused_states=sum(b['state_reused'] for b in bindings.values()),fixed_timing_task=pilot)),flush=True)
    if args.audit_only:
        return
    selected = [pilot] if args.timing_task else list(tasks)
    model,active_building = None,None
    for task_id in selected:
        task,truth = tasks[task_id]
        binding = bindings[task_id]
        target = args.output/'events/refi_GP'/(task_id+'.json')
        gp_target = args.output/'events/retained_gp'/(task_id+'.json')
        if target.exists():
            old = read(target)
            matched(old,task,truth)
            assert old['input_binding'] == binding
            if not gp_target.exists():
                export_gp(gp_target,target,old,truth)
            gp_old = read(gp_target)
            matched(gp_old,task,truth)
            assert gp_old['input_binding'] == binding and gp_old['derived_from']['sha256'] == run._sha(target)
            continue
        building = task.building
        if building != active_building:
            sources = [dict(dataset='uji',building=b,floors=buildings[b],ap_ids=axes[b]) for b in folds[building]['source_data']]
            model = ext_gp.ReFiGP(source_run=folds[building]['source_run'],
                gp_options=folds[building]['gp_options']).fit_source(sources)
            active_building = building
        times = {}
        original_adapt = gp_prediction.adapt
        if binding['state_reused']:
            state_event = read(Path(binding['state_source']))
            assert run._sha(binding['state_source']) == binding['state_event_sha256']
            state = state_event['diagnostics']['state']
            adaptation = lambda *a,**k:state
        else:
            adaptation = original_adapt
        tick = time.monotonic()
        try:
            with (
                patch.object(gp_prediction,'adapt',timed(adaptation,times,'adaptation_seconds')),
                patch.object(position_fusion_gp,'prepare_retained_fields',timed(position_fusion_gp.prepare_retained_fields,times,'retained_construction_seconds')),
                patch.object(cached_retained_readout,'score_cached_fields',timed(cached_retained_readout.score_cached_fields,times,'Query_scoring_seconds')),
            ):
                prediction = model.predict(task)
            assert all(f['cache']!='fitted' for f in model.gp.diagnostics['history_fields'])
            common = dict(task_id=task_id,dataset='uji',building=building,floor=task.floor,repeat=task.repeat,K=task.K,
                meta=task.meta,status='completed',query_row_indices=truth['row_indices'].tolist(),
                query_groups=truth['groups'].tolist(),input_binding=binding)
            record = dict(common,method='refi_GP',prediction=prediction.tolist(),scores=score(prediction,truth),
                diagnostics=model.diagnostics,timings=times,seconds=time.monotonic()-tick)
            run._write(target,record)
            export_gp(gp_target,target,record,truth)
            print(json.dumps(dict(stage='task_completed',task_id=task_id,seconds=record['seconds'],
                timings=times,state_reused=binding['state_reused'])),flush=True)
        except Exception as error:
            run._write(args.output/'failures'/(task_id+'.json'),dict(task_id=task_id,status='failed',
                error=repr(error),seconds=time.monotonic()-tick,timings=times,input_binding=binding))
            raise
    if args.timing_task:
        pilot_record = read(args.output/'events/refi_GP'/(pilot+'.json'))
        old_seconds = sum(read(BASE/f'results/paper_uji_{b}_ext_20261111/events/refi_GP'/(t.task_id+'.json'))['seconds']
            for t,_ in tasks.values() for b in [t.building] if t.K in (1,3,5))
        forecast = 1.3*(old_seconds+78*pilot_record['seconds'])/3600
        run._write(args.output/'timing_estimate.json',dict(task_id=pilot,formal_task_reused=True,
            task_seconds=pilot_record['seconds'],stage_timings=pilot_record['timings'],
            original117_task_seconds=old_seconds,new_high_budget_tasks=78,margin=1.3,
            forecast_hours=forecast,rule='Historical full lower-K task cost plus78 times the fixed K20 timing, with30% margin. Single-task timing is a planning estimate.'))
        return
    summary = {}
    for method in ('refi_GP','retained_gp'):
        records = [read(args.output/'events'/method/(task_id+'.json')) for task_id in tasks]
        summary[method] = dict(method=method,tasks=195,by_K=aggregate(records),
            by_building={b:aggregate([r for r in records if r['building']==b]) for b in folds})
    run._write(args.output/'summary.json',summary)
    run._write(args.output/'complete.json',dict(complete=True,unique_task_scenarios=195,events=390,
        freeze_sha256=run._sha(args.output/'freeze.json'),source_review_sha256=run._sha(args.output/'source_review.json'),
        summary_sha256=run._sha(args.output/'summary.json'),training_performed=False,new_history_fields=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--timing-task',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    with threadpool_limits(limits=2):
        main(args)
