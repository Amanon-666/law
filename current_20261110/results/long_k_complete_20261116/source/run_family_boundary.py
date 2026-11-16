"""Four post-failure budgets in the same frozen B2F4/r0 long-K family."""
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

from run_long_k import (BASE, BUDGETS, read, matched, context_identity, timed,
    run, load, make_tasks, history_key, score, ext_gp, gp_prediction,
    position_fusion_gp, cached_retained_readout, torch, threadpool_limits)

WORK = BASE/'results/long_k_complete_20261116'
OUT = WORK/'boundary_B2F4_r0'


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    original_freeze = read(WORK/'data/freeze.json')
    fold = original_freeze['folds']['B2']
    original_review = read(WORK/'data/source_review.json')
    bindings = {b['task_id']:b for b in original_review['task_bindings']}
    buildings,axes = load('uji')
    family = make_tasks('uji','B2',buildings['B2'],axes['B2'],4,budgets=BUDGETS,
        repeat=0,seed=20261111,R=3,query_positions=60)
    scope = dict(post_failure_boundary_check=True,family='uji/B2/F4/r0',budgets=[1,3,5,10],
        full_sampling_budgets=list(BUDGETS),Query='Keep the exact original max-K20 Query for every budget.',
        source_freeze_sha256=run._sha(WORK/'data/freeze.json'),
        original_K20_failure_sha256=run._sha(WORK/'data/failures/uji/B2/F4/K20/r0.json'),
        code_sha256=run._sha(Path(__file__)),solver_or_seed_or_tolerance_changed=False,
        source_training=False,new_history_fitting=False,CPU_threads=2,maximum_wall_seconds=1800,
        rule='Retain each success or failure at the predeclared same family. No task substitution or posterior-solver change.')
    run._write(OUT/'scope.json',scope)
    model = ext_gp.ReFiGP(source_run=fold['source_run'],gp_options=fold['gp_options']).fit_source([
        dict(dataset='uji',building=b,floors=buildings[b],ap_ids=axes[b]) for b in fold['source_data']])
    results = []
    for task,truth in family[:-1]:
        original = read(Path(fold['source_run'])/'events/refi'/(task.task_id+'.json'))
        matched(original,task,truth)
        binding = bindings[task.task_id]
        keys = {str(f):history_key(task,f,fold['gp_options']['provenance']['parameters_sha256']) for f in task.H}
        assert context_identity(task,keys,fold['gp_options']['provenance']['parameters_sha256']) == binding['GP_context_sha256']
        adaptation = gp_prediction.adapt
        if binding['state_reused']:
            state_source = Path(binding['state_source'])
            assert run._sha(state_source) == binding['state_event_sha256']
            state = read(state_source)['diagnostics']['state']
            adaptation = lambda *a,**k:state
        times = {}
        record = dict(task_id=task.task_id,K=task.K,meta=task.meta,input_binding=binding,
            query_row_indices=truth['row_indices'].tolist(),query_groups=truth['groups'].tolist(),
            learning_reference=dict(original_event_sha256=binding['original_learning_event_sha256'],scores=original['scores']))
        tick = time.monotonic()
        try:
            with (
                patch.object(gp_prediction,'adapt',timed(adaptation,times,'adaptation_seconds')),
                patch.object(position_fusion_gp,'prepare_retained_fields',timed(position_fusion_gp.prepare_retained_fields,times,'retained_construction_seconds')),
                patch.object(cached_retained_readout,'score_cached_fields',timed(cached_retained_readout.score_cached_fields,times,'Query_scoring_seconds')),
            ):
                prediction = model.predict(task)
            assert all(f['cache']!='fitted' for f in model.gp.diagnostics['history_fields'])
            record.update(status='completed',complete_prediction=prediction.tolist(),
                complete_scores=score(prediction,truth),GP_prediction=model.diagnostics['GP_projected'],
                GP_scores=score(model.diagnostics['GP_projected'],truth))
        except Exception as error:
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        record.update(seconds=time.monotonic()-tick,timings=times,diagnostics=model.diagnostics)
        run._write(OUT/'events'/(task.task_id+'.json'),record)
        results.append(dict(task_id=task.task_id,K=task.K,status=record['status'],seconds=record['seconds'],
            complete_mean=record.get('complete_scores',{}).get('mean'),GP_mean=record.get('GP_scores',{}).get('mean'),
            error=record.get('error')))
        print(json.dumps(results[-1]),flush=True)
    run._write(OUT/'summary.json',dict(post_failure_boundary_check=True,family='uji/B2/F4/r0',
        requested_tasks=4,completed_tasks=sum(r['status']=='completed' for r in results),
        failed_tasks=sum(r['status']=='failed' for r in results),results=results,
        seconds=time.monotonic()-started,Query_binding='Original long-K Query, unchanged across all four budgets.',
        original_195_batch_status='stopped after the original K20 fixed-budget numerical rejection'))


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    with threadpool_limits(limits=2):
        main()
