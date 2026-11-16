"""Post-failure replay of the same AP, with original sampling and no Query score."""
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

BASE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110')
WORK = BASE/'results/long_k_complete_20261116'
sys.path.insert(0,str(BASE/'src'))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from law_bench import run
from law_bench.data import load
from law_bench.protocol import make_tasks
from law_bench.method_retained_gp import RetainedGPMethod,history_key
from fields import InterceptBankPrior
from observations import gaussian_update
import position_fusion_gp
import posterior
import sampling
from run_long_k import context_identity

AP,SEED = 143,1134527432
DEST = WORK/'diagnostic_failed_AP143'


class DiagnosticCaptured(Exception):
    pass


def values(tensor):
    return tensor.detach().cpu().tolist()


def main():
    started = time.monotonic()
    DEST.mkdir(exist_ok=False)
    failure = json.loads((WORK/'data/failures/uji/B2/F4/K20/r0.json').read_text())
    frozen = json.loads((WORK/'data/freeze.json').read_text())
    options = frozen['folds']['B2']['gp_options']
    buildings,axes = load('uji')
    task = make_tasks('uji','B2',buildings['B2'],axes['B2'],4,budgets=(1,3,5,10,20),
        repeat=0,seed=20261111,R=3,query_positions=60)[-1][0]
    keys = {str(f):history_key(task,f,options['provenance']['parameters_sha256']) for f in task.H}
    identity = context_identity(task,keys,options['provenance']['parameters_sha256'])
    assert identity == failure['input_binding']['GP_context_sha256']
    model = RetainedGPMethod(**options).fit_source([
        dict(dataset='uji',building=b,floors=buildings[b],ap_ids=axes[b]) for b in ('B0','B1')])
    result = dict(status='post_failure_diagnostic',task_id=task.task_id,AP_index=AP,AP_name=axes['B2'][AP],
        seed=SEED,GP_context_sha256=identity,original_failure=failure,
        original_failure_sha256=run._sha(WORK/'data/failures/uji/B2/F4/K20/r0.json'),
        observations=model.obs,base=model.base,Query_truth_saved=False,Query_scored=False,
        numerical_settings_changed=False,proposal_changed=False,
        source_code_sha256={str(p):run._sha(p) for p in [Path(__file__),Path(posterior.__file__),Path(sampling.__file__),
            Path(position_fusion_gp.__file__),Path(sys.modules[gaussian_update.__module__].__file__)]})

    def capture(context,state,event_id,k,**kwargs):
        assert event_id == task.task_id and k == 20 and AP in context['cases']
        case = context['cases'][AP]
        prior = InterceptBankPrior(case['histories'],state['weights'],model.base,state['sigma']['intercept_std'])
        mean,features = prior.features(case['sx'])
        covariance = prior.covariance(case['sx'],case['sx'],features,features)
        stats = case['stats']
        updated = gaussian_update(mean,covariance,stats,model.obs)
        result.update(adaptation_state=state,
            support_xy=values(case['sx']),support_counts={name:values(value) for name,value in stats.items()},
            actual_S_RSS_column=task.S['rss'][:,AP].tolist(),actual_S_XY=task.S['xy'].tolist(),
            prior_mean=values(mean),prior_covariance=values(covariance),
            RSS_updated_mean=values(updated['mean']),RSS_updated_covariance=values(updated['cov']),
            detection_summary=dict(positions=len(mean),total_scans=float(stats['n'].sum()),
                detected_scans=float(stats['k'].sum()),positions_with_RSS=int((stats['k']>0).sum()),
                mixed_positions=int(((stats['k']>0)&(stats['k']<stats['n'])).sum()),
                all_missing_positions=int((stats['k']==0).sum())),
            prior_RSS_shift_max_abs=float((updated['mean']-mean).abs().max()),
            prior_variance_range=[float(covariance.diag().min()),float(covariance.diag().max())],
            RSS_variance_range=[float(updated['cov'].diag().min()),float(updated['cov'].diag().max())],
            RSS_covariance_eigenvalues=values(torch.linalg.eigvalsh(updated['cov'])))
        run._write(DEST/'context_before_sampling.json',result)
        original_setup,original_sample = posterior.setup,posterior.sample_reference

        def traced_setup(*args,**kwargs):
            output = original_setup(*args,**kwargs)
            _,initial,center,root,modes = output
            result['Laplace_initialization'] = dict(center=center.tolist(),root=root.tolist(),modes=modes.tolist(),
                weights=initial.weights.tolist(),means=[c.mu.tolist() for c in initial.components],
                covariances=[c.sigma.tolist() for c in initial.components])
            return output

        def traced_sample(*args,**kwargs):
            output = original_sample(*args,**kwargs)
            result['PMC_info'] = output[2]
            run._write(DEST/'PMC_before_fallback.json',result)
            return output

        prepared = dict(sm=mean,sk=covariance,phi_s=mean.new_zeros((len(mean),10)),**stats)
        tick = time.monotonic()
        with patch.object(posterior,'setup',traced_setup),patch.object(posterior,'sample_reference',traced_sample):
            try:
                _,info = posterior.proposal(prepared,model.obs,mean.new_zeros(10),SEED)
                result.update(replay_status='accepted',sampler=info)
            except RuntimeError as error:
                result.update(replay_status='rejected',replay_error=str(error))
        result['same_AP_proposal_seconds'] = time.monotonic()-tick
        result['total_diagnostic_seconds'] = time.monotonic()-started
        run._write(DEST/'diagnostic.json',result)
        raise DiagnosticCaptured

    with patch.object(position_fusion_gp,'prepare_retained_fields',capture):
        try:
            model.predict_scores(task)
        except DiagnosticCaptured:
            pass
    assert (DEST/'diagnostic.json').is_file()
    assert all(field['cache']!='fitted' for field in model.diagnostics['history_fields'])
    run._write(DEST/'complete.json',dict(complete=True,post_failure_diagnostic=True,
        same_failed_task_only=True,AP_index=AP,Query_scored=False,Query_truth_saved=False,
        new_history_fields=0,training_performed=False,numerical_settings_changed=False,
        diagnostic_sha256=run._sha(DEST/'diagnostic.json'),seconds=time.monotonic()-started))
    print(json.dumps(dict(diagnostic_complete=True,replay_status=result['replay_status'],
        ESS=result['PMC_info']['effective_samples'],observations=model.obs,
        detection_summary=result['detection_summary'])),flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    with threadpool_limits(limits=2):
        main()
