"""Reuse the completed B0 same-objective solver continuation, without shared edits.

The original UTS solver stopped at F-2/K3 with all16 starts saturated in
softmax coordinates. Reuse the earlier successful B0 continuation verbatim:
same objective, histories, parameters, stopping criteria and output rule.
It leaves successful original paths unchanged and only extends the specified
all-starts saturation failure. A prospective execution amendment retains the
original freeze/failure and binds the four already completed tasks.
"""
import argparse
import importlib.util
import socket
from unittest.mock import patch

import torch
from threadpoolctl import threadpool_limits
import adaptation
from candidates import uts_official_gp_20261111 as G

SOLVER = G.ROOT/'results/law_bench_B0_softmax_continuation_20261111/source_snapshot/adaptation.py'
SOLVER_SHA = 'ea59beed0a92d6a5ecab00ce5729b8e59a9e4fff1d837cab8ef5cca214c1ae87'
EVIDENCE = SOLVER.parent.parent/'complete.json'


def run(seconds,max_events):
    G.check()
    assert G.digest(SOLVER)==SOLVER_SHA
    evidence=G.read(EVIDENCE)
    assert evidence['complete'] and evidence['final']['stationary']
    assert all(v==0 for v in evidence['original_four_trial_max_abs_differences'].values())
    path=G.OUTPUT/'solver_continuation_execution.json'
    binding=dict(runner_sha256=G.digest(__file__),solver_path=str(SOLVER),solver_sha256=SOLVER_SHA,
        prior_evidence_path=str(EVIDENCE),prior_evidence_sha256=G.digest(EVIDENCE),
        original_freeze_sha256=G.digest(G.OUTPUT/'freeze.json'),
        failure_log=str(G.ROOT/'results/uts_official_A02_20261111_source/gp_predict2.log'),
        rule='Verbatim existing mass-transfer continuation only after all original starts fail by saturation',
        objective_changed=False,history_changed=False,parameters_changed=False,query_XY_opened=False)
    binding['failure_log_sha256']=G.digest(binding['failure_log'])
    if path.exists():
        assert G.read(path)['binding']==binding
    else:
        reused=[]
        for p in sorted(G.OUTPUT.glob('F*/complete.json')):
            record=G.read(p)
            diagnostics=G.read(p.parent/'diagnostics.json')
            assert any(t['stationary'] for t in diagnostics['state']['trials'])
            reused.append(dict(event=record['event']['event'],path=str(p),sha256=G.digest(p)))
        assert {r['event'] for r in reused}=={'F-3_K1','F-3_K3','F-3_K5','F-2_K1'}
        G.write(path,dict(created=G.now(),binding=binding,reused_successful_tasks=reused))
    spec=importlib.util.spec_from_file_location('uts_existing_solver_continuation',SOLVER)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with patch.object(adaptation,'fit_source_weights',module.fit_source_weights):
        code=G.predict(seconds,max_events)
    if code==0:
        complete=G.read(G.OUTPUT/'predictions_complete.json')
        complete['solver_execution']=dict(path=str(path),sha256=G.digest(path),
            policy='Original valid tasks retained; later tasks use the bound saturation continuation')
        G.write(G.OUTPUT/'predictions_complete.json',complete)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=540)
    parser.add_argument('--max-events',type=int,default=1)
    args=parser.parse_args()
    assert socket.gethostname()=='ray-web' and 0<args.seconds<=540 and args.max_events>0
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    with threadpool_limits(limits=2):
        raise SystemExit(run(args.seconds,args.max_events))
