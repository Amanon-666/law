"""One bounded source trial of history-conditioned GP recomposition."""
import gc
import json
import socket
import time
import numpy as np
import torch
from neural_history_operator import (NeuralHistoryOperator,history_context,prepare_operator,
                                    operator_prior,evaluate_operator)
from conditional_history_field import prepare_context
from conditional_budget_trial import history_reader
from episodic_tasks import task_inputs
from fingerprint_posterior import propose_support,propose_qmc_support,support_summary
from fingerprint_change_continue import seed_for
from neighbor_gpu_check import source_task
from neighbor_stream import prepare_cpu_proposals
from location_decision import geometric_median
from recomposition_trial import ROOT,OLD,PLAN,CSV,digest,write,now,subset
from uji import load_uji

OUT=ROOT/'results/neural_history_learning_20261105'
STRUCTURE=ROOT/'results/neural_history_operator_structure_20261105'
TASKS=ROOT/'results/fingerprint_task_variation_cpu_20261105'
DECISION=ROOT/'results/conditional_field_decision_20261105'
CODE=('neural_history_trial.py','neural_history_operator.py','conditional_history_field.py',
      'conditional_field_objective.py','fields.py','recomposition.py','posterior.py','observations.py',
      'fingerprint_posterior.py','episodic_tasks.py','conditional_budget_trial.py',
      'location_decision.py','neighbor_gpu_check.py','neighbor_stream.py','uji.py')


def progress(stage,**kw):
    row=dict(created=now(),stage=stage,**kw)
    write(OUT/'progress.json',row);print(json.dumps(row),flush=True)


def support_prior(model,ep,ap):
    case=ep['cases'][ap];p=case['prior'];h=case['history']
    if h is None:return dict(sm=p['sm'],sk=p['sk'])
    b=model.coefficients(ep['context'],ap,ep['support'])
    return dict(sm=p['sm']+b@h['zm'],
                sk=p['sk']+b@h['zs']+h['zs'].T@b.T+b@h['zz']@b.T)


class NumericalPolicy:
    def __init__(self):self.repairs=set();self.records=[]

    @torch.no_grad()
    def ensure(self,model,ep,event):
        failed=[];esses=[]
        for ap,c in ep['cases'].items():
            p=support_prior(model,ep,ap)
            ess=support_summary(p,c,ep['obs'])['ESS'];esses.append(ess)
            if ess<1000:failed.append(ap)
        self.records.append(dict(event=event,minimum_ESS=min(esses),failed_APs=failed))
        write(OUT/'numerical.json',self.records)
        if len(self.repairs)+len(failed)>4:raise RuntimeError('Four task/AP repair budget exceeded')
        for ap in failed:
            if (event,ap) in self.repairs:raise RuntimeError('Repeated proposal failure')
            c=ep['cases'][ap];p=support_prior(model,ep,ap);summaries=[]
            for repeat in (1,2):
                proposal,_=propose_qmc_support(p,c['stats'],ep['obs'],seed_for(f'neural-history/{event}/{ap}/{repeat}'))
                summaries.append(support_summary(p,dict(stats=c['stats'],proposal=proposal),ep['obs']))
                if repeat==1:first=proposal
            a,b=summaries
            mg=float((a['mean']-b['mean']).abs().max())
            cg=float(torch.linalg.matrix_norm(a['cov']-b['cov'])/torch.linalg.matrix_norm((a['cov']+b['cov'])/2))
            accepted=min(a['ESS'],b['ESS'])>=1000 and mg<=.1 and cg<=.01
            self.records.append(dict(event=event,AP=ap,ESS=[a['ESS'],b['ESS']],mean_gap=mg,relative_cov_gap=cg,accepted=accepted))
            write(OUT/'numerical.json',self.records)
            if not accepted:raise RuntimeError('Two-scramble posterior precision failed')
            c['proposal']=first;self.repairs.add((event,ap))


def metric(logp,grid,query):
    p=logp.numpy();x=grid.numpy();mass=np.exp(p)
    mean=mass@x
    pairs=[geometric_median(x,w) for w in mass]
    median=np.asarray([a for a,_ in pairs])
    weights=1/(len(query['positions'])*query['stats']['n'][query['group']])
    errors={k:np.linalg.norm(v-query['xy'],axis=1) for k,v in [('mean',mean),('median',median)]}
    return dict(mde={k:float(weights@v) for k,v in errors.items()},
                errors={k:v.tolist() for k,v in errors.items()},maximum_decision_gap=max(v for _,v in pairs))


def run():
    started=time.monotonic();OUT.mkdir(exist_ok=False)
    sf=json.loads((STRUCTURE/'freeze.json').read_text())
    assert json.loads((STRUCTURE/'complete.json').read_text())['complete']
    assert all(digest(ROOT/n)==h for n,h in sf['code'].items())
    assert all(digest(p)==h for p,h in sf['inputs'].items())
    tf=json.loads((TASKS/'freeze.json').read_text());events=tf['varied_schedule']
    assert len(events)==12 and all(e['role']=='train' and e['floor'].startswith('B0') for e in events)
    assert all(digest(p)==h for p,h in tf['inputs'].items())
    old_history=json.loads((TASKS/'history_inputs.json').read_text())
    assert all(digest(p)==h for p,h in old_history.items())
    paths=[PLAN,CSV,STRUCTURE/'freeze.json',TASKS/'freeze.json',TASKS/'history_inputs.json',
           DECISION/'complete.json',DECISION/'fixed_scores.npz']
    paths += [TASKS/(e['id']+'_adaptation.json') for e in events]
    write(OUT/'freeze.json',dict(created=now(),code={n:digest(ROOT/n) for n in CODE},
        inputs={str(p):digest(p) for p in paths},events=events,steps_per_event=2,updates=24,seed=1150527,
        optimizer='Adam lr=.001 weight_decay=.00001, original model, final24 only',
        objective='Position-balanced complete-scan location NLL through transformed historical GP and censored S update',
        task_adaptation='Reuse exactly matching frozen H/S weights/intercept; original-density proposals regenerated',
        numerical='ESS>=1000; <=4 task/AP one-time double power18 QMC repairs; mean gap<=.1 dB, relative covariance gap<=.01',
        readout='Certified geometric median for both GP and learned model; mean recorded, never selected instead',
        gate='Final median MDE improves >=3% vs same-readout GP and historical+S 3-WKNN',
        scope='Known B0F1 newS K5; source labels and nuisance statistics already exposed; not independent confirmation',
        wall_seconds=1200,threads=2,GPU=0,downloads=0,B1_B2_evaluations=0))
    for n in CODE:(OUT/n).write_bytes((ROOT/n).read_bytes())
    torch.manual_seed(1150527);model=NeuralHistoryOperator()
    initial={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.00001)
    torch.save(dict(model=initial,step=0),OUT/'initial.pt')
    policy=NumericalPolicy();used={};prepared=[];step=0
    plan=json.loads(PLAN.read_text());origin=np.array(plan['origin']);floors=load_uji(CSV)
    for e in events:
        begin=time.monotonic()
        meta=json.loads((OLD/f"results/decision_local_pilot_20261030/{e['parent_event']}/meta.json").read_text())
        read=history_reader(meta,plan['base'],origin,used)
        histories={f:subset(floors[f],e['history_rows'][f],origin) for f in e['histories']}
        support=subset(floors[e['floor']],e['support_rows'],origin)
        task=task_inputs(histories,support,plan['base'],plan['obs'],read)
        task['state']=json.loads((TASKS/(e['id']+'_adaptation.json')).read_text())
        candidate=prepare_context(histories,support,plan['base'],plan['obs'])
        assert list(task['cases'])==candidate['APs']
        task['grid']=candidate['grid'];del candidate
        context=history_context(histories,task['state']['weights'],plan['base'],plan['obs'])
        ep=prepare_operator(task,{ap:None for ap in task['cases']},context)
        records={}
        for ap,c in ep['cases'].items():
            c['proposal'],records[ap]=propose_support(c['prior'],c['stats'],ep['obs'],seed_for(f'episodic-refi/{e["id"]}/{ap}'))
        write(OUT/(e['id']+'_proposals.json'),records)
        # Source locations enter only now, after all H/S preparations are fixed.
        query=subset(floors[e['floor']],e['query_rows'],origin)
        assert not (query['positions'][:,None]==support['positions'][None]).all(-1).any()
        ap=next(a for a,c in ep['cases'].items() if c['history'] is not None)
        with torch.no_grad():
            a=support_prior(model,ep,ap);b=operator_prior(model,ep,ap)
            assert max(float((a[k]-b[k]).abs().max()) for k in ('sm','sk'))<1e-10
        for _ in range(2):
            policy.ensure(model,ep,e['id']);optimizer.zero_grad()
            result=evaluate_operator(model,ep,query,backward=True)
            assert np.isfinite(result['nll'])
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            encoder_gradient=float(model.encode[0].weight.grad.norm())
            optimizer.step();step+=1
            row=dict(step=step,event=e['id'],nll=result['nll'],minimum_ESS=result['minimum_ESS'],
                     encoder_gradient_norm=encoder_gradient,seconds=time.monotonic()-started)
            with (OUT/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,event=e['id']),OUT/'latest.pt')
        prepared.append(dict(event=e['id'],seconds=time.monotonic()-begin,APs=len(ep['cases']),grid=len(ep['grid'])))
        write(OUT/'preparation.json',prepared);write(OUT/'history_inputs.json',used)
        progress('learning',step=step,seconds=time.monotonic()-started)
        del ep,task,context,result;gc.collect()
    assert step==24 and all(old_history[p]==h for p,h in used.items())
    torch.save(dict(model=model.state_dict(),step=step),OUT/'final.pt')
    cpu,task,event,delta,query=source_task()
    histories={f:subset(floors[f],event['history_rows'][f],origin) for f in event['histories']}
    support=subset(floors[event['floor']],event['support_rows'],origin)
    task['grid']=prepare_context(histories,support,plan['base'],plan['obs'])['grid']
    context=history_context(histories,task['state']['weights'],plan['base'],plan['obs'])
    ep=prepare_operator(task,prepare_cpu_proposals(cpu,task,delta,event['id']),context)
    baseline=NeuralHistoryOperator();baseline.load_state_dict(initial)
    replay=evaluate_operator(baseline,ep,query)
    with np.load(DECISION/'fixed_scores.npz') as old:
        assert np.array_equal(old['grid'],ep['grid'].numpy())
        assert float((replay['log_probability']-torch.tensor(old['gp_logp'])).abs().max())<1e-8
    metrics=dict(gp=metric(replay['log_probability'],ep['grid'],query))
    policy.ensure(model,ep,event['id']);model.eval()
    final=evaluate_operator(model,ep,query)
    metrics['learned']=metric(final['log_probability'],ep['grid'],query)
    previous=json.loads((DECISION/'complete.json').read_text())
    assert abs(metrics['gp']['mde']['median']-previous['mde']['gp']['median'])<1e-8
    wknn=previous['wknn_mde'];value=metrics['learned']['mde']['median']
    gains=dict(gp=1-value/metrics['gp']['mde']['median'],wknn=1-value/wknn)
    result=dict(complete=True,created=now(),seconds=time.monotonic()-started,updates=step,
        mde={k:v['mde'] for k,v in metrics.items()},wknn_mde=wknn,gains=gains,
        passed=all(g>=.03 for g in gains.values()),numerical_repairs=len(policy.repairs),
        B1_B2_evaluations=0,scope='Known source configuration only, no independent generalization evidence')
    write(OUT/'source_gate.json',dict(metrics=metrics,gp_nll=replay['nll'],learned_nll=final['nll'],**result))
    write(OUT/'complete.json',result)
    progress('complete',seconds=result['seconds'],updates=step,passed=result['passed'])


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    try:run()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),error=repr(error)))
        raise
