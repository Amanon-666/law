"""Finish the stopped 24-step source trial under a recorded numerical revision.

The old failure remains authoritative. No extra optimization steps or target
development scores are permitted; numerical adaptation sees H/S only.
"""
import hashlib
import json
import socket
import time
import numpy as np
import torch
from fingerprint_change import FingerprintChange,prepare_fingerprint,evaluate_fingerprint,changed_prior
from fingerprint_posterior import support_summary,propose_qmc_support
from fingerprint_change_trial import OUT as STOPPED,report_parameters
from neighbor_learning_trial import PREP,load_original
from neighbor_gpu_check import source_task
from neighbor_stream import prepare_cpu_proposals
from recomposition_trial import ROOT,PLAN,CSV,digest,write,now,subset
from uji import load_uji

OUT=ROOT/'results/fingerprint_change_continued_20261105'
REPAIR=ROOT/'results/fingerprint_posterior_qmc_20261105'


def seed_for(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'little')


class NumericalPolicy:
    def __init__(self):
        self.refreshed={('train_4',75)}
        self.new_count=0
        self.records=[]

    @torch.no_grad()
    def ensure(self,model,episode,event):
        kernels=model.kernels(episode['geometry']);failed=[];esses=[]
        for ap,c in episode['cases'].items():
            p=changed_prior(c,episode['geometry'],kernels)
            ess=support_summary(p,c,episode['obs'])['ESS'];esses.append(ess)
            if ess<1000:failed.append(ap)
        self.records.append(dict(event=event,stage='H_S_check',minimum_ESS=min(esses),failed_APs=failed))
        write(OUT/'numerical.json',self.records)
        if self.new_count+len(failed)>4:raise RuntimeError('Four-new-AP numerical budget exceeded')
        for ap in failed:
            if (event,ap) in self.refreshed:raise RuntimeError(f'Repeated invalid proposal: {event}/{ap}')
            c=episode['cases'][ap];p=changed_prior(c,episode['geometry'],kernels)
            cp={k:p[k].cpu() for k in ('sm','sk')};stats={k:v.cpu() for k,v in c['stats'].items()}
            summaries=[];first=None
            for repeat in (1,2):
                seed=seed_for(f'fingerprint-continue/{event}/{ap}/{repeat}')
                proposal,_=propose_qmc_support(cp,stats,episode['obs'],seed)
                summaries.append(support_summary(cp,dict(stats=stats,proposal=proposal),episode['obs']))
                if repeat==1:first=proposal
            a,b=summaries
            mg=float((a['mean']-b['mean']).abs().max())
            cg=float(torch.linalg.matrix_norm(a['cov']-b['cov'])/torch.linalg.matrix_norm((a['cov']+b['cov'])/2))
            accepted=min(a['ESS'],b['ESS'])>=1000 and mg<=.1 and cg<=.01
            self.records.append(dict(event=event,AP=ap,ESS=[a['ESS'],b['ESS']],mean_gap=mg,
                                     relative_cov_gap=cg,accepted=accepted))
            write(OUT/'numerical.json',self.records)
            if not accepted:raise RuntimeError(f'Fixed QMC precision not established: {event}/{ap}')
            c['proposal']=tuple(v.to(p['sm'].device) for v in first)
            self.refreshed.add((event,ap));self.new_count+=1


def progress(stage,**kwargs):
    row=dict(created=now(),stage=stage,**kwargs)
    write(OUT/'progress.json',row);print(json.dumps(row),flush=True)


def run():
    OUT.mkdir(exist_ok=False);started=time.monotonic()
    oldfreeze=json.loads((STOPPED/'freeze.json').read_text())
    assert all(digest(ROOT/n)==h for n,h in oldfreeze['code'].items())
    repairfreeze=json.loads((REPAIR/'freeze.json').read_text())
    assert all(digest(ROOT/n)==h for n,h in repairfreeze['code'].items())
    assert json.loads((REPAIR/'complete.json').read_text())['passed']
    checkpoint=STOPPED/'latest.pt';checkpoint_hash=digest(checkpoint)
    assert checkpoint_hash==repairfreeze['checkpoint_sha256']
    saved=torch.load(checkpoint,weights_only=False,map_location='cpu');assert saved['step']==22
    cached=torch.load(REPAIR/'repaired_proposals.pt',weights_only=False,map_location='cpu')
    assert cached['checkpoint_sha256']==checkpoint_hash and cached['event']=='train_4'
    batches=oldfreeze['schedule'][22:];assert [b['step'] for b in batches]==[23,24]
    write(OUT/'freeze.json',dict(created=now(),code={n:digest(ROOT/n) for n in
        ('fingerprint_change_continue.py','fingerprint_change.py','fingerprint_posterior.py','fingerprint_change_trial.py')},
        inputs={str(p):digest(p) for p in (STOPPED/'freeze.json',checkpoint,REPAIR/'freeze.json',
                REPAIR/'complete.json',REPAIR/'repaired_proposals.pt',PLAN,CSV,STOPPED/'gate_before_training.json')},
        revision='After stopped trial: validated S-only numerical refresh; old failure not overwritten',
        schedule=batches,optimizer='Original Adam full state, lr=.03, original penalty',
        gradient_rule='Step23 full-loss gradient: two validated scrambles differ <=1% max norm + 1e-6',
        numerical='At most4 new task/AP QMC repairs; one two-scramble power18 per pair, ESS>=1000, mean<=.1, cov<=.01',
        gate='Final24 beats identity AND untrained by >=3% MDE on existing B0F1 newS K5',
        B1_B2_evaluations=0,threads=2,wall_seconds=300,downloads=0))
    for name in ('fingerprint_change_continue.py','fingerprint_change.py','fingerprint_posterior.py'):
        (OUT/name).write_bytes((ROOT/name).read_bytes())
    plan=json.loads(PLAN.read_text());floors=load_uji(CSV);origin=np.array(plan['origin'])
    events={e['id']:e for e in plan['events']}
    rows={r['id']:r for r in json.loads((PREP/'complete.json').read_text())['episodes']}
    model=FingerprintChange().cuda();model.load_state_dict(saved['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=.03);optimizer.load_state_dict(saved['optimizer'])
    assert torch.equal(model.theta.cpu().detach(),saved['model']['theta'])
    assert int(optimizer.state[model.theta]['step'])==22
    policy=NumericalPolicy();used={}
    for batch in batches:
        ident=batch['event']
        task,proposals,delta,unused=load_original(events[ident],rows[ident],plan,floors,used)
        del unused
        episode=prepare_fingerprint(task,proposals,'cuda')
        if ident==cached['event']:
            for ap,proposal in cached['proposals'].items():episode['cases'][ap]['proposal']=tuple(v.cuda() for v in proposal)
        policy.ensure(model,episode,ident)
        query=subset(floors[events[ident]['floor']],batch['query_rows'],origin)
        optimizer.zero_grad()
        metric=evaluate_fingerprint(model,episode,query,backward=True);model.penalty().backward()
        first_gradient=model.theta.grad.detach().clone()
        if batch['step']==23:
            frozen=model.theta.detach().clone();c=episode['cases'][75];first=c['proposal']
            with torch.no_grad():
                p=changed_prior(c,episode['geometry'],model.kernels(episode['geometry']))
                cp={k:p[k].cpu() for k in ('sm','sk')};stats={k:v.cpu() for k,v in c['stats'].items()}
                other,_=propose_qmc_support(cp,stats,episode['obs'],seed_for('fingerprint-posterior/train_4/75/2'))
            c['proposal']=tuple(v.cuda() for v in other)
            optimizer.zero_grad();second=evaluate_fingerprint(model,episode,query,backward=True);model.penalty().backward()
            second_gradient=model.theta.grad.detach().clone()
            difference=float((first_gradient-second_gradient).norm())
            scale=max(float(first_gradient.norm()),float(second_gradient.norm()))
            accepted=difference<=.01*scale+1e-6
            write(OUT/'gradient_precision.json',dict(first=first_gradient.tolist(),second=second_gradient.tolist(),
                norm_difference=difference,relative_difference=difference/scale,accepted=accepted,
                nll_gap=abs(metric['nll']-second['nll']),optimizer_step=int(optimizer.state[model.theta]['step'])))
            assert torch.equal(frozen,model.theta.detach())
            if not accepted:raise RuntimeError('Two-scramble full gradient precision failed')
            c['proposal']=first;model.theta.grad=first_gradient
        assert bool(torch.isfinite(model.theta.grad).all())
        optimizer.step()
        assert int(optimizer.state[model.theta]['step'])==batch['step']
        row=dict(step=batch['step'],event=ident,nll=metric['nll'],mde=metric['mde'],
                 minimum_ESS=metric['minimum_ESS'],gradient=first_gradient.tolist(),parameters=report_parameters(model))
        with (OUT/'train.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=batch['step']),OUT/'latest.pt')
        progress('continued',step=batch['step'],seconds=time.monotonic()-started)
        del episode,task,proposals
    write(OUT/'history_inputs.json',used)
    cpu,task,event,delta,query=source_task()
    proposals=prepare_cpu_proposals(cpu,task,delta,event['id'])
    episode=prepare_fingerprint(task,proposals,'cuda')
    untrained=FingerprintChange().cuda();metrics={}
    metrics['identity']=evaluate_fingerprint(untrained,episode,query,identity=True)
    policy.ensure(untrained,episode,event['id']+'/untrained')
    metrics['untrained']=evaluate_fingerprint(untrained,episode,query)
    old=json.loads((STOPPED/'gate_before_training.json').read_text())
    for key in metrics:
        gap=max(abs(a-b) for a,b in zip(metrics[key]['errors'],old[key]['errors']))
        assert gap<1e-6;metrics[key]['old_error_gap']=gap
    policy.ensure(model,episode,event['id']+'/trained')
    metrics['trained']=evaluate_fingerprint(model,episode,query)
    gains={k:1-metrics['trained']['mde']/metrics[k]['mde'] for k in ('identity','untrained')}
    passed=all(v>=.03 for v in gains.values())
    write(OUT/'source_gate.json',dict(metrics=metrics,gains=gains,passed=passed,parameters=report_parameters(model)))
    write(OUT/'complete.json',dict(complete=True,created=now(),seconds=time.monotonic()-started,
        status='24_steps_completed_with_recorded_numerical_revision',steps_added=2,total_steps=24,
        mde={k:v['mde'] for k,v in metrics.items()},gains=gains,passed=passed,parameters=report_parameters(model),
        new_numerical_repairs=policy.new_count,reused_validated_repairs=1,B1_B2_evaluations=0,
        scope='Known source support configuration only; not independent cross-environment confirmation'))
    print((OUT/'complete.json').read_text(),flush=True)


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    try:run()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),error=repr(error)))
        raise
