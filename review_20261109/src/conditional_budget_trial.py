"""One UJI-only 48-step support-budget coverage comparison, with a source gate."""
import json
import socket
import time
from pathlib import Path
import numpy as np
import torch
from recomposition_trial import ROOT,OLD,PLAN,CSV,digest,write,now,subset
from recomposition_multibudget_natural import OUT as NATURAL,load_parent
from conditional_budgets import conditioned_episode
from support_recomposition import load_parameters,prepare_context,context_prior,context_scores
from recomposition import evaluate
from episode_queries import set_queries
from episodic_tasks import task_inputs,prepare_adapted,attach_queries
from source_context_replay import Scorer
from uji import load_uji,load_history,tensor,ap_stats

OUT=ROOT/'results/conditional_budget_coverage_20261105'
SHAPE=ROOT/'results/conditional_shape_uji_k5_20261105'
FIXED=ROOT/'results/support_recomposition_uji_k5_refreshed_20261105'
ROTATED=ROOT/'results/query_rotation_uji_k5_20261105'
TASKS=ROOT/'results/episodic_task_integration_20261105'
REPLAY=ROOT/'results/source_context_replay_20261105'


def history_reader(meta,base,origin,used):
    entries={(r['floor'],r['ap']):r for r in meta['history_records']}
    def read(floor,ap):
        r=entries[(floor,ap)]
        p=OLD/r['reuse_path'].split('/Law-Probabilistic-Transfer-20261023/',1)[1] if 'reuse_path' in r else OLD/f'results/decision_local_pilot_20261030/histories/{floor}/ap{ap}.npz'
        value=digest(p);assert value==r['sha256'];used[str(p)]=value
        with np.load(p) as a:shift=tuple(a['origin']-origin)
        return load_history(str(p),shift,tuple(base.items()))
    return read


def run():
    OUT.mkdir(exist_ok=False);start=time.monotonic()
    ref=json.loads((ROTATED/'freeze.json').read_text())
    for name in ('support_recomposition.py','conditional_shape.py','recomposition.py','observations.py','posterior.py'):
        assert digest(ROOT/name)==ref['code'][name]
    assert digest(CSV)==ref['csv_sha256'] and digest(PLAN)==ref['plan_sha256']
    records=json.loads((NATURAL/'progress.json').read_text())['records']
    assert digest(ROOT/'recomposition_multibudget_natural.py')==json.loads((NATURAL/'freeze.json').read_text())['code_sha256']
    allowed={f'train_{i}' for i in range(6)}|{'validation_0','validation_1'}
    records=[r for r in records if r['id'] in allowed];assert {r['id'] for r in records}==allowed
    schedule=[];visits={f'train_{i}':0 for i in range(6)}
    batches=json.loads((ROTATED/'queries.json').read_text())['batches']
    for b in batches:
        ident=b['event'];round_=visits[ident];visits[ident]+=1
        schedule.append(dict(b,K=(1,3,5)[(round_+int(ident.split('_')[1]))%3]))
    assert [sum(b['K']==k for b in schedule) for k in (1,3,5)]==[16,16,16]
    inputs=[PLAN,CSV,NATURAL/'progress.json',SHAPE/'prepared.json',FIXED/'initial.pt',
            ROTATED/'final48.pt',ROTATED/'queries.json',ROTATED/'complete.json',TASKS/'tasks.json',REPLAY/'complete.json']
    inputs += [REPLAY/f'K{k}_{suffix}.json' for k in (1,3,5) for suffix in ('adaptation','metrics')]
    freeze=dict(created=now(),code={n:digest(ROOT/n) for n in ('conditional_budget_trial.py','conditional_budgets.py',
        'recomposition_multibudget_natural.py','support_recomposition.py','conditional_shape.py','recomposition.py',
        'episode_queries.py','episodic_tasks.py','source_context_replay.py','observations.py','posterior.py','uji.py')},
        inputs={str(p):digest(p) for p in inputs},schedule=schedule,steps=48,initial='Original 341-parameter initial.pt',
        change='S budget and its existing deployment-correct adaptation only; same query batches, task order, model, optimizer and steps as rotation control.',
        optimizer='Adam',learning_rate=.001,regularizer='Unchanged model.penalty',
        source_gate='All K means no worse than identity; mean relative improvement >=3%; mean MDE lower than rotating-Q model. Source labels already seen; only S combinations withheld.',
        development_gate='Only if source gate passes. All K means no worse than identity; mean relative gain >=3% vs identity and rotated control; >=4/6 draws better identity.',
        selection='Final48 only. No new seeds, tuning, model promotion, or additional training on failure.',
        scope='UJI only. 18 nested source configurations from six original maximal S sets, not arbitrary S resampling or new buildings.',
        numerical='Existing per-budget proposals; one S-only refresh/task/AP, <=64 total, same scorer policy; refreshed state survives episode reconstruction.',
        threads=2,wall_limit_seconds=900,downloads=0,new_large_disk_cache=0)
    write(OUT/'freeze.json',freeze)
    plan=json.loads(PLAN.read_text());events={e['id']:e for e in plan['events']};origin=np.array(plan['origin']);floors=load_uji(CSV)
    additions={r['id']:r for r in json.loads((SHAPE/'prepared.json').read_text())['records']}
    parents={};left={};used={}
    for r in records:
        ident=r['id'];a=additions[ident];parents[ident]=load_parent(r)
        assert parents[ident][1]['base_sha256']==a['parent_sha256']
        assert digest(SHAPE/a['path'])==a['sha256'];left[ident]=torch.load(SHAPE/a['path'],weights_only=True)
        # The budget cache must describe exactly the original H/S/Q rows.
        p=parents[ident][1]['provenance'];e=events[ident]
        assert p['support_rows']==e['support_rows'] and p['query_rows']==e['query_rows'] and p['history_rows']==e['history_rows']
        raw_s=subset(floors[e['floor']],e['support_rows'],origin)
        assert torch.equal(parents[ident][1]['support_positions'],tensor(raw_s['positions']))
        for ap,c in parents[ident][0]['cases'].items():
            assert all(torch.equal(value,c['stats'][key]) for key,value in ap_stats(raw_s,ap).items())
    initial=load_parameters(FIXED/'initial.pt');model=load_parameters(FIXED/'initial.pt')
    # Nonzero trained rule: sliced full covariance must equal direct H/current-S construction.
    checkmodel=load_parameters(ROTATED/'final48.pt');ident='train_1';event=events[ident]
    meta=json.loads((OLD/f'results/decision_local_pilot_20261030/{ident}/meta.json').read_text())
    reader=history_reader(meta,plan['base'],origin,used)
    support=subset(floors[event['floor']],event['support_rows'],origin)
    positions=[tensor(subset(floors[f],event['history_rows'][f],origin)['positions']) for f in event['histories']]
    delta=[int(event['floor'].split('F')[1])-int(f.split('F')[1]) for f in event['histories']]
    errors={}
    for k in (1,3):
        ep=conditioned_episode(parents[ident],left[ident],k);state=parents[ident][1]['states'][k];u=state['indices']
        candidates=[ap for ap,c in ep['cases'].items() if c['blocks']['active']][:2]
        for ap in candidates:
            hs=[reader(f,ap) for f in event['histories']]
            c=ep['cases'][ap];sx=tensor(support['positions'])[u]
            direct=prepare_context(hs,positions,delta,sx,ep['grid'],ep['base'],state['intercept'],c['stats'])
            with torch.no_grad():
                a=context_prior(checkmodel,c['blocks'],ep['weights']);b=context_prior(checkmodel,direct,ep['weights'])
                error=max(float((a[key]-b[key]).abs().max()) for key in a)
            assert error<1e-7,(k,ap,error);errors[f'K{k}/AP{ap}']=error
    del checkmodel,ep
    # Check source-query row/position exclusion for all 48 updates before optimizing.
    queries=[]
    for b in schedule:
        e=events[b['event']];q=subset(floors[e['floor']],b['query_rows'],origin)
        full_s=subset(floors[e['floor']],e['support_rows'],origin)
        assert not (q['positions'][:,None]==full_s['positions'][None]).all(-1).any()
        queries.append(q)
    write(OUT/'pretraining_audit.json',dict(created=now(),nonzero_prior_errors=errors,
        query_excluded_maximal_support=True,current_budget_proposals_preserved=True,training_updates=0))
    scorer=Scorer();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    with (OUT/'train.jsonl').open('w') as log:
        for b,q in zip(schedule,queries):
            ident,k=b['event'],b['K'];ep=conditioned_episode(parents[ident],left[ident],k);set_queries(ep,q)
            optimizer.zero_grad(set_to_none=True)
            try:metric=evaluate(model,ep,backward=True,score_fn=scorer)
            except Exception:
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_steps=b['step']-1),OUT/'failure_checkpoint.pt');raise
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            # Rebuilt episodes retain their own adapted numerical proposals.
            parents[ident][1]['states'][k]['proposals']={ap:c['proposal'] for ap,c in ep['cases'].items()}
            log.write(json.dumps(dict(step=b['step'],event=ident,K=k,**metric))+'\n');log.flush()
            if b['step']%12==0:print(json.dumps(dict(step=b['step'],seconds=time.monotonic()-start)),flush=True)
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=48),OUT/'final48.pt')
    write(OUT/'training_refreshes.json',scorer.records)
    # Independent S configurations; same seen source floor and previously exposed labels.
    family=json.loads((TASKS/'tasks.json').read_text())['family'];source=[]
    source_scorer=Scorer()
    for e in [e for e in family if len(e['histories'])==2]:
        assert e['floor']==event['floor'] and e['histories']==event['histories']
        k=e['K'];s=subset(floors[e['floor']],e['support_rows'],origin);q=subset(floors[e['floor']],e['query_rows'],origin)
        hs={f:subset(floors[f],e['history_rows'][f],origin) for f in e['histories']}
        task=task_inputs(hs,s,plan['base'],plan['obs'],reader)
        task['state']=json.loads((REPLAY/f'K{k}_adaptation.json').read_text())
        ep=prepare_adapted(initial,task,delta,e['id']);attach_queries(ep,q,s['positions'])
        with torch.no_grad():
            baseline=evaluate(initial,ep,score_fn=context_scores)
            reference=json.loads((REPLAY/f'K{k}_metrics.json').read_text())
            assert abs(baseline['mde']-reference['identity']['mde'])<1e-7
            result=evaluate(model,ep,score_fn=source_scorer)
        write(OUT/f'source_K{k}.json',result)
        source.append(dict(K=k,identity=baseline['mde'],rotated=reference['rotated']['mde'],candidate=result['mde']))
        del ep,task
    gain=lambda rows,key:100*np.mean([(v[key]-v['candidate'])/v[key] for v in rows])
    source_pass=all(v['candidate']<=v['identity'] for v in source) and gain(source,'identity')>=3 and np.mean([v['candidate']-v['rotated'] for v in source])<0
    write(OUT/'source_gate.json',dict(rows=source,mean_relative_gain_identity=gain(source,'identity'),passed=bool(source_pass)))
    write(OUT/'source_refreshes.json',source_scorer.records)
    development=[];development_pass=False
    if source_pass:
        control=load_parameters(ROTATED/'final48.pt');devscorers={n:Scorer() for n in ('identity','rotated','candidate')}
        for ident in ('validation_0','validation_1'):
            for k in (1,3,5):
                ep=conditioned_episode(parents[ident],left[ident],k);proposals={a:c['proposal'] for a,c in ep['cases'].items()};row=dict(event=ident,K=k)
                for n,m in (('identity',initial),('rotated',control),('candidate',model)):
                    for a,c in ep['cases'].items():c['proposal']=proposals[a]
                    with torch.no_grad():value=evaluate(m,ep,score_fn=devscorers[n])
                    row[n]=value['mde']
                development.append(row)
        byk=[{n:float(np.mean([v[n] for v in development if v['K']==k])) for n in ('identity','rotated','candidate')} for k in (1,3,5)]
        development_pass=all(v['candidate']<=v['identity'] for v in byk) and gain(byk,'identity')>=3 and gain(byk,'rotated')>=3 and sum(v['candidate']<v['identity'] for v in development)>=4
        write(OUT/'development_refreshes.json',{n:s.records for n,s in devscorers.items()})
    write(OUT/'history_inputs.json',used)
    write(OUT/'complete.json',dict(complete=True,created=now(),seconds=time.monotonic()-start,steps=48,
        source=source,source_gate_passed=bool(source_pass),development=development,
        development_gate_passed=bool(development_pass),target_evaluations=len(development),
        conclusion='Bounded task-coverage trial; no generalization claim. Source gate failure skips B1, not evidence of B1 failure.'))


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    try:run()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),error=repr(error)))
        raise
