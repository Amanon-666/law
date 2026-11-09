"""Frozen model replay on new source support configurations, not a new test set."""
import hashlib
import json
import socket
import time
import numpy as np
import torch
from recomposition_trial import ROOT, OLD, PLAN, CSV, digest, write, now, subset
from uji import load_uji, load_history
from episodic_tasks import task_inputs, prepare_adapted, attach_queries
from predict import adapt
from support_recomposition import load_parameters, context_scores, refresh_context_case
from recomposition import evaluate

PREP=ROOT/'results/episodic_task_integration_20261105'
FIXED=ROOT/'results/support_recomposition_uji_k5_refreshed_20261105'
ROTATED=ROOT/'results/query_rotation_uji_k5_20261105'
OUT=ROOT/'results/source_context_replay_20261105'
CHECKPOINTS=dict(identity=FIXED/'initial.pt',fixed=FIXED/'conditioned_final48.pt',rotated=ROTATED/'final48.pt')


class Scorer:
    def __init__(self):self.seen=set();self.records=[]

    def __call__(self,model,case,w,obs):
        try:return context_scores(model,case,w,obs)
        except RuntimeError as error:
            if not str(error).startswith('proposal ESS '):raise
            key=case['numerical_key']
            if key in self.seen or len(self.seen)>=64:raise
            self.seen.add(key)
            seed=int.from_bytes(hashlib.sha256(f'support-recomposition/proposal/{key}'.encode()).digest()[:4],'little')
            fresh,info=refresh_context_case(model,case,w,obs,seed)
            case['proposal']=fresh['proposal']
            result=context_scores(model,case,w,obs)
            self.records.append(dict(key=key,seed=seed,original_error=str(error),ESS=result[1],proposal=info))
            return result


def run():
    OUT.mkdir(exist_ok=False);started=time.monotonic()
    plan=json.loads(PLAN.read_text());oldfreeze=json.loads((PREP/'freeze.json').read_text())
    assert digest(PLAN)==oldfreeze['plan_sha256'] and digest(CSV)==oldfreeze['csv_sha256']
    for folder in (FIXED,ROTATED):
        reference=json.loads((folder/'freeze.json').read_text())
        for name in ('support_recomposition.py','conditional_shape.py','recomposition.py','observations.py','posterior.py'):
            assert digest(ROOT/name)==reference['code'][name]
    original=PREP/'frozen_episodic_tasks.py'
    assert digest(original)==oldfreeze['code']['episodic_tasks.py']
    tasks=json.loads((PREP/'tasks.json').read_text())['family']
    events=[e for e in tasks if len(e['histories'])==2]
    assert [e['K'] for e in events]==[1,3,5]
    assert all(e['role']=='train' and e['floor']=='B0F1' and e['histories']==['B0F0','B0F2'] for e in events)
    k3state=PREP/(events[1]['id']+'_adaptation.json')
    meta_path=OLD/'results/decision_local_pilot_20261030/train_1/meta.json'
    assert digest(meta_path)==oldfreeze['history_metadata_sha256']
    freeze=dict(created=now(),code={n:digest(ROOT/n) for n in ('source_context_replay.py','episodic_tasks.py',
        'support_recomposition.py','conditional_shape.py','recomposition.py','predict.py','adaptation.py',
        'posterior.py','observations.py','uji.py','episode_queries.py')},
        checkpoints={n:digest(p) for n,p in CHECKPOINTS.items()},
        inputs={str(p):digest(p) for p in (PLAN,CSV,PREP/'freeze.json',PREP/'complete.json',PREP/'tasks.json',
            PREP/'history_inputs.json',k3state,FIXED/'complete.json',ROTATED/'complete.json',ROTATED/'queries.json')},
        events=[e['id'] for e in events],training_updates=0,target_evaluations=0,
        interpretation='One already-seen source floor, one new nested S family. K5 keeps training H/K; K1/K3 also change budget. No independent label/region generalization.',
        decision='Descriptive per-K replay only. No checkpoint selection, model promotion or automatic training follows.',
        numerical='Identical initial S-only proposals per K; each model restores them. One S-only refresh per task/AP, <=64 per model, ESS>=1000; repeated failure stops.',
        threads=2,wall_limit_seconds=600,downloads=0)
    write(OUT/'freeze.json',freeze)
    floors=load_uji(CSV);origin=np.array(plan['origin']);raw=floors['B0F1']
    byid={e['id']:e for e in plan['events']}
    oldq={r for e in plan['events'] if e['role']=='train' and e['floor']=='B0F1' for r in e['query_rows']}
    rotatedq={r for b in json.loads((ROTATED/'queries.json').read_text())['batches']
              if byid[b['event']]['floor']=='B0F1' for r in b['query_rows']}
    historical={r for e in plan['events'] if e['role']=='train' for r in e['history_rows'].get('B0F1',[])}
    query=subset(raw,events[0]['query_rows'],origin)
    def overlap(rows):
        xy=subset(raw,sorted(rows),origin)['positions']
        return int((query['positions'][:,None]==xy[None]).all(-1).any(1).sum())
    write(OUT/'label_exposure.json',dict(query_positions=len(query['positions']),
        fixed_training_query_position_overlap=overlap(oldq),rotating_training_query_position_overlap=overlap(rotatedq),
        all_query_rows_seen_as_training_history=set(events[0]['query_rows'])<=historical,
        new_support_combination=all(e['support_rows']!=t['support_rows'] for e in events for t in plan['events'] if t['role']=='train'),
        interpretation='New adaptation configuration, NOT held-out source labels or an independent test environment.'))
    meta=json.loads(meta_path.read_text());entries={(r['floor'],r['ap']):r for r in meta['history_records']}
    oldinputs=json.loads((PREP/'history_inputs.json').read_text());used={}
    def history(floor,ap):
        record=entries[(floor,ap)]
        p=OLD/record['reuse_path'].split('/Law-Probabilistic-Transfer-20261023/',1)[1] if 'reuse_path' in record else OLD/f'results/decision_local_pilot_20261030/histories/{floor}/ap{ap}.npz'
        h=digest(p);assert h==record['sha256']==oldinputs[str(p)];used[str(p)]=h
        with np.load(p) as a:shift=tuple(a['origin']-origin)
        return load_history(str(p),shift,tuple(plan['base'].items()))
    models={n:load_parameters(p).requires_grad_(False) for n,p in CHECKPOINTS.items()}
    scorers={n:Scorer() for n in models};records=[]
    for event in events:
        t=time.monotonic();print(json.dumps(dict(stage='adapt',K=event['K'])),flush=True)
        support=subset(raw,event['support_rows'],origin)
        histories={f:subset(floors[f],event['history_rows'][f],origin) for f in event['histories']}
        task=task_inputs(histories,support,plan['base'],plan['obs'],history)
        if event['K']==3:
            task['state']=json.loads(k3state.read_text())
            old=json.loads((PREP/'complete.json').read_text())['executed_tasks'][1]
            assert len(task['cases'])==old['APs'] and len(task['grid'])==old['grid_points']
        else:task['state']=adapt(task['cases'],plan['base'],plan['obs'])
        write(OUT/f"K{event['K']}_adaptation.json",task['state'])
        delta=[int(event['floor'].split('F')[1])-int(f.split('F')[1]) for f in event['histories']]
        ep=prepare_adapted(models['identity'],task,delta,event['id'])
        attach_queries(ep,query,support['positions'])
        initial={ap:c['proposal'] for ap,c in ep['cases'].items()};metrics={}
        for name,model in models.items():
            for ap,c in ep['cases'].items():c['proposal']=initial[ap]
            with torch.no_grad():metrics[name]=evaluate(model,ep,score_fn=scorers[name])
            write(OUT/(name+'_refreshes.json'),scorers[name].records)
        write(OUT/f"K{event['K']}_metrics.json",metrics)
        row=dict(K=event['K'],APs=len(ep['cases']),grid_points=len(ep['grid']),
            mde={n:m['mde'] for n,m in metrics.items()},nll={n:m['nll'] for n,m in metrics.items()},
            improvement_pct={n:100*(metrics['identity']['mde']-metrics[n]['mde'])/metrics['identity']['mde'] for n in ('fixed','rotated')},
            reused_adaptation=event['K']==3,seconds=time.monotonic()-t)
        records.append(row);write(OUT/'records.json',records);print(json.dumps(row),flush=True)
        del ep,task
    write(OUT/'history_inputs.json',used)
    write(OUT/'complete.json',dict(complete=True,created=now(),seconds=time.monotonic()-started,records=records,
        training_updates=0,target_evaluations=0,history_refits=0,large_cache_files=0,
        K_curve_nonincreasing={n:all(records[i+1]['mde'][n]<=records[i]['mde'][n] for i in range(2)) for n in models},
        limitations=freeze['interpretation']))


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    try:run()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),error=repr(error)))
        raise
