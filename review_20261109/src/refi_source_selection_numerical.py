"""Evaluate frozen ReFi checkpoints with a common, checked numerical proposal.

No parameter updates: proposal mixtures are quadrature, not model mixtures.
All queries and the original H/S domain remain unchanged.
"""
import gc
import json
import resource
import socket
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import refi_source_selection_trial as trial
from recomposition import Parameters
from recomposition_coupled_trial import episode
from source_environment_validation import select_checkpoint
from paired_assets import read_allowed
from recomposition_trial import ROOT,CSV,digest,write,now,subset
from uji import tensor,group_scans
from recomposition_adaptation import adapted_prior
from operator_proposals import propose_states, summarize_states, compare_scrambles
from recomposition_multibudget_natural import wknn

PARENT=trial.OUT
PREVIOUS=ROOT/'results/refi_source_selection_learning_20261107_selection_completion'
OUT=ROOT/'results/refi_source_selection_learning_20261107_checked_selection'
CODES=('refi_source_selection_numerical.py','refi_source_selection_trial.py',
       'source_environment_validation.py','operator_proposals.py','recomposition_adaptation.py',
       'recomposition_multibudget_natural.py')


def attach_all_queries(data):
    ids=set(i for e in data['events'] for i in e['query_rows'])
    raw=read_allowed(CSV,ids);qs=[];coverage={}
    lo=data['grid'].min(0).values.numpy();hi=data['grid'].max(0).values.numpy()
    for e in data['events']:
        q=subset(raw[e['floor']],e['query_rows'],data['origin'])
        assert not (q['positions'][:,None]==data['support_positions'][None]).all(-1).any()
        outside=((q['positions']<lo)|(q['positions']>hi)).any(1)
        distance=np.linalg.norm(np.maximum(np.maximum(lo-q['xy'],q['xy']-hi),0),axis=1)
        weights=1/(len(q['positions'])*q['stats']['n'][q['group']])
        coverage[e['id']]=dict(outside_positions=int(outside.sum()),positions=len(q['positions']),
            rectangle_distance_lower_bound_MDE=float(weights@distance),
            meaning='Geometric floor for outputs inside the fixed rectangular grid hull, not a GP performance upper bound')
        qs.append(q)
    q=group_scans(np.concatenate([a['rss'] for a in qs]),np.concatenate([a['xy'] for a in qs]))
    weights=1/(len(q['positions'])*q['stats']['n'][q['group']])
    ep=episode(data['cases'],data['grid'],data['obs'],q['rss'],q['xy'],weights)
    return dict(data=data,ep=ep,q=q,cells=qs),coverage


@torch.no_grad()
def prepare_checkpoint_proposals(models,data,floor,remaining):
    # H/S and frozen checkpoints only. Never inspect Q or prediction scores.
    pending={};coverage=[];report=dict(floor=floor,coverage_before=coverage,checks=[],accepted=False)
    path=OUT/f'{floor}_numerical.json'
    for ap,case in data['cases'].items():
        priors=[adapted_prior(m,case['blocks'],data['state']) for m in models]
        if not len(priors[0]['sm']):continue
        summaries=summarize_states(priors,case['stats'],data['obs'],case['proposal'])
        esses=[s['ESS'] for s in summaries]
        if not np.isfinite(esses).all():raise RuntimeError(f'Nonfinite ESS {floor} AP {ap}')
        coverage.append(dict(AP=ap,ESS=esses))
        if min(esses)<1000:pending[ap]=priors
    report['failing_APs']=list(pending);write(path,report)
    if len(pending)>remaining:
        raise RuntimeError(f'Numerical repair budget exceeded: {floor} {list(pending)}, remaining {remaining}')
    replacements={}
    for ap,priors in pending.items():
        case=data['cases'][ap];summaries=[];first=None
        seed=117080+1000000*int(floor.split('F')[1])+100*ap
        for repeat in (0,1):
            nodes=propose_states(priors,case['stats'],data['obs'],power=16,seed=seed+repeat*100000)
            summaries.append(summarize_states(priors,case['stats'],data['obs'],nodes))
            if repeat==0:first=nodes
        checked=compare_scrambles(*summaries)
        report['checks'].append(dict(AP=ap,seed=seed,**checked));write(path,report)
        if not checked['accepted']:raise RuntimeError(f'Proposal precision rejected: {floor} AP {ap}')
        replacements[ap]=first
    # Install atomically for this context after all proposals pass.
    for ap,nodes in replacements.items():data['cases'][ap]=dict(data['cases'][ap],proposal=nodes)
    report.update(accepted=True,repaired_APs=list(replacements),samples_per_repair=3*2**16)
    write(path,report)
    return len(replacements)


def score_wknn(data,item):
    e=data['events'][0]
    allowed=set(e['support_rows']).union(*(set(r) for r in e['history_rows'].values()))
    raw=read_allowed(CSV,allowed)
    support=subset(raw[e['floor']],e['support_rows'],data['origin'])
    histories=[subset(raw[h],e['history_rows'][h],data['origin']) for h in e['histories']]
    columns=np.array(list(data['cases']))
    indices=torch.arange(len(support['positions']))
    return {event['id']:wknn(support,q,histories,columns,indices,data['grid'])
            for event,q in zip(data['events'],item['cells'])}


def run():
    start=time.monotonic();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    old=json.loads((PARENT/'freeze.json').read_text());fixed=json.loads((PARENT/'models_fixed.json').read_text())
    assert (PARENT/'exit_code.txt').read_text().strip()=='1'
    failure=json.loads((PARENT/'failure.json').read_text())
    assert 'attach_queries' in failure['traceback'] and 'AssertionError' in failure['traceback']
    for n,h in old['code'].items():assert digest(ROOT/n)==h and digest(PARENT/'code'/n)==h,n
    for p,h in old['inputs'].items():assert digest(p)==h,p
    for s,h in fixed['checkpoints'].items():assert digest(PARENT/f'step{s}.pt')==h
    rows=[json.loads(x) for x in (PARENT/'train.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,17))
    plan=json.loads((trial.PLAN/'tasks.json').read_text())
    assert [r['floor'] for r in rows]==[sorted({e['floor'] for e in plan['fit']})[i] for i in old['optimizer']['schedule']]
    assert (PREVIOUS/'exit_code.txt').read_text().strip()=='1'
    assert 'proposal ESS' in json.loads((PREVIOUS/'failure.json').read_text())['traceback']
    OUT.mkdir(exist_ok=False)
    inputs=[PARENT/n for n in ('freeze.json','models_fixed.json','failure.json','train.jsonl','histories.json','selection_B2F0_support.pt','step0.pt','step8.pt','step16.pt')]
    inputs += [PREVIOUS/n for n in ('freeze.json','failure.json','exit_code.txt')]
    write(OUT/'freeze.json',dict(created=now(),parent=str(PARENT),parent_exit=1,
        code={n:digest(ROOT/n) for n in CODES},
        inputs={str(p):digest(p) for p in inputs},
        correction='Common H/S-only proposal across fixed checkpoints; all Q and H/S grid retained',
        numerical=dict(max_total_AP_repairs=4,power=16,scrambles=2,seed=117080,minimum_ESS=1000,
            mean_gap=.1,relative_covariance_gap=.01,log_evidence_gap=.01,
            policy='Only APs failing original ESS; all three checkpoint priors equally represented; no threshold relaxation'),
        unchanged='All 16 training updates, checkpoints 0/8/16, task rows, H/S fit policy, readout, 3% criterion',
        limits='No B1, no retraining, no query-conditioned geometry, no AP/position filtering',
        resources=dict(cpu=2,gpu=0,seconds=430,previous_wall_seconds=466.12,total_limit_seconds=960)))
    for n in CODES:
        (OUT/n).write_bytes((ROOT/n).read_bytes())
    manifest=json.loads((PARENT/'histories.json').read_text())
    original_manifest=json.loads((trial.ASSETS/'histories.json').read_text())
    params=json.loads(Path(original_manifest['parameters_path']).read_text())
    bank=trial.AssetBank(manifest,params['base'],params['obs'])
    zero=Parameters(torch.zeros(9),torch.ones(9));model=Parameters(torch.zeros(9),torch.ones(9))
    model.load_state_dict(torch.load(PARENT/'step0.pt',weights_only=True))
    models=[]
    for step in trial.CHECKPOINTS:
        frozen=Parameters(torch.zeros(9),torch.ones(9))
        frozen.load_state_dict(torch.load(PARENT/f'step{step}.pt',weights_only=True))
        models.append(frozen)
    trial.OUT=OUT
    records=[dict(step=s,checkpoint=str(PARENT/f'step{s}.pt'),fitted_buildings=['B0'],event_mde={}) for s in trial.CHECKPOINTS]
    details={str(s):{} for s in trial.CHECKPOINTS};coverage={};cache_paths={};baselines={};repairs=0
    for floor in sorted({e['floor'] for e in plan['selection']}):
        events=[e for e in plan['selection'] if e['floor']==floor]
        if floor=='B2F0':
            path=PARENT/'selection_B2F0_support.pt';data=torch.load(path,weights_only=False)
            assert data['events']==events
        else:
            data=trial.prepare_pair(events,bank,zero,'selection');path=OUT/f'selection_{floor}_support.pt'
        original_path=path
        repairs+=prepare_checkpoint_proposals(models,data,floor,4-repairs)
        path=OUT/f'selection_{floor}_checked_support.pt';torch.save(data,path)
        cache_paths[floor]=dict(path=str(path),sha256=digest(path),original_path=str(original_path),original_sha256=digest(original_path))
        write(OUT/'support_caches.json',cache_paths)
        item,covered=attach_all_queries(data);coverage.update(covered)
        torch.save(dict(q=item['q'],cells=item['cells']),OUT/f'selection_{floor}_query.pt')
        for record in records:
            model.load_state_dict(torch.load(record['checkpoint'],weights_only=True))
            result=trial.score(model,item)
            for eid,m in result['cells'].items():record['event_mde'][eid]=m['mde']['median']
            details[str(record['step'])][floor]={k:v for k,v in result.items() if k!='log_probability'}
            np.savez_compressed(OUT/f'{floor}_step{record["step"]}_prediction.npz',
                log_probability=result['log_probability'].numpy(),grid=data['grid'].numpy())
        baselines.update(score_wknn(data,item))
        write(OUT/'selection_progress.json',dict(records=records,details=details,coverage=coverage,WKNN=baselines))
        del data,item,result;gc.collect()
        trial.progress('selection_floor_finished',floor=floor,seconds=time.monotonic()-start)
    selected=select_checkpoint(records,plan['selection'])
    weighted={str(r['step']):sum(e['selection_weight']*r['event_mde'][e['id']] for e in plan['selection']) for r in records}
    gain=1-selected['source_selection_mde']/weighted['0']
    baseline_mde=sum(e['selection_weight']*baselines[e['id']]['mde'] for e in plan['selection'])
    write(OUT/'selection.json',dict(selected=selected,records=records,details=details,coverage=coverage,
        aggregate_MDE=weighted,WKNN_MDE=baseline_mde,WKNN=baselines,relative_gain=gain,worth_continuing=selected['step']>0 and gain>=.03,B1_data_read=False))
    assert all(digest(PARENT/f'step{s}.pt')==h for s,h in fixed['checkpoints'].items())
    complete=dict(complete=True,seconds=time.monotonic()-start,training_updates=0,total_original_training_steps=16,
        selected_step=selected['step'],aggregate_source_selection_MDE=weighted,WKNN_MDE=baseline_mde,relative_gain=gain,
        repaired_APs=repairs,
        worth_continuing=selected['step']>0 and gain>=.03,new_history_fields=bank.new_fields,
        max_new_history_VI_gradient=bank.max_gradient,B1_data_read=False,
        peak_memory_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file()})
    write(OUT/'complete.json',complete);print(json.dumps(complete),flush=True)

if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    try:run()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),traceback=traceback.format_exc()))
        raise
