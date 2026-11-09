"""Same six source updates as the full-model trial, learning only final matching.

Uses the exact existing B0/B1 development tasks, full history and prior baseline
predictions. No target update, intermediate selection or budget extension.
"""
import json
import socket
import time

import numpy as np
import torch

from acquisition_tasks import load_acquisition_rows, episode_scans
from check_tabicl_history_learning import bounds_and_aps
from tabicl_history import CHECKPOINT, PACKAGE, state_hash
from tabicl_context_matching import ContextMatchingRegressor
from check_tabicl_context_matching import frozen_hash
from recomposition_trial import ROOT, PLAN, CSV, digest, write, now

OUT=ROOT/'results/tabicl_context_matching_trial_20261105'
SOURCE=ROOT/'results/acquisition_retention_trial_20261105'
TARGET=ROOT/'results/acquisition_retention_transfer_20261105'
FULL_SOURCE=ROOT/'results/tabicl_history_source_trial_20261105'
FULL_TARGET=ROOT/'results/tabicl_history_transfer_20261105'
SEED=1150541
STEPS=6


def prepare(directory, source):
    events=json.loads((directory/'freeze.json').read_text())['events']
    reference=json.loads((directory/'evaluation.json').read_text())
    assert len(events)==2
    assert all(e['floor']==('B0F1' if source else 'B1F1') for e in events)
    assert all(e['role']==('train' if source else 'validation') for e in events)
    rows=load_acquisition_rows(CSV,{'B0F0','B0F1','B0F2'} if source else {'B1F0','B1F1','B1F3'})
    origin=np.asarray(json.loads(PLAN.read_text())['origin'])
    packed=[];fit_positions=set();evaluation_positions=set();support_positions=set()
    for e,ref in zip(events,reference):
        qrows=e['evaluation_rows'] if source else e['query_rows']
        assert ref['event']==e['id'] and ref['query_rows']==qrows
        p=episode_scans(rows,dict(e,query_rows=qrows),origin)
        assert np.array_equal(p['query']['xy'],np.asarray(ref['truth']))
        cache=torch.load(directory/f'{e["id"]}_episode.pt',mmap=True,weights_only=False)
        grid=cache['grid'].numpy()
        bounds=np.array([grid.min(0),grid.max(0)])
        _,aps=bounds_and_aps(p['histories'],p['support'])
        assert aps.tolist()==sorted(cache['cases'])
        item=dict(p,event=e,query_rows=qrows,bounds=bounds,aps=aps)
        if source:
            item['fit']=episode_scans(rows,dict(e,query_rows=e['fit_rows']),origin)['query']
            fit_positions.update(map(tuple,item['fit']['xy']))
        evaluation_positions.update(map(tuple,p['query']['xy']))
        support_positions.update(map(tuple,p['support']['xy']))
        packed.append(item)
        del cache
    assert not fit_positions.intersection(evaluation_positions)
    assert not support_positions.intersection(fit_positions|evaluation_positions)
    return packed,reference


def fit_context(model,p):
    model.fit(p['histories'],p['support'],p['event']['floor'],p['aps'],p['bounds'],SEED)
    count=sum(len(v['rss']) for v in p['histories'].values())+len(p['support']['rss'])
    assert all(m.n_samples_in_==count and m.model_ is model.loaded.model_ for m in model.estimators)


@torch.no_grad()
def evaluate(model,packed,scope,old_directory):
    records=[]
    for p in packed:
        fit_context(model,p)
        q=p['query'];eid=p['event']['id']
        previous=np.load(old_directory/f'frozen_{eid}.npz')
        weights=1/(len(q['positions'])*q['stats']['n'][q['group']])
        assert np.array_equal(previous['query_rows'],p['query_rows'])
        assert np.array_equal(previous['truth'],q['xy']) and np.array_equal(previous['weights'],weights)
        predictions=model.predict(q['rss'])
        errors={k:np.linalg.norm(v-q['xy'],axis=1) for k,v in predictions.items()}
        record=dict(event=eid,mde={k:float(weights@v) for k,v in errors.items()},
            frozen=float(weights@np.linalg.norm(previous['projected']-q['xy'],axis=1)),
            context_rows=model.estimators[0].n_samples_in_,APs=len(p['aps']))
        np.savez(OUT/f'{scope}_{eid}.npz',**predictions,truth=q['xy'],weights=weights,
            query_rows=p['query_rows'],**{k+'_errors':v for k,v in errors.items()})
        records.append(record)
        write(OUT/f'{scope}.json',records)
        print(json.dumps(dict(stage='evaluated',scope=scope,**record)),flush=True)
    return records


def summary(records,reference,retention):
    means=dict(matching=float(np.mean([r['mde']['projected'] for r in records])),
        frozen=float(np.mean([r['frozen'] for r in records])),
        GP=float(np.mean([r['GP']['mde']['median'] for r in reference])),
        retained=float(np.mean([r[retention]['mde']['median'] for r in reference])),
        WKNN=float(np.mean([r['WKNN']['mde'] for r in reference])))
    gains={k:1-means['matching']/v for k,v in means.items() if k!='matching'}
    learned=all(r['mde']['projected']<=r['frozen'] for r in records) and gains['frozen']>=.03
    return dict(means=means,gains=gains,learning_criterion=learned,
        replacement_criterion=learned and all(g>=.03 for g in gains.values()))


def run():
    started=time.monotonic();OUT.mkdir(exist_ok=False)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    source_freeze=json.loads((FULL_SOURCE/'freeze.json').read_text())
    source_complete=json.loads((FULL_SOURCE/'complete.json').read_text())
    assert source_complete['complete'] and source_complete['steps']==STEPS and source_freeze['seed']==SEED
    assert json.loads((FULL_TARGET/'complete.json').read_text())['complete']
    for name in ['tabicl_history.py','tabicl_history_learning.py','acquisition_tasks.py','uji.py']:
        assert digest(ROOT/name)==source_freeze['code'][name]
    assert digest(CSV)==source_freeze['inputs'][str(CSV)]
    assert digest(CHECKPOINT)==source_freeze['inputs'][str(CHECKPOINT)]
    package={str(p.relative_to(PACKAGE)):digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package==source_freeze['package']
    names=['tabicl_context_matching_trial.py','tabicl_context_matching.py','check_tabicl_context_matching.py',
           'tabicl_history.py','tabicl_history_learning.py','check_tabicl_history_learning.py',
           'acquisition_tasks.py','uji.py','recomposition_trial.py']
    paths=[PLAN,CSV,CHECKPOINT]
    for directory in [SOURCE,TARGET,FULL_SOURCE,FULL_TARGET]:
        paths.extend(p for p in directory.iterdir() if p.suffix in {'.json','.npz','.pt'})
    write(OUT/'freeze.json',dict(created=now(),seed=SEED,code={n:digest(ROOT/n) for n in names},
        inputs={str(p):digest(p) for p in sorted(paths)},package=package,
        optimizer=dict(name='AdamW',lr=1e-5,weight_decay=.01,clip_norm=1.,steps=STEPS,amp_initial=1024.),
        learning='Only last context query_mlp, 8320 parameters; inherited joint axes and eight views',
        criterion='Target both directions no worse and mean at least 3% lower than frozen; replacement also 3% lower than GP/retained/WKNN',
        limits='Same six-step early learning budget, not optimization convergence. Known B1 development tasks, not blind.',
        resources=dict(cpu=2,gpu=1,wall_seconds=600,downloads=0,max_gpu_fraction=.8),
        target_updates=0,checkpoint_selection='final six steps only'))
    for n in names:(OUT/n).write_bytes((ROOT/n).read_bytes())
    prepared,reference=prepare(SOURCE,True)
    model=ContextMatchingRegressor(device='cuda:0')
    assert state_hash(model.loaded.model_)==source_complete['original_state_sha256']
    frozen=frozen_hash(model)
    model.enable_recompute()
    parameters=[p for p in model.loaded.model_.parameters() if p.requires_grad]
    assert sum(p.numel() for p in parameters)==8320
    optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    training=[]
    for step in range(STEPS):
        tick=time.monotonic();p=prepared[step%2]
        fit_context(model,p);optimizer.zero_grad(set_to_none=True)
        loss=model.backward(p['fit']['rss'],p['fit']['xy'],scaler)
        scaler.unscale_(optimizer)
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
        assert norm>0
        assert all(v.grad is None for v in model.loaded.model_.parameters() if not v.requires_grad)
        scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        assert scaler.get_scale()>=scale,'AMP overflow; no retries'
        assert int(optimizer.state[parameters[0]]['step'])==step+1
        training.append(dict(step=step+1,event=p['event']['id'],pinball=loss,
            gradient_norm=norm,seconds=time.monotonic()-tick))
        write(OUT/'training.json',training);print(json.dumps(training[-1]),flush=True)
    assert frozen_hash(model)==frozen
    final_hash=state_hash(model.loaded.model_)
    assert final_hash!=source_complete['original_state_sha256']
    checkpoint=OUT/'matching_step_6.ckpt'
    model.save(checkpoint,source_updates=STEPS,provenance=dict(freeze_sha256=digest(OUT/'freeze.json'),
        source_only=True,source_floor='B0F1',trainable_names=model.matching_parameter_names,converged=False))
    del optimizer,parameters,model
    torch.cuda.empty_cache()
    model=ContextMatchingRegressor(checkpoint,device='cuda:0')
    assert state_hash(model.loaded.model_)==final_hash and frozen_hash(model)==frozen
    model.loaded.model_.requires_grad_(False).eval()
    source=evaluate(model,prepared,'source',FULL_SOURCE)
    source_result=summary(source,reference,'uniform')
    del prepared
    target_prepared,target_reference=prepare(TARGET,False)
    target=evaluate(model,target_prepared,'target',FULL_TARGET)
    target_result=summary(target,target_reference,'retained')
    assert state_hash(model.loaded.model_)==final_hash
    result=dict(complete=True,created=now(),seconds=time.monotonic()-started,steps=STEPS,
        source=source_result,target=target_result,training=training,
        previous_full_parameter_source=source_complete['means'],
        previous_full_parameter_target=json.loads((FULL_TARGET/'complete.json').read_text())['means'],
        trained_state_sha256=final_hash,frozen_state_sha256=frozen,checkpoint_sha256=digest(checkpoint),
        target_updates=0,converged=False,blind_test=False,
        peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20)
    write(OUT/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    assert socket.gethostname()=='ray-web' and torch.cuda.is_available()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.8,0)
    run()
