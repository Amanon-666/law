"""Single-UJI B0 fitting/B2 checkpoint selection; B1 is never opened.

Original 12-parameter ReFi transformations, fixed S nuisance fits, full S
posterior differentiation. No B0+B2 pretrained weights, new architecture,
query-conditioned adaptation, implicit-gradient refitting, or target tuning.
"""
import gc
import json
import resource
import socket
import time
import traceback
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import history_fit
from building_source_assets import load_field
from episodic_tasks import task_inputs
from neural_history_trial import metric
from paired_assets import read_allowed
from recomposition import Parameters
from recomposition_coupled import prepare_cases,adapt_cases,evaluate
from recomposition_coupled_trial import episode
from recomposition_trial import ROOT,CSV,digest,write,now,subset
from source_environment_validation import select_checkpoint
from source_episode_catalog import read_metadata
from uji import tensor,group_scans

OUT=ROOT/'results/refi_source_selection_learning_20261107'
PLAN=ROOT/'results/source_environment_validation_20261107'
ASSETS=ROOT/'results/building_source_assets_20261106'
CALIBRATION=ROOT/'results/paired_assets_20261106'
CHECKPOINTS=(0,8,16)


def progress(stage,**kwargs):
    row=dict(created=now(),stage=stage,**kwargs)
    write(OUT/'progress.json',row);print(json.dumps(row),flush=True)


class AssetBank:
    def __init__(self,manifest,base,obs):
        self.manifest,self.base,self.obs=manifest,base,obs
        self.records={(r['floor'],r['ap']):r for r in manifest['fields']}
        self.rows=dict(manifest['history_rows']);self.new_fields=0;self.max_gradient=0.

    def ensure(self,floor,rows,scans,origin):
        if floor in self.rows:
            assert rows==self.rows[floor]
            aps=set(map(int,np.flatnonzero((scans['stats']['k']>0).any(0))))
            assert aps=={a for h,a in self.records if h==floor}
            return
        assert floor in ('B0F0','B2F0'),floor
        original=history_fit.minimize
        def monitored(*args,**kwargs):
            r=original(*args,**kwargs);g=r.jac.copy()
            for i,(lo,hi) in enumerate(kwargs['bounds']):
                if lo is not None and r.x[i]<=lo+1e-10 and g[i]>0:g[i]=0
                if hi is not None and r.x[i]>=hi-1e-10 and g[i]<0:g[i]=0
            residual=float(np.abs(g).max())
            with (OUT/'history_solver.jsonl').open('a') as f:
                f.write(json.dumps(dict(floor=floor,ap=ap,success=bool(r.success),iterations=int(r.nit),projected_gradient=residual))+'\n')
            if not np.isfinite(r.fun) or not np.isfinite(r.x).all() or residual>1e-3:
                raise RuntimeError(f'Historical VI unfinished: {floor} AP {ap}, gradient {residual}')
            self.max_gradient=max(self.max_gradient,residual);return r
        for ap in map(int,np.flatnonzero((scans['stats']['k']>0).any(0))):
            with patch.object(history_fit,'minimize',monitored):
                field=history_fit.single_history(dict(scans=scans),ap,self.base,self.obs,steps=640)
            path=OUT/'histories'/floor/f'ap{ap}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,z=field.z.numpy(),mean=field.m.numpy(),cov=field.c.numpy(),origin=origin)
            self.records[floor,ap]=dict(floor=floor,ap=ap,path=str(path.relative_to(ROOT)),sha256=digest(path))
            self.new_fields+=1
        self.rows[floor]=rows
        write(OUT/'histories.json',dict(parameters_sha256=self.manifest['parameters_sha256'],
            fields=list(self.records.values()),history_rows=self.rows,new_fields=self.new_fields))
        progress('history_added',floor=floor,new_fields=self.new_fields)


def prepare_pair(events,bank,zero,role):
    e=events[0];assert len(events)==2 and events[0]['support_rows']==events[1]['support_rows']
    allowed=set(e['support_rows']).union(*(set(ids) for ids in e['history_rows'].values()))
    assert all(not allowed.intersection(x['query_rows']) for x in events)
    raw=read_allowed(CSV,allowed)
    # All rows within each historical floor are exactly its frozen H rows.
    origin=np.concatenate([raw[h]['xy'] for h in e['histories']]).mean(0)
    hs={h:subset(raw[h],ids,origin) for h,ids in e['history_rows'].items()}
    support=subset(raw[e['floor']],e['support_rows'],origin)
    for h,scans in hs.items():bank.ensure(h,e['history_rows'][h],scans,origin)
    @lru_cache(maxsize=None)
    def loader(h,ap):return load_field(bank.records[h,ap],origin,bank.base)
    task=task_inputs(hs,support,bank.base,bank.obs,loader)
    assert task['names']==e['histories']
    delta=[int(e['floor'].split('F')[1])-int(h.split('F')[1]) for h in task['names']]
    blocks=prepare_cases(task,delta)
    state,cases,_=adapt_cases(zero,blocks,bank.obs,e['floor'],False)
    result=dict(cases=cases,state=state,grid=task['grid'],base=bank.base,obs=bank.obs,
        events=events,origin=origin,support_positions=support['positions'])
    path=OUT/f'{role}_{e["floor"]}_support.pt';torch.save(result,path)
    progress('support_prepared',role=role,floor=e['floor'],APs=len(cases),grid=len(task['grid']),
        weights=state['weights'].tolist(),variance=float(state['intercept_variance']))
    return result


def attach_queries(data):
    rows=sum((e['query_rows'] for e in data['events']),[])
    raw=read_allowed(CSV,set(rows));qs=[]
    for e in data['events']:
        q=subset(raw[e['floor']],e['query_rows'],data['origin'])
        assert not (q['positions'][:,None]==data['support_positions'][None]).all(-1).any()
        assert not ((q['xy']<data['grid'].min(0).values.numpy())|(q['xy']>data['grid'].max(0).values.numpy())).any()
        qs.append(q)
    q=group_scans(np.concatenate([x['rss'] for x in qs]),np.concatenate([x['xy'] for x in qs]))
    w=1/(len(q['positions'])*q['stats']['n'][q['group']])
    ep=episode(data['cases'],data['grid'],data['obs'],q['rss'],q['xy'],w)
    return dict(data=data,ep=ep,q=q,cells=qs)


def score(model,item,backward=False):
    result=evaluate(model,item['ep'],item['data']['state'],backward)
    if backward:return result
    cells={};start=0
    for e,q in zip(item['data']['events'],item['cells']):
        end=start+len(q['rss']);cells[e['id']]=metric(result['log_probability'][start:end],item['data']['grid'],q)
        start=end
    return dict(nll=result['nll'],minimum_ESS=result['min_ess'],cells=cells,log_probability=result['log_probability'])


def run():
    started=time.monotonic();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    plan=json.loads((PLAN/'tasks.json').read_text());pd=json.loads((PLAN/'complete.json').read_text())
    assert pd['complete'] and digest(PLAN/'tasks.json')==pd['artifacts']['tasks.json']
    pf=json.loads((PLAN/'freeze.json').read_text())
    for n,h in pf['code'].items():assert digest(ROOT/n)==h
    manifest=json.loads((ASSETS/'histories.json').read_text())
    ad=json.loads((ASSETS/'complete.json').read_text());assert digest(ASSETS/'histories.json')==ad['histories_sha256']
    assert digest(manifest['parameters_path'])==manifest['parameters_sha256']
    calibration=json.loads((CALIBRATION/'freeze.json').read_text())
    cd=json.loads((CALIBRATION/'complete.json').read_text())
    assert digest(CALIBRATION/'freeze.json')==cd['freeze_sha256']
    metadata=read_metadata(CSV,set(plan['fit'][0]['histories']+[e['floor'] for e in plan['fit']]))
    assert all(metadata[i]['floor'].startswith('B0F') for i in calibration['allowed_rows'])
    assert digest(CSV)==calibration['inputs'][str(CSV)]
    params=json.loads(Path(manifest['parameters_path']).read_text());base,obs=params['base'],params['obs']
    OUT.mkdir(exist_ok=False);(OUT/'code').mkdir()
    schedule=[];rng=np.random.default_rng(117079)
    for epoch in range(4):schedule+=rng.permutation(4).tolist()
    frozen=dict(created=now(),code={p.name:digest(p) for p in ROOT.glob('*.py')},
        inputs={str(p):digest(p) for p in [CSV,PLAN/'freeze.json',PLAN/'tasks.json',PLAN/'complete.json',
            ASSETS/'histories.json',ASSETS/'complete.json',CALIBRATION/'freeze.json',CALIBRATION/'parameters.json']},
        learning_object='Original ReFi12; zero initialization; fixed zero-theta S weights/variance; full conditional-field derivative',
        optimizer=dict(name='Adam',lr=.02,steps=16,selection_steps=CHECKPOINTS,schedule=schedule,penalty=.001),
        roles=dict(shared_fit='B0',source_selection='B2',unopened='B1'),
        source_normalization='B0 H anchor features, equally weighted target floors; no B2 statistics',
        validation='B2 protected Q; equal floor and AA/AB; position-balanced geometric-median MDE; include identity',
        worthwhile='Selected step > 0 and B2 aggregate MDE >=3% below identity; source selection only, not target transfer evidence',
        history_policy='Reuse exact existing H rows/base; fit only missing B0F0/B2F0 with VI640 and gradient<=1e-3',
        numeric='Fixed zero-theta proposals, original posterior reweighting, ESS>=1000; fail without auto-repair',
        limits=['One fitting and one source-selection building, correlated floors; not sufficient environmental coverage',
            'B2 was used in previous development; only current gradients and checkpoint decisions are separated',
            'Base/observation calibration uses B0 source Q; no B2 calibration or shared normalization',
            'S nuisance rule is the checked joint VI fit frozen at theta0, not exactly the original discrete-variance fitter',
            'Finite 16 steps; no claim of convergence, K stability or innovation from validation alone'],
        resources=dict(cpu=2,gpu=0,hard_seconds=900,downloads=0,new_history_floors=['B0F0','B2F0']))
    write(OUT/'freeze.json',frozen)
    for n in frozen['code']:(OUT/'code'/n).write_bytes((ROOT/n).read_bytes())
    bank=AssetBank(manifest,base,obs);zero=Parameters(torch.zeros(9),torch.ones(9))
    source=[];moments=[]
    for floor in sorted({e['floor'] for e in plan['fit']}):
        data=prepare_pair([e for e in plan['fit'] if e['floor']==floor],bank,zero,'fit')
        z=torch.cat([c['blocks']['za'].reshape(-1,9) for c in data['cases'].values()])
        moments.append((z.mean(0),z.square().mean(0)));source.append(attach_queries(data))
    center=torch.stack([m for m,v in moments]).mean(0);v=torch.stack([v for m,v in moments]).mean(0)-center.square()
    assert float(v.min())>-1e-10
    scale=v.clamp_min(0).sqrt();scale[v.abs()<1e-12]=0
    model=Parameters(center,scale);torch.save(model.state_dict(),OUT/'step0.pt')
    opt=torch.optim.Adam(model.parameters(),lr=.02)
    for step,index in enumerate(schedule,1):
        opt.zero_grad();result=score(model,source[index],True)
        assert torch.isfinite(model.theta.grad).all()
        row=dict(step=step,floor=source[index]['data']['events'][0]['floor'],nll=result['nll'],minimum_ESS=result['min_ess'],
                 gradient_norm=float(model.theta.grad.norm()),seconds=time.monotonic()-started)
        opt.step()
        with (OUT/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step),OUT/'latest.pt')
        if step in CHECKPOINTS:torch.save(model.state_dict(),OUT/f'step{step}.pt')
        progress('shared_update',**row)
    write(OUT/'models_fixed.json',dict(created=now(),checkpoints={str(s):digest(OUT/f'step{s}.pt') for s in CHECKPOINTS},
        fitted_buildings=['B0'],B2_task_data_read=False,B1_data_read=False,seconds=time.monotonic()-started))
    del source,data,z,result;gc.collect()
    records=[dict(step=s,checkpoint=str(OUT/f'step{s}.pt'),fitted_buildings=['B0'],event_mde={}) for s in CHECKPOINTS]
    details={str(s):{} for s in CHECKPOINTS}
    for floor in sorted({e['floor'] for e in plan['selection']}):
        data=prepare_pair([e for e in plan['selection'] if e['floor']==floor],bank,zero,'selection')
        # S fit and numerical nodes are fixed before source-selection Q is read.
        item=attach_queries(data)
        torch.save(dict(q=item['q'],cells=item['cells']),OUT/f'selection_{floor}_query.pt')
        for r in records:
            model.load_state_dict(torch.load(r['checkpoint'],weights_only=True))
            result=score(model,item)
            for eid,m in result['cells'].items():r['event_mde'][eid]=m['mde']['median']
            details[str(r['step'])][floor]={k:v for k,v in result.items() if k!='log_probability'}
            np.savez_compressed(OUT/f'{floor}_step{r["step"]}_prediction.npz',log_probability=result['log_probability'].numpy(),grid=data['grid'].numpy())
        write(OUT/'selection_progress.json',dict(records=records,details=details))
        del data,item,result;gc.collect()
        progress('selection_floor_finished',floor=floor,seconds=time.monotonic()-started)
    selected=select_checkpoint(records,plan['selection'])
    weighted={str(r['step']):sum(e['selection_weight']*r['event_mde'][e['id']] for e in plan['selection']) for r in records}
    gain=1-selected['source_selection_mde']/weighted['0']
    write(OUT/'selection.json',dict(selected=selected,records=records,details=details,aggregate_MDE=weighted,
        relative_gain=gain,worth_continuing=selected['step']>0 and gain>=.03,B1_data_read=False))
    fixed=json.loads((OUT/'models_fixed.json').read_text())
    assert all(digest(OUT/f'step{s}.pt')==fixed['checkpoints'][str(s)] for s in CHECKPOINTS)
    complete=dict(complete=True,seconds=time.monotonic()-started,steps=16,selected_step=selected['step'],
        aggregate_source_selection_MDE=weighted,relative_gain=gain,worth_continuing=selected['step']>0 and gain>=.03,
        new_history_fields=bank.new_fields,max_new_history_VI_gradient=bank.max_gradient,
        source_contexts=4,selection_contexts=5,B1_data_read=False,peak_memory_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file()})
    write(OUT/'complete.json',complete);print(json.dumps(complete),flush=True)

if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    try:run()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),traceback=traceback.format_exc()))
        raise
