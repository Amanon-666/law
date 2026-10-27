"""Repair context thinning for learning a cross-AP sharing prior.

Same source floors, physical scans, H/S/Q position split and hierarchical
model. Use the full H+S-visible AP catalog; preserve the four-AP experiment.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket

import numpy as np
import torch

import inference
from data import group_scans
from multi_history import single_history
from partial_source_pooling import conditional_pooling,partial_ap_log_prob,fit_probability
from run import ap_stats,tensor
from run_multi_history import digest,write
from source_weight_mixture import conditional_field,mixture_log_prob
from uji_probability_study import DATA,load_raw
from uji_history_bank_study import cached_history
import run_partial_source_pooling as common

ROOT=common.ROOT
SPARSE=common.OUT
SOURCE=common.SOURCE
OUT=ROOT/'results/uji_partial_pooling_full_scan_v1_20261026'


def configure():common.OUT=OUT


def prepare():
    configure();common.prepare()
    freeze=json.loads((OUT/'freeze.json').read_text())
    original=json.loads((SOURCE/'manifest.json').read_text())
    floors=load_raw('trainingData.csv',set(freeze['source_floors']));events=[]
    for e in original['events']:
        f=floors[e['floor']];lookup={int(r):i for i,r in enumerate(f['row_ids'])}
        sr=np.array([lookup[r] for r in e['support_rows']])
        h=np.concatenate([floors[n]['rss'] for n in e['histories']])
        aps=np.flatnonzero((h!=100).any(0)|(f['rss'][sr]!=100).any(0))
        events.append(dict(e,ap_columns=aps.tolist(),previous_ap_subset=e['aps']))
    write(OUT/'source_manifest.json',dict(events=events,origin=original['origin']))
    freeze.update(development_order='After the completed 17-node baseline and four-AP source fit; before full-scan source inference/fit. No original output overwritten.',
        hypothesis='Learning an inter-AP sharing rule requires deployment-like full-scan adaptation information; valid AP marginals alone do not guarantee a matching conditional training task',
        source_APs_per_task=[len(e['ap_columns']) for e in events],
        source_training='Exact same 18 B0 tasks and original S5x3/Q12x3 scans; full H+S-visible AP catalog; all existing model parameters except lambda fixed',
        source_objective='Position-balanced conditional complete-scan probability with full AP Support; same predictive formula as deployment',
        AP_subsampling='Removed from the primary fit. Old four AP component fields reused exactly; all additional AP evidence/prediction comes from the same previously available scans.',
        matched_context_diagnostic='Also fit on the original four Query APs with full Support AP evidence, isolating conditioning information from Query output dimension',
        old_run=str(SPARSE),history_solver='Original VI, maxiter640, actual optimizer success recorded for newly fitted source histories')
    for p in [Path(__file__),ROOT/'data.py',ROOT/'multi_history.py',ROOT/'uji_probability_study.py']:
        freeze['code_sha256'][str(p)]=digest(p);shutil.copy2(p,OUT/'code'/p.name)
    extra=[SOURCE/'manifest.json',DATA/'trainingData.csv',SPARSE/'source_fit.json']
    extra += [SPARSE/'source'/f'task{e["id"]:02}'/'fields.npz' for e in events]
    freeze['inputs'].update({str(p):digest(p) for p in extra})
    write(OUT/'freeze.json',freeze)
    common.log(stage='full_scan_prepared',AP_counts=freeze['source_APs_per_task'],new_positions=0,new_scans=0)


def histories(freeze):
    meta=json.loads((OUT/'source_manifest.json').read_text());origin=np.array(meta['origin'])
    setting=freeze['source_setting'];base,obs=setting['base'],setting['obs']
    floors=load_raw('trainingData.csv',set(freeze['source_floors']));records=[];real=inference.minimize;context={}
    def tracked(*a,**kw):
        result=real(*a,**kw)
        records.append(dict(**context,success=bool(result.success),iterations=int(result.nit),gradient=float(np.abs(result.jac).max()),message=str(result.message)))
        if not result.success:raise RuntimeError(f'Unfinished source history: {records[-1]}')
        return result
    inference.minimize=tracked
    for name,f in floors.items():
        aps=sorted({ap for e in meta['events'] if name in e['histories'] for ap in e['ap_columns']})
        dest=OUT/'histories'/name;dest.mkdir(parents=True,exist_ok=True)
        centered=group_scans(f['rss'],f['xy']-origin)
        for ap in aps:
            p=dest/f'ap{ap}.npz'
            if p.exists():continue
            previous=SOURCE/'histories'/name/f'ap{ap}.npz'
            if previous.exists():shutil.copy2(previous,p);continue
            context.update(floor=name,ap=ap)
            h=single_history(dict(scans=centered),ap,base,obs,steps=640)
            value=dict(empty=np.array(h is None),origin=origin)
            if h is not None:value.update(z=h.z.numpy(),mean=h.m.numpy(),cov=h.c.numpy())
            np.savez_compressed(p,**value)
        common.log(stage='full_source_histories',floor=name,APs=len(aps))
    files=list((OUT/'histories').glob('B*/ap*.npz'))
    write(OUT/'histories/completed.json',dict(completed=True,origin=origin.tolist(),new_solver_records=records,
        hashes={str(p):digest(p) for p in files}))


def source_fields(freeze,task):
    assert (OUT/'histories/completed.json').exists()
    dest=OUT/'source'/f'task{task:02}';dest.mkdir(exist_ok=True)
    if (dest/'completed.json').exists():return
    meta=json.loads((OUT/'source_manifest.json').read_text());e=meta['events'][task];origin=np.array(meta['origin'])
    f=load_raw('trainingData.csv',{e['floor']})[e['floor']];lookup={int(r):i for i,r in enumerate(f['row_ids'])}
    sr=np.array([lookup[r] for r in e['support_rows']]);qr=np.array([lookup[r] for r in e['query_rows']])
    s=group_scans(f['rss'][sr],f['xy'][sr]-origin);q=group_scans(f['rss'][qr],f['xy'][qr]-origin)
    assert len(s['positions'])==5 and len(q['positions'])==12
    assert not (s['positions'][:,None]==q['positions'][None]).all(-1).any()
    base,obs=freeze['source_setting']['base'],freeze['source_setting']['obs']
    old_meta=json.loads((SPARSE/'source'/f'task{task:02}'/'completed.json').read_text())
    old_indices={x['ap']:i for i,x in enumerate(old_meta['episodes'])}
    with np.load(SPARSE/'source'/f'task{task:02}'/'fields.npz') as a:old={k:a[k] for k in a.files}
    sx=tensor(s['positions']);qx=tensor(q['positions']);group=torch.tensor(q['group'])
    for index,ap in enumerate(e['ap_columns']):
        path=dest/f'ap{ap}.npz'
        if path.exists():continue
        if ap in old_indices:
            i=old_indices[ap];value={k:old[k][i] for k in ['log_z','log_prob','ess']}
        else:
            hh=[cached_history(OUT,n,ap,base) for n in e['histories']];empty=all(h is None for h in hh)
            z=[];lp=[];ess=[]
            with torch.no_grad():
                for w in freeze['nodes']:
                    if empty and z:z.append(z[0]);lp.append(lp[0]);ess.append(ess[0]);continue
                    field=conditional_field(hh,float(w),base,obs,sx,ap_stats(s,ap),qx,261527+task*520+ap)
                    z.append(field['log_evidence']);ess.append(field['effective_samples'])
                    lp.append(inference.observation_log_prob(tensor(q['rss'][:,ap]),field['mean'][group],field['variance'][group],obs).numpy())
            value=dict(log_z=np.array(z),log_prob=np.array(lp),ess=np.array(ess))
        np.savez_compressed(path,**value)
        if index%20==0:common.log(stage='full_source_fields',task=task,APs=index+1,total=len(e['ap_columns']))
    values={k:[] for k in ['log_z','log_prob','ess']}
    for ap in e['ap_columns']:
        with np.load(dest/f'ap{ap}.npz') as a:
            for k in values:values[k].append(a[k])
    values={k:np.stack(v) for k,v in values.items()}
    subset=np.array([e['ap_columns'].index(x['ap']) for x in old_meta['episodes']])
    for k in ['log_z','log_prob']:np.testing.assert_array_equal(values[k][subset],old[k])
    weights=1/(len(q['positions'])*q['stats']['n'][q['group']])
    np.testing.assert_array_equal(weights,old['weights'])
    np.savez_compressed(dest/'fields.npz',**values,weights=weights,old_query_AP_indices=subset,ap_columns=e['ap_columns'])
    write(dest/'completed.json',dict(completed=True,task=task,APs=len(e['ap_columns']),old_components_identical=True,
        fields_sha256=digest(dest/'fields.npz'),min_ess=float(values['ess'].min())))
    common.log(stage='full_source_task_completed',task=task,APs=len(e['ap_columns']),min_ess=float(values['ess'].min()))


def fit(freeze):
    configure();common.fit_source(freeze)
    mass=tensor(freeze['prior_mass']);data=[]
    for task in range(freeze['source_tasks']):
        with np.load(OUT/'source'/f'task{task:02}'/'fields.npz') as a:
            data.append((tensor(a['log_z']),tensor(a['log_prob']),tensor(a['weights']),a['old_query_AP_indices']))
    def loss(lam):
        result=[]
        for z,lp,weights,indices in data:
            gamma,shared,local,independent,_=conditional_pooling(z,mass,lam)
            score=torch.zeros_like(lp[0])
            for i in indices:score+=partial_ap_log_prob(lp[i],shared[i],local[i],independent[i])
            result.append(-(weights*mixture_log_prob(score,gamma)).sum())
        return float(torch.stack(result).mean())
    diagnostic=fit_probability(loss)
    diagnostic.update(completed=True,role='Same four Query APs as old run; only Support AP information restored',
        old_source_fit=json.loads((SPARSE/'source_fit.json').read_text())['unpooled_probability'],
        half_loss=loss(.5),shared_loss=loss(0.),independent_loss=loss(1.))
    write(OUT/'context_only_fit.json',diagnostic)
    common.log(stage='context_only_fit',**{k:v for k,v in diagnostic.items() if k not in ['grid','profile']})


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on server')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','histories','source','fit','event','summarize']);p.add_argument('--task',type=int);p.add_argument('--event');args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:
        configure();freeze=json.loads((OUT/'freeze.json').read_text())
        for group in ['code_sha256','inputs']:
            for path,expected in freeze[group].items():assert digest(Path(path))==expected,path
        if args.stage=='histories':histories(freeze)
        elif args.stage=='source':source_fields(freeze,args.task)
        elif args.stage=='fit':fit(freeze)
        elif args.stage=='event':common.event(freeze,args.event)
        else:common.summarize()
