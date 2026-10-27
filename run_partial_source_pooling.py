"""Fixed, source-learned and local empirical-Bayes partial pooling.

One shared hyperparameter, no new network. Historical source-coefficient
fields are reused. This is a controlled application of hierarchical Bayes,
not a claim of a new statistical construction.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from full_ap_study import metrics
from inference import observation_log_prob
from partial_source_pooling import check,conditional_pooling,partial_ap_log_prob,scan_log_prob,fit_probability
from run_multi_history import digest,write
from source_weight_mixture import conditional_field,mixture_log_prob
from run_source_weight_mixture import OUT as PARENT,SOURCE,ROOT

OUT=ROOT/'results/uji_partial_source_pooling_v1_20261026'


def log(**row):print(json.dumps(row,allow_nan=False),flush=True)


def prepare():
    OUT.mkdir(exist_ok=False)
    parent=json.loads((PARENT/'freeze.json').read_text());source=json.loads((SOURCE/'freeze.json').read_text())
    files=[Path(__file__),ROOT/'partial_source_pooling.py',ROOT/'source_weight_mixture.py',ROOT/'history_bank.py',ROOT/'reference_pmc.py',ROOT/'inference.py']
    freeze=dict(created_unix=time.time(),development_order='Designed after two initial B1 source-weight events; before any partial-pooling source fit or target evaluation. Original study continues unchanged.',
        hypothesis='Partial sharing protects weakly observed APs while allowing source disagreements; source tasks may learn a useful shared deviation probability',
        generative_model='gamma~Uniform[0,1]; independently per AP, w_a=gamma with probability 1-lambda, otherwise w_a~Uniform[0,1]; f_a|w_a is unchanged LAW',
        learned_object='One scalar lambda: prior probability that an AP has a separate source relation',
        variants=dict(fixed_half='lambda=.5, no extra fit',source_learned='Fit lambda using B0 conditional held-out scan probabilities',target_eb='Fit lambda from all target Support AP evidence only'),
        endpoints='lambda=0 exactly shared_bma; lambda=1 exactly ap_bma. Existing continuous shared-w LAW remains a strong control.',
        source_tasks=source['source_tasks'],source_APs_per_task=source['aps_per_task'],source_floors=source['source_floors'],
        source_training='Reuse 18 existing B0 tasks, each with 4 aligned APs, S5x3 and Q12x3; no B1 labels used',
        source_objective='Position-balanced sum of single-scan predictive log probabilities, joint across the four sampled APs; not full joint likelihood across Q scans or a position-ranking loss',
        AP_subsampling='Four APs per training task are a consistent marginal of the exchangeable hierarchical model, not an unbiased estimate of the full-AP joint log score; deployment AP count differs and generalization must be measured',
        source_geometry='Existing two-thirds outside-source-floor-range tasks are retained; no B1-driven source task filtering',
        scalar_fit='101-point scalar profile, local bounded refinement including edge intervals, compare endpoints; no query-based target fit or architecture sweep',
        source_parameter_updates='One scalar learned by bounded optimization; report objective calls, not fictitious neural training steps',
        target_source_inputs='Same original H/S/AP/grid/query; reused 17-component fields and S evidences',
        numerical_scope='Same 17-point source-coefficient approximation; no Gaussian collapse across different coefficients',
        novelty='Partial pooling, empirical Bayes and mixture priors are mature; this is a focused learning/control test, not a novelty claim',
        reverse_audit=['B0 H/S -> fixed per-coefficient evidence and Q conditional probabilities',
            'Aligned B0 Q scans -> conditional joint predictive score -> one shared lambda',
            'Target H/S -> same fixed component fields -> gamma posterior and AP shared/local responsibilities',
            'Target Query RSS -> joint predictive mixture -> same coordinate posterior mean; Query XY scoring only'],
        counterexamples=dict(empty_support='No data estimates lambda; source/fixed prior remains. Flat target-EB objective reports no identification.',
            no_source_information='If component fields are identical, all variants predict identically',
            wrong_histories='Cannot generate an absent spatial pattern; no negative-transfer guarantee',
            source_AP_permutation='Exchangeable AP prior and symmetric source-coefficient prior',
            sharing_limits='Exact reduction to the two existing endpoints, verified by enumeration',
            weak_local_AP='Other APs inform gamma, but cannot invent that AP measurements',
            training_mismatch='Source AP subsets and floor geometry differ from full target scans; success not assumed'),
        checks=check(),nodes=parent['coefficient_nodes'],prior_mass=parent['prior_mass'],
        source_setting=source,
        code_sha256={str(p):digest(p) for p in files},
        inputs={str(p):digest(p) for p in [PARENT/'freeze.json',PARENT/'manifest.json',SOURCE/'freeze.json',SOURCE/'source_cases.pt']})
    write(OUT/'freeze.json',freeze);(OUT/'code').mkdir();(OUT/'source').mkdir()
    for p in files:shutil.copy2(p,OUT/'code'/p.name)
    shutil.copy2(PARENT/'manifest.json',OUT/'manifest.json')
    log(stage='prepared',checks=freeze['checks'],new_neural_parameters=0)


def source_fields(freeze,task):
    dest=OUT/'source'/f'task{task:02}';dest.mkdir(exist_ok=True)
    if (dest/'completed.json').exists():return
    batch=torch.load(SOURCE/'source_cases.pt',weights_only=False)[task]
    setting=freeze['source_setting'];base,obs=setting['base'],setting['obs'];means=[];logs=[];ess=[]
    # APs must describe the same physical query scans for a joint scan score.
    for c in batch:
        torch.testing.assert_close(c['qx'],batch[0]['qx']);torch.testing.assert_close(c['group'],batch[0]['group'])
        torch.testing.assert_close(c['weights'],batch[0]['weights'])
    with torch.no_grad():
        for a,c in enumerate(batch):
            z=[];lp=[];effective=[]
            for w in freeze['nodes']:
                f=conditional_field(c['histories'],float(w),base,obs,c['sx'],c['stats'],c['qx'],261327+task*4+a)
                g=c['group'];lp.append(observation_log_prob(c['rss'],f['mean'][g],f['variance'][g],obs).numpy())
                z.append(f['log_evidence']);effective.append(f['effective_samples'])
            means.append(z);logs.append(lp);ess.append(effective)
    np.savez_compressed(dest/'fields.npz',log_z=np.array(means),log_prob=np.array(logs),weights=batch[0]['weights'].numpy(),ess=np.array(ess))
    write(dest/'completed.json',dict(completed=True,task=task,episodes=[c['episode'] for c in batch],
        fields_sha256=digest(dest/'fields.npz'),min_ess=float(np.min(ess))))
    log(stage='source_fields',task=task,min_ess=float(np.min(ess)))


def fit_source(freeze):
    data=[]
    for task in range(freeze['source_tasks']):
        d=OUT/'source'/f'task{task:02}'; assert (d/'completed.json').exists()
        with np.load(d/'fields.npz') as a: data.append(tuple(torch.tensor(a[k]) for k in ['log_z','log_prob','weights']))
    mass=torch.tensor(freeze['prior_mass'])
    def loss(value):
        return float(torch.stack([-(weights*scan_log_prob(z,lp,mass,value)).sum() for z,lp,weights in data]).mean())
    result=fit_probability(loss)
    result.update(completed=True,fixed_half_loss=loss(.5),shared_loss=loss(0.),independent_loss=loss(1.),
        source_fields_sha256={str(OUT/'source'/f'task{t:02}'/'fields.npz'):digest(OUT/'source'/f'task{t:02}'/'fields.npz') for t in range(freeze['source_tasks'])})
    write(OUT/'source_fit.json',result)
    log(stage='source_fit',**{k:v for k,v in result.items() if k not in ['grid','profile','source_fields_sha256']})


def event(freeze,event_id):
    old=PARENT/event_id; assert (old/'completed.json').exists()
    dest=OUT/event_id;dest.mkdir(exist_ok=True)
    if (dest/'completed.json').exists():return
    fitted=json.loads((OUT/'source_fit.json').read_text())
    meta=next(e for e in json.loads((OUT/'manifest.json').read_text())['events'] if e['id']==event_id)
    z=[]
    for ap in meta['ap_columns']:
        with np.load(old/f'ap{ap}.npz') as a:z.append(a['log_evidence'])
    z=torch.tensor(np.array(z));mass=torch.tensor(freeze['prior_mass'])
    eb=fit_probability(lambda x:-float(conditional_pooling(z,mass,x)[-1]))
    values=dict(fixed_half=.5,source_learned=fitted['unpooled_probability'],target_eb=eb['unpooled_probability'])
    write(dest/'adaptation.json',dict(values=values,target_eb=eb,source_fit_sha256=digest(OUT/'source_fit.json')))
    # Reuse exact parent query order and H+S grid; do not reconstruct bounds.
    from run_source_weight_mixture import load_raw
    f=load_raw('trainingData.csv',{meta['floor']})[meta['floor']];q=load_raw('validationData.csv',{meta['floor']})[meta['floor']]
    fi={int(r):i for i,r in enumerate(f['row_ids'])};qi={int(r):i for i,r in enumerate(q['row_ids'])}
    ir=np.array([fi[r] for r in meta['internal_rows']]);orr=np.array([qi[r] for r in meta['official_rows']])
    rss=torch.tensor(np.concatenate((f['rss'][ir],q['rss'][orr])));truth=np.concatenate((f['xy'][ir],q['xy'][orr]))
    with np.load(old/'predictions.npz') as a:
        grid=torch.tensor(a['grid']);np.testing.assert_allclose(a['query_xy'],truth)
        endpoints=dict(shared_bma=a['shared_bma'],ap_bma=a['ap_bma'])
    obs=json.loads((PARENT/'parameters.json').read_text())['obs'];device=torch.device('cuda:0')
    g=grid.to(device); predictions={};pending=[];posterior={}
    for method,lam in values.items():
        if lam==0.:predictions[method]=endpoints['shared_bma']
        elif lam==1.:predictions[method]=endpoints['ap_bma']
        else:pending.append(method);posterior[method]=conditional_pooling(z,mass,lam)
    scores={m:torch.zeros((len(mass),len(rss),len(grid)),device=device) for m in pending}
    with torch.no_grad():
        for a,ap in enumerate(meta['ap_columns']):
            if not pending:break
            with np.load(old/f'ap{ap}.npz') as f:
                mean=torch.tensor(f['mean'],device=device);variance=torch.tensor(f['variance'],device=device)
            for start in range(0,len(rss),64):
                sl=slice(start,start+64);r=rss[sl,ap].to(device)
                lp=observation_log_prob(r[None,:,None],mean[:,None,:],variance[:,None,:],obs)
                for method in pending:
                    _,shared,local,independent,_=posterior[method]
                    scores[method][:,sl]+=partial_ap_log_prob(lp,shared[a].to(device),local[a].to(device),independent[a].to(device))
        for method in pending:
            lp=mixture_log_prob(scores[method],posterior[method][0].to(device))
            # Center before the matrix product for precision with large UJI coordinates.
            origin=g.mean(0);predictions[method]=(torch.softmax(lp,1)@(g-origin)+origin).cpu().numpy()
    rows=[]
    for split,sl in [('internal',slice(0,len(ir))),('official',slice(len(ir),None))]:
        xy=truth[sl];_,group,count=np.unique(xy,axis=0,return_inverse=True,return_counts=True);sw=1/(len(count)*count[group.ravel()])
        for method,pred in predictions.items():rows.append(dict(method=method,split=split,**metrics(pred[sl],dict(xy=xy,scan_weights=sw))))
    np.savez_compressed(dest/'predictions.npz',**predictions,query_xy=truth,internal_count=len(ir))
    write(dest/'completed.json',dict(completed=True,event=event_id,values=values,results=rows,predictions_sha256=digest(dest/'predictions.npz')))
    log(stage='event',event=event_id,values=values,official=[r for r in rows if r['split']=='official'])


def summarize():
    rows=[]
    for e in json.loads((OUT/'manifest.json').read_text())['events']:
        d=json.loads((OUT/e['id']/'completed.json').read_text());rows.extend(dict(event=e['id'],floor=e['floor'],**r) for r in d['results'])
    summary=[dict(method=m,split=s,mde=float(np.mean([r['mde'] for r in rows if r['method']==m and r['split']==s])))
        for m in ['fixed_half','source_learned','target_eb'] for s in ['internal','official']]
    write(OUT/'summary.json',dict(completed=True,summary=summary,rows=rows));log(stage='summary',summary=summary)


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('All numerical work is server-only')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','source','fit','event','summarize']);p.add_argument('--task',type=int);p.add_argument('--event');args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:
        freeze=json.loads((OUT/'freeze.json').read_text())
        for group in ['code_sha256','inputs']:
            for path,expected in freeze[group].items():assert digest(Path(path))==expected,path
        if args.stage=='source':source_fields(freeze,args.task)
        elif args.stage=='fit':fit_source(freeze)
        elif args.stage=='event':event(freeze,args.event)
        else:summarize()
