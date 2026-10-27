"""Compare shared vs AP-specific source relations on the frozen LAW bank.

No neural training, new observation likelihood, AP selection, or target split.
Posterior source probabilities are estimated from Support only. The AP-point
and AP-mixture controls share all conditional fields and evidence estimates.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from data import group_scans
from full_ap_study import metrics
from history_bank import BankPrior
from inference import observation_log_prob
from run import ap_stats, tensor
from run_multi_history import digest, write
from source_weight_mixture import quadrature, coefficient_posteriors, mixture_log_prob, conditional_field, check
from uji_history_bank_study import OUT as BANK, SOURCES, cached_history, load_raw

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/uji_source_weight_mixture_v1_20261026'
SOURCE=ROOT/'results/uji_bank_crossfloor_learning_v1_20261026'
ORDER=17
METHODS=['shared_bma','ap_point','ap_bma']


def log(**row): print(json.dumps(row,allow_nan=False),flush=True)


def source_audit():
    OUT.mkdir(exist_ok=False)
    setting=json.loads((SOURCE/'freeze.json').read_text());base,obs=setting['base'],setting['obs']
    batches=torch.load(SOURCE/'source_cases.pt',weights_only=False)
    cases=[c for b in batches for c in b]; selected=[]
    for floor in setting['source_floors']:
        candidates=[]
        for i,c in enumerate(cases):
            if c['episode']['floor']!=floor: continue
            p0=BankPrior(c['histories'],[1.,0.],base);p1=BankPrior(c['histories'],[0.,1.],base)
            m0,_=p0.features(c['sx']);m1,_=p1.features(c['sx'])
            candidates.append((float((m0-m1).square().mean()),i))
        candidates.sort()
        selected.extend([candidates[len(candidates)//2][1],candidates[-1][1]])
    pre=dict(created_unix=time.time(),scope='Source-only numerical resolution check, never target selection',
        rules=check(),orders=[17,33],selection='Within each B0 floor, median and maximum H-only contrast at S',
        source_case_indices=selected,coefficient_mean_tolerance=.005,query_log_probability_tolerance=.02,
        shared_scope='Tests one AP coefficient integral; continuous-global EB remains the stronger control against coarse shared quadrature',
        source_cases_sha256=digest(SOURCE/'source_cases.pt'),
        code_sha256={str(p):digest(p) for p in [Path(__file__),ROOT/'source_weight_mixture.py']})
    write(OUT/'source_audit_prefreeze.json',pre);rows=[]
    with torch.no_grad():
        for i in selected:
            c=cases[i]; result={}
            for order in pre['orders']:
                nodes,mass=quadrature(order);fields=[]
                for w in nodes:
                    fields.append(conditional_field(c['histories'],float(w),base,obs,c['sx'],c['stats'],c['qx'],261127+i))
                evidence=tensor([a['log_evidence'] for a in fields])
                logp=torch.log_softmax(evidence+tensor(mass).log(),0)
                m=torch.stack([a['mean'] for a in fields])[:,c['group']]
                v=torch.stack([a['variance'] for a in fields])[:,c['group']]
                lp=observation_log_prob(c['rss'][None],m,v,obs)
                predictive=mixture_log_prob(lp,logp)
                result[order]=dict(mean_coefficient=float(logp.exp()@tensor(nodes)),logp=predictive,
                    min_ess=min(a['effective_samples'] for a in fields))
            a,b=result[17],result[33]
            rows.append(dict(case=i,episode=c['episode'],mean_coefficient_17=a['mean_coefficient'],mean_coefficient_33=b['mean_coefficient'],
                coefficient_mean_difference=abs(a['mean_coefficient']-b['mean_coefficient']),
                max_predictive_log_probability_difference=float((a['logp']-b['logp']).abs().max()),
                min_ess=min(a['min_ess'],b['min_ess'])))
            log(stage='source_resolution',**{k:v for k,v in rows[-1].items() if k!='episode'})
    passed=all(r['coefficient_mean_difference']<pre['coefficient_mean_tolerance'] and
               r['max_predictive_log_probability_difference']<pre['query_log_probability_tolerance'] for r in rows)
    write(OUT/'source_audit.json',dict(completed=True,passed=passed,rows=rows))
    if not passed: raise RuntimeError('17-node source resolution check failed; no target run authorized by this code')


def prepare():
    audit=json.loads((OUT/'source_audit.json').read_text()); assert audit['passed']
    pre=json.loads((OUT/'source_audit_prefreeze.json').read_text())
    for path,expected in pre['code_sha256'].items(): assert digest(Path(path))==expected,path
    nodes,mass=quadrature(ORDER)
    files=[Path(__file__),*[ROOT/p for p in ['source_weight_mixture.py','history_bank.py','transfer.py','inference.py',
        'reference_pmc.py','uji_history_bank_study.py','uji_probability_study.py','full_ap_study.py','data.py']]]
    parents=sorted((BANK/'histories').glob('B*/ap*.npz'))+sorted(BANK.glob('B*/ap*.npz'))+sorted(BANK.glob('B*/ap*.json'))
    freeze=dict(created_unix=time.time(),hypothesis='Allowing AP-specific historical relations improves localization at the same sparse Support budget; preserving source-coefficient uncertainty may help beyond independent point selection',
        model='Same conditional f_a=w_a*h_0a+(1-w_a)*h_3a+delta_a; uniform prior on w in [0,1]',
        methods=METHODS,source_names=SOURCES,coefficient_nodes=nodes.tolist(),prior_mass=mass.tolist(),
        numerical_scope='17-point Gauss-Legendre coefficient integral; M-PMC within each coefficient; Gaussian field moments within each coefficient, mixture retained across coefficients',
        strong_control='Existing continuous shared coefficient empirical-Bayes LAW; improvement only over shared_bma is insufficient',
        ap_point='Grid likelihood maximizer under uniform continuous prior; quadrature cell masses do not enter the point-mode choice',
        shared_bma='One coefficient shared across APs; all Support AP evidences determine its posterior; mix whole-scan probabilities once',
        ap_bma='Independent AP coefficients under the same fixed uniform prior; each AP uses its own Support evidence; product of AP mixture probabilities',
        parameter_learning='Zero new source learning and zero target gradient updates; posterior probability update only',
        history_absence='All histories undetected -> identical original local-GP field at every coefficient',
        midpoint_reuse='Existing equal_bank field/evidence at w=.5, exact same model/seed/grid; all-H-undetected uses original local_gp',
        source_audit=audit, k=5,scans_per_position=3,events='Same six previously inspected B1F1/F2 development cases',
        input_rights='Same H_bank, Support rows, AP columns and H+S candidate rectangle; no extra unlabeled target scans',
        correspondence='Native shared UJI XY/AP identity in one building; no arbitrary-building registration claim',
        reverse_audit=['H posterior + each fixed coefficient -> same GP prior at S/grid',
            'S-only exact RSS update and sampled detection evidence -> coefficient evidence and conditional field',
            'S evidence -> shared or AP-specific coefficient posterior; never Query or its position',
            'Query RSS -> conditional component probabilities -> proper mixture -> coordinate posterior -> XY expectation',
            'Query coordinates -> final score only'],
        counterexamples=dict(no_support='Coefficient posterior equals prior; uncertainty remains',
            one_support='Cannot distinguish sources that agree at the measured location',
            zero_residual='Source coefficient still changes mean and historical uncertainty, unlike correction-only kernels',
            wrong_sources='Convex mixtures do not create missing target spatial information; no negative-transfer guarantee',
            source_swap='Nodes/masses symmetric and source weights transform w to 1-w',
            duplicate_sources='Independent-source covariance must not double-count a copied physical acquisition',
            new_AP='All-H-undetected common local GP; no automatic AP matching',
            identifiability='Equal source functions and covariance can leave coefficient evidence weak; do not call diffuse posterior learning failure',
            quadrature='Source-only resolution check passed; cannot assume arbitrary coefficients/source counts are covered'),
        code_sha256={str(p):digest(p) for p in files},parent_cache_sha256={str(p):digest(p) for p in parents},
        inputs={str(p):digest(p) for p in [BANK/'freeze.json',BANK/'manifest.json',BANK/'parameters.json',OUT/'source_audit.json']})
    write(OUT/'freeze.json',freeze);shutil.copy2(BANK/'manifest.json',OUT/'manifest.json');shutil.copy2(BANK/'parameters.json',OUT/'parameters.json')
    (OUT/'code').mkdir()
    for p in files: shutil.copy2(p,OUT/'code'/p.name)
    log(stage='prepared',methods=METHODS,nodes=ORDER,new_source_updates=0)


def event(event_id):
    freeze=json.loads((OUT/'freeze.json').read_text())
    for group in ['code_sha256','parent_cache_sha256','inputs']:
        for path,expected in freeze[group].items(): assert digest(Path(path))==expected,path
    destination=OUT/event_id;destination.mkdir(exist_ok=True)
    if (destination/'completed.json').exists(): return
    params=json.loads((OUT/'parameters.json').read_text());base,obs=params['base'],params['obs']
    meta=next(e for e in json.loads((OUT/'manifest.json').read_text())['events'] if e['id']==event_id)
    f=load_raw('trainingData.csv',{meta['floor']})[meta['floor']];q=load_raw('validationData.csv',{meta['floor']})[meta['floor']]
    fi={int(r):i for i,r in enumerate(f['row_ids'])};qi={int(r):i for i,r in enumerate(q['row_ids'])}
    sr=np.array([fi[r] for r in meta['support_rows']]);ir=np.array([fi[r] for r in meta['internal_rows']]);orr=np.array([qi[r] for r in meta['official_rows']])
    origin=np.array(json.loads((BANK/'histories/completed.json').read_text())['origin'])
    support=group_scans(f['rss'][sr],f['xy'][sr]-origin);sx=tensor(support['positions'])
    with np.load(BANK/event_id/'predictions.npz') as old: grid=tensor(old['grid']-origin)
    nodes=np.array(freeze['coefficient_nodes']);mass=tensor(freeze['prior_mass']); all_evidence=[]; component_records=[]
    for index,ap in enumerate(meta['ap_columns']):
        path=destination/f'ap{ap}.npz';record=destination/f'ap{ap}.json'
        if not (path.exists() and record.exists()):
            histories=[cached_history(BANK,name,ap,base) for name in SOURCES];empty=all(h is None for h in histories)
            old_meta=json.loads((BANK/event_id/f'ap{ap}.json').read_text())
            saved=[];rows=[];old_method='local_gp' if empty else 'equal_bank'
            for j,w in enumerate(nodes):
                start=time.perf_counter()
                if empty or abs(w-.5)<1e-14:
                    with np.load(BANK/event_id/f'ap{ap}.npz') as a:
                        m=tensor(a[old_method+'_mean']);v=tensor(a[old_method+'_variance'])
                    info=next(r for r in old_meta['results'] if r['method']==old_method)
                    result=dict(mean=m,variance=v,log_evidence=info['support_log_evidence'],effective_samples=info['effective_samples'])
                    reused=True
                else:
                    result=conditional_field(histories,float(w),base,obs,sx,ap_stats(support,ap),grid,261026+meta['seed']);reused=False
                saved.append((result['mean'].numpy(),result['variance'].numpy()))
                rows.append(dict(node=j,coefficient=float(w),log_evidence=result['log_evidence'],effective_samples=result['effective_samples'],
                    reused_original=reused,seconds=time.perf_counter()-start))
            np.savez_compressed(path,mean=np.stack([a for a,b in saved]),variance=np.stack([b for a,b in saved]),log_evidence=[r['log_evidence'] for r in rows])
            write(record,dict(ap=ap,all_history_undetected=empty,results=rows))
        record_data=json.loads(record.read_text());component_records.extend(dict(ap=ap,**r) for r in record_data['results'])
        all_evidence.append([r['log_evidence'] for r in record_data['results']])
        if index%10==0: log(stage='fields',event=event_id,aps=index+1,total=len(meta['ap_columns']))
    evidence=tensor(all_evidence);shared,separate=coefficient_posteriors(evidence,mass)
    # Likelihood-mode point estimates; center resolves exact ties deterministically.
    point=np.array([np.lexsort((np.abs(nodes-.5),-row))[0] for row in np.array(all_evidence)])
    write(destination/'adaptation.json',dict(nodes=nodes.tolist(),shared_posterior=shared.exp().tolist(),ap_posterior=separate.exp().tolist(),
        ap_columns=meta['ap_columns'],point_coefficients=nodes[point].tolist(),ap_mean_coefficients=(separate.exp()@tensor(nodes)).tolist()))
    # Query scans enter only after the source posteriors are finalized above.
    rss=tensor(np.concatenate((f['rss'][ir],q['rss'][orr])));truth=np.concatenate((f['xy'][ir],q['xy'][orr]))
    device=torch.device('cuda:0');g=grid.to(device)
    shared_scores=torch.zeros((ORDER,len(rss),len(grid)),device=device)
    separate_scores=torch.zeros((len(rss),len(grid)),device=device);point_scores=torch.zeros_like(separate_scores)
    with torch.no_grad():
        for a,ap in enumerate(meta['ap_columns']):
            with np.load(destination/f'ap{ap}.npz') as arrays:
                m=tensor(arrays['mean']).to(device);v=tensor(arrays['variance']).to(device)
            for start in range(0,len(rss),64):
                sl=slice(start,start+64);r=rss[sl,ap].to(device)
                lp=observation_log_prob(r[None,:,None],m[:,None,:],v[:,None,:],obs)
                shared_scores[:,sl]+=lp
                separate_scores[sl]+=mixture_log_prob(lp,separate[a].to(device))
                point_scores[sl]+=lp[point[a]]
        scores={'shared_bma':mixture_log_prob(shared_scores,shared.to(device)),'ap_bma':separate_scores,'ap_point':point_scores}
        predictions={name:(torch.softmax(score,1)@g).cpu().numpy()+origin for name,score in scores.items()}
    rows=[]
    for split,sl in [('internal',slice(0,len(ir))),('official',slice(len(ir),None))]:
        xy=truth[sl];_,group,count=np.unique(xy,axis=0,return_inverse=True,return_counts=True)
        weights=1/(len(count)*count[group.ravel()])
        for name,pred in predictions.items(): rows.append(dict(method=name,split=split,**metrics(pred[sl],dict(xy=xy,scan_weights=weights))))
    np.savez_compressed(destination/'predictions.npz',**predictions,query_xy=truth,internal_count=len(ir),support_xy=support['positions']+origin,
        grid=grid.numpy()+origin,ap_columns=meta['ap_columns'])
    write(destination/'completed.json',dict(completed=True,event=event_id,results=rows,
        min_effective_samples=min(r['effective_samples'] for r in component_records),
        predictions_sha256=digest(destination/'predictions.npz')))
    log(stage='completed',event=event_id,official=[r for r in rows if r['split']=='official'])


def summarize():
    rows=[]
    for meta in json.loads((OUT/'manifest.json').read_text())['events']:
        d=json.loads((OUT/meta['id']/'completed.json').read_text())
        rows.extend(dict(event=meta['id'],floor=meta['floor'],**r) for r in d['results'])
    summary=[dict(method=m,split=s,events=6,mde=float(np.mean([r['mde'] for r in rows if r['method']==m and r['split']==s])))
        for m in METHODS for s in ['internal','official']]
    write(OUT/'summary.json',dict(completed=True,summary=summary,rows=rows));log(stage='summary',summary=summary)


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('All numerical work is server-only')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['source-audit','prepare','event','summarize']);p.add_argument('--event');args=p.parse_args()
    if args.stage=='source-audit':source_audit()
    elif args.stage=='prepare':prepare()
    elif args.stage=='event':event(args.event)
    else:summarize()
