"""One source-task feasibility check, not a shared-learning experiment."""
import argparse
import json
import time
from functools import lru_cache
import numpy as np
import torch
from building_source_assets import load_field
from check_recomposition_adaptation import cases_for
from episodic_tasks import task_inputs
from fields import InterceptBankPrior
from paired_assets import read_allowed
from recomposition import Parameters
from recomposition_adaptation import adapted_prior
from recomposition_coupled import prepare_cases, adapt_cases, evaluate, scores_for
from recomposition_trial import ROOT, CSV, digest, write, now, subset
from uji import tensor

CHECK = ROOT/'results/recomposition_coupled_structure_20261107'
OUT = ROOT/'results/recomposition_coupled_source_20261107'
ASSETS = ROOT/'results/building_source_assets_20261106'
RULE = ROOT/'results/stacking_shared_learning_20261107'


def episode(cases, grid, obs, rss, truth, query_weights):
    return dict(cases={ap:dict(c, rss=tensor(rss[:, ap])) for ap,c in cases.items()},
        grid=grid, obs=obs, truth=tensor(truth), query_weights=tensor(query_weights),
        nearest=torch.cdist(tensor(truth), grid).argmin(1))


def snapshot(out, **extra):
    out.mkdir(exist_ok=False)
    paths=list(ROOT.glob('*.py'))
    write(out/'freeze.json',dict(created=now(),code={p.name:digest(p) for p in paths},**extra))
    (out/'code').mkdir()
    for name in ('recomposition_coupled.py','recomposition_coupled_trial.py','recomposition_adaptation.py'):
        (out/'code'/name).write_bytes((ROOT/name).read_bytes())


def structure():
    started=time.monotonic()
    snapshot(CHECK, scope='Two-pass gradient vs monolithic full-scan derivative; synthetic only',
             tolerance=1e-9, shared_updates=0)
    model=Parameters(torch.zeros(9),torch.ones(9))
    with torch.no_grad(): model.theta.copy_(torch.linspace(-.08,.08,12))
    cases,grid=cases_for(model)
    obs=dict(alpha=.85,beta=-85.,detect_scale=4.,noise_std=5.)
    state,prepared,_=adapt_cases(model,cases,obs,'structure',True)
    ep=episode(prepared,grid,obs,np.array([[-77.,-79.],[-74.,100.]]),
               np.array([[1.,1.],[2.,1.]]),np.array([.5,.5]))
    score=sum(scores_for(model,c,state,obs)[0] for c in ep['cases'].values())
    lp=score.log_softmax(1)
    loss=-(ep['query_weights']*lp[torch.arange(2),ep['nearest']]).sum()+model.penalty()
    expected,=torch.autograd.grad(loss,model.theta,retain_graph=True)
    result=evaluate(model,ep,state,backward=True)
    gap=float((expected-model.theta.grad).abs().max())
    assert gap<1e-9
    probability_gap=float((result['log_probability']-lp.detach()).abs().max())
    assert probability_gap<1e-12
    write(CHECK/'complete.json',dict(passed=True,seconds=time.monotonic()-started,
        maximum_gradient_difference=gap,log_probability_difference=probability_gap,
        nll=result['nll'],shared_updates=0,real_queries=0,
        code={n:digest(ROOT/n) for n in ('recomposition_coupled.py','recomposition_adaptation.py')}))


def run():
    started=time.monotonic()
    check=json.loads((CHECK/'complete.json').read_text());assert check['passed']
    for n,h in check['code'].items(): assert digest(ROOT/n)==h
    source=json.loads((ASSETS/'freeze.json').read_text())
    manifest=json.loads((ASSETS/'histories.json').read_text())
    spec=json.loads((RULE/'freeze.json').read_text())['source_tasks'][0]
    assert spec['floor']=='B0F0'
    assert digest(manifest['parameters_path'])==manifest['parameters_sha256']
    assert digest(CSV)==source['inputs'][str(CSV)]
    parameters=json.loads(open(manifest['parameters_path']).read())
    base,obs=parameters['base'],parameters['obs']
    allowed=set(spec['support_rows']).union(*(set(v) for v in spec['history_rows'].values()))
    qrows=sum((spec['queries'][s] for s in ('A','B')),[])
    assert not allowed & set(qrows)
    assert not (allowed|set(qrows)) & set(source['protected_rows'])
    snapshot(OUT, inputs={str(p):digest(p) for p in (CSV,ASSETS/'freeze.json',ASSETS/'histories.json',
        RULE/'freeze.json',CHECK/'complete.json')},spec=spec,
        normalization='H anchor features only, equal AP/source/anchor rows; no S/Q statistical fit',
        policy='Original manuscript missing-history policy; no observed-negative augmentation',
        object='All-AP joint adaptation and one full-scan gradient, no optimizer update',
        decision='Proceed only if stationary positive-curvature IFT, finite full gradient, identity moment parity <1e-7, total <300 s',
        resources=dict(cpu=2,gpu=0,hard_seconds=300,shared_updates=0,downloads=0),
        scope='Previously used source task; not performance, transfer, K stability or learning benefit')
    def progress(stage,**more):
        record=dict(stage=stage,seconds=time.monotonic()-started,**more)
        write(OUT/'progress.json',record);print(json.dumps(record),flush=True)
    raw=read_allowed(CSV,allowed);origin=np.array(spec['origin'])
    hs={name:subset(raw[name],ids,origin) for name,ids in spec['history_rows'].items()}
    support=subset(raw[spec['floor']],spec['support_rows'],origin)
    records={(r['floor'],r['ap']):r for r in manifest['fields'] if r['floor'] in hs}
    @lru_cache(maxsize=None)
    def loader(name,ap):return load_field(records[name,ap],origin,base)
    task=task_inputs(hs,support,base,obs,loader)
    delta=[-int(name.split('F')[1]) for name in task['names']]
    cases=prepare_cases(task,delta)
    rows=torch.cat([c['blocks']['za'].reshape(-1,9) for c in cases.values()])
    center=rows.mean(0);variance=rows.var(0,unbiased=False)
    scale=variance.sqrt();scale[variance<1e-12]=0
    model=Parameters(center,scale)
    torch.save(dict(cases=cases,grid=task['grid'],base=base,obs=obs,center=center,scale=scale,
        support_positions=support['positions'],spec=spec),OUT/'support_cases.pt')
    progress('prepared',APs=len(cases),grid_points=len(task['grid']),histories=loader.cache_info().currsize)
    start_fit=time.monotonic()
    state,prepared,samplers=adapt_cases(model,cases,obs,'B0F0_source_cost',True)
    adaptation_seconds=time.monotonic()-start_fit
    write(OUT/'adaptation.json',dict(weights=state['weights'].detach().tolist(),
        variance=float(state['intercept_variance'].detach()),diagnostics=state['diagnostics'],
        seconds=adaptation_seconds,query_RSS_parsed=False))
    # Independent original identity GP construction, with the SAME fitted w/v.
    maximum=0.
    with torch.no_grad():
        for ap,c in task['cases'].items():
            old=InterceptBankPrior(c['histories'],state['weights'].tolist(),base,
                                   float(state['intercept_variance'].sqrt()))
            sm,sf=old.features(c['sx']);gm,gf=old.features(task['grid'])
            expected=dict(sm=sm,gm=gm,sk=old.covariance(c['sx'],c['sx'],sf,sf),
                cross=old.covariance(task['grid'],c['sx'],gf,sf),gv=old.diagonal(task['grid'],gf))
            actual=adapted_prior(model,cases[ap]['blocks'],state)
            maximum=max(maximum,max(float((actual[k]-v).abs().max()) for k,v in expected.items()))
    assert maximum<1e-7
    progress('adaptation_fixed',adaptation_seconds=adaptation_seconds,identity_moment_gap=maximum)
    qr=read_allowed(CSV,set(qrows));query=subset(qr[spec['floor']],qrows,origin)
    assert not (query['positions'][:,None]==support['positions'][None]).all(-1).any()
    weights=1/(len(query['positions'])*query['stats']['n'][query['group']])
    ep=episode(prepared,task['grid'],obs,query['rss'],query['xy'],weights)
    # Only one source gradient; no optimizer, checkpoint selection or target evaluation.
    start_eval=time.monotonic(); result=evaluate(model,ep,state,True)
    eval_seconds=time.monotonic()-start_eval
    assert torch.isfinite(model.theta.grad).all()
    assert torch.equal(model.theta.detach(),torch.zeros(12))
    torch.save(dict(model=model.state_dict(),gradient=model.theta.grad.detach(),
        log_probability=result['log_probability'],grid=task['grid'],query=query),OUT/'source_gradient.pt')
    complete=dict(passed=True,seconds=time.monotonic()-started,APs=len(cases),
        grid_points=len(task['grid']),query_scans=len(qrows),adaptation_seconds=adaptation_seconds,
        full_scan_gradient_seconds=eval_seconds,identity_moment_gap=maximum,
        gradient=model.theta.grad.tolist(),gradient_norm=float(model.theta.grad.norm()),
        min_ess=result['min_ess'],source_nll=result['nll'],shared_updates=0,
        artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file()})
    if OUT.name.endswith('_batch'):
        reference=ROOT/'results/recomposition_coupled_source_20261107'
        old=json.loads((reference/'complete.json').read_text())
        old_state=json.loads((reference/'adaptation.json').read_text())
        differences=dict(weights=float((state['weights'].detach()-tensor(old_state['weights'])).abs().max()),
            variance=abs(float(state['intercept_variance'].detach())-old_state['variance']),
            gradient=float((model.theta.grad-tensor(old['gradient'])).abs().max()),
            nll=abs(result['nll']-old['source_nll']))
        limits=dict(weights=1e-5,variance=1e-3,gradient=1e-4,nll=1e-5)
        assert all(differences[k]<limits[k] for k in limits),differences
        complete['reference_differences']=differences
        complete['reference_tolerances']=limits
        complete['reference_complete_sha256']=digest(reference/'complete.json')
    write(OUT/'complete.json',complete);print(json.dumps(complete),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--structure',action='store_true')
    parser.add_argument('--batch',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    if args.batch:
        CHECK=ROOT/'results/recomposition_adaptation_batch_20261107_attempt2'
        OUT=ROOT/'results/recomposition_coupled_source_20261107_batch'
    structure() if args.structure else run()
