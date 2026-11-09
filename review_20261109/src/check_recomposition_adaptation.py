"""Server-only small structural check; no real query or shared training."""
import json
import time
from unittest.mock import patch
import numpy as np
import torch
import recomposition_adaptation as joint
from fields import History
from recomposition import Parameters, prepare, condition_field
from posterior import proposal
from observations import scan_log_prob
from recomposition_trial import ROOT, digest, write, now

OUT=ROOT/'results/recomposition_adaptation_structure_20261107_attempt2'


def cases_for(model, sources=2, count=2, order=None, identical=False, shift=0.):
    base=dict(mean=-90.,amplitude=7.,length=3.,drift=2.)
    anchors=torch.tensor([[0.,0.],[2.,0.],[0.,3.]])+shift
    sx=torch.tensor([[.5,.2],[2.,2.]])[:count]+shift
    grid=torch.tensor([[0.,1.],[1.,1.],[2.,1.],[3.,1.]])+shift
    order=list(range(sources)) if order is None else order
    out={}
    for ap in range(2):
        means=[torch.tensor([-70.,-76.,-88.])+ap*3,
               torch.tensor([-90.,-80.,-72.])-ap*2,
               torch.tensor([-77.,-82.,-79.])+ap]
        hs=[History(anchors,dict(mean=means[0 if identical else i],cov=torch.eye(3)*8),base) for i in range(sources)]
        stats=dict(n=torch.tensor([3.,3.])[:count],k=torch.tensor([3.,2.])[:count],
                   s1=torch.tensor([-231.-ap*6,-150.-ap*4])[:count],
                   s2=torch.tensor([17789.+ap*936,11252.+ap*608])[:count])
        # RSS triples (-78,-77,-76) / pairs (-76,-74), shifted -2*ap.
        out[ap]=dict(blocks=prepare([hs[i] for i in order],[anchors]*sources,
            [0]*sources,sx,grid,base,0.),stats=stats)
    return out,grid


def run():
    started=time.monotonic();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    OUT.mkdir(exist_ok=False)
    files=['recomposition_adaptation.py','check_recomposition_adaptation.py','recomposition.py',
           'adaptation.py','observations.py','posterior.py','sampling.py','fields.py']
    write(OUT/'freeze.json',dict(created=now(),code={n:digest(ROOT/n) for n in files},
        limits='Synthetic only; CPU2/GPU0; 120 seconds; no source or target training',
        finite_difference=dict(step=.0002,tolerance='0.002*(1+abs(fd))')))
    for n in files:(OUT/n).write_bytes((ROOT/n).read_bytes())
    model=Parameters(torch.zeros(9),torch.ones(9))
    with torch.no_grad(): model.theta.copy_(torch.linspace(-.08,.08,12))
    obs=dict(alpha=.85,beta=-85.,detect_scale=4.,noise_std=5.)
    cases,grid=cases_for(model)
    captured=[];original=joint.minimize
    def capture(*args,**kwargs):
        r=original(*args,**kwargs);captured.append(r);return r
    with patch.object(joint,'minimize',capture): state=joint.fit(model,cases,obs,differentiable=True)
    chosen=min(captured,key=lambda r:abs(r.fun-state['diagnostics']['selected_objective']))
    objective=joint.SupportObjective(model,cases,obs)
    point=torch.tensor(chosen.x,requires_grad=True)
    F=lambda p:objective.value(p[:2],p[2:].reshape(2,5))
    H=torch.autograd.functional.hessian(F,point).detach()
    gp,=torch.autograd.grad(F(point),point,create_graph=True)
    mixed=torch.stack([torch.autograd.grad(gp[i],model.theta,retain_graph=True)[0][10] for i in range(len(point))]).detach()
    free=list(range(len(point)))
    if state['diagnostics']['variance_bound']!='interior': free.remove(1)
    response=torch.zeros_like(point)
    response[free]=-torch.linalg.solve(H[free][:,free],mixed[free])
    w=state['weights'].detach()
    dense=float(w[0]*(1-w[0])*response[0])
    implicit=float(torch.autograd.grad(state['weights'][0],model.theta,retain_graph=True)[0][10])
    assert abs(dense-implicit)<1e-8
    prepared={}
    for ap,c in cases.items():
        p=joint.adapted_prior(model,c['blocks'],state)
        with torch.no_grad():
            numerical,_=proposal(dict(sm=p['sm'],sk=p['sk'],phi_s=torch.zeros((2,0)),**c['stats']),obs,torch.empty(0),117030+ap)
        prepared[ap]=dict(c,proposal=(numerical['samples'],numerical['log_proposal']))
    rss=torch.tensor([[-77.,-79.],[-74.,100.]])
    def loss(adapted):
        score=torch.zeros((2,len(grid)))
        for ap,c in prepared.items():
            field,_=condition_field(joint.adapted_prior(model,c['blocks'],adapted),c,obs)
            score=score+scan_log_prob(rss[:,ap],field['mean'],field['variance'],obs)
        lp=score.log_softmax(1)
        return -(lp[0,1]+lp[1,2])/2
    outer=loss(state)
    analytic=torch.autograd.grad(outer,model.theta)[0].detach()
    fixed={**state,'weights':state['weights'].detach(),'intercept_variance':state['intercept_variance'].detach()}
    partial=torch.autograd.grad(loss(fixed),model.theta)[0].detach()
    checks=[]
    for index in (9,10):
        value=float(model.theta[index].detach());answers=[];weights=[]
        for direction in (-1,1):
            with torch.no_grad(): model.theta[index]=value+direction*.0002
            fitted=joint.fit(model,cases,obs)
            answers.append(float(loss(fitted).detach()));weights.append(float(fitted['weights'][0]))
        with torch.no_grad(): model.theta[index]=value
        fd=(answers[1]-answers[0])/.0004
        assert abs(float(analytic[index])-fd)<.002*(1+abs(fd))
        checks.append(dict(parameter=index,implicit=float(analytic[index]),finite_difference=fd,
                           frozen_adaptation_partial=float(partial[index]),weight_response=(weights[1]-weights[0])/.0004))
    assert float((analytic-partial).abs().max())>1e-5
    three,_=cases_for(model,sources=3);t=joint.fit(model,three,obs)
    swapped,_=cases_for(model,sources=3,order=[2,0,1]);s=joint.fit(model,swapped,obs)
    permutation=float((t['weights'][[2,0,1]]-s['weights']).abs().max());assert permutation<1e-4
    empty,_=cases_for(model,count=0);zero=joint.fit(model,empty,obs,differentiable=True)
    assert torch.equal(zero['weights'],torch.tensor([.5,.5])) and float(zero['intercept_variance'])==0
    one,_=cases_for(model,count=1);single=joint.fit(model,one,obs,differentiable=True)
    same,_=cases_for(model,identical=True);equal=joint.fit(model,same,obs,differentiable=True)
    # Identical independent GP marginals still give sum(w^2) C, so uniform
    # weights are NOT required. Only a genuinely w-independent prior does.
    absent={ap:dict(c,blocks=dict(c['blocks'],active=False)) for ap,c in cases.items()}
    with torch.no_grad(): uninformed=joint.fit(model,absent,obs)
    assert float((uninformed['weights']-.5).abs().max())<1e-5
    moved,_=cases_for(model,shift=100.);translated=joint.fit(model,moved,obs)
    translation=float((translated['weights']-state['weights'].detach()).abs().max());assert translation<1e-5
    result=dict(passed=True,seconds=time.monotonic()-started,dense_IFT=dense,block_IFT=implicit,
        full_scan_gradient=checks,permutation_gap=permutation,translation_gap=translation,
        weights=state['weights'].detach().tolist(),intercept_variance=float(state['intercept_variance'].detach()),
        diagnostics=state['diagnostics'],single_anchor_diagnostics=single['diagnostics'],
        identical_source_weights=equal['weights'].detach().tolist(),
        no_history_weights=uninformed['weights'].detach().tolist(),
        shared_training_steps=0,real_query_scans=0,
        scope='Local stationary derivatives and synthetic deployment path; not localization performance or all boundary cases')
    write(OUT/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__': run()
