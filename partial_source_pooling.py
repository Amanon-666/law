"""A standard hierarchical mixture between shared and independent AP relations.

Given a common source coefficient gamma, an AP uses gamma with probability
1-lambda and an independent coefficient from the same prior with probability
lambda. Existing source fields are reused; this only changes the joint prior.
"""
import itertools
import math

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from source_weight_mixture import coefficient_posteriors,mixture_log_prob


def conditional_pooling(log_z,prior_mass,unpooled_probability):
    lam=float(unpooled_probability)
    log_shared=-math.inf if lam==1 else math.log1p(-lam)
    log_local=-math.inf if lam==0 else math.log(lam)
    log_prior=prior_mass.log()
    independent_evidence=torch.logsumexp(log_z+log_prior,dim=1)
    conditional_evidence=torch.logaddexp(log_shared+log_z,log_local+independent_evidence[:,None])
    gamma_log_mass=torch.log_softmax(log_prior+conditional_evidence.sum(0),0)
    shared_responsibility=log_shared+log_z-conditional_evidence
    local_responsibility=log_local+independent_evidence[:,None]-conditional_evidence
    independent_log_mass=torch.log_softmax(log_z+log_prior,1)
    log_evidence=torch.logsumexp(log_prior+conditional_evidence.sum(0),0)
    return gamma_log_mass,shared_responsibility,local_responsibility,independent_log_mass,log_evidence


def partial_ap_log_prob(component_log_prob,shared_log_mass,local_log_mass,independent_log_mass):
    independent=mixture_log_prob(component_log_prob,independent_log_mass)
    shape=(len(shared_log_mass),)+(1,)*(component_log_prob.ndim-1)
    return torch.logaddexp(component_log_prob+shared_log_mass.reshape(shape),
        independent[None]+local_log_mass.reshape(shape))


def scan_log_prob(log_z,component_log_prob,prior_mass,unpooled_probability):
    gamma,shared,local,independent,_=conditional_pooling(log_z,prior_mass,unpooled_probability)
    score=torch.zeros_like(component_log_prob[0])
    for a in range(len(log_z)):
        score+=partial_ap_log_prob(component_log_prob[a],shared[a],local[a],independent[a])
    return mixture_log_prob(score,gamma)


def fit_probability(objective):
    """Bounded scalar fit; brackets come only from the declared fitting loss."""
    grid=np.linspace(0.,1.,101);values=np.array([float(objective(x)) for x in grid])
    if not np.isfinite(values).all(): raise RuntimeError('Nonfinite pooling objective')
    trials=[(values[0],0.),(values[-1],1.),(values[50],.5)];evaluations=len(grid)
    brackets=[(0.,grid[1]),(grid[-2],1.)]
    for i in range(1,len(grid)-1):
        if values[i]<=values[i-1] and values[i]<=values[i+1] and (values[i]<values[i-1] or values[i]<values[i+1]):
            brackets.append((grid[i-1],grid[i+1]))
    for bounds in brackets:
        result=minimize_scalar(objective,bounds=bounds,method='bounded',options=dict(xatol=1e-8))
        if not result.success: raise RuntimeError('Unfinished scalar pooling fit')
        trials.append((float(result.fun),float(result.x)));evaluations+=result.nfev
    loss,value=min(trials,key=lambda pair:(pair[0],abs(pair[1]-.5)))
    return dict(unpooled_probability=value,objective=loss,evaluations=int(evaluations),
        objective_range=float(values.max()-values.min()),grid=grid.tolist(),profile=values.tolist())


def check():
    z=torch.tensor([[0.,-3.],[-2.,0.],[-.3,-.8]])
    lp=torch.log(torch.tensor([[[.8,.2],[.1,.7]],[[.2,.6],[.9,.2]],[[.7,.2],[.3,.5]]]))
    pi=torch.tensor([.5,.5]);shared,independent=coefficient_posteriors(z,pi)
    torch.testing.assert_close(scan_log_prob(z,lp,pi,0.),mixture_log_prob(lp.sum(0),shared))
    expected=sum(mixture_log_prob(lp[a],independent[a]) for a in range(len(z)))
    torch.testing.assert_close(scan_log_prob(z,lp,pi,1.),expected)
    # With only one AP, its marginal coefficient prior is identical for every
    # lambda; learning sharing requires relationships between APs.
    torch.testing.assert_close(scan_log_prob(z[:1],lp[:1],pi,.37),scan_log_prob(z[:1],lp[:1],pi,0.))
    lam=.37;numerator=torch.zeros(lp.shape[-1]);denominator=0.
    for gamma in range(2):
        for weights in itertools.product(range(2),repeat=len(z)):
            probability=pi[gamma].item()
            for w in weights: probability*=((1-lam) if w==gamma else 0.)+lam*pi[w].item()
            evidence=probability*torch.exp(sum(z[a,w] for a,w in enumerate(weights)))
            denominator+=evidence
            numerator+=evidence*torch.exp(sum(lp[a,w] for a,w in enumerate(weights)))
    torch.testing.assert_close(scan_log_prob(z,lp,pi,lam).exp(),numerator/denominator)
    torch.testing.assert_close(scan_log_prob(z.flip(1),lp.flip(1),pi.flip(0),lam),scan_log_prob(z,lp,pi,lam))
    torch.testing.assert_close(scan_log_prob(z.flip(0),lp.flip(0),pi,lam),scan_log_prob(z,lp,pi,lam))
    no_information=conditional_pooling(torch.zeros_like(z),pi,lam)
    torch.testing.assert_close(no_information[0].exp(),pi)
    torch.testing.assert_close(no_information[1].exp(),torch.full_like(z,1-lam))
    return dict(shared_endpoint_exact=True,independent_endpoint_exact=True,single_AP_cannot_identify_sharing=True,
        enumerated_joint_model_agreement=True,AP_and_source_permutation=True,empty_support_does_not_estimate_hyperparameter=True)
