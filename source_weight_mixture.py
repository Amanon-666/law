"""Bayesian averaging over an existing LAW source coefficient.

The source GP and target discrepancy GP are unchanged. Conditional on w,
f_a = w*h_0a + (1-w)*h_1a + delta_a. Only w's sharing/integration changes.
This is a mature Bayesian model-averaging construction, not a new kernel.
"""
import numpy as np
import torch

from history_bank import BankPrior
from inference import gaussian_update, predict_from_posterior
from reference_pmc import setup, sample_reference


def quadrature(order):
    nodes, mass = np.polynomial.legendre.leggauss(order)
    return (nodes+1)/2, mass/2


def coefficient_posteriors(log_evidence, prior_mass):
    """Evidence [AP,node]; each likelihood enters exactly once."""
    log_prior = prior_mass.log()
    separate = torch.log_softmax(log_evidence+log_prior, dim=-1)
    shared = torch.log_softmax(log_evidence.sum(0)+log_prior, dim=-1)
    return shared, separate


def mixture_log_prob(component_log_prob, log_mass):
    """Mix probabilities before normalizing over candidate positions."""
    shape = (len(log_mass),)+(1,)*(component_log_prob.ndim-1)
    return torch.logsumexp(component_log_prob+log_mass.reshape(shape), dim=0)


@torch.no_grad()
def conditional_field(histories, coefficient, base, obs, sx, stats, points, seed):
    prior = (BankPrior(histories,[coefficient,1-coefficient],base)
             if any(h is not None for h in histories) else BankPrior([],[],base))
    return field_from_prior(prior,obs,sx,stats,points,seed)


@torch.no_grad()
def field_from_prior(prior, obs, sx, stats, points, seed, defensive_fraction=0.):
    """Same frozen LAW update/readout, accepting a history-conditioned prior."""
    qm,qf = prior.features(points); qv = prior.diagonal(points,qf)
    if not len(sx):
        return dict(mean=qm,variance=qv,log_evidence=0.,effective_samples=None)
    sm,sf = prior.features(sx); sk = prior.covariance(sx,sx,sf,sf)
    target,proposal,center,root,_ = setup(sm,sk,stats,obs)
    samples,logw,info = sample_reference(target,proposal,center,root,seed,defensive_fraction=defensive_fraction)
    w=logw.exp(); pm=w@samples; c=samples-pm
    posterior=dict(mean=pm,cov=(c.T*w)@c)
    cross=prior.covariance(points,sx,qf,sf)
    prediction=predict_from_posterior(qm,qv,cross,sm,sk,posterior)
    if not torch.isfinite(prediction['mean']).all() or not torch.isfinite(prediction['variance']).all() or (prediction['variance']<0).any():
        raise RuntimeError('Invalid conditional field')
    return dict(**prediction,
        log_evidence=info['log_detection_evidence']-gaussian_update(sm,sk,stats,obs)['loss'].item(),
        effective_samples=info['effective_samples'])


def check():
    # Opposite AP evidence must not silently collapse to one global source.
    prior=torch.tensor([.5,.5]); z=torch.tensor([[0.,-8.],[-8.,0.]])
    shared,separate=coefficient_posteriors(z,prior)
    torch.testing.assert_close(shared.exp(),prior)
    assert separate[0].exp()[0]>.999 and separate[1].exp()[1]>.999
    # No S evidence leaves the original prior, not an estimated hard source.
    empty_shared,empty_separate=coefficient_posteriors(torch.zeros_like(z),prior)
    torch.testing.assert_close(empty_shared.exp(),prior)
    torch.testing.assert_close(empty_separate.exp(),prior.expand_as(z))
    lp=torch.tensor([[[-1.,-4.],[-2.,-3.]],[[-3.,-1.],[-4.,-2.]]])
    mixed=mixture_log_prob(lp,shared)
    torch.testing.assert_close(mixed.exp(),(lp.exp()*prior[:,None,None]).sum(0))
    # Shared latent source is integrated AFTER multiplying AP probabilities.
    other=lp.flip(0)
    common=mixture_log_prob(lp+other,shared)
    independent=mixture_log_prob(lp,separate[0])+mixture_log_prob(other,separate[1])
    assert (common-independent).abs().max()>1
    reverse_s,reverse_a=coefficient_posteriors(z.flip(1),prior.flip(0))
    torch.testing.assert_close(reverse_s,shared.flip(0))
    torch.testing.assert_close(reverse_a,separate.flip(1))
    nodes,mass=quadrature(17)
    np.testing.assert_allclose(nodes,1-nodes[::-1],atol=1e-15)
    np.testing.assert_allclose(mass.sum(),1.,atol=1e-15)
    return dict(opposite_AP_preferences=True,empty_evidence_keeps_prior=True,
        probability_mixture_not_logscore_average=True,shared_latent_integrated_once=True,
        source_permutation=True,uniform_prior_quadrature=True)
