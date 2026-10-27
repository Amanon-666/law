"""Bounded structural and one-source derivative audit, no model training."""
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from history_bank import BankPrior
from inference import observation_log_prob, predict_from_posterior
from profile_prior import ProfileMean, ProfileBankPrior
from reference_pmc import setup, sample_reference, log_proposal
from run_multi_history import digest, write
from source_weight_mixture import conditional_field, field_from_prior
from support_integral import ImportanceSupport
from transfer import History, rbf
from calibration_relation import CalibrationRelation


def structural():
    torch.manual_seed(261026)
    base = dict(mean=-95., amplitude=16., length=8., drift=7.)
    z = torch.tensor([[0., 0.], [4., 2.], [9., 3.]])
    h = History(z, dict(mean=torch.tensor([-70., -75., -90.]), cov=.2*rbf(z,z,16.,8.)), base)
    x = torch.tensor([[1., 1.], [5., 3.], [11., 4.], [40., 20.]])
    model = ProfileMean().double()
    for histories, weights, dz in [([h],[1.],[1]), ([h,None],[.3,.7],[-1,2]),
                                   ([h,None,h,h],[.1,.2,.3,.4],[-2,-1,1,4])]:
        old = BankPrior(histories,weights,base)
        prior = ProfileBankPrior(histories,weights,base,dz,model)
        om,of = old.features(x); m,f = prior.features(x)
        torch.testing.assert_close(m,om,rtol=0,atol=0)
        torch.testing.assert_close(prior.covariance(x,x,f,f),old.covariance(x,x,of,of),rtol=0,atol=0)
        torch.testing.assert_close(prior.diagonal(x,f),old.diagonal(x,of),rtol=0,atol=0)
    with torch.no_grad():
        model.readout[-1].weight.fill_(.01)
        model.readout[-1].bias.fill_(.2)
    p = ProfileBankPrior([h,None],[.3,.7],base,[-1,2],model)
    r = ProfileBankPrior([None,h],[.7,.3],base,[2,-1],model)
    pm,pf=p.features(x); rm,rf=r.features(x)
    torch.testing.assert_close(pm,rm)
    torch.testing.assert_close(pm,torch.cat([p.features(x[:2])[0],p.features(x[2:])[0]]))
    move = torch.tensor([100.,-60.])
    moved_h = History(z+move,dict(mean=h.m,cov=h.c),base)
    moved = ProfileBankPrior([moved_h,None],[.3,.7],base,[-1,2],model)
    torch.testing.assert_close(pm,moved.features(x+move)[0])
    relation = CalibrationRelation('uniform',base['length']).double()
    p = ProfileBankPrior([h,None],[.3,.7],base,[-1,2],model,relation)
    _,f = p.features(x); cov=p.covariance(x,x,f,f)
    torch.testing.assert_close(cov.diag(),p.diagonal(x,f),rtol=1e-6,atol=1e-5)
    assert torch.linalg.eigvalsh(cov).min()>0
    empty=ProfileBankPrior([None,None],[.5,.5],base,[-1,1],model)
    em,ef=empty.features(x)
    torch.testing.assert_close(em,torch.full_like(em,base['mean']))
    torch.testing.assert_close(empty.diagonal(x,ef),torch.full_like(em,base['amplitude']**2))
    return dict(zero_head_exact_bank=True,source_permutation_nonzero=True,
        point_batch_independent=True,coordinate_translation=True,variable_J=[1,2,4],
        missing_sources=True,existing_constant_local_covariance=True)


def main():
    if socket.gethostname()!='ray-web': raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    root=Path(__file__).resolve().parent
    out=root/'results/profile_prior_core_audit_v2_20261026';out.mkdir(exist_ok=False)
    source=root/'results/uji_bank_crossfloor_learning_v1_20261026'
    original=root/'results/uji_source_weight_mixture_v1_20261026/code/source_weight_mixture.py'
    files=[Path(__file__),root/'profile_prior.py',root/'source_weight_mixture.py',
           root/'history_bank.py',root/'bank_calibration.py',root/'calibration_relation.py',
           root/'transfer.py',root/'inference.py',root/'reference_pmc.py',root/'support_integral.py']
    write(out/'freeze.json',dict(created_unix=time.time(),scope=__doc__,
        limitation='No localization evaluation, learned checkpoint or general-J integration/training',
        hypothesis='A zero-initialized history-only profile mean can preserve the mature bank and support update exactly while admitting a differentiable non-convex mean correction',
        files={str(p):digest(p) for p in files},
        inputs={str(p):digest(p) for p in [source/'freeze.json',source/'source_cases.pt',original]}))
    (out/'code').mkdir()
    for p in files: shutil.copy2(p,out/'code'/p.name)
    checks=structural()
    setting=json.loads((source/'freeze.json').read_text());base,obs=setting['base'],setting['obs']
    case=torch.load(source/'source_cases.pt',weights_only=False)[0][0]
    # Read the preserved implementation, never overwrite the baseline artifact.
    spec=importlib.util.spec_from_file_location('frozen_source_weight',original)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    args=(case['histories'],.37,base,obs,case['sx'],case['stats'],case['qx'],261026)
    before=legacy.conditional_field(*args)
    np.random.seed(9);np.random.normal(size=100)
    after=conditional_field(*args)
    model=ProfileMean().double()
    dz=[int(n.split('F')[1])-int(case['episode']['floor'].split('F')[1]) for n in case['episode']['histories']]
    prior=ProfileBankPrior(case['histories'],[.37,.63],base,dz,model)
    candidate=field_from_prior(prior,obs,case['sx'],case['stats'],case['qx'],261026)
    for key in ['mean','variance','log_evidence']:
        for result in [after,candidate]:
            torch.testing.assert_close(torch.as_tensor(result[key]),torch.as_tensor(before[key]),rtol=0,atol=0)
    empty=field_from_prior(prior,obs,case['sx'][:0],None,case['qx'],261026)
    assert empty['log_evidence']==0 and empty['effective_samples'] is None
    with torch.no_grad():
        sm,sf=prior.features(case['sx']);sk=prior.covariance(case['sx'],case['sx'],sf,sf)
        target,proposal,center,root_cov,_=setup(sm,sk,case['stats'],obs)
        samples,_,info=sample_reference(target,proposal,center,root_cov,261027)
        logq=log_proposal(samples,center,root_cov,info)
    def loss():
        sm,sf=prior.features(case['sx']);sk=prior.covariance(case['sx'],case['sx'],sf,sf)
        post=ImportanceSupport(sm,sk,case['stats'],obs,samples,logq)
        w=post.log_weights.exp(); m=w@samples; c=samples-m
        qm,qf=prior.features(case['qx'])
        prediction=predict_from_posterior(qm,prior.diagonal(case['qx'],qf),
            prior.covariance(case['qx'],case['sx'],qf,sf),sm,sk,dict(mean=m,cov=(c.T*w)@c))
        group=case['group']
        lp=observation_log_prob(case['rss'],prediction['mean'][group],prediction['variance'][group],obs)
        return -(case['weights']*lp).sum()
    value=loss(); parameter=model.readout[-1].bias
    gradient,=torch.autograd.grad(value,parameter)
    eps=1e-5
    with torch.no_grad():
        parameter.fill_(eps);plus=loss().item()
        parameter.fill_(-eps);minus=loss().item();parameter.zero_()
    numerical=(plus-minus)/(2*eps)
    torch.testing.assert_close(gradient[0],torch.tensor(numerical),rtol=2e-3,atol=2e-5)
    assert gradient.abs().max()>1e-8
    result=dict(completed=True,structural=checks,zero_model_source_replay_exact=True,
        shared_sampler='Local RNG forwarded to all pypmc Gaussian components; both paths use the repaired sampler, not a replay of historical random draws',
        independent_of_global_numpy_state=True,
        source_case=case['episode'],empty_support=True,head_gradient=gradient.item(),
        finite_difference=numerical,effective_samples=info['effective_samples'],parameter_updates=0)
    write(out/'audit.json',result);print(json.dumps(result))


if __name__=='__main__': main()
