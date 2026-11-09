"""ReFi identity baseline: support adaptation, field posterior, scan likelihood.

Candidate coordinates are permitted geometry; no true query coordinates enter.
The identity setting is a compatibility baseline, not the full learned ReFi.
"""
import hashlib
import torch
from adaptation import fit_episode, choose_intercept
from fields import InterceptBankPrior
from posterior import proposal, ImportanceSupport
from observations import predict_from_posterior, observation_log_prob


def adapt(cases, base, obs):
    weights, trials = fit_episode(cases, base, obs)
    sigma = choose_intercept(cases, weights, base, obs)
    return dict(weights=weights, trials=trials, sigma=sigma)


@torch.no_grad()
def predict(cases, base, obs, state, points, query_rss, event_id, k, progress=None):
    scores = points.new_zeros((len(query_rss),len(points)))
    diagnostics = []
    for column,(ap,case) in enumerate(cases.items()):
        prior=InterceptBankPrior(case['histories'],state['weights'],base,state['sigma']['intercept_std'])
        sm,sf=prior.features(case['sx']);qm,qf=prior.features(points)
        sk=prior.covariance(case['sx'],case['sx'],sf,sf)
        qv=prior.diagonal(points,qf)
        cross=prior.covariance(points,case['sx'],qf,sf)
        prepared=dict(sm=sm,sk=sk,phi_s=sm.new_zeros((k,10)),**case['stats'])
        seed=int.from_bytes(hashlib.sha256(f'multifloor/{event_id}/K{k}/{ap}'.encode()).digest()[:4],'little')
        numerical,info=proposal(prepared,obs,sm.new_zeros(10),seed)
        posterior=ImportanceSupport(sm,sk,case['stats'],obs,numerical['samples'],numerical['log_proposal'])
        weights=posterior.log_weights.exp();ess=float(weights.square().sum().reciprocal())
        if ess<1000:raise RuntimeError(f'AP {ap}: ESS {ess}')
        mean=weights@posterior.samples;centered=posterior.samples-mean
        value=predict_from_posterior(qm,qv,cross,sm,sk,dict(mean=mean,cov=(centered.T*weights)@centered))
        if not torch.isfinite(value['mean']).all() or not torch.isfinite(value['variance']).all() or (value['variance']<0).any():
            raise RuntimeError(f'AP {ap}: invalid propagated moments')
        scores+=observation_log_prob(torch.as_tensor(query_rss[:,column],dtype=torch.float64)[:,None],value['mean'][None],value['variance'][None],obs)
        diagnostics.append(dict(AP=ap,ESS=ess,sampler=info['sampler']))
        if progress is not None:progress(column+1,len(cases))
    if not torch.isfinite(scores).all():raise RuntimeError('Invalid complete scan scores')
    return dict(scores=scores,predictions=scores.softmax(1)@points,diagnostics=diagnostics)
