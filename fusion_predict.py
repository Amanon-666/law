"""Common H/S-only adaptation and full-scan prediction for mature controls.

The four fusion arms use one Gaussian-VI posterior backend. This is a newly
specified scalable comparison, not a claim of equality to old PMC results.
No function here accepts Q coordinates. A failed arm emits no partial score.
"""
import time

import numpy as np
import torch

from data import group_scans
from external_baselines import tpm_adapter
from full_ap_study import wknn
from fusion_moments import source_moments,support_matrices,fuse
from fusion_adaptation import fit_joint
from fusion_posterior import infer_batch
from fusion_prior import combine,fit_weights
from fusion_task_data import histories_for_AP
from inference import observation_log_prob,predict_from_posterior
from operator_prior import marginal_loglik
from run import ap_stats,tensor
from transfer import rbf

ARMS={'global_identity':(False,False),'global_affine':(False,True),
      'local_identity':(True,False),'local_affine':(True,True)}
FLAGS={**ARMS,'legacy_global_identity':(False,False)}
SIGMAS=(0.,5.,10.,15.,20.)


def adapt(task,parameters,transfer,cross_floor,parameter_sha):
    """All nuisance fitting ends here, before any query signal is supplied."""
    base,obs=parameters['base'],parameters['obs'];s=tensor(task['support']['positions'])
    bins=[]
    for h in task['histories']:
        bins.append(cross_floor.ap_bin(dict(xy=h['positions'],**h['stats'])))
    cases=[]
    for column,ap in enumerate(task['AP_indices']):
        histories=histories_for_AP(task,ap,parameter_sha)
        cases.append(dict(histories=histories,stats=ap_stats(task['support'],column),
            moments=source_moments(histories,s,s.new_empty((0,2)),base),
            affine=[(float(transfer['rho'][b[column]]),float(transfer['bbar'][b[column]])) for b in bins]))
    stats={key:torch.stack([c['stats'][key] for c in cases]) for key in ('n','k','s1','s2')}
    states={};failures={}
    for name,(local,affine) in FLAGS.items():
        started=time.perf_counter()
        try:
            transforms=[c['affine'] if affine else [(1.,0.)]*len(c['histories']) for c in cases]
            matrices=[support_matrices(c['moments'],t) for c,t in zip(cases,transforms)]
            means,covariances,residual=(torch.stack(parts) for parts in zip(*matrices))
            if name!='legacy_global_identity':
                state=fit_joint(means,covariances,residual,stats,obs,local=local,transfer_variance=base['drift']**2)
                state.update(transforms=transforms,seconds=time.perf_counter()-started,criterion='Joint S-only full-covariance ELBO; continuous sigma in [0,20]')
                states[name]=state
                continue
            weights,trials=fit_weights(means,covariances,residual,stats,obs,local=local,transfer_variance=base['drift']**2)
            sm,sk=combine(means,covariances,residual,weights,local,base['drift']**2)
            sigma_scores=[float(marginal_loglik(sm,sk.diagonal(dim1=-2,dim2=-1)+sigma**2,**stats,obs=obs).sum()) for sigma in SIGMAS]
            sigma=SIGMAS[int(np.argmax(sigma_scores))]
            sk=sk+sigma**2
            posterior=infer_batch(sm,sk,stats,obs)
            states[name]=dict(weights=weights,trials=trials,sigma=sigma,sigma_scores=sigma_scores,
                sm=sm,sk=sk,posterior=posterior,transforms=transforms,seconds=time.perf_counter()-started,
                criterion='Legacy two-stage weights at sigma=0, then composite-marginal sigma grid')
        except (RuntimeError,ValueError,FloatingPointError) as error:
            failures[name]=dict(stage='S_adaptation',reason=repr(error))
    try:
        sm=s.new_full((len(cases),len(s)),base['mean'])
        sk=rbf(s,s,base['amplitude'],base['length']).expand(len(cases),-1,-1)
        states['GP_S']=dict(sm=sm,sk=sk,posterior=infer_batch(sm,sk,stats,obs))
    except (RuntimeError,ValueError,FloatingPointError) as error:
        failures['GP_S']=dict(stage='S_adaptation',reason=repr(error))
    return dict(cases=cases,states=states,failures=failures,parameters=parameters)


def finish_scores(scores,points):
    probabilities=torch.softmax(scores,dim=1)
    predictions=probabilities@points
    distance=torch.linalg.vector_norm(points[None]-predictions[:,None],dim=-1)
    sorted_distance,order=distance.sort(1)
    cumulative=probabilities.gather(1,order).cumsum(1)
    index=(cumulative<.9).sum(1).clamp_max(len(points)-1)
    radius=sorted_distance.gather(1,index[:,None]).squeeze(1)
    return dict(predictions=predictions.numpy().tolist(),radius90=radius.numpy().tolist(),
                radius_meaning='Disk around reported mean containing at least 90% of the model discrete candidate mass; empirical coverage must be checked separately.')


def predict(task,query_rss,adapted,transfer,cross_floor,reference_selection):
    """Score each full scan independently; only the caller evaluates coordinates."""
    parameters=adapted['parameters'];base,obs=parameters['base'],parameters['obs']
    s=tensor(task['support']['positions']);points=task['points'];q=tensor(query_rss)
    states=adapted['states'];failures=dict(adapted['failures']);results={}
    scores={name:points.new_zeros((len(q),len(points))) for name in states}
    for column,case in enumerate(adapted['cases']):
        moments=source_moments(case['histories'],s,points,base)
        for name,state in states.items():
            if name in failures:continue
            try:
                if name=='GP_S':
                    field=dict(qm=points.new_full((len(points),),base['mean']),
                               qv=points.new_full((len(points),),base['amplitude']**2),
                               cross=rbf(points,s,base['amplitude'],base['length']))
                else:
                    field=fuse(moments,state['transforms'][column],state['weights'],
                               local=FLAGS[name][0],transfer_variance=base['drift']**2,intercept_std=state['sigma'])
                posterior={k:state['posterior'][k][column] for k in ('mean','cov')}
                pred=predict_from_posterior(field['qm'],field['qv'],field['cross'],state['sm'][column],state['sk'][column],posterior)
                if not all(torch.isfinite(v).all() for v in pred.values()) or bool((pred['variance']<0).any()):
                    raise FloatingPointError('Invalid propagated moments')
                scores[name]+=observation_log_prob(q[:,column,None],pred['mean'][None],pred['variance'][None],obs)
            except (RuntimeError,ValueError,FloatingPointError) as error:
                failures[name]=dict(stage='full_scan_field',AP=task['AP_indices'][column],reason=repr(error))
    for name,state in states.items():
        if name in failures:continue
        if not torch.isfinite(scores[name]).all():
            failures[name]=dict(stage='full_scan_score',reason='Nonfinite complete score');continue
        results[name]=finish_scores(scores[name],points)
        results[name]['posterior']=dict(approximation='full Gaussian VI',
            max_projected_gradient=max(state['posterior']['projected_gradient']),
            optimizer_success=state['posterior']['optimizer_success'],iterations=state['posterior']['iterations'])
        if name!='GP_S':
            results[name]['adaptation']={k:state[k] for k in ['weights','trials','sigma','sigma_scores','seconds','criterion'] if k in state}
    for name in ('TPM_pool','SCMT_pool'):
        selected=reference_selection['selections'][name]
        try:
            prior=tpm_adapter.history_map(task['histories'],transfer if name=='TPM_pool' else None,cross_floor)
            prototypes,locations=tpm_adapter.calibrate(prior,task['support'],selected['field'])
            probability=tpm_adapter.tobit_weights(query_rss,prototypes,*selected['match'])
            results[name]=dict(predictions=(probability@locations).tolist())
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as error:
            failures[name]=dict(stage='reference_prediction',reason=repr(error))
    for name in ('WKNN_HS','WKNN_S'):
        reference=task['support']
        if name=='WKNN_HS':
            reference=group_scans(np.concatenate([h['rss'] for h in task['histories']]+[reference['rss']]),
                                  np.concatenate([h['xy'] for h in task['histories']]+[reference['xy']]))
        prediction=wknn(reference,dict(rss=query_rss),np.arange(len(task['AP_indices'])),False)
        results[name]=dict(predictions=prediction.tolist())
    assert not (set(results)&set(failures))
    return dict(results=results,failures=failures,scope='Predictions only; Q coordinates and outcome metrics are absent')
