"""Complete measured H evidence on the existing differentiable ReFi path.

No new transformation or observation likelihood. Historical negative fields
are loaded explicitly; the caller owns their row/parameter/cache identity.
"""
import torch
from observed_history import observed_history_task
from recomposition_coupled import prepare_cases
from recomposition_adaptation import adapted_prior
from recomposition import condition_field


def prepare(history_scans, support, base, obs, delta, load_history, load_nondetection):
    """H/S-only input assembly, with the same cases used by ReFi learning."""
    task=observed_history_task(history_scans,support,base,obs,load_history,load_nondetection)
    cases=prepare_cases(task,delta)
    return task,cases


@torch.no_grad()
def build_field(model,cases,state,obs,grid):
    """Deployment from the SAME checked S proposals and current prior as training.

    Inputs contain no query RSS or coordinates. Returns the established
    recomposition_predict.locate cache format; all candidate nodes remain.
    """
    values=[];diagnostics=[]
    for ap,c in cases.items():
        value,ess=condition_field(adapted_prior(model,c['blocks'],state),c,obs)
        values.append(value);diagnostics.append(dict(AP=ap,ESS=ess))
    return dict(APs=list(cases),grid=grid,obs=obs,diagnostics=diagnostics,
        **{k:torch.stack([v[k] for v in values]) for k in ('mean','variance')})
