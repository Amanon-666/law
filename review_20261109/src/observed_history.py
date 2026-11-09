"""Retain all-nondetection H evidence under ReFi's existing observation model.

UJI's 100 means an acquired scan did not detect the AP, not an unobserved
location. This is an explicit alternative to v0.26's documented availability
approximation; it is not a change to the original replay/default path.
With shared base/observation parameters, all-zero detection columns within
one source have identical sufficient statistics. Fit that posterior once per
source and reuse its distribution for independent AP fields, not as a shared
cross-AP random variable. No pseudo RSS or new learnable parameter is added.
"""
import numpy as np
import torch

from episodic_tasks import task_inputs
from fields import History, rbf
from history_fit import infer
from uji import tensor


def nondetection_history(scans, base, obs, steps=640):
    """H-only posterior for a column known to be undetected in every scan.

    The caller establishes that the column has this observation pattern.
    This routine does not infer AP existence, sensor failures or a target's
    future availability. As for detected histories, the posterior is a full
    Gaussian VI approximation using the original censored likelihood.
    """
    z = tensor(scans['positions'])
    if not len(z):
        return None
    counts = tensor(scans['stats']['n'])
    zeros = torch.zeros_like(counts)
    stats = dict(n=counts, k=zeros, s1=zeros, s2=zeros)
    posterior = infer(z.new_full((len(z),), base['mean']),
                      rbf(z, z, base['amplitude'], base['length']),
                      stats, obs, steps=steps)
    return History(z, posterior, base)


def observed_history_task(history_scans, support, base, obs,
                          load_history, load_nondetection):
    """Same H/S task as ReFi, with observed negatives replacing H fallbacks.

    Only the existing H/S-selected AP catalog is used. Each source is loaded
    at most once for its negative-only field. Missing positive-history cache
    files still fail in load_history; no exception turns them into negatives.
    A loaded field must correspond to this source's retained rows and origin,
    just as positive-history assets must. Query inputs are not accepted.
    Nonempty S and shared AP/coordinate identities follow task_inputs.
    """
    task = task_inputs(history_scans, support, base, obs, load_history)
    evidence = {}
    for j, name in enumerate(task['names']):
        scans = history_scans[name]
        aps = [ap for ap, case in task['cases'].items()
               if case['histories'][j] is None]
        if not aps or not len(scans['positions']):
            continue
        assert not np.any(scans['stats']['k'][:, aps])
        field = load_nondetection(name)
        if field is None:
            raise ValueError(f'{name}: observed nondetections need a fitted field')
        np.testing.assert_allclose(field.z.detach().cpu().numpy(),
                                   scans['positions'], rtol=0, atol=1e-8)
        for ap in aps:
            task['cases'][ap]['histories'][j] = field
        evidence[name] = dict(APs=aps, positions=len(scans['positions']),
                              scans=int(scans['stats']['n'].sum()))
    task['nondetection_evidence'] = evidence
    return task
