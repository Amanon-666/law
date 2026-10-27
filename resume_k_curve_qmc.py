"""Bounded numerical continuation after observed K20 PMC degeneracy.

The frozen driver/model remain intact. Only an ESS failure delegates to the
existing RSS-Gaussian Sobol integrator, using two fixed 2**18 scrambles.
No Query observations enter the numerical decision or posterior moments.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import socket

import numpy as np
import torch

import run_k_curve as original
import support_integral
from inference import predict_from_posterior

OUT = original.OUT / 'numerical_continuation'
FROZEN_FIELD = original.field
EVENT = None
BUILDING = None
POWER = 18
SEED_MASKS = (0x13579BDF, 0x2468ACE0)


@torch.no_grad()
def field(prior, case, points, obs, seed, correction=None):
    try:
        return FROZEN_FIELD(prior, case, points, obs, seed, correction)
    except RuntimeError as error:
        if not str(error).startswith('Insufficient ESS:'):
            raise
        initial_failure = str(error)
    sx, stats = case['sx'], case['stats']
    sm, sf = prior.features(sx)
    sk = prior.covariance(sx, sx, sf, sf)
    qm, qf = prior.features(points)
    qv = prior.diagonal(points, qf)
    cross = prior.covariance(points, sx, qf, sf)
    if correction is not None:
        sm = sm + correction(sx)
        qm = qm + correction(points)
    means, covariances, evidence, effective = [], [], [], []
    for mask in SEED_MASKS:
        posterior = support_integral.SupportIntegral(sm, sk, stats, obs, power=POWER,
                        seed=seed ^ mask, proposal='rss_gaussian')
        w = posterior.log_weights.exp()
        pm = w @ posterior.samples
        centered = posterior.samples-pm
        means.append(pm)
        covariances.append((centered.T*w)@centered)
        evidence.append(posterior.log_detection_evidence)
        effective.append(w.square().sum().reciprocal())
    mean_gap = float((means[0]-means[1]).abs().max())
    scale = ((covariances[0].diag()+covariances[1].diag())/2).sqrt()
    covariance_gap = float(((covariances[0]-covariances[1]).abs()/(scale[:,None]*scale[None])).max())
    evidence_gap = float((evidence[0]-evidence[1]).abs())
    accepted = (min(effective) >= 1000 and mean_gap <= .1*obs['noise_std']
                and covariance_gap <= .1 and evidence_gap <= .05)
    record = dict(event=EVENT, seed=seed, original_failure=initial_failure,
                  scramble_seeds=[seed ^ mask for mask in SEED_MASKS], samples_per_scramble=2**POWER,
                  ESS=[float(v) for v in effective], mean_gap_dBm=mean_gap,
                  covariance_gap_normalized=covariance_gap, log_evidence_gap=evidence_gap,
                  accepted=bool(accepted), query_used=False)
    with (OUT/f'{BUILDING}_fields.jsonl').open('a') as stream:
        import json
        stream.write(json.dumps(record)+'\n')
    if not accepted:
        raise RuntimeError(f'Fixed-budget QMC consistency failed: {record}')
    # Pool unnormalized importance estimates; do not average normalized
    # posterior moments equally when the two evidence estimates differ.
    mass = torch.stack(evidence).softmax(0)
    pm = sum(p*m for p,m in zip(mass,means))
    pc = sum(p*(c+(m-pm)[:,None]*(m-pm)[None]) for p,m,c in zip(mass,means,covariances))
    prediction = predict_from_posterior(qm,qv,cross,sm,sk,dict(mean=pm,cov=pc))
    if not all(torch.isfinite(v).all() for v in prediction.values()) or bool((prediction['variance']<0).any()):
        raise FloatingPointError('Invalid QMC predictive moments')
    original.log(stage='numerical_recovery', **record)
    return dict(mean=prediction['mean'].numpy(),variance=prediction['variance'].numpy(),
                ess=np.array(float(1/sum(p*p/e for p,e in zip(mass,effective)))),
                sampler=np.array('rss_gaussian_sobol_two_scrambles'),samples=np.array(2**(POWER+1)))


def freeze():
    original.verify()
    OUT.mkdir(exist_ok=True)
    target=OUT/'freeze.json'
    if target.exists():
        raise FileExistsError(target)
    check=original.OUT/'solver_check/ess_failure_ap494/qmc_check.json'
    original.write(target,dict(created=datetime.now(timezone.utc).isoformat(),
        chronology='After original ESS failures; original freeze and successful field caches unchanged.',
        code_sha256={str(p):original.digest(p) for p in [Path(__file__),Path(support_integral.__file__)]},
        input_sha256={str(p):original.digest(p) for p in [original.OUT/'freeze.json',check]},
        entry='resume_k_curve_qmc.py; explicitly substitutes only run_k_curve.field for this process',
        changes='On Insufficient ESS only, use existing SupportIntegral RSS-Gaussian proposal; same prior, likelihood, support and Gaussian-moment grid readout.',
        policy=dict(power=POWER,scrambles=2,seed_xor=list(SEED_MASKS),minimum_ESS_each=1000,
            max_mean_gap_noise_fraction=.1,max_covariance_gap_normalized=.1,max_log_evidence_gap=.05),
        criteria_scope='Engineering consistency gates fixed before continuation, chosen after the two S-only diagnostics; not preregistered in v1 and not a proof against unseen modes.',
        fail='One fixed-budget attempt. Stop failing event and retain diagnostics; no power/seed search, clipping, relaxed gates or Query-selected recovery.',
        evidence='AP494 and Pisa AP63 fixed-budget diagnostics; pure RSS-Gaussian proposal uses broad mass wasted by collapsed adapted components.',
        interpretation='QMC ESS is weight concentration, not an iid confidence interval. Repeated scrambles give a finite numerical check only. No training or method tuning.'))


def run(building):
    global EVENT, BUILDING
    original.verify()
    f=original.read(OUT/'freeze.json')
    for key in ('code_sha256','input_sha256'):
        for path,expected in f[key].items():
            if original.digest(path)!=expected:
                raise ValueError(f'Continuation input changed: {path}')
    BUILDING=building
    path=OUT/f'{building}.json'
    if path.exists():
        raise FileExistsError(path)
    m=original.manifest()
    events=[e for e in m['events'] if e['building']==building]
    record=dict(building=building,started=datetime.now(timezone.utc).isoformat(),
                completed=[],failures={},previously_completed=[e['id'] for e in events
                    if (original.OUT/e['id']/'completed.json').exists()])
    original.write(path,record)
    original.field=field
    for event in events:
        if event['id'] in record['previously_completed']:
            continue
        EVENT=event['id']
        try:
            original.run_event(event,m['base'],m['obs'])
            record['completed'].append(EVENT)
        except Exception as error:
            record['failures'][EVENT]=repr(error)
            original.log(stage='numerical_continuation_failed',event=EVENT,error=repr(error))
        original.write(path,record)
    record['finished']=datetime.now(timezone.utc).isoformat()
    record['all_events_complete']=all((original.OUT/e['id']/'completed.json').exists() for e in events)
    original.write(path,record)


if __name__=='__main__':
    if socket.gethostname()!='ray-web':
        raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','run']);parser.add_argument('--building')
    args=parser.parse_args()
    freeze() if args.stage=='freeze' else run(args.building)
