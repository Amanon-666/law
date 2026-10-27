"""Post-v2 S-only local information of shared linear content coefficients.

This is conditional on frozen w/sigma/base/obs and the model's independent-AP
assumption. Observed curvature need not be positive. Rank is not a claim of
accurate localization, and redundant feature coefficients are not functions.
No Q arrays are loaded and no target coefficients are fitted.
"""
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from inference import JITTER, gaussian_update
from support_integral import ImportanceSupport
from residual_probe.prepare import ROOT
from residual_probe.learning import FEATURES

SOURCE = ROOT/'results/source_residual_probe_v1_20261026'


def load(path):
    with np.load(path,allow_pickle=False) as data:
        case={k:torch.from_numpy(data[k]).double() for k in
              ['sm','sk','phi_s','samples','log_proposal','n','k','s1','s2']}
    with np.load(SOURCE/'paired_features'/path.parent.name/path.name,allow_pickle=False) as data:
        case['phi_s']=torch.cat((case['phi_s'],torch.from_numpy(data['cross_s']).double()),1)
    return case


def evidence(theta,case,obs):
    sm=case['sm']+20*case['phi_s']@theta
    posterior=ImportanceSupport(sm,case['sk'],{k:case[k] for k in ['n','k','s1','s2']},obs,
                                case['samples'],case['log_proposal'])
    return -posterior.rss['loss']+posterior.log_detection_evidence


@torch.no_grad()
def derivatives(case,obs):
    sm,sk,D=case['sm'],case['sk'],20*case['phi_s']
    stats={k:case[k] for k in ['n','k','s1','s2']}
    post=ImportanceSupport(sm,sk,stats,obs,case['samples'],case['log_proposal'])
    w=post.log_weights.exp();mean=w@post.samples
    centered=post.samples-mean;cov=(centered.T*w)@centered
    keep=case['k']>0
    dm=D.clone();score=torch.zeros(D.shape[1]);rss_info=torch.zeros((D.shape[1],D.shape[1]))
    if keep.any():
        V=sk[keep][:,keep]+torch.diag(obs['noise_std']**2/case['k'][keep]+JITTER)
        solve=torch.linalg.solve(V,D[keep])
        dm-=sk[:,keep]@solve
        residual=case['s1'][keep]/case['k'][keep]-sm[keep]
        score+=D[keep].T@torch.linalg.solve(V,residual)
        rss_info=D[keep].T@solve
    B=torch.linalg.solve(post.rss['cov']+JITTER*torch.eye(len(sm)),dm)
    score+=B.T@(mean-post.rss['mean'])
    information=rss_info+dm.T@B-B.T@cov@B
    return score,(information+information.T)/2,D.T@D,float(w.square().sum().reciprocal())


def run():
    out=SOURCE/'support_information.json'
    if out.exists():raise FileExistsError(out)
    obs=json.loads((SOURCE/'prepared/parameters.json').read_text())['obs']
    events=json.loads((SOURCE/'prepared/manifest.json').read_text())['events']
    example=load(Path(events[0]['case_paths'][0]));theta=torch.zeros(10,requires_grad=True)
    analytic_g,analytic_i,_,_=derivatives(example,obs)
    auto_g,=torch.autograd.grad(evidence(theta,example,obs),theta)
    auto_i=-torch.autograd.functional.hessian(lambda t:evidence(t,example,obs),theta)
    torch.testing.assert_close(analytic_g,auto_g,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(analytic_i,auto_i,atol=2e-6,rtol=2e-5)
    checks=dict(score_max_error=float((analytic_g-auto_g).abs().max()),
                curvature_max_error=float((analytic_i-auto_i).abs().max()))
    rows=[];started=time.time()
    for event in events:
        score=torch.zeros(10);info=torch.zeros((10,10));gram=torch.zeros((10,10));minimum=32768.
        for path in event['case_paths']:
            g,I,G,ess=derivatives(load(Path(path)),obs)
            score+=g;info+=I;gram+=G;minimum=min(minimum,ess)
        arms={}
        for kind,indices in FEATURES.items():
            ids=torch.tensor(indices);I=info[ids[:,None],ids[None]];G=gram[ids[:,None],ids[None]]
            eigen,U=torch.linalg.eigh(G);active=eigen>eigen.max()*1e-10
            basis=U[:,active]
            local=torch.linalg.eigvalsh(basis.T@I@basis)
            arms[kind]=dict(dimensions=len(indices),S_design_rank=int(active.sum()),
                design_eigenvalues=eigen.tolist(),observed_information_eigenvalues=local.tolist(),
                positive_directions=int((local>1e-6).sum()),negative_directions=int((local< -1e-6).sum()),
                evidence_gradient_norm=float(score[ids].norm()))
        row=dict(event=event['id'],building=event['building'],J=event['J'],APs=len(event['APs']),
                 arms=arms,min_ESS=minimum)
        rows.append(row);print(json.dumps(row),flush=True)
    result=dict(scope=__doc__,chronology='Diagnostic after learning-v2 results, before any S-coefficient fitting',
        parameter_point='theta=0; conditional on frozen S-only w/sigma and source base/obs',
        derivative='Exact derivative of current fixed-proposal evidence estimate, including RSS evidence and normalized detection weights; checked against autograd',
        query_read=False,training_updates=0,checks=checks,events=rows,seconds=time.time()-started,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    run()
