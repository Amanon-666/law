"""S-only calibration of the SAME ten content coefficients, two prior means.

Motivated after the v2 held-building failures and S-information calculation.
This is a mature regularized composite-likelihood control, not a claim of full
Bayesian coefficient inference or a new network. Shared coefficients pool S
evidence across APs; final AP field updates remain the existing LAW backend.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch
from scipy.optimize import minimize

from inference import observation_log_prob
from residual_probe.prepare import ROOT
from residual_probe import localization as location
from run_multi_history import digest, write

SOURCE=ROOT/'results/source_residual_probe_v1_20261026'
OUT=SOURCE/'support_calibration_v1'
RIDGE=.001
ARMS=('S_only','source_prior')


def load_support(event):
    means=[];variances=[];features=[];observations=[]
    for filename in event['case_paths']:
        path=Path(filename)
        with np.load(path,allow_pickle=False) as a:
            n,k,s1=a['n'],a['k'],a['s1']
            if not np.all(n==1) or not np.isin(k,[0,1]).all():
                raise ValueError('This frozen K5/R1 comparison requires actual single scans')
            means.append(a['sm']);variances.append(np.diag(a['sk']))
            phi=a['phi_s'];observations.append(np.where(k==1,s1,100.))
        with np.load(SOURCE/'paired_features'/path.parent.name/path.name,allow_pickle=False) as a:
            features.append(np.concatenate((phi,a['cross_s']),axis=1))
    return tuple(torch.from_numpy(np.stack(v)).double() for v in [means,variances,features,observations])


def fit(data,obs,center,starts):
    sm,sv,phi,rss=data
    def objective(value):
        theta=torch.from_numpy(value).requires_grad_()
        loss=-observation_log_prob(rss,sm+20*(phi@theta),sv,obs).mean()
        loss=loss+RIDGE*(theta-center).square().sum()
        gradient,=torch.autograd.grad(loss,theta)
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite S-calibration objective/gradient')
        return float(loss.detach()),gradient.detach().numpy()
    runs=[]
    for start in starts:
        result=minimize(objective,start.numpy(),jac=True,method='L-BFGS-B',
            options=dict(maxiter=300,maxls=30,ftol=1e-12,gtol=1e-7))
        value,gradient=objective(result.x)
        runs.append(dict(parameters=result.x.tolist(),objective=value,start=start.tolist(),
            success=bool(result.success),message=str(result.message),iterations=int(result.nit),
            gradient_max=float(np.abs(gradient).max()),
            converged=bool(result.success and np.abs(gradient).max()<1e-4)))
    completed=[r for r in runs if r['converged']]
    if not completed:raise RuntimeError(f'S calibration did not converge: {runs}')
    best=min(completed,key=lambda r:r['objective'])
    return dict(theta=best['parameters'],objective=best['objective'],runs=runs)


def freeze():
    location.verify()
    OUT.mkdir(exist_ok=False)
    manifest=json.loads((SOURCE/'prepared/manifest.json').read_text())
    models={b:SOURCE/'learning_v2'/b/'paired/model.json' for b in sorted({e['building'] for e in manifest['events']})}
    for b,p in models.items():
        model=json.loads(p.read_text())
        assert b not in model['training_buildings'] and model['feature_indices']==list(range(10))
    for event in manifest['events']:
        load_support(event)
    write(OUT/'freeze.json',dict(created=location.now(),chronology=__doc__,arms=ARMS,
        model='Existing paired10 mean correction only; w/sigma/base/obs fixed; final S posterior and full-scan localization unchanged',
        prior_means='S_only:zero; source_prior:the completed g-only leave-building-out paired coefficients',
        objective='Mean exact one-position censored observation NLL across all AP/S pairs + .001 ||theta-center||^2; ignores cross-position covariance during coefficient fitting, not a joint marginal likelihood',
        ridge=RIDGE,normalization='Same .001 coefficient penalty and 20*Phi scale as the original linear probe; fixed before these Q scores, no validation selection',
        starts='Both arms use the same two starts: zero and the source-trained vector; choose lowest S objective among converged runs',
        hypothesis='S can recalibrate strongly informed content directions and reduce source mean negative transfer; prior-only directions may retain source value. This is a candidate, not guaranteed improvement.',
        reverse_audit='Only sm,diag(sk),phi_s,n,k,s1 and H-only paired cross_s enter fits. No Q arrays loaded. Fit both models before score_event sees Q RSS/XY. All APs share one coefficient vector per event.',
        limits='Empirical-Bayes/composite point calibration; does not propagate coefficient uncertainty. Four development buildings, reused query sets; no independent confirmation.',
        code_sha256={str(p):digest(p) for p in [Path(__file__),ROOT/'residual_probe/localization.py',ROOT/'inference.py']},
        source_models={b:dict(path=str(p),sha256=digest(p)) for b,p in models.items()},
        prepared_manifest_sha256=digest(SOURCE/'prepared/manifest.json'),
        motivation_information_sha256=digest(SOURCE/'support_information.json')))
    print(json.dumps(dict(stage='frozen',events=24)),flush=True)


def run():
    record=json.loads((OUT/'freeze.json').read_text());location.verify()
    for path,sha in record['code_sha256'].items():assert digest(Path(path))==sha,path
    for item in record['source_models'].values():assert digest(Path(item['path']))==item['sha256']
    assert digest(SOURCE/'prepared/manifest.json')==record['prepared_manifest_sha256']
    manifest=json.loads((SOURCE/'prepared/manifest.json').read_text())
    obs=json.loads((SOURCE/'prepared/parameters.json').read_text())['obs']
    for event in manifest['events']:
        data=load_support(event)
        source=torch.tensor(json.loads(Path(record['source_models'][event['building']]['path']).read_text())['theta'])
        starts=[torch.zeros(10),source]
        for arm in ARMS:
            directory=OUT/event['id']/arm;directory.mkdir(parents=True,exist_ok=True)
            if not (directory/'model.json').exists():
                start=time.time();result=fit(data,obs,torch.zeros(10) if arm=='S_only' else source,starts)
                result.update(kind=arm,feature_indices=list(range(10)),numerical_policy='S_only_ESS_refresh',
                              fit_seconds=time.time()-start,query_used=False)
                write(directory/'model.json',result)
        for arm in ARMS:
            directory=OUT/event['id']/arm
            if (directory/'result.json').exists():continue
            model=json.loads((directory/'model.json').read_text());model['sha256']=digest(directory/'model.json')
            location.score_event(event,model,directory)
        summarize()


def summarize():
    rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('*/*/result.json'))]
    groups=[]
    for building in sorted({r['building'] for r in rows}):
        for arm in ARMS:
            selected=[r for r in rows if r['building']==building and r['kind']==arm]
            if selected:groups.append(dict(building=building,kind=arm,events=len(selected),
                coordinate_unit=selected[0]['coordinate_unit'],mde=float(np.mean([r['mde'] for r in selected])),
                min_ESS=min(r['min_ESS'] for r in selected)))
    location.write(OUT/'summary.json',dict(completed=len(rows)==48,groups=groups,events=rows))


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','summary']);args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='run':run()
    else:summarize()
