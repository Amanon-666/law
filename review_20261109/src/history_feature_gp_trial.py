"""One source-only comparison of mature fitting rules on frozen H features.

Joint and conditional likelihoods use the SAME site-average Gaussian model.
This is a deep-kernel baseline experiment, not an originality claim or a ReFi
RSS-field-transform implementation. No network updates or target evaluation.
"""
import json
import math
import signal
import socket
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from history_feature_gp import HistoryFeatureGP, squared_distance, site_average
from history_feature_cache import build_cache
from source_episode_catalog import make_episode
from tabicl_history import PACKAGE
from tabicl_multitask_full_trial import events_for, aggregate
from recomposition_trial import ROOT, CSV, digest, write, now

SOURCE = ROOT/'results/tabicl_multitask_full_trial_20261107_checked'
CATALOG = ROOT/'results/source_episode_catalog_20261105/catalog.json'
CHECKPOINT = SOURCE/'full_step_24.ckpt'
OUT = ROOT/'results/history_feature_gp_trial_20261107'
SEED = 1150546  # Same official ensemble seed as the retained full H+S model.
WALL_SECONDS = 900
STARTS = [(10.,1.,5.), (5.,.1,5.), (20.,10.,10.)]
BOUNDS = [(math.log(.1),math.log(100.)),
          (math.log(.01),math.log(100.)),(math.log(.1),math.log(100.))]
CODE = ('history_feature_gp_trial.py','history_feature_gp.py','history_feature_cache.py',
        'tabicl_history_features.py','tabicl_history.py','tabicl_history_learning.py',
        'tabicl_multitask_full_trial.py','source_episode_catalog.py','recomposition_trial.py')


def gaussian_nll(residual, covariance):
    """Two independent coordinate outputs with one shared covariance."""
    root = torch.linalg.cholesky(covariance)
    return (len(residual)*math.log(2*math.pi) + 2*root.diag().log().sum()
            + .5*(residual*torch.cholesky_solve(residual,root)).sum())


class CachedEpisode:
    def __init__(self, event, cache, history_bounds):
        self.event = event
        lookup = {int(row):i for i,row in enumerate(cache['row_ids'])}
        si = [lookup[i] for i in event['support_rows']]
        qi = [lookup[i] for i in event['query_rows']]
        self.sm = torch.from_numpy(cache['mean'][si]).double()
        self.qm = torch.from_numpy(cache['mean'][qi]).double()
        self.sy = torch.from_numpy(cache['xy'][si]).double()
        self.qy = torch.from_numpy(cache['xy'][qi]).double()
        phi = torch.from_numpy(cache['phi'][si+qi]).double()
        self.distance = squared_distance(phi,phi)
        self.ns = len(si)
        sl, sa = site_average(self.sy); ql, qa = site_average(self.qy)
        assert len(sl)==event['K']
        assert not set(map(tuple,sl.numpy())).intersection(map(tuple,ql.numpy()))
        self.average = torch.block_diag(sa,qa)
        self.residual = torch.cat((sl,ql))-self.average@torch.cat((self.sm,self.qm))
        self.sites = len(sl); self.qsites = len(ql)
        self.identity = torch.eye(len(self.residual),dtype=torch.float64)
        bounds = np.asarray(history_bounds)
        self.bounds = np.stack((np.minimum(bounds[0],self.sy.numpy().min(0)),
                                np.maximum(bounds[1],self.sy.numpy().max(0))))
        _, group, counts = np.unique(self.qy.numpy(),axis=0,return_inverse=True,return_counts=True)
        self.weights = 1/(len(counts)*counts[group])

    def loss(self, model, objective):
        kernel = model.kernel_from_distance(self.distance)
        variance = self.average@kernel@self.average.T + model.log_scales[2].exp().square()*self.identity
        joint = gaussian_nll(self.residual,variance)
        if objective=='joint':
            return joint/len(self.residual)
        assert objective=='conditional'
        marginal = gaussian_nll(self.residual[:self.sites],variance[:self.sites,:self.sites])
        return (joint-marginal)/self.qsites

    def predict(self, model):
        return model.posterior_from_distance(self.sm,self.sy,self.qm,
            self.distance[:self.ns,:self.ns],self.distance[self.ns:,:self.ns])


def verify_objectives():
    """Necessary check: cached update and difference-of-likelihood objectives."""
    model=HistoryFeatureGP()
    xy=np.array([[0.,0.],[0.,0.],[2.,1.],[2.,1.]])
    cache=dict(row_ids=np.arange(4),xy=xy,mean=np.array([[.3,.1],[.5,.2],[1.,2.],[2.,3.]]),
               phi=np.array([[0.,1.],[1.,0.],[.5,-1.],[2.,1.]]))
    event=dict(support_rows=[0,1],query_rows=[2,3],K=1)
    p=CachedEpisode(event,cache,[[-10,-10],[10,10]])
    direct=model.posterior(p.sm,torch.from_numpy(cache['phi'][:2]),p.sy,
                           p.qm,torch.from_numpy(cache['phi'][2:]))
    cached=p.predict(model)
    for key in direct:torch.testing.assert_close(direct[key],cached[key],atol=1e-11,rtol=0)
    V=p.average@model.kernel_from_distance(p.distance)@p.average.T+25*p.identity
    cross=V[1:,:1]; root=torch.linalg.cholesky(V[:1,:1])
    residual=p.residual[1:]-cross@torch.cholesky_solve(p.residual[:1],root)
    cov=V[1:,1:]-cross@torch.cholesky_solve(cross.T,root)
    direct_conditional=gaussian_nll(residual,cov)
    difference=p.loss(model,'conditional')
    torch.testing.assert_close(difference,direct_conditional,atol=1e-11,rtol=0)
    errors={}
    initial=model.log_scales.detach().clone(); direction=torch.tensor([.2,-.7,.4],dtype=torch.float64)
    for objective in ('joint','conditional'):
        loss=p.loss(model,objective)
        grad,=torch.autograd.grad(loss,model.log_scales)
        values=[]
        for sign in (1,-1):
            with torch.no_grad():model.log_scales.copy_(initial+sign*1e-5*direction)
            values.append(float(p.loss(model,objective).detach()))
        with torch.no_grad():model.log_scales.copy_(initial)
        errors[objective]=abs((values[0]-values[1])/2e-5-float(grad@direction))
        assert errors[objective]<1e-7
    return dict(cached_update=True,conditional_identity=True,directional_gradient_errors=errors)


def optimize(episodes, objective):
    runs=[]
    for start in STARTS:
        model=HistoryFeatureGP(); evaluations=0
        def value_and_gradient(theta):
            nonlocal evaluations
            with torch.no_grad():model.log_scales.copy_(torch.from_numpy(theta))
            loss=torch.stack([p.loss(model,objective) for p in episodes]).mean()
            gradient,=torch.autograd.grad(loss,model.log_scales)
            assert torch.isfinite(loss) and torch.isfinite(gradient).all()
            evaluations+=1
            return float(loss.detach()),gradient.detach().numpy()
        result=minimize(value_and_gradient,np.log(start),jac=True,method='L-BFGS-B',bounds=BOUNDS,
                        options=dict(maxiter=120,maxls=30,ftol=1e-11,gtol=1e-6))
        value,gradient=value_and_gradient(result.x)
        projected=gradient.copy()
        for j,(low,high) in enumerate(BOUNDS):
            if result.x[j]<=low+1e-8 and gradient[j]>0:projected[j]=0
            if result.x[j]>=high-1e-8 and gradient[j]<0:projected[j]=0
        runs.append(dict(start=start,theta=result.x.tolist(),scales=np.exp(result.x).tolist(),
                         loss=value,success=bool(result.success),message=str(result.message),
                         iterations=int(result.nit),evaluations=evaluations,
                         gradient_max=float(np.abs(gradient).max()),
                         projected_gradient_max=float(np.abs(projected).max()),
                         at_boundary=[bool(min(abs(result.x[j]-a),abs(result.x[j]-b))<1e-6)
                                      for j,(a,b) in enumerate(BOUNDS)]))
    acceptable=[r for r in runs if r['success'] and r['projected_gradient_max']<=1e-4]
    write(OUT/f'{objective}_optimization.json',dict(runs=runs,accepted=len(acceptable)))
    assert acceptable, f'{objective}: no converged source fit; no validation-driven retry'
    selected=min(acceptable,key=lambda r:r['loss'])
    model=HistoryFeatureGP()
    with torch.no_grad():model.log_scales.copy_(torch.tensor(selected['theta'],dtype=torch.float64))
    return model,dict(selected=selected,runs=runs)


def score(episodes,models):
    records=[]
    for p in episodes:
        event=p.event; predictions={'history_mean':p.qm.numpy()}
        with torch.no_grad():
            for label,model in models.items():predictions[label]=p.predict(model)['mean'].numpy()
        with np.load(SOURCE/f'step_24_{event["id"]}.npz') as old:
            assert np.array_equal(old['query_rows'],event['query_rows'])
            assert np.array_equal(old['truth'],p.qy.numpy())
            assert np.array_equal(old['weights'],p.weights)
            predictions['full_HS']=old['raw'].copy()
            assert np.allclose(np.clip(old['raw'],*p.bounds),old['projected'],rtol=0,atol=1e-10)
        errors={label:np.linalg.norm(np.clip(pred,*p.bounds)-p.qy.numpy(),axis=1)
                for label,pred in predictions.items()}
        path=OUT/f'validation_{event["id"]}.npz'
        np.savez(path,truth=p.qy.numpy(),weights=p.weights,query_rows=event['query_rows'],
                 bounds=p.bounds,**predictions,**{label+'_error':v for label,v in errors.items()})
        records.append(dict(event=event['id'],floor=event['floor'],K=event['K'],
                            mde={k:float(p.weights@v) for k,v in errors.items()},sha256=digest(path)))
    return records


def run():
    tick=time.monotonic();OUT.mkdir(exist_ok=False)
    catalog=json.loads(CATALOG.read_text());source=json.loads((SOURCE/'complete.json').read_text())
    source_freeze=json.loads((SOURCE/'freeze.json').read_text())
    assert source['complete'] and source['formal_step']==24
    assert digest(CHECKPOINT)==source['checkpoint_sha256']['24']
    schedule,validation=events_for(catalog); fit_specs=[s for batch in schedule for s in batch]
    assert schedule==source_freeze['schedule'] and validation==source_freeze['validation']
    package={str(p.relative_to(PACKAGE)):digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package==source_freeze['package']
    for p in (CSV,CATALOG):assert digest(p)==source_freeze['inputs'][str(p)]
    freeze=dict(created=now(),seed=SEED,code={n:digest(ROOT/n) for n in CODE},package=package,
        inputs={str(p):digest(p) for p in (CSV,CATALOG,CHECKPOINT,SOURCE/'complete.json',SOURCE/'freeze.json')},
        fit_specs=fit_specs,validation_specs=validation,source_events=72,validation_events=27,
        optimizer=dict(method='L-BFGS-B',starts=STARTS,log_bounds=BOUNDS,maxiter=120,maxls=30,
                       ftol=1e-11,gtol=1e-6,accepted_projected_gradient=1e-4),
        objectives='Same site-average Gaussian; joint NLL/all sites vs (joint-S-marginal NLL)/Q sites; equal episodes/floors.',
        selection='Lowest converged SOURCE fitting objective. No validation selection; no target evaluation.',
        criterion='Conditional mean improves >=3% over both full H+S and joint GP; no floor or K mean worse than either.',
        boundaries=['Deep-kernel GP and conditional fitting are mature rules, not originality evidence',
                    'Full H-only preprocessing differs from existing H+S: system comparison, not single-module attribution',
                    'Previously exposed source validation, no unseen-building or independent-trial claim',
                    'No matching strong-GP result for this source catalog; old scores cannot substitute'],
        resources=dict(cpu=2,gpu=1,wall_seconds=WALL_SECONDS,max_gpu_fraction=.75,
                       downloads=0,network_updates=0))
    write(OUT/'freeze.json',freeze)
    for name in CODE:(OUT/name).write_bytes((ROOT/name).read_bytes())
    write(OUT/'objective_check.json',verify_objectives())
    manifest=build_cache(OUT/'features',catalog,fit_specs,validation,CHECKPOINT,SEED)
    assert manifest['model_state_sha256']==source['trained_state_sha256']
    caches={}
    for floor in catalog['source_floors']:
        with np.load(OUT/'features'/f'{floor}.npz') as cache:caches[floor]=dict(cache)
    def prepare(specs):
        return [CachedEpisode(e,caches[e['floor']],manifest['floors'][e['floor']]['history_bounds'])
                for e in (make_episode(catalog,**s) for s in specs)]
    fit_episodes=prepare(fit_specs)
    models={'fixed_gp':HistoryFeatureGP()}; fits={}
    for objective in ('joint','conditional'):
        models[objective],fits[objective]=optimize(fit_episodes,objective)
        print(json.dumps(dict(stage='fit_done',objective=objective,selected=fits[objective]['selected'])),flush=True)
    # Protected labels are only loaded into scoring objects after both fits end.
    records=score(prepare(validation),models)
    summaries={name:aggregate(records,lambda r:r['mde'][name]) for name in records[0]['mde']}
    summaries['WKNN']=source['source']['controls']['WKNN']
    assert abs(summaries['full_HS']['mean']-source['source']['full']['mean'])<1e-10
    candidate=summaries['conditional']; gains={}
    for label in ('full_HS','joint','fixed_gp','history_mean'):
        baseline=summaries[label]
        gains[label]=dict(mean=1-candidate['mean']/baseline['mean'],
                         by_floor={k:1-v/baseline['by_floor'][k] for k,v in candidate['by_floor'].items()},
                         by_K={k:1-v/baseline['by_K'][k] for k,v in candidate['by_K'].items()})
    passed=all(gains[n]['mean']>=.03 and min(gains[n]['by_floor'].values())>=0
               and min(gains[n]['by_K'].values())>=0 for n in ('full_HS','joint'))
    write(OUT/'validation.json',records)
    done=dict(complete=True,created=now(),seconds=time.monotonic()-tick,summaries=summaries,
              conditional_relative_gains=gains,source_gate_passed=passed,fits=fits,
              feature_manifest_sha256=digest(OUT/'features'/'manifest.json'),
              freeze_sha256=digest(OUT/'freeze.json'),model_state_sha256=manifest['model_state_sha256'],
              network_updates=0,target_opened=False,blind_test=False)
    write(OUT/'complete.json',done);print(json.dumps(done),flush=True)


def deadline(signum,frame):
    raise TimeoutError(f'{WALL_SECONDS}-second source-only budget exhausted')


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    torch.set_num_threads(2)
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,WALL_SECONDS)
    try:run()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),error=repr(error)))
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)
