"""Small synthetic algebra/identifiability audit, never a localization result."""
import hashlib
import json
from pathlib import Path

import torch
from scipy.optimize import minimize_scalar

from fusion_prior import FusionPrior, combine, fit_weights
from history_bank import BankPrior, gaussian_batch
from inference import predict_from_posterior
from transfer import History, Prior, Transfer


def run():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.manual_seed(261027)
    base = dict(mean=-104., amplitude=16., length=6., drift=4.)
    z0 = torch.tensor([[0., 0.], [2., 0.]])
    z1 = z0 + torch.tensor([40., 0.])
    def history(z, mean):
        return History(z, dict(mean=torch.tensor(mean), cov=.1*torch.eye(2)), base)
    histories = [history(z0, [-50., -52.]), history(z1, [-70., -72.])]
    x = torch.cat((z0, z1, torch.tensor([[20., 0.], [0., 15.]])))
    weights = [.4, .6]
    identity = [(1., 0.)] * 2
    old = BankPrior(histories, weights, base)
    global_prior = FusionPrior(histories, weights, base, identity)
    om, of = old.features(x)
    gm, gf = global_prior.features(x)
    torch.testing.assert_close(gm, om)
    torch.testing.assert_close(global_prior.covariance(x,x,gf,gf), old.covariance(x,x,of,of))
    local = FusionPrior(histories, weights, base, identity, local=True, transfer_variance=16.)
    lm, lf = local.features(x)
    covariance = local.covariance(x,x,lf,lf)
    torch.testing.assert_close(covariance.diag(), local.diagonal(x,lf))
    assert torch.linalg.eigvalsh(covariance).min() > 0
    assert lf[0][0,0] > weights[0] and lf[0][1,2] > weights[1]
    # Precisely measured wrong source: a counterexample, not a performance task.
    wrong_source_target = -80.
    assert abs(lm[0]-wrong_source_target) > abs(gm[0]-wrong_source_target)
    diffuse = FusionPrior(histories,weights,base,identity,local=True,transfer_variance=1e12)
    dm,df = diffuse.features(x)
    torch.testing.assert_close(dm,gm,atol=1e-7,rtol=0)
    mats = local.source_moments(x)
    cm, cc = combine(*mats, weights, True, 16.)
    torch.testing.assert_close(cm, lm)
    torch.testing.assert_close(cc, covariance)
    # Appending another candidate must not change any previous point's field.
    part, pf = local.features(x[:2])
    torch.testing.assert_close(part,lm[:2])
    torch.testing.assert_close(local.covariance(x[:2],x[:2],pf,pf),covariance[:2,:2])
    reverse = FusionPrior(histories[::-1], weights[::-1], base, identity[::-1], local=True, transfer_variance=16.)
    rm, rf = reverse.features(x)
    torch.testing.assert_close(rm,lm)
    torch.testing.assert_close(reverse.covariance(x,x,rf,rf),covariance)
    shift = torch.tensor([100., -30.])
    moved = [history(z0+shift,[-50.,-52.]), history(z1+shift,[-70.,-72.])]
    translated = FusionPrior(moved,weights,base,identity,local=True,transfer_variance=16.)
    tm, tf = translated.features(x+shift)
    torch.testing.assert_close(tm,lm)
    torch.testing.assert_close(translated.covariance(x+shift,x+shift,tf,tf),covariance)
    # A source affine map has the same full covariance as the existing Prior.
    model = Transfer('affine').double()
    with torch.no_grad():
        model.a.fill_(1.3)
        model.b.fill_(.4)
    single = Prior(model,histories[0],base)
    affine = FusionPrior(histories[:1],[1.],base,[(1.3,8.)],local=True,transfer_variance=16.)
    am, af = affine.features(x)
    sm, sf = single.features(x)
    torch.testing.assert_close(am,sm)
    torch.testing.assert_close(affine.covariance(x,x,af,af),single.covariance(x,x,sf,sf))
    for hs, ws in [([],[]),([None,None],weights),(histories,[0.,0.])]:
        empty = FusionPrior(hs,ws,base,identity,local=True,transfer_variance=16.)
        em, ef = empty.features(x)
        expected = BankPrior([],[],base)
        bm, bf = expected.features(x)
        torch.testing.assert_close(em,bm)
        torch.testing.assert_close(empty.covariance(x,x,ef,ef),expected.covariance(x,x,bf,bf))
    # Empty S and zero posterior residual leave the prior mean unchanged.
    result = predict_from_posterior(lm,covariance.diag(),x.new_zeros((len(x),0)),
                                   x.new_zeros(0),x.new_zeros((0,0)),{})
    torch.testing.assert_close(result['mean'],lm)
    result = predict_from_posterior(lm,covariance.diag(),covariance[:,:1],lm[:1],covariance[:1,:1],
                                   dict(mean=lm[:1],cov=covariance[:1,:1]))
    torch.testing.assert_close(result['mean'],lm)
    # Check the normalized local coefficient derivative, including covariance.
    mats = tuple(v.detach() for v in mats)
    w = torch.tensor(weights,requires_grad=True)
    m,c = combine(*mats,w,True,16.)
    objective = m.square().mean()+c.square().mean()
    gradient, = torch.autograd.grad(objective,w)
    numerical=[]
    for j in range(2):
        d=torch.zeros(2);d[j]=1e-5
        vals=[]
        for sign in (1.,-1.):
            m,c=combine(*mats,w.detach()+sign*d,True,16.)
            vals.append(m.square().mean()+c.square().mean())
        numerical.append((vals[0]-vals[1])/2e-5)
    torch.testing.assert_close(gradient,torch.stack(numerical),rtol=1e-6,atol=1e-5)
    # The same prior moments are used by S fitting and deployment.
    means = -65 + 8*torch.randn(5,3,3)
    covs = .02*torch.eye(3).expand(5,3,3,3)
    residual = .02*torch.eye(3).expand(5,3,3)
    truth = torch.tensor([.6,.3,.1])
    rss,_ = combine(means,covs,residual,truth,True,16.)
    stats = dict(n=torch.full_like(rss,3),k=torch.full_like(rss,3),s1=3*rss,s2=3*rss.square())
    obs = dict(alpha=.9,beta=-100.,detect_scale=3.,noise_std=1.)
    empty_stats={key:value[:,:0] for key,value in stats.items()}
    no_s,_=fit_weights(means[:,:,:0],covs[:,:,:0,:0],residual[:,:0,:0],empty_stats,obs,
                       local=True,transfer_variance=16.)
    assert no_s==[1/3]*3
    no_h,_=fit_weights(means[:,:0],covs[:,:0],residual,stats,obs,local=True,transfer_variance=16.)
    assert no_h==[]
    fitted,trials = fit_weights(means,covs,residual,stats,obs,local=True,transfer_variance=16.)
    torch.testing.assert_close(torch.tensor(fitted),truth,atol=.02,rtol=0)
    # Equal source variance makes local normalization exactly global mixing.
    ordinary, ordinary_trials = fit_weights(means,covs,residual,stats,obs,local=False,transfer_variance=16.)
    torch.testing.assert_close(torch.tensor(fitted),torch.tensor(ordinary),atol=1e-4,rtol=0)
    # Heterogeneous variances exercise actual LOCAL fitting. With certain
    # detection the exact Gaussian marginal likelihood is an independent oracle.
    hm = means[:,:2]
    hc = torch.diag_embed(torch.tensor([[[.1,.8,2.],[2.,.6,.1]]]).expand(5,-1,-1))
    hrss,_ = combine(hm,hc,residual,torch.tensor([.6,.4]),True,1.)
    hs = dict(n=torch.full_like(hrss,3),k=torch.full_like(hrss,3),s1=3*hrss,s2=3*hrss.square())
    ho = dict(alpha=1.,beta=-10000.,detect_scale=3.,noise_std=1.)
    hw,ht = fit_weights(hm,hc,residual,hs,ho,local=True,transfer_variance=1.)
    def exact_loss(w):
        m,c = combine(hm,hc,residual,torch.tensor([w,1-w]),True,1.)
        return float(gaussian_batch(m,c,hs,1.)[2].mean())
    exact = minimize_scalar(exact_loss,bounds=(0.,1.),method='bounded',options={'xatol':1e-10})
    assert exact.success and abs(hw[0]-exact.x)<1e-4
    assert abs(min(v['objective'] for v in ht if v['stationary'])-exact.fun)<1e-7
    root = Path(__file__).resolve().parents[1]
    out = root/'results/fusion_candidate_20261027'
    out.mkdir(exist_ok=True)
    record = dict(scope=__doc__, passed=True, source_weights=fitted, true_weights=truth.tolist(),
        trials=trials, global_trials=ordinary_trials,
        local_coefficients_at_source_centres=lf[0][:,[0,2]].tolist(),
        current_global_means=gm[[0,2]].tolist(), local_means=lm[[0,2]].tolist(),
        covariance_min_eigenvalue=float(torch.linalg.eigvalsh(covariance).min()),
        derivative_max_gap=float((gradient-torch.stack(numerical)).abs().max()),
        heterogeneous_local_fit=dict(weights=hw,exact_gaussian_weight=float(exact.x),
                                     objective_gap=abs(exact_loss(hw[0])-exact.fun),trials=ht),
        wrong_source_counterexample=dict(target=wrong_source_target,global_error=float(abs(gm[0]-wrong_source_target)),
                                         local_error=float(abs(lm[0]-wrong_source_target))),
        checks=['identity reduction','single-source affine reduction','consistent diagonal/cross-covariance',
                'positive covariance','source permutation','coordinate translation','query-set invariance',
                'empty/absent/zero-weight history','empty S','single-anchor zero residual','weight derivatives',
                'S/deployment moment identity','same-variance fit reduction',
                'large transfer variance reduces to global','precise wrong source counterexample',
                'heterogeneous local S fit matches exact Gaussian likelihood'],
        limits=['No localization performance or innovation demonstrated.',
                'A precisely measured wrong source may still get excessive weight: coverage is not domain relevance.',
                'Independent-source assumption does not handle duplicated/correlated acquisitions.',
                'Absent-source fallback preserves old semantics; measured non-detections are not reconstructed from a None cache.',
                'K=1 cannot identify per-AP affine slope and offset. Source parameters must be frozen outside the target building.'],
        code_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__),root/'fusion_prior.py']})
    (out/'check.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    run()
