"""Complete a missed endpoint bracket in the existing source GP profile.

Only handles the observed inward-descent boundary failure. It neither changes
quadrature/tolerances nor certifies global optimality of the profiled search.
"""
import copy
import numpy as np
from scipy.optimize import minimize_scalar

from scalar_gp_adaptation import evidence_at


def complete_boundary(fitted, blocks, stats, obs):
    if fitted['passed']:
        return fitted
    result = copy.deepcopy(fitted)
    aps = list(blocks)
    ss = {k:np.array([float(stats[a][k][0]) for a in aps]) for k in ('n','k','s1','s2')}
    means = np.stack([b['sm'][:,0].numpy() if b['active'] else np.full(2,b['base_mean']) for b in blocks.values()])
    variances = np.stack([b['ss'][:,0,0,0].numpy() if b['active'] else np.zeros(2) for b in blocks.values()])
    residual = np.array([float(b['rs'][0,0]) for b in blocks.values()])
    repairs = []
    for order in (128,256):
        fit = result['fits'][order] if order in result['fits'] else result['fits'][str(order)]
        selected = fit['selected']
        if selected['projected_derivative'] <= 1e-3:
            continue
        w, sigma = selected['weight'], selected['sigma']
        assert w in (0.,1.), 'Only the observed missed boundary bracket is handled'
        profile = next(p for p in fit['profiles'] if p['sigma']==sigma)
        grid = profile['weights']
        bracket = (grid[0],grid[1]) if w==0. else (grid[-2],grid[-1])
        def objective(value):
            return -float(evidence_at(value,sigma,means,variances,residual,ss,obs,order).sum())
        optimized = minimize_scalar(objective,bounds=bracket,method='bounded',
                                    options=dict(xatol=1e-10,maxiter=100))
        candidate = dict(sigma=sigma,weight=float(optimized.x),nll=float(optimized.fun),
            success=bool(optimized.success),kind='completed_endpoint_bracket',iterations=int(optimized.nit))
        fit['candidates'].append(candidate)
        best = min(fit['candidates'],key=lambda c:c['nll'])
        bw, bs = best['weight'],best['sigma']
        lo,hi = max(0.,bw-1e-5),min(1.,bw+1e-5)
        def selected_objective(value):
            return -float(evidence_at(value,bs,means,variances,residual,ss,obs,order).sum())
        derivative = (selected_objective(hi)-selected_objective(lo))/(hi-lo)
        projected = 0. if (bw==0. and derivative>0.) or (bw==1. and derivative<0.) else abs(derivative)
        fit['selected'] = dict(best,derivative=derivative,projected_derivative=projected)
        repairs.append(dict(order=order,bracket=bracket,old=selected,new=fit['selected']))
    low,high = [result['fits'][o]['selected'] if o in result['fits'] else result['fits'][str(o)]['selected'] for o in (128,256)]
    evidence = {o:float(evidence_at(high['weight'],high['sigma'],means,variances,residual,ss,obs,o).sum()) for o in (128,256)}
    result.update(selected=high,evidence=evidence,boundary_repairs=repairs,
        passed=(low['success'] and high['success'] and max(low['projected_derivative'],high['projected_derivative'])<=1e-3
                and low['sigma']==high['sigma'] and abs(low['weight']-high['weight'])<=.005
                and abs(evidence[128]-evidence[256])<=.01))
    return result
