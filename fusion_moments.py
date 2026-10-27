"""Reusable exact moments for the existing affine/local-fusion prior.

No approximation or new model: for identity History, P=K_xz(K_zz+eps I)^-1
and C(x,y)=K_xy+P_x(sym(C_H)-K_zz)P_y^T. The two eps terms in the existing
conditional residual and cubature covariance cancel. This avoids repeatedly
forming 2H cubature functions for every comparison arm.
"""
import torch

from fusion_prior import mixing_weights
from transfer import rbf


def source_moments(histories, support, points, base):
    kss=rbf(support,support,base['amplitude'],base['length'])
    kxs=rbf(points,support,base['amplitude'],base['length'])
    present=[h is not None for h in histories]
    ms,cs,mx,vx,cross=[],[],[],[],[]
    for h in histories:
        if h is None:
            ms.append(support.new_full((len(support),),base['mean']))
            mx.append(points.new_full((len(points),),base['mean']))
            cs.append(kss)
            vx.append(points.new_full((len(points),),base['amplitude']**2))
            cross.append(kxs)
        else:
            ps,px=h.projection(support),h.projection(points)
            delta=(h.c+h.c.T)/2-h.kernel(h.z,h.z)
            ms.append(base['mean']+ps@(h.m-base['mean']))
            mx.append(base['mean']+px@(h.m-base['mean']))
            cs.append(kss+ps@delta@ps.T)
            projected=px@delta
            vx.append(base['amplitude']**2+(projected*px).sum(1))
            cross.append(kxs+projected@ps.T)
    def stack(values,shape):
        return torch.stack(values) if values else support.new_empty(shape)
    return dict(present=present,support=support,points=points,base=base,
        source_ms=stack(ms,(0,len(support))),source_cs=stack(cs,(0,len(support),len(support))),
        source_mx=stack(mx,(0,len(points))),source_vx=stack(vx,(0,len(points))),
        source_cross=stack(cross,(0,len(points),len(support))))


def transformed(moments,transforms):
    """Freeze source transformations; absent-source fallback remains identity."""
    m=moments
    rho=m['support'].new_tensor([r if present else 1. for (r,_),present in zip(transforms,m['present'])])
    beta=m['support'].new_tensor([b if present else 0. for (_,b),present in zip(transforms,m['present'])])
    return dict(ms=-75+rho[:,None]*(m['source_ms']+75)+beta[:,None],
                cs=rho[:,None,None].square()*m['source_cs'],
                mx=-75+rho[:,None]*(m['source_mx']+75)+beta[:,None],
                vx=rho[:,None].square()*m['source_vx'],
                cross=rho[:,None,None].square()*m['source_cross'])


def fuse(moments,transforms,weights,*,local,transfer_variance,intercept_std=0.):
    m=moments;base=m['base'];s=m['support'];x=m['points']
    has_history=any(m['present']) and float(torch.as_tensor(weights).sum())>0
    scale=base['drift'] if has_history else base['amplitude']
    ss=rbf(s,s,scale,base['length'])+intercept_std**2
    xs=rbf(x,s,scale,base['length'])+intercept_std**2
    diagonal=x.new_full((len(x),),scale**2+intercept_std**2)
    if not has_history:
        return dict(sm=s.new_full((len(s),),base['mean']),sk=ss,
                    qm=x.new_full((len(x),),base['mean']),qv=diagonal,cross=xs)
    f=transformed(m,transforms)
    sa=mixing_weights(weights,f['cs'].diagonal(dim1=-2,dim2=-1),local,transfer_variance)
    xa=mixing_weights(weights,f['vx'],local,transfer_variance)
    return dict(sm=(sa*f['ms']).sum(0),
                sk=ss+(sa[:,:,None]*f['cs']*sa[:,None,:]).sum(0),
                qm=(xa*f['mx']).sum(0),qv=diagonal+(xa.square()*f['vx']).sum(0),
                cross=xs+(xa[:,:,None]*f['cross']*sa[:,None,:]).sum(0))


def support_matrices(moments,transforms):
    """Inputs to the existing S fitter, including its all-H-absent convention."""
    m=moments;base=m['base'];s=m['support']
    if not any(m['present']):
        return (s.new_full(m['source_ms'].shape,base['mean']),torch.zeros_like(m['source_cs']),
                rbf(s,s,base['amplitude'],base['length']))
    f=transformed(m,transforms)
    return f['ms'],f['cs'],rbf(s,s,base['drift'],base['length'])


if __name__=='__main__':
    import json
    from pathlib import Path
    import hashlib
    from fusion_prior import FusionPrior
    from transfer import History
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    base=dict(mean=-104.,amplitude=16.,length=6.,drift=4.)
    z=torch.tensor([[0.,0.],[3.,1.],[8.,0.]])
    h=History(z,dict(mean=torch.tensor([-50.,-65.,-85.]),cov=torch.eye(3)),base)
    s=torch.tensor([[1.,1.],[5.,0.]])
    x=torch.tensor([[0.,0.],[2.,4.],[20.,0.]])
    checks=[];maximum=0.
    for histories in ([h,None],[None,None],[],[h]):
        weights=[1/len(histories)]*len(histories) if histories else []
        transforms=[(1.2,7.)]*len(histories)
        moments=source_moments(histories,s,x,base)
        for local in (False,True):
            for intercept in (0.,5.):
                actual=fuse(moments,transforms,weights,local=local,transfer_variance=16.,intercept_std=intercept)
                prior=FusionPrior(histories,weights,base,transforms,local=local,transfer_variance=16.,intercept_std=intercept)
                sm,sf=prior.features(s);qm,qf=prior.features(x)
                expected=dict(sm=sm,sk=prior.covariance(s,s,sf,sf),qm=qm,qv=prior.diagonal(x,qf),cross=prior.covariance(x,s,qf,sf))
                for key in expected:
                    torch.testing.assert_close(actual[key],expected[key],atol=1e-9,rtol=1e-10)
                    maximum=max(maximum,float((actual[key]-expected[key]).abs().max()))
                checks.append(dict(sources=len(histories),local=local,intercept=intercept))
        for left,right in zip(support_matrices(moments,transforms),FusionPrior(histories,weights,base,transforms).source_moments(s)):
            torch.testing.assert_close(left,right,atol=1e-9,rtol=1e-10)
    out=Path(__file__).resolve().parent/'results/fusion_candidate_20261027'
    result=dict(passed=True,scope='Exact reorganization of prior moments, no new fitting or localization',
                max_gap=maximum,cases=checks,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'moment_reuse_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
