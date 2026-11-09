"""Explicit correlated source GP, recycled H evidence, and original ReFi T/gates.

R is a supplied valid source correlation matrix, not an inferred relationship.
All source cross-covariances survive the transform. This research interface does
not replace default ReFi adaptation or establish learned localization gains.
"""
import torch
from fields import rbf
from module_evidence import GaussianEvidence, combine_whitened
from recomposition import features, prepare, prior


class JointBank:
    def __init__(self, histories, correlation, base):
        self.histories, self.R, self.base = histories, correlation, base
        self.observed = [j for j,h in enumerate(histories) if h is not None]
        hs = [histories[j] for j in self.observed]
        modules = [GaussianEvidence(h.m.new_full(h.m.shape,base['mean']),h.chol,h.m,
                    h.c+1e-6*torch.eye(len(h.z),dtype=h.c.dtype,device=h.c.device)) for h in hs]
        rows=[]
        for j,h in zip(self.observed,hs):
            row=[]
            for k,g in zip(self.observed,hs):
                if j==k:
                    block=torch.eye(len(h.z),dtype=h.z.dtype,device=h.z.device)
                else:
                    cross=self.R[j,k]*rbf(h.z,g.z,base['amplitude'],base['length'])
                    left=torch.linalg.solve_triangular(h.chol,cross,upper=False)
                    block=torch.linalg.solve_triangular(g.chol,left.T,upper=False).T
                row.append(block)
            rows.append(torch.cat(row,1))
        self.B=torch.cat(rows,0)
        self.root=torch.linalg.cholesky(self.B)
        posterior=combine_whitened(modules,self.B)
        self.mean=posterior['mean'];self.change=posterior['cov']-self.B
        self.log_normalizer=posterior['log_normalizer']

    def projection(self,x):
        rows=[]
        for j in range(len(self.histories)):
            cross=[]
            for k in self.observed:
                h=self.histories[k]
                raw=self.R[j,k]*rbf(x,h.z,self.base['amplitude'],self.base['length'])
                cross.append(torch.linalg.solve_triangular(h.chol,raw.T,upper=False).T)
            cross=torch.cat(cross,1)
            rows.append(torch.cholesky_solve(cross.T,self.root).T)
        return torch.stack(rows)


class Geometry:
    def __init__(self,bank,positions,delta):
        self.bank,self.positions,self.delta=bank,positions,delta
        self.base=bank.base
        self.anchors=torch.unique(torch.cat(positions),dim=0)
        self.anchor_projection=bank.projection(self.anchors)
        self.kaa=rbf(self.anchors,self.anchors,self.base['amplitude'],self.base['length'])

    def points(self,x):
        base=self.base;bank=self.bank
        k=rbf(x,self.anchors,1.,base['length']);den=1+k.sum(1)
        w=k/den[:,None];b=den.reciprocal()-1
        projection=bank.projection(x)
        changed=b[None,:,None]*projection+torch.einsum('na,jad->jnd',w,self.anchor_projection)
        means=base['mean']+projection@bank.mean
        variations=changed@bank.mean
        variances=base['amplitude']**2+(projection@bank.change*projection).sum(-1)
        parts=[dict(m=means[j],d=variations[j],v=variances[j],missing=h is None)
               for j,h in enumerate(bank.histories)]
        z=features(x,parts,self.positions,self.delta,base)
        return dict(x=x,w=w,b=b,projection=projection,changed=changed,z=z,
                    ka=rbf(x,self.anchors,base['amplitude'],base['length']))

    def kernel_terms(self,left,right,diagonal=False):
        b,w=left['b'],left['w'];c,v=right['b'],right['w']
        if diagonal:
            raw=b.new_full(b.shape,self.base['amplitude']**2)
            side=(w*left['ka']).sum(1)
            linear=2*(b*raw+side)
            quadratic=b.square()*raw+2*b*side+((w@self.kaa)*w).sum(1)
        else:
            raw=rbf(left['x'],right['x'],self.base['amplitude'],self.base['length'])
            xa=left['ka']@v.T;ay=w@right['ka'].T
            linear=b[:,None]*raw+ay+raw*c[None]+xa
            quadratic=b[:,None]*raw*c[None]+b[:,None]*xa+ay*c[None]+w@self.kaa@v.T
        return torch.stack((raw,linear,quadratic))


@torch.no_grad()
def prepare_joint(histories,positions,delta,sx,grid,base,intercept,correlation,chunk=128,bank=None):
    if not any(h is not None for h in histories):
        return dict(active=False,legacy=prepare(histories,positions,delta,sx,grid,base,intercept))
    bank=JointBank(histories,correlation,base) if bank is None else bank
    geometry=Geometry(bank,positions,delta)
    support=geometry.points(sx)
    return dict(active=True,bank=bank,geometry=geometry,support=support,
                za=geometry.points(geometry.anchors)['z'],grid=grid,base=base,
                intercept=intercept,chunk=chunk,
                ss=geometry.kernel_terms(support,support))


def joint_prior(model,block,weights):
    if not block['active']:return prior(model,block['legacy'],weights)
    bank=block['bank'];geometry=block['geometry'];base=block['base']
    _,eta,lg,ls=model.values()
    coefficients=torch.stack((eta.new_ones(()),eta,eta.square()))

    def mixed(point):
        alpha=model.weights(point['z'],block['za'],weights)
        f=(alpha[:,:,None]*(point['projection']+eta*point['changed'])).sum(0)
        mean=-75+lg.exp()*(base['mean']+f@bank.mean+75)
        return alpha,f,mean

    def covariance(a,f,b,g,terms,diagonal=False):
        kernel=torch.einsum('i,i...->...',coefficients,terms)
        if diagonal:
            factor=(a*(bank.R@a)).sum(0)
            low_rank=(f@bank.change*f).sum(1)
        else:
            factor=a.T@bank.R@b
            low_rank=f@bank.change@g.T
        return ls.mul(2).exp()*(factor*kernel+low_rank)

    support=block['support'];sx=support['x']
    a,f,sm=mixed(support)
    sk=covariance(a,f,a,f,block['ss'])+rbf(sx,sx,base['drift'],base['length'])+block['intercept']**2
    means=[];crosses=[];variances=[]
    for x in block['grid'].split(block['chunk']):
        point=geometry.points(x)
        q,g,mean=mixed(point)
        cross=covariance(q,g,a,f,geometry.kernel_terms(point,support))
        cross=cross+rbf(x,sx,base['drift'],base['length'])+block['intercept']**2
        var=covariance(q,g,q,g,geometry.kernel_terms(point,point,True),True)
        var=var+base['drift']**2+block['intercept']**2
        means.append(mean);crosses.append(cross);variances.append(var)
    return dict(sm=sm,sk=sk,gm=torch.cat(means),cross=torch.cat(crosses),gv=torch.cat(variances))
