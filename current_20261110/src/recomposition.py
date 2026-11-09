"""ReFi v0.26 deterministic 12-parameter prior and full-scan training path.
Historical geometry/moments are cached, never target query labels. No new kernel,
observation model, source-weight solver, or query readout is introduced.
"""
import math
import torch
from torch import nn
from fields import Prior, Transfer, rbf
from observations import predict_from_posterior, observation_log_prob, scan_log_prob
from posterior import ImportanceSupport

class Parameters(nn.Module):
    def __init__(self, center, scale, mode="full"):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(12, dtype=torch.float64))
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        mask = torch.ones(12, dtype=torch.float64)
        if mode == "scales": mask[:10] = 0
        elif mode == "shape_scales": mask[:9] = 0
        elif mode != "full": raise ValueError(mode)
        self.register_buffer("mask", mask)

    def values(self):
        x = self.theta*self.mask
        return x[:9], .5*x[9].tanh(), math.log(2)*x[10].tanh(), math.log(2)*x[11].tanh()

    def normalize(self, z):
        safe = torch.where(self.scale>0, self.scale, torch.ones_like(self.scale))
        return torch.where(self.scale>0, ((z-self.center)/safe).clamp(-3,3), torch.zeros_like(z))

    def weights(self, z, za, w):
        phi, _, _, _ = self.values()
        center = (self.normalize(za)@phi).tanh().mean(1)[:,None] if za.shape[1] else 0.
        r = (self.normalize(z)@phi).tanh()-center
        # Exactly zero global source weights remain zero, as specified in v0.26.
        a = w[:,None]*r.exp()
        return a/a.sum(0)

    def penalty(self):
        phi, eta, lg, ls = self.values()
        return .001*(phi.square().sum()+eta.square()+lg.square()+ls.square())

class Source:
    def __init__(self, h, anchors, base):
        self.p = Prior(Transfer("identity"),h,dict(base,drift=0.))
        self.anchors, self.base, self.missing = anchors, base, h is None
        self.ma,self.fa = self.p.features(anchors)
        self.aa = self.p.covariance(anchors,anchors,self.fa,self.fa)

    def at(self,x):
        m,f=self.p.features(x)
        k=rbf(x,self.anchors,1.,self.base["length"])
        den=1+k.sum(1); w=k/den[:,None]; b=den.reciprocal()-1
        ca=self.p.covariance(x,self.anchors,f,self.fa)
        v=self.p.diagonal(x,f)
        return dict(x=x,m=m,f=f,w=w,b=b,ca=ca,d=b*m+w@self.ma,v=v)

    def block(self,l,r,diagonal=False):
        if diagonal:
            c=l["v"]
            lin=2*(l["b"]*c+(l["w"]*l["ca"]).sum(1))
            quad=l["b"].square()*c+2*l["b"]*(l["w"]*l["ca"]).sum(1)+((l["w"]@self.aa)*l["w"]).sum(1)
        else:
            c=self.p.covariance(l["x"],r["x"],l["f"],r["f"])
            xa=l["ca"]@r["w"].T; ay=l["w"]@r["ca"].T
            lin=l["b"][:,None]*c+ay+c*r["b"][None]+xa
            quad=l["b"][:,None]*c*r["b"][None]+l["b"][:,None]*xa+ay*r["b"][None]+l["w"]@self.aa@r["w"].T
        return torch.stack((c,lin,quad))

@torch.no_grad()
def features(points, parts, positions, delta, base):
    means=torch.stack([p["m"] for p in parts]); d=torch.stack([p["d"] for p in parts])
    difference=means-means.mean(0); contrast=-(d-d.mean(0))
    coverage=torch.stack([-torch.expm1(-rbf(points,torch.unique(x,dim=0),1.,base["length"]).sum(1)) for x in positions])
    variance=torch.stack([p["v"] for p in parts])
    missing=means.new_tensor([p["missing"] for p in parts])[:,None].expand_as(means)
    dz=means.new_tensor([0. if x is None else x for x in delta])[:,None]
    unknown=means.new_tensor([x is None for x in delta])[:,None].expand_as(means)
    return torch.stack((coverage,torch.log1p(variance),missing,difference,contrast,dz*difference,dz*contrast,dz.abs()*contrast,unknown),-1)

@torch.no_grad()
def prepare(histories, positions, delta, sx, grid, base, intercept, chunk=256):
    active=any(h is not None for h in histories)
    if not histories:
        return dict(active=False,sm=sx.new_empty((0,len(sx))),base_mean=base["mean"],
                    rs=rbf(sx,sx,base["amplitude"],base["length"])+intercept**2,
                    rg=rbf(grid,sx,base["amplitude"],base["length"])+intercept**2,
                    rv=grid.new_full((len(grid),),base["amplitude"]**2+intercept**2))
    anchors=torch.unique(torch.cat(positions),dim=0)
    sources=[Source(h,anchors,base) for h in histories]
    def at(x):
        out=[s.at(x) for s in sources]
        for p,s in zip(out,sources): p["missing"]=s.missing
        return out
    aa=at(anchors); ss=at(sx)
    za=features(anchors,aa,positions,delta,base); zs=features(sx,ss,positions,delta,base)
    out=dict(active=active,za=za,zs=zs,sm=torch.stack([s["m"] for s in ss]),
             sd=torch.stack([s["d"] for s in ss]),
             ss=torch.stack([p.block(s,s) for p,s in zip(sources,ss)]),
             rs=rbf(sx,sx,base["drift"] if active else base["amplitude"],base["length"])+intercept**2,
             base_mean=base["mean"])
    chunks=[]
    for x in grid.split(chunk):
        gg=at(x)
        chunks.append(dict(zg=features(x,gg,positions,delta,base),gm=torch.stack([g["m"] for g in gg]),
            gd=torch.stack([g["d"] for g in gg]),
            gs=torch.stack([p.block(g,s) for p,g,s in zip(sources,gg,ss)]),
            gv=torch.stack([p.block(g,g,True) for p,g in zip(sources,gg)]),
            rg=rbf(x,sx,base["drift"] if active else base["amplitude"],base["length"])+intercept**2))
    for key,axis in (("zg",1),("gm",1),("gd",1),("gs",2),("gv",2),("rg",0)):
        out[key]=torch.cat([c[key] for c in chunks],dim=axis)
    out["rv"]=grid.new_full((len(grid),),(base["drift"] if active else base["amplitude"])**2+intercept**2)
    return out

def prior(model,b,w):
    if not b["active"]:
        return dict(sm=b["sm"].new_full((b["sm"].shape[1],),b["base_mean"]),
                    gm=b["rv"].new_full(b["rv"].shape,b["base_mean"]),sk=b["rs"],cross=b["rg"],gv=b["rv"])
    phi,e,lg,ls=model.values()
    a=model.weights(b["zs"],b["za"],w); q=model.weights(b["zg"],b["za"],w)
    sm=-75+lg.exp()*((a*(b["sm"]+e*b["sd"])).sum(0)+75)
    gm=-75+lg.exp()*((q*(b["gm"]+e*b["gd"])).sum(0)+75)
    coef=torch.stack((e.new_ones(()),e,e.square()))
    sk=(a[:,:,None]*a[:,None,:]*(b["ss"]*coef[None,:,None,None]).sum(1)).sum(0)*ls.mul(2).exp()+b["rs"]
    cross=(q[:,:,None]*a[:,None,:]*(b["gs"]*coef[None,:,None,None]).sum(1)).sum(0)*ls.mul(2).exp()+b["rg"]
    gv=(q.square()*(b["gv"]*coef[None,:,None]).sum(1)).sum(0)*ls.mul(2).exp()+b["rv"]
    return dict(sm=sm,gm=gm,sk=sk,cross=cross,gv=gv)

def field(model,case,w,obs,min_ess=1000.):
    return condition_field(prior(model,case["blocks"],w),case,obs,min_ess)

def condition_field(p,case,obs,min_ess=1000.):
    """The same support posterior for any explicitly constructed joint GP prior."""
    if not len(p["sm"]): return dict(mean=p["gm"],variance=p["gv"]),None
    samples,logq=case["proposal"]
    post=ImportanceSupport(p["sm"],p["sk"],case["stats"],obs,samples,logq)
    mass=post.log_weights.exp(); ess=float(mass.detach().square().sum().reciprocal())
    if ess<min_ess: raise RuntimeError(f"proposal ESS {ess:g} below {min_ess:g}")
    mu=mass@post.samples; d=post.samples-mu
    out=predict_from_posterior(p["gm"],p["gv"],p["cross"],p["sm"],p["sk"],dict(mean=mu,cov=(d.T*mass)@d))
    if not bool(torch.isfinite(out["mean"]).all() and torch.isfinite(out["variance"]).all() and (out["variance"]>=0).all()):
        raise RuntimeError("Invalid field moments")
    return out,ess

def ap_scores(model,case,w,obs):
    f,ess=field(model,case,w,obs)
    return scan_log_prob(case["rss"],f["mean"],f["variance"],obs),ess

def evaluate(model,ep,backward=False,score_fn=ap_scores):
    with torch.no_grad():
        total=ep["grid"].new_zeros((len(ep["truth"]),len(ep["grid"])))
        esses=[]
        for c in ep["cases"].values():
            s,e=score_fn(model,c,ep["weights"],ep["obs"]);total+=s
            if e is not None: esses.append(e)
    scores=total.requires_grad_(backward)
    logp=scores.log_softmax(1)
    loss=-(ep["query_weights"]*logp[torch.arange(len(logp)),ep["nearest"]]).sum()
    xy=logp.exp()@ep["grid"]; errors=(xy-ep["truth"]).norm(dim=1)
    if backward:
        adjoint,=torch.autograd.grad(loss,scores)
        for c in ep["cases"].values():
            s,_=score_fn(model,c,ep["weights"],ep["obs"])
            if s.requires_grad:(s*adjoint).sum().backward()
        model.penalty().backward()
    return dict(nll=float(loss.detach()),mde=float(ep["query_weights"]@errors.detach()),
                min_ess=min(esses) if esses else None, errors=errors.detach().tolist())

