"""History-conditioned local contrast in the existing ReFi prior.

Replace scalar eta by eta_j(x)=.5*tanh(theta_eta+z_j(x) @ psi).
Nine extra shared parameters; psi=0 exactly contains the old 12-param model.
Every source's mean AND full covariance use I+diag(eta_j)(P_H-I).
This is a constrained conditional linear GP operator, not a coordinate warp.
It has no demonstrated localization benefit yet.
"""
import torch
from torch import nn
from fields import Prior, Transfer, rbf
from recomposition import Parameters, prepare, prior, condition_field
from observations import scan_log_prob


class ConditionalShape(Parameters):
    def __init__(self, center, scale):
        super().__init__(center, scale)
        self.shape = nn.Parameter(torch.zeros(9, dtype=torch.float64))

    def local_eta(self, features):
        return .5*(self.theta[9]+self.normalize(features)@self.shape).tanh()

    def penalty(self):
        return super().penalty()+.001*self.shape.square().sum()


@torch.no_grad()
def geometry(positions, sx, grid, base):
    """H-only anchors and same P_H as ReFi; reusable across APs/sources."""
    anchors = torch.unique(torch.cat(positions), dim=0)
    result = dict(anchors=anchors, sx=sx)
    for name,x in (("s",sx),("g",grid)):
        k = rbf(x,anchors,1.,base["length"])
        den = 1+k.sum(1)
        result[name+"w"] = k/den[:,None]
        result[name+"b"] = den.reciprocal()-1
    return result


@torch.no_grad()
def augment(block, histories, geo, base):
    """Add only DC at SS/GS; old symmetric contractions stay untouched."""
    if not block["active"]:
        return block
    covariances = []
    for history in histories:
        p = Prior(Transfer("identity"),history,dict(base,drift=0.))
        _,af = p.features(geo["anchors"])
        _,sf = p.features(geo["sx"])
        covariances.append(p.covariance(geo["anchors"],geo["sx"],af,sf))
    cross = torch.stack(covariances)
    left_ss = geo["sb"][None,:,None]*block["ss"][:,0]+torch.einsum("ia,jak->jik",geo["sw"],cross)
    left_gs = geo["gb"][None,:,None]*block["gs"][:,0]+torch.einsum("ia,jak->jik",geo["gw"],cross)
    return dict(block,left_ss=left_ss,left_gs=left_gs)


@torch.no_grad()
def prepare_shape(histories, positions, delta, sx, grid, base, intercept):
    block = prepare(histories,positions,delta,sx,grid,base,intercept)
    if not block["active"]:
        return block
    return augment(block,histories,geometry(positions,sx,grid,base),base)


def transformed(block, left, eta_x, eta_y):
    """C + E_x DC + CD* E_y + E_x DCD* E_y, source-batched."""
    right = block[:,1]-left
    ex,ey = eta_x[:,:,None],eta_y[:,None,:]
    return block[:,0]+ex*left+ey*right+ex*ey*block[:,2]


def shape_prior(model, block, weights):
    if not block["active"]:
        return prior(model,block,weights)
    _,_,lg,ls = model.values()
    a = model.weights(block["zs"],block["za"],weights)
    q = model.weights(block["zg"],block["za"],weights)
    es,eg = model.local_eta(block["zs"]),model.local_eta(block["zg"])
    sm = -75+lg.exp()*((a*(block["sm"]+es*block["sd"])).sum(0)+75)
    gm = -75+lg.exp()*((q*(block["gm"]+eg*block["gd"])).sum(0)+75)
    ss = transformed(block["ss"],block["left_ss"],es,es)
    gs = transformed(block["gs"],block["left_gs"],eg,es)
    gv = block["gv"][:,0]+eg*block["gv"][:,1]+eg.square()*block["gv"][:,2]
    scale = ls.mul(2).exp()
    return dict(sm=sm,gm=gm,
        sk=(a[:,:,None]*a[:,None,:]*ss).sum(0)*scale+block["rs"],
        cross=(q[:,:,None]*a[:,None,:]*gs).sum(0)*scale+block["rg"],
        gv=(q.square()*gv).sum(0)*scale+block["rv"])


def shape_scores(model, case, weights, obs):
    field,ess = condition_field(shape_prior(model,case["blocks"],weights),case,obs)
    return scan_log_prob(case["rss"],field["mean"],field["variance"],obs),ess


@torch.no_grad()
def refresh_case(model, case, weights, obs, seed):
    """Construct an S-only numerical proposal using this conditional prior."""
    from posterior import proposal
    p = shape_prior(model,case["blocks"],weights)
    k = len(p["sm"])
    if not k:
        return dict(case,proposal=None),dict(sampler="no_support")
    inputs = dict(sm=p["sm"],sk=p["sk"],phi_s=p["sm"].new_zeros((k,0)),**case["stats"])
    numerical,info = proposal(inputs,obs,p["sm"].new_empty(0),seed)
    return dict(case,proposal=(numerical["samples"],numerical["log_proposal"])),info


@torch.no_grad()
def build_field(model, prepared_cases, weights, obs, grid):
    """Cache H/S conditional fields; query RSS/coordinates are not required.

    Cases contain prepare_shape blocks, S stats and S-only proposal samples.
    Output uses the existing recomposition_predict.locate interface unchanged.
    """
    means,variances,diagnostics = [],[],[]
    for ap,case in prepared_cases.items():
        prediction,ess = condition_field(shape_prior(model,case["blocks"],weights),case,obs)
        means.append(prediction["mean"]);variances.append(prediction["variance"])
        diagnostics.append(dict(ap=ap,ESS=ess))
    return dict(APs=list(prepared_cases),grid=grid,mean=torch.stack(means),
                variance=torch.stack(variances),obs=obs,diagnostics=diagnostics)


def load_parameters(path):
    state = torch.load(path,weights_only=True)["model"]
    model = ConditionalShape(state["center"],state["scale"])
    model.load_state_dict(state)
    return model.eval()
