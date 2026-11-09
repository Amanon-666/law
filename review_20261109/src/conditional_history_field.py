"""Source-preserving conditional neural field, an experimental ReFi alternative.

ConvCNP-style density/value encoding; shared source-set processing; the existing
censored observation model and position readout. S conditions the predictive
field once: this is NOT a GP prior followed by another S likelihood update.
No localization benefit or Bayesian sequential consistency is established.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from observations import scan_log_prob


def _point_values(scans, aps, base):
    st = scans['stats']
    n = st['n'][:, None]
    k = st['k'][:, aps]
    seen = k > 0
    mean = st['s1'][:, aps] / np.where(seen, k, 1)
    variance = st['s2'][:, aps] / np.where(seen, k, 1) - mean**2
    assert not variance.size or variance.min() > -1e-7
    return np.stack((k/n,
        np.where(seen, (mean-base['mean'])/base['amplitude'], 0),
        np.where(seen, np.maximum(variance, 0)/base['amplitude']**2, 0),
        np.broadcast_to(np.log1p(n), k.shape), seen.astype(float)), axis=-1)


def prepare_context(histories, support, base, obs, step=2., padding=2, device='cpu'):
    """Only grouped H/S acquisitions enter AP selection, geometry and features.

    Inputs use uji.group_scans semantics, including actual non-detections and
    repeat counts. All sources must already share coordinate and AP alignment.
    The relative lattice differs from the old globally rounded ReFi lattice.
    """
    scans = list(histories.values()) + [support]
    locations = np.concatenate([c['positions'] for c in scans])
    if not len(locations): raise ValueError('H/S contain no positions defining a domain')
    seen = np.logical_or.reduce([(c['rss'] != 100).any(0) for c in scans])
    aps = np.flatnonzero(seen).tolist()
    low = locations.min(0)
    cells = (locations.max(0)-low)/step
    # Avoid adding a whole cell when a translated exact lattice span rounds up.
    nearest = np.rint(cells)
    cells = np.where(np.abs(cells-nearest)<1e-10,nearest,cells)
    nx, ny = (np.ceil(cells).astype(int)+1).tolist()
    axes = [np.arange(-padding, n+padding)*step+low[d] for d,n in enumerate((nx,ny))]
    xy = np.stack(np.meshgrid(*axes,indexing='xy'),-1)
    shape = xy.shape[:2]
    grid = xy[padding:padding+ny,padding:padding+nx].reshape(-1,2)
    tensor = lambda x:torch.as_tensor(x,dtype=torch.float64,device=device)
    raster_points = tensor(xy.reshape(-1,2))

    def raster(c):
        if not len(c['positions']):
            return raster_points.new_zeros((len(aps),6,*shape))
        points = tensor(c['positions'])
        weight = (-.5*((raster_points[:,None]-points[None])/step).square().sum(-1)).exp()
        density = weight.sum(-1)
        # Empty/underflowed cells carry explicit zero density, not fake RSS.
        normal = weight/density.clamp_min(torch.finfo(weight.dtype).tiny)[:,None]
        values = tensor(_point_values(c,aps,base))
        features = torch.einsum('gp,pac->agc',normal,values)
        count = torch.log1p(density)[None,:,None].expand(len(aps),-1,1)
        return torch.cat((count,features),-1).transpose(1,2).reshape(len(aps),6,*shape)

    encoded = [raster(c) for c in scans]
    return dict(histories=encoded[:-1],support=encoded[-1],APs=aps,grid=tensor(grid),
                support_positions=tensor(support['positions']),source_names=list(histories),
                base=base,obs=obs,padding=padding,shape=(ny,nx))


def support_pool(values, support):
    """Source-specific evidence at S can affect the entire map, not only local cells."""
    density = support[:,0:1].expm1()
    mass = density.sum((-2,-1),keepdim=True)
    return (values*density).sum((-2,-1),keepdim=True)/mass.clamp_min(torch.finfo(values.dtype).tiny)


class ConditionalHistoryField(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = nn.Sequential(nn.Conv2d(12,16,3,padding=1),nn.SiLU(),
                                    nn.Conv2d(16,16,3,padding=1),nn.SiLU())
        self.relate = nn.Sequential(nn.Conv2d(32,16,1),nn.SiLU())
        self.select = nn.Conv2d(16,1,1)
        self.decode = nn.Sequential(nn.Conv2d(27,16,3,padding=1),nn.SiLU(),
                                    nn.Conv2d(16,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,2,1))
        # Nonzero output weights preserve gradients to both encoders immediately.
        nn.init.normal_(self.decode[-1].weight,std=.01)
        nn.init.zeros_(self.decode[-1].bias)
        self.double()

    def forward(self, context, start=0, stop=None):
        s = context['support'][start:stop]
        representations, logits = [], []
        for h in context['histories']:
            local = self.source(torch.cat((h[start:stop],s),1))
            global_at_support = support_pool(local,s).expand_as(local)
            r = self.relate(torch.cat((local,global_at_support),1))
            representations.append(r);logits.append(self.select(r))
        if representations:
            weights = torch.stack(logits).softmax(0)
            history = (weights*torch.stack(representations)).sum(0)
        else:
            history = s.new_zeros((len(s),16,*s.shape[-2:]))
        global_support = support_pool(s[:,1:],s).expand(-1,-1,*s.shape[-2:])
        out = self.decode(torch.cat((history,s,global_support),1))
        p = context['padding']; ny,nx = context['shape']
        out = out[:,:,p:p+ny,p:p+nx].flatten(2)
        base = context['base']
        mean = base['mean']+base['amplitude']*out[:,0]
        variance = base['amplitude']**2*F.softplus(out[:,1]+math.log(math.expm1(1.)))
        return dict(mean=mean,variance=variance)

    def penalty(self):
        return sum(p.square().sum() for p in self.parameters())


@torch.no_grad()
def build_field(model, context, chunk=8):
    fields = [model(context,start,start+chunk) for start in range(0,len(context['APs']),chunk)]
    empty = context['grid'].new_empty((0,len(context['grid'])))
    return dict(APs=context['APs'],grid=context['grid'],obs=context['obs'],
        **{key:torch.cat([v[key] for v in fields]) if fields else empty for key in ('mean','variance')})


def score_chunk(field, rss, obs):
    score = field['mean'].new_zeros((len(rss),field['mean'].shape[-1]))
    for j,(mean,variance) in enumerate(zip(field['mean'],field['variance'])):
        score = score+scan_log_prob(rss[:,j],mean,variance,obs)
    return score


def predict(field, query_rss):
    """Query RSS columns follow the H/S-selected AP catalog; no true coordinates."""
    rss = torch.as_tensor(query_rss,dtype=field['grid'].dtype,device=field['grid'].device)
    scores = score_chunk(field,rss,field['obs'])
    logp = scores.log_softmax(1)
    return dict(log_probability=logp,coordinates=logp.exp()@field['grid'])


def training_loss(model, context, query, backward=False, chunk=8):
    """Position-balanced complete-scan NLL; two passes bound AP graph memory.

    Only source training calls this function with true query positions. Target
    deployment uses build_field/predict, whose APIs have no true-position input.
    This function applies no optimizer step or implicit regularizer.
    """
    grid=context['grid']; aps=context['APs']
    truth=torch.as_tensor(query['xy'],dtype=grid.dtype,device=grid.device)
    assert not (truth[:,None]==context['support_positions'][None]).all(-1).any()
    nearest=torch.cdist(truth.cpu(),grid.cpu()).argmin(1).to(grid.device)
    rss=torch.as_tensor(query['rss'][:,aps],dtype=grid.dtype,device=grid.device)
    qw=torch.as_tensor(1/(len(query['positions'])*query['stats']['n'][query['group']]),
                       dtype=grid.dtype,device=grid.device)
    with torch.no_grad():
        scores=grid.new_zeros((len(truth),len(grid)))
        for start in range(0,len(aps),chunk):
            scores+=score_chunk(model(context,start,start+chunk),rss[:,start:start+chunk],context['obs'])
    scores.requires_grad_(backward)
    logp=scores.log_softmax(1)
    loss=-(qw*logp[torch.arange(len(truth),device=grid.device),nearest]).sum()
    if backward:
        adjoint,=torch.autograd.grad(loss,scores)
        for start in range(0,len(aps),chunk):
            contribution=score_chunk(model(context,start,start+chunk),rss[:,start:start+chunk],context['obs'])
            (contribution*adjoint).sum().backward()
    return float(loss.detach())
