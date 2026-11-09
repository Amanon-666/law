"""Content-conditioned linear recomposition of the fused historical GP.

The neural output is a zero-sum point-to-history-anchor operator, not an S
posterior. Historical mean and full covariance undergo the same transform;
the original target discrepancy and censored S update are retained.
"""
import math
import numpy as np
import torch
from torch import nn
from conditional_history_field import prepare_context
from conditional_field_objective import at_locations
from fields import InterceptBankPrior
from fingerprint_change import prepare_fingerprint
from recomposition import condition_field
from observations import scan_log_prob
from uji import group_scans


def history_context(histories, weights, base, obs, rank=8):
    if not histories:
        return None
    columns = next(iter(histories.values()))['rss'].shape[1]
    empty = group_scans(np.empty((0, columns)), np.empty((0, 2)))
    context = prepare_context(histories, empty, base, obs)
    points = torch.as_tensor(np.unique(np.concatenate([h['positions'] for h in histories.values()]), axis=0))
    selected = [0]
    distance = (points-points[0]).square().sum(1)
    for _ in range(1, min(rank, len(points))):
        # Farthest-point ties use the lexicographic H order, also after translation.
        tolerance = 1e-12*max(1.,float(distance.max()))
        index = int(torch.nonzero(distance >= distance.max()-tolerance)[0,0])
        selected.append(index)
        distance = torch.minimum(distance, (points-points[index]).square().sum(1))
    return dict(context, weights=context['grid'].new_tensor(weights), anchors=points[selected],
                ap_index={ap:i for i,ap in enumerate(context['APs'])})


class NeuralHistoryOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Conv2d(6,8,3,padding=1),nn.SiLU(),
                                    nn.Conv2d(8,8,3,padding=1),nn.SiLU())
        self.edge = nn.Sequential(nn.Linear(19,16),nn.SiLU(),nn.Linear(16,1,bias=False))
        nn.init.zeros_(self.edge[-1].weight)
        self.double()

    def coefficients(self, context, ap, locations):
        if context is None:
            return locations.new_zeros((len(locations),0))
        z = context['anchors']; rank = len(z)
        if ap not in context['ap_index']:
            return locations.new_zeros((len(locations),rank))
        index = context['ap_index'][ap]
        encoded = sum(w*self.encode(h[index:index+1]) for w,h in zip(context['weights'], context['histories']))
        p = context['padding']; ny,nx = context['shape']
        encoded = encoded[0,:,p:p+ny,p:p+nx].flatten(1)
        points = torch.cat((locations,z))
        features = at_locations(dict(mean=encoded),context,points)['mean'].T
        local,anchor = features[:len(locations)],features[len(locations):]
        relative = (locations[:,None]-z[None])/context['base']['length']
        pairs = torch.cat((local[:,None].expand(-1,rank,-1),
                           anchor[None].expand(len(local),-1,-1),relative,
                           relative.square().sum(-1,keepdim=True)),dim=-1)
        raw = self.edge(pairs).squeeze(-1).tanh()/math.sqrt(rank)
        return raw-raw.mean(1,keepdim=True)


@torch.no_grad()
def prepare_operator(task, proposals, context):
    original = prepare_fingerprint(task,proposals)
    cases = {}
    for ap,c in original['cases'].items():
        active = (context is not None and ap in context['ap_index']
                  and any(h is not None for h in task['cases'][ap]['histories']))
        if active:
            source = task['cases'][ap]
            history = InterceptBankPrior(source['histories'],task['state']['weights'],
                                         dict(task['base'],drift=0.),0.)
            z = context['anchors']; s = source['sx']; g = task['grid']
            zm,zf = history.features(z); _,sf = history.features(s); _,gf = history.features(g)
            blocks = dict(zm=zm,zz=history.covariance(z,z,zf,zf),
                          zs=history.covariance(z,s,zf,sf),gz=history.covariance(g,z,gf,zf))
        else:
            blocks = None
        cases[ap] = dict(c,history=blocks)
    return dict(cases=cases,context=context,grid=original['grid'],support=original['support'],obs=original['obs'])


def transform_prior(prior, blocks, bs, bg):
    if blocks is None:
        return prior
    z,zz,zs,gz = (blocks[k] for k in ('zm','zz','zs','gz'))
    return dict(sm=prior['sm']+bs@z,gm=prior['gm']+bg@z,
        sk=prior['sk']+bs@zs+zs.T@bs.T+bs@zz@bs.T,
        cross=prior['cross']+bg@zs+gz@bs.T+bg@zz@bs.T,
        gv=prior['gv']+2*(bg*gz).sum(1)+(bg@zz*bg).sum(1))


def operator_prior(model, episode, ap):
    case = episode['cases'][ap]
    if case['history'] is None:
        return case['prior']
    s,g = episode['support'],episode['grid']
    coefficients = model.coefficients(episode['context'],ap,torch.cat((s,g)))
    return transform_prior(case['prior'],case['history'],coefficients[:len(s)],coefficients[len(s):])


def evaluate_operator(model, episode, query, backward=False):
    grid = episode['grid']; truth = grid.new_tensor(query['xy'])
    assert not (truth[:,None] == episode['support'][None]).all(-1).any()
    nearest = torch.cdist(truth,grid).argmin(1)
    weights = grid.new_tensor(1/(len(query['positions'])*query['stats']['n'][query['group']]))
    def score(ap,case):
        field,ess = condition_field(operator_prior(model,episode,ap),case,episode['obs'])
        return scan_log_prob(grid.new_tensor(query['rss'][:,ap]),field['mean'],field['variance'],episode['obs']),ess
    total = grid.new_zeros((len(truth),len(grid))); esses=[]
    with torch.no_grad():
        for ap,case in episode['cases'].items():
            value,ess = score(ap,case);total += value
            if ess is not None:esses.append(ess)
    total.requires_grad_(backward)
    logp = total.log_softmax(1)
    loss = -(weights*logp[torch.arange(len(truth)),nearest]).sum()
    if backward:
        adjoint, = torch.autograd.grad(loss,total)
        for ap,case in episode['cases'].items():
            value,_ = score(ap,case)
            if value.requires_grad:(value*adjoint).sum().backward()
    return dict(nll=float(loss.detach()),log_probability=logp.detach(),minimum_ESS=min(esses) if esses else None)


@torch.no_grad()
def build_operator_field(model, episode):
    """Deploy from H/S-prepared state; the query scan and its location are absent."""
    fields=[];diagnostics=[]
    for ap,case in episode['cases'].items():
        field,ess=condition_field(operator_prior(model,episode,ap),case,episode['obs'])
        fields.append(field);diagnostics.append(dict(ap=ap,ESS=ess))
    return dict(APs=list(episode['cases']),grid=episode['grid'],obs=episode['obs'],
        mean=torch.stack([f['mean'] for f in fields]),
        variance=torch.stack([f['variance'] for f in fields]),diagnostics=diagnostics)
