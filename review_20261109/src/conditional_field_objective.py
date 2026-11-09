"""Censored observation training for the source-preserving conditional field."""
import torch
from torch.nn import functional as F
from observations import observation_log_prob


def select_aps(context, indices):
    """AP minibatches leave the H/S-only geometry and each AP's context intact."""
    idx=torch.as_tensor(indices,dtype=torch.long,device=context['grid'].device)
    return dict(context,APs=[context['APs'][i] for i in indices],
                histories=[h.index_select(0,idx) for h in context['histories']],
                support=context['support'].index_select(0,idx))


def at_locations(field, context, locations):
    """Bilinear marginal prediction, with explicit constant boundary extension.

    The encoder grid is determined solely by H/S, never expanded using Q labels.
    At grid nodes this is exactly the field used for localization. Outside the
    rectangle it cannot express new spatial variation; no extrapolation claim.
    This interpolates marginal parameters, not a joint Gaussian process.
    """
    grid=context['grid'];ny,nx=context['shape']
    xy=torch.as_tensor(locations,dtype=grid.dtype,device=grid.device)
    low,high=grid.min(0).values,grid.max(0).values
    normal=torch.stack([2*(xy[:,d]-low[d])/(high[d]-low[d])-1 if n>1
                        else torch.zeros_like(xy[:,d]) for d,n in enumerate((nx,ny))],-1)
    request=normal[None,None].expand(len(field['mean']),-1,-1,-1)
    return {key:F.grid_sample(value.reshape(-1,1,ny,nx),request,
                mode='bilinear',padding_mode='border',align_corners=True)[:,0,0]
            for key,value in field.items() if key in ('mean','variance')}


def observation_loss(model, context, query):
    """Position-balanced mean per-AP predictive NLL, including non-detections.

    Uniform AP subsampling gives an unbiased estimate of this mean objective.
    Repeated target scans are marginal predictive terms, not a claimed joint
    likelihood integrating one shared latent value for all future repeats.
    """
    grid=context['grid']
    truth=torch.as_tensor(query['xy'],dtype=grid.dtype,device=grid.device)
    assert not (truth[:,None]==context['support_positions'][None]).all(-1).any()
    value=at_locations(model(context),context,truth)
    rss=torch.as_tensor(query['rss'][:,context['APs']].T,dtype=grid.dtype,device=grid.device)
    logp=observation_log_prob(rss,value['mean'],value['variance'],context['obs'])
    weights=torch.as_tensor(1/(len(query['positions'])*query['stats']['n'][query['group']]),
                            dtype=grid.dtype,device=grid.device)
    return -(logp.mean(0)*weights).sum()


def interpolation_check():
    """Check the new continuous readout, including its actual outside-domain rule."""
    grid=torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float64)
    context=dict(grid=grid,shape=(2,2))
    mean=torch.tensor([[0.,1.,2.,3.]],dtype=torch.float64,requires_grad=True)
    field=dict(mean=mean,variance=torch.ones_like(mean)*2)
    points=torch.tensor([[.25,.5],[2.,.5]],dtype=torch.float64)
    read=at_locations(field,context,points)
    assert torch.allclose(read['mean'],torch.tensor([[1.25,2.]],dtype=torch.float64),atol=1e-14,rtol=0)
    assert torch.equal(at_locations(field,context,grid)['mean'],mean)
    obs=dict(alpha=.9,beta=-85.,detect_scale=4.,noise_std=5.)
    # A non-saturated RSS-scale example tests the observation-loss chain.
    signal=mean-78
    def loss(m):
        r=at_locations(dict(mean=m,variance=field['variance']),context,points)
        return -observation_log_prob(torch.tensor([[-77.,100.]],dtype=torch.float64),r['mean'],r['variance'],obs).mean()
    reference=loss(signal);gradient,=torch.autograd.grad(reference,mean)
    delta=torch.zeros_like(mean);delta[0,1]=1e-5
    difference=float((loss(signal.detach()+delta)-loss(signal.detach()-delta))/(2e-5))
    gap=abs(difference-float(gradient[0,1]));assert gap<1e-7
    single=dict(grid=grid[:1],shape=(1,1))
    only=at_locations(dict(mean=mean[:,:1],variance=field['variance'][:,:1]),single,points)
    assert torch.equal(only['mean'],torch.zeros_like(only['mean']))
    return dict(exact_grid_nodes=True,bilinear_linear_field=True,explicit_border_extension=True,
                singleton_axis=True,observation_gradient_gap=gap)
