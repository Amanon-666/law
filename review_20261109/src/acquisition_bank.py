"""ReFi history configurations and shared acquisition offsets adapted jointly.

Configurations are an explicit finite prior, NOT a fitted continuous simplex.
The original spatial operator and source gates remain inside each configuration.
Source weights and AP-intercept scales are never fitted under a different
observation model before this call. Generic BMA is not novel. With
independent_residual=True, the AP residual field and intercept are specific to
each acquisition; historical uncertainty still links Support and Query.
"""
import torch

from acquisition_offset import AcquisitionOffset
from recomposition import prepare, prior


class AcquisitionBank:
    def __init__(self, model, blocks, stats, grid, obs, source_weights,
                 intercept_stds, config_log_mass, offsets, offset_log_mass, *,
                 independent_residual=False, **numerics):
        assert source_weights.ndim == 2 and len(source_weights) > 0
        assert intercept_stds.shape == config_log_mass.shape == (len(source_weights),)
        assert bool(torch.isfinite(source_weights).all() and (source_weights >= 0).all())
        assert bool(torch.isfinite(intercept_stds).all() and (intercept_stds >= 0).all())
        assert bool(torch.isfinite(config_log_mass).all())
        assert torch.allclose(source_weights.sum(1), torch.ones_like(config_log_mass))
        self.grid = grid
        self.independent_residual = independent_residual
        self.aps = list(blocks)
        self.source_weights = source_weights
        self.log_prior = config_log_mass.log_softmax(0)
        self.models = []
        for w, sigma in zip(source_weights, intercept_stds):
            priors = {}
            for ap, block in blocks.items():
                assert block['sm'].shape[0] == len(w)
                p = prior(model, block, w)
                cross = (p['cross']-block['rg'] if independent_residual
                         else p['cross']+sigma.square())
                priors[ap] = dict(p, sk=p['sk']+sigma.square(),
                                 cross=cross, gv=p['gv']+sigma.square())
            self.models.append(AcquisitionOffset(priors, stats, grid, obs, offsets,
                                                 offset_log_mass, **numerics))
        joint = self.log_prior + torch.stack([m.log_evidence for m in self.models])
        self.log_evidence = torch.logsumexp(joint, dim=0)
        self.log_config_mass = joint - self.log_evidence
        self.log_joint_mass = self.log_config_mass[:, None] + torch.stack(
            [m.log_support_mass for m in self.models])

    def locate(self, rss, **query_settings):
        """Mix complete scan likelihoods BEFORE location normalization."""
        if self.independent_residual and query_settings.get('same_acquisition', False):
            raise ValueError('Independent residual bank predicts a different acquisition only')
        components = torch.stack([m.locate(rss, **query_settings)['log_likelihood']
                                  for m in self.models])
        scores = torch.logsumexp(components + self.log_config_mass[:, None, None], dim=0)
        logp = scores.log_softmax(1)
        return dict(xy=logp.exp() @ self.grid, log_probability=logp, log_likelihood=scores)


def prepare_bank_blocks(task, target_minus_source):
    """Consume unadapted H/S task_inputs; never an old fitted adaptation state."""
    assert 'state' not in task
    assert len(target_minus_source) == len(task['names'])
    return {ap: prepare(case['histories'], task['positions'], target_minus_source,
                        case['sx'], task['grid'], task['base'], 0.)
            for ap, case in task['cases'].items()}


def adapt_acquisition_bank(history_scans, support, base, obs, load_history, model,
                           target_minus_source, source_weights, intercept_stds,
                           config_log_mass, offsets, offset_log_mass, **numerics):
    """H/S -> original ReFi geometry/priors -> joint acquisition adaptation.

    The loader must provide H-only fitted fields; source-learned base/obs and
    configuration priors must respect the caller's data split. S contains one
    homogeneous acquisition. No query or old S-fitted state is accepted.
    """
    from episodic_tasks import task_inputs
    task = task_inputs(history_scans, support, base, obs, load_history)
    blocks = prepare_bank_blocks(task, target_minus_source)
    return AcquisitionBank(model, blocks, {a: c['stats'] for a, c in task['cases'].items()},
                           task['grid'], obs, source_weights, intercept_stds, config_log_mass,
                           offsets, offset_log_mass, **numerics)
