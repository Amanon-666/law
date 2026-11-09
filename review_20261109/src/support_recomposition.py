"""S-conditioned ReFi rules, using a neural-embedding GP prediction pattern.

Learn (H, S) -> per-AP history recomposition, rather than a fixed H-only rule.
This is a conditional predictor: S generates the rule AND enters field updating.
It is not a full Bayesian posterior for an observation-independent prior.
No localization benefit has been demonstrated. Original defaults stay intact.

The 341 shared parameters comprise an 18->8 tanh set encoder, an 8->21
bias-free head and the original 21 parameters. No target parameter fitting.
Reference: Iwata & Tanaka, Machine Learning 111 (2022), 1239-1257, §§3.3-3.4.
This implementation is independent; the original paper did not model Wi-Fi H.
"""
import math
import torch
from torch import nn
from conditional_shape import ConditionalShape, prepare_shape, shape_prior
from recomposition import Parameters, condition_field
from observations import scan_log_prob


class Rule:
    """A differentiable view; do not construct Parameters or detach its tensors."""
    values = Parameters.values
    normalize = Parameters.normalize
    weights = Parameters.weights
    local_eta = ConditionalShape.local_eta

    def __init__(self, model, increment):
        self.theta = model.theta + increment[:12]
        self.shape = model.shape + increment[12:]
        self.center, self.scale, self.mask = model.center, model.scale, model.mask


class SupportRecomposition(ConditionalShape):
    def __init__(self, center, scale):
        super().__init__(center, scale)
        self.encoder = nn.Sequential(nn.Linear(18, 8), nn.Tanh()).double()
        self.head = nn.Linear(8, 21, bias=False).double()
        nn.init.zeros_(self.head.weight)

    def condition(self, tokens):
        if not tokens.shape[1]:
            return Rule(self, self.theta.new_zeros(21))
        normalized = torch.cat((self.normalize(tokens[..., :9]), tokens[..., 9:]), -1)
        representation = self.encoder(normalized).mean((0, 1))
        return Rule(self, self.head(representation))

    def penalty(self):
        return super().penalty() + .001 * sum(
            p.square().sum() for module in (self.encoder, self.head) for p in module.parameters())


@torch.no_grad()
def context_tokens(block, stats, sx, length):
    """Only H at S and measured S statistics; no query inputs or fitted labels.

    Detected-RSS differences are features, NOT unbiased latent RSS residuals.
    k=0 has an explicit mask and zero innovation; no fictitious RSS is used.
    Empty or all-missing H bypasses this encoder in context_prior.
    """
    sources, k = block['sm'].shape
    if not k:
        return sx.new_empty((sources, 0, 18))
    n, detected, s1, s2 = (stats[key] for key in ('n', 'k', 's1', 's2'))
    has = detected > 0
    repeated = detected > 1
    denominator = detected.clamp_min(1)
    average = s1 / denominator
    variance = ((s2 - s1.square() / denominator) / denominator).clamp_min(0)
    innovation = torch.where(has[None], average[None] - block['sm'], 0.) / 20
    scan_features = torch.stack((detected/n, has, n.log1p()/math.log(4),
                                torch.where(repeated, variance.sqrt()/20, 0.), repeated), -1)
    relative = (sx - sx.mean(0)) / length
    return torch.cat((block['zs'], relative[None].expand(sources, -1, -1),
                      ((block['sm']+75)/20)[..., None], innovation[..., None],
                      scan_features[None].expand(sources, -1, -1)), -1)


@torch.no_grad()
def attach_context(block, stats, sx, length):
    if not block['active']:
        return block
    return dict(block, context=context_tokens(block, stats, sx, length))


@torch.no_grad()
def prepare_context(histories, positions, delta, sx, grid, base, intercept, stats):
    block = prepare_shape(histories, positions, delta, sx, grid, base, intercept)
    return attach_context(block, stats, sx, base['length'])


def context_prior(model, block, weights):
    rule = model.condition(block['context']) if block['active'] else model
    return shape_prior(rule, block, weights)


def context_scores(model, case, weights, obs):
    prediction, ess = condition_field(context_prior(model, case['blocks'], weights), case, obs)
    return scan_log_prob(case['rss'], prediction['mean'], prediction['variance'], obs), ess


@torch.no_grad()
def refresh_context_case(model, case, weights, obs, seed):
    """S-only numerical proposal for the conditional predictor's field update."""
    from posterior import proposal
    p = context_prior(model, case['blocks'], weights)
    k = len(p['sm'])
    if not k:
        return dict(case, proposal=None), dict(sampler='no_support')
    numerical, info = proposal(dict(sm=p['sm'], sk=p['sk'],
        phi_s=p['sm'].new_zeros((k, 0)), **case['stats']), obs, p['sm'].new_empty(0), seed)
    return dict(case, proposal=(numerical['samples'], numerical['log_proposal'])), info


@torch.no_grad()
def build_context_field(model, prepared_cases, weights, obs, grid):
    """Same cached-field interface as recomposition_predict.locate; no Q data."""
    means, variances, diagnostics = [], [], []
    for ap, case in prepared_cases.items():
        prediction, ess = condition_field(context_prior(model, case['blocks'], weights), case, obs)
        means.append(prediction['mean']); variances.append(prediction['variance'])
        diagnostics.append(dict(ap=ap, ESS=ess))
    return dict(APs=list(prepared_cases), grid=grid, mean=torch.stack(means),
                variance=torch.stack(variances), obs=obs, diagnostics=diagnostics)


def load_parameters(path):
    state = torch.load(path, weights_only=True)['model']
    model = SupportRecomposition(state['center'], state['scale'])
    model.load_state_dict(state)
    return model.eval()
