"""Candidate floor-only prior. One shared EB attenuation; uncertain AP relation.

No AP position/peak estimate, no learned network, no localization runner.
Run on the existing LAW Python path. See design.md for working assumptions.
"""
import math

import torch

from history_bank import BankPrior
from inference import JITTER, observation_log_prob
from operator_prior import marginal_loglik
from source_weight_mixture import field_from_prior, mixture_log_prob
from transfer import Prior, Transfer


class ShiftedBankPrior(BankPrior):
    def __init__(self, histories, weights, base, shift):
        super().__init__(histories, weights, base)
        self.shift = shift

    def features(self, x):
        mean, features = super().features(x)
        return mean + self.shift, features


class FloorProfile:
    """One AP. Probe coordinates must be H-only, registered and jointly covered.

    Components are the supplied building floors plus one null relation.
    A source-level profile discrepancy uses base['drift']; it does not shrink
    when extra correlated probe points are supplied. Missing histories do not
    contribute either floor evidence or a physical correction.
    """
    def __init__(self, histories, weights, base, source_floors, target_floor,
                 candidate_floors, probes, prior_mass=None):
        self.base = base
        self.histories = histories if any(h is not None for h in histories) else []
        self.weights = weights if self.histories else []
        self.bank = BankPrior(self.histories, self.weights, base)
        self.labels = list(candidate_floors) + [None]
        mass = probes.new_ones(len(self.labels)) if prior_mass is None else probes.new_tensor(prior_mass)
        self.log_prior = (mass / mass.sum()).log()
        available = [j for j, h in enumerate(histories) if h is not None]
        units = [sum(weights[j] * (abs(f - source_floors[j]) - abs(f - target_floor))
                     for j in available) for f in candidate_floors] + [0.]
        self.offset_units = probes.new_tensor(units)
        self.contrast = self.expected = self.root = None
        if len(available) < 2 or not len(probes):
            return
        levels, variances = [], []
        for j in available:
            p = Prior(Transfer('identity'), histories[j], dict(base, drift=0.))
            m, features = p.features(probes)
            cov = p.covariance(probes, probes, features, features)
            levels.append(m.mean())
            variances.append(cov.mean() + base['drift'] ** 2)
        levels, variances = torch.stack(levels), torch.stack(variances)
        count = len(available)
        d = torch.cat((-torch.ones((count - 1, 1), dtype=probes.dtype),
                       torch.eye(count - 1, dtype=probes.dtype)), dim=1)
        distance = probes.new_tensor([[abs(f - source_floors[j]) for j in available]
                                     for f in candidate_floors] + [[0.] * count])
        self.contrast = d @ levels
        self.expected = -distance @ d.T
        # Add numerical variance in source coordinates, preserving permutation.
        cov = (d * (variances + JITTER)[None]) @ d.T
        self.root = torch.linalg.cholesky(cov)

    def history_log_evidence(self, attenuation):
        if self.contrast is None:
            return torch.zeros_like(self.log_prior)
        residual = self.contrast[None] - attenuation * self.expected
        whitened = torch.linalg.solve_triangular(self.root, residual.T, upper=False)
        normalizer = len(self.contrast) * math.log(2 * math.pi) + 2 * self.root.diag().log().sum()
        return -.5 * (whitened.square().sum(0) + normalizer)

    def history_log_mass(self, attenuation):
        return torch.log_softmax(self.log_prior + self.history_log_evidence(attenuation), 0)

    def component(self, attenuation, index):
        return ShiftedBankPrior(self.histories, self.weights, self.base,
                                attenuation * self.offset_units[index])

    def prior_moments(self, x, attenuation):
        """H-only mixture moments; covariance retains the rank-one ambiguity."""
        mean, features = self.bank.features(x)
        cov = self.bank.covariance(x, x, features, features)
        mass = self.history_log_mass(attenuation).exp()
        shifts = attenuation * self.offset_units
        shift = mass @ shifts
        return mean + shift, cov + mass @ (shifts - shift).square()


@torch.no_grad()
def fit_attenuation(profiles, sx, statistics, obs, grid, empty_support_default):
    """All-AP conditional composite likelihood, f shared across S positions.

    The score curve is a diagnostic, not a calibrated posterior over a.
    Source weights and all profile-prior settings must already be fixed.
    """
    if not len(sx):
        return dict(attenuation=float(empty_support_default), scores=None,
                    grid=list(grid), support_identified=False)
    scores = sx.new_zeros(len(grid))
    for profile, stats in zip(profiles, statistics):
        mean, features = profile.bank.features(sx)
        var = profile.bank.diagonal(sx, features)
        for i, attenuation in enumerate(grid):
            means = mean[None] + attenuation * profile.offset_units[:, None]
            logz = marginal_loglik(means, var[None], **stats, obs=obs).sum(-1)
            scores[i] += torch.logsumexp(profile.history_log_mass(attenuation) + logz, 0)
    best = int(scores.argmax())
    return dict(attenuation=float(grid[best]), scores=scores.tolist(), grid=list(grid),
                support_identified=None, score_range=float(scores.max() - scores.min()))


@torch.no_grad()
def field_from_profile(profile, attenuation, obs, sx, stats, points, seed,
                       defensive_fraction=.1):
    """Reuse LAW Support solver per f, then use each returned S evidence ONCE.

    Keep components for the scan likelihood; Gaussian mixture moments alone
    do not specify that likelihood. a is a shared EB point estimate.
    """
    shifts = attenuation * profile.offset_units
    if bool((shifts == 0).all()):
        value = field_from_prior(profile.bank, obs, sx, stats, points, seed,
                                 defensive_fraction=defensive_fraction)
        return dict(**value, component_means=value['mean'][None],
                    component_variances=value['variance'][None],
                    log_mass=points.new_zeros(1), labels=['identity'])
    values = [field_from_prior(profile.component(attenuation, i), obs, sx, stats,
                               points, seed, defensive_fraction=defensive_fraction)
              for i in range(len(profile.labels))]
    evidence = points.new_tensor([v['log_evidence'] for v in values])
    joint = profile.history_log_mass(attenuation) + evidence
    log_mass = torch.log_softmax(joint, 0)
    means = torch.stack([v['mean'] for v in values])
    variances = torch.stack([v['variance'] for v in values])
    mean = log_mass.exp() @ means
    variance = log_mass.exp() @ (variances + (means - mean).square())
    return dict(mean=mean, variance=variance, component_means=means,
                component_variances=variances, log_mass=log_mass,
                log_evidence=float(torch.logsumexp(joint, 0)), labels=profile.labels,
                effective_samples=[v['effective_samples'] for v in values])


def scan_log_prob(rss, field, obs):
    """rss [scan,1] (or [scan,point]); output [scan,point] for ONE AP."""
    scores = observation_log_prob(rss[None], field['component_means'][:, None],
                                  field['component_variances'][:, None], obs)
    return mixture_log_prob(scores, field['log_mass'])
