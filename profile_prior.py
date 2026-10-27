"""History-profile mean correction on the existing LAW bank.

The encoder sees source GP moments and relative FLOOR INDEX, not AP IDs,
target RSS, physical AP positions or a metric floor height. It is pointwise
in x and permutation invariant over complete source records. The correction
is deterministic given H, so this defines a conditional GP; it does not claim
to push the nonlinear network through the random source functions.

Only the mean rule is new. Historical full covariances and the selected mature
calibration relation are reused. Model uncertainty is not added in this core.
"""
import math

import torch
from torch import nn

from bank_calibration import CalibratedBankPrior
from history_bank import BankPrior


class ProfileMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh())
        self.readout = nn.Sequential(nn.Linear(17, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, tokens):
        """[source, position, feature] -> additive dB correction [position]."""
        pooled = self.encode(tokens).mean(0)
        count = pooled.new_full((len(pooled), 1), math.log1p(len(tokens)))
        return 20 * self.readout(torch.cat((pooled, count), -1)).squeeze(-1)


class ProfileBankPrior(CalibratedBankPrior):
    def __init__(self, histories, weights, base, relative_floors, model, relation=None):
        super().__init__(histories, weights, base, relation)
        self.relative_floors = relative_floors
        self.model = model

    def features(self, x):
        if not self.sources:
            return BankPrior.features(self, x)
        parts = [p.features(x) for p in self.sources]
        mean = sum(w*m for w, (m, _) in zip(self.weights, parts))
        features = [f for _, f in parts]
        tokens = []
        for source, (m, f), dz in zip(self.sources, parts, self.relative_floors):
            variance = source.diagonal(x, f)
            if bool((variance < -1e-7).any()):
                raise RuntimeError('Invalid historical marginal variance')
            # Only numerical roundoff is removed, not missing-history uncertainty.
            tokens.append(torch.stack(((m+75)/20, variance.clamp_min(0).sqrt()/20,
                torch.full_like(m, float(dz)), torch.full_like(m, abs(float(dz))),
                torch.full_like(m, float(source.history is not None))), -1))
        return mean + self.model(torch.stack(tokens)), features

    def covariance(self, x, y, fx, fy):
        if self.relation is None:
            return BankPrior.covariance(self, x, y, fx, fy)
        # Reuse the original bank relation, not the corrected mean as a new input.
        return CalibratedBankPrior(self.histories, self.weights, self.base,
                                   self.relation).covariance(x, y, fx, fy)

    @property
    def histories(self):
        return [p.history for p in self.sources]
