"""Count-conditioned original Hurdle working score; no field/reader tuning.

Chen & Liu (1997), Statistica Sinica 7:875-892, equations (1),(3),(7),(9).
For each candidate x, miss_a=(1-EPS)(1-p_a)+EPS*U_MISS and q_a=1-miss_a.
The original detected factor t_a(o) factors as q_a * [t_a(o)/q_a].
Thus L_raw = log Bern_q(D) + log mean_o prod_detected[t_a(o)/q_a].
Subtract log P_q(sum D=M) BEFORE the unchanged BETA/A temperature.
Given M, a common log-odds shift cancels in the Bernoulli factor if the
conditional intensity score t/q is held fixed. AP-specific reporting shifts
and changed conditional RSS are outside this statement. This is a working
score decomposition, not a normalized joint RSS contamination model.

Wrap an existing PreparedFrozenHurdle. Precompute with cache_counts over all
used Query receipts (both logs) once per H/S context before batching. This
uses only RSS masks; no Query coordinates/GT or original-model predictions.
Candidate multiplicities, p/m/s2, offsets, robust terms, temperature, bounds,
posterior mean and the downstream fixed GP pool remain unchanged.
"""
import numpy as np
from scipy.special import logsumexp

from frozen_hurdle_translator import PreparedFrozenHurdle, BETA, EPS, U_MISS


def poisson_binomial_logpmf(q, max_count):
    """Exact log DP, q[candidate, AP] -> log P(M=0..max_count).

    Cost O(candidates * APs * max_count), storage O(candidates * max_count).
    No Gaussian/Poisson approximation, clipping or missing-factor weighting.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or not 0 <= max_count <= q.shape[1]:
        raise ValueError('Expected candidate-by-AP probabilities and 0 <= count <= APs')
    if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError('Invalid Bernoulli probabilities')
    with np.errstate(divide='ignore'):
        detected, missing = np.log(q), np.log1p(-q)
    dp = np.full((len(q), max_count+1), -np.inf)
    dp[:, 0] = 0.
    for ap in range(q.shape[1]):
        upper = min(ap+1, max_count)
        # RHS uses only the preceding AP's states before either assignment.
        dp[:, 1:upper+1] = np.logaddexp(
            dp[:, 1:upper+1] + missing[:, ap, None],
            dp[:, :upper] + detected[:, ap, None])
        dp[:, 0] += missing[:, ap]
    return dp


class CountConditionedHurdle(PreparedFrozenHurdle):
    """Original normalized scores plus the count correction; arrays are immutable."""
    def __init__(self, prepared):
        super().__init__(*(getattr(prepared, key) for key in
            ('cand_xy', 'aps', 'p', 'm', 's2', 'bounds', 'ap_count')))
        self.q_eff = 1-((1-EPS)*(1-self.p)+EPS*U_MISS)
        self._log_counts = None

    def cache_counts(self, counts):
        """Cache every count up to the required maximum; repeated M does no DP.

        A later larger maximum rebuilds the truncated DP. For one DP per context,
        supply all needed counts once before passing the original 16-row batches.
        """
        counts = np.asarray(counts)
        if not len(counts):
            return
        if not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any() or (counts > len(self.aps)).any():
            raise ValueError('Counts must be integers on the original prepared AP axis')
        maximum = int(counts.max())
        if self._log_counts is None or maximum >= self._log_counts.shape[1]:
            self._log_counts = poisson_binomial_logpmf(self.q_eff, maximum)

    def predict(self, query_rss):
        query = np.asarray(query_rss)
        assert query.ndim == 2 and query.shape[1] == self.ap_count
        counts = (query[:, self.aps] != 100).sum(1)
        self.cache_counts(counts)
        original = super().predict(query)
        if not len(query):
            return original
        # log_position_mass = BETA/A * L_raw - a query-only normalizer.
        # Removing the count term here is algebraically the same as doing so
        # before temperature; renormalization removes the old normalizer.
        scores = original['log_position_mass'] - (BETA/len(self.aps))*self._log_counts[:, counts].T
        log_mass = scores-logsumexp(scores, axis=1, keepdims=True)
        return dict(projected=np.clip(np.exp(log_mass) @ self.cand_xy, self.bounds[0], self.bounds[1]),
                    log_position_mass=log_mass, cand_xy=self.cand_xy.copy(), aps=self.aps.copy(),
                    original_projected=original['projected'])


def enumeration_check():
    """One tiny enumeration check for DP indexing, including M=0/N and truncation.

    Execute on the server only; does not load models, data, or run localization.
    """
    from itertools import product
    q = np.array([[.1, .3, .6, .9], [.8, .2, .4, .7]])
    exact = np.zeros((len(q), 5))
    for bits in product((0, 1), repeat=4):
        d = np.array(bits, dtype=bool)
        exact[:, sum(bits)] += np.prod(np.where(d[None], q, 1-q), axis=1)
    for maximum in (0, 2, 4):
        np.testing.assert_allclose(np.exp(poisson_binomial_logpmf(q, maximum)),
                                   exact[:, :maximum+1], rtol=1e-13, atol=1e-15)
    return dict(passed=True, APs=4, candidates=2, enumerated_patterns=16,
                checked_max_counts=[0, 2, 4], fits=0, localization_predictions=0)


if __name__ == '__main__':
    import argparse
    import socket
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-enumeration', action='store_true', required=True)
    parser.parse_args()
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Run the enumeration check on ray-web only')
    print(enumeration_check())
