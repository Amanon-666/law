"""AP-shared finite acquisition offsets around a history-conditioned GP.

This is an inference kernel, not a fitted device model. Priors must precede
Support adaptation; old S-fitted weights/posteriors are not H-only priors.
Both detection and reported RSS depend on f + offset. The finite mixture is
exact outside the existing QMC SupportIntegral. Query rows get individual
marginals: they do not update the Support posterior or calibrate each other.
"""
import torch

from observations import observation_log_prob
from posterior import SupportIntegral


class AcquisitionOffset:
    def __init__(self, priors, stats, grid, obs, offsets, log_mass, *,
                 power=13, seed=1150548, min_ess=1000., scalar_order=None, query_conditioned=False):
        assert not query_conditioned or scalar_order is not None
        self.query_conditioned = query_conditioned
        self.priors, self.grid, self.obs = priors, grid, obs
        self.aps = list(priors)
        self.offsets = offsets
        self.log_prior = log_mass.log_softmax(0)
        assert offsets.ndim == log_mass.ndim == 1 and len(offsets) == len(log_mass) > 0
        assert bool(torch.isfinite(offsets).all() and torch.isfinite(log_mass).all())
        self.posts = []
        evidence = []
        self.minimum_ess = None
        for offset in offsets:
            posts = {}
            total = grid.new_zeros(())
            for ap, p in priors.items():
                if scalar_order is None:
                    post = SupportIntegral(p['sm'] + offset, p['sk'], stats[ap], obs,
                                           power=power, seed=(seed + int(ap)) % (2**31))
                else:
                    from scalar_support import ScalarSupportIntegral
                    post = ScalarSupportIntegral(p['sm'] + offset, p['sk'], stats[ap], obs, scalar_order)
                if post.analytic:
                    total = total - post.rss['loss']
                else:
                    if scalar_order is None:
                        ess = float(post.log_weights.detach().exp().square().sum().reciprocal())
                        if ess < min_ess:
                            raise RuntimeError(f'Acquisition offset ESS {ess:g} below {min_ess:g}')
                        self.minimum_ess = ess if self.minimum_ess is None else min(self.minimum_ess, ess)
                    total = total + post.log_detection_evidence - post.rss['loss']
                posts[ap] = post
            self.posts.append(posts)
            evidence.append(total)
        joint = self.log_prior + torch.stack(evidence)
        self.log_evidence = torch.logsumexp(joint, dim=0)
        self.log_support_mass = joint - self.log_evidence

    def _conditional_score(self, posts, offset, rss, chunk_size):
        """Integrate per-AP field conditional, keeping the common offset fixed."""
        total = self.grid.new_zeros((len(rss), len(self.grid)))
        for column, ap in enumerate(self.aps):
            p = self.priors[ap]
            values, inverse = torch.unique(rss[:, column], sorted=True, return_inverse=True)
            if self.query_conditioned:
                pieces = []
                for start in range(0, len(self.grid), chunk_size):
                    sl = slice(start, start+chunk_size)
                    pieces.append(posts[ap].conditioned_log_prob(values, p['gm'][sl]+offset,
                                                                p['gv'][sl], p['cross'][sl]))
                total = total+torch.cat(pieces, 1)[inverse]
                continue
            # Component nodes/weights are fixed by S; Query does not refit them.
            means, variance, mass = posts[ap].components(p['gm'] + offset, p['gv'], p['cross'])
            chunks = []
            for start in range(0, len(self.grid), chunk_size):
                sl = slice(start, start + chunk_size)
                lp = observation_log_prob(values[:, None, None], means[None, :, sl],
                                          variance[None, None, sl], self.obs)
                chunks.append(torch.logsumexp(mass[None, :, None] + lp, dim=1))
            total = total + torch.cat(chunks, dim=1)[inverse]
        return total

    def locate(self, rss, *, same_acquisition=False, query_offsets=None,
               query_log_mass=None, chunk_size=32):
        """RSS columns follow self.aps; no query coordinates or fitted Q offset.

        A new acquisition defaults to the same population prior, independently
        of S. An explicitly known different population can supply its own prior.
        Same-acquisition queries share the actual latent S state.
        """
        assert rss.ndim == 2 and rss.shape[1] == len(self.aps) and len(rss) > 0
        if same_acquisition:
            assert query_offsets is None and query_log_mass is None
        else:
            assert (query_offsets is None) == (query_log_mass is None)
            if query_offsets is None:
                query_offsets, query_log_mass = self.offsets, self.log_prior
            assert query_offsets.ndim == query_log_mass.ndim == 1
            assert len(query_offsets) == len(query_log_mass) > 0
            assert bool(torch.isfinite(query_offsets).all() and torch.isfinite(query_log_mass).all())
            query_log_mass = query_log_mass.log_softmax(0)
        states = []
        for i, posts in enumerate(self.posts):
            if same_acquisition:
                score = self._conditional_score(posts, self.offsets[i], rss, chunk_size)
            else:
                components = torch.stack([self._conditional_score(posts, b, rss, chunk_size)
                                          for b in query_offsets])
                score = torch.logsumexp(components + query_log_mass[:, None, None], dim=0)
            states.append(score + self.log_support_mass[i])
        score = torch.logsumexp(torch.stack(states), dim=0)
        logp = score.log_softmax(1)
        return dict(xy=logp.exp() @ self.grid, log_probability=logp,
                    log_likelihood=score)
