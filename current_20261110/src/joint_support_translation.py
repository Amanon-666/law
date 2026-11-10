"""Arbitrary-XY total GP moments for a future historical-field translator.

Collect the actual per-AP prior and complete support posterior from ONE existing
retained-GP pass. No adaptation, sampling, model fitting, Query RSS/labels, file
I/O or network hook is performed here. This is not wired into translator
training or evaluation and has not been numerically validated.

Usage with an already legal, adapted task/state on its final device::

    collector = JointSupportTranslation(task)
    grid_fields = prepare_retained_fields(
        task, state, event_id, k, relation_collector=collector)
    fields = collector.at(candidate_xy_absolute)

prepare_retained_fields is the existing position_fusion_gp entry point. Its
unchanged posterior/ESS policy supplies the callback; repeated at() calls do
not condition on S again. AP order follows task['cases'], not a new Q-derived
directory. The caller binds H cache rows, parameters, adaptation state, event
seed and coordinate origin to the same permitted H/S context. Priors/history
objects are retained by reference and must remain immutable after collection;
support coordinates and posterior tensors are copied on the original device.

at() returns total prior/posterior marginal means and variances [XY, AP],
float64 tensors on that device. Means are latent RSS in dBm; variances are
dB**2. These are not encoded missing=-105 means, observed-RSS variances,
detection probabilities or calibrated real-world confidence. Full K-by-K
posterior covariance is used internally; no XY-by-XY covariance is returned.
Query acquisition drift/intercept remain in both prior and posterior marginal
variance, while their cross covariance with S is removed exactly as in the
retained path. A total cross is propagated once, retaining cross-source terms
induced by conditioning; separately propagated source variances are NOT added.
The original predict_from_posterior solve/jitter convention is unchanged.

Future source supervision may evaluate at permitted source-Q coordinates;
deployment evaluates at H/S-derived candidates, never true target-Q positions.
Choosing/normalizing consumer features and training that consumer remain caller
work. Do not treat this moment representation as a lossless copy of all H/S.
"""
import numpy as np
import torch

from fields import rbf
from observations import predict_from_posterior


FIELD_NAMES = ('prior_mean', 'prior_variance', 'posterior_mean', 'posterior_variance')
FIELD_UNITS = ('dBm', 'dB**2', 'dBm', 'dB**2')


class JointSupportTranslation:
    """Existing relation_collector callback plus arbitrary absolute-XY readout."""

    def __init__(self, task):
        self.aps = np.asarray(list(task['cases']), dtype=np.int64)
        assert len(self.aps), 'A prepared nonempty H/S task is required'
        self.origin = np.asarray(task['origin'], dtype=np.float64).copy()
        self.support = {int(ap): case['sx'].detach().clone()
                        for ap, case in task['cases'].items()}
        self.posterior = {}

    @torch.no_grad()
    def __call__(self, ap, prior, sm, sk, moments):
        """Receive the exact prior and full moments used by _retained_field."""
        ap = int(ap)
        assert ap in self.support and ap not in self.posterior
        sx = self.support[ap]
        k = len(sx)
        assert k > 0 and sm.shape == moments['mean'].shape == (k,)
        assert sk.shape == moments['cov'].shape == (k, k)
        assert all(t.dtype == torch.float64 and t.device == sx.device
                   for t in (sx, sm, sk, moments['mean'], moments['cov']))
        _, sf = prior.features(sx)
        self.posterior[ap] = dict(
            prior=prior, sf=sf, sm=sm.detach().clone(), sk=sk.detach().clone(),
            moments={name: moments[name].detach().clone() for name in ('mean', 'cov')})

    @torch.no_grad()
    def at(self, xy):
        """Evaluate registered absolute XY metres, preserving input row order.

        Return FIELD_NAMES as [N,AP] tensors, plus aps, xy_absolute and units.
        There is no clipping, standardization or second S update. All expected
        AP callbacks must have completed before any field can be consumed.
        """
        assert set(self.posterior) == set(self.support), 'Incomplete GP posterior pass'
        anchor = self.support[int(self.aps[0])]
        absolute = torch.as_tensor(xy, dtype=torch.float64, device=anchor.device)
        assert absolute.ndim == 2 and absolute.shape[1] == 2
        points = absolute - anchor.new_tensor(self.origin)
        output = {name: [] for name in FIELD_NAMES}
        for ap in self.aps:
            saved, sx = self.posterior[int(ap)], self.support[int(ap)]
            prior = saved['prior']
            qm, qf = prior.features(points)
            qv = prior.diagonal(points, qf)
            cross = prior.covariance(points, sx, qf, saved['sf'])
            amplitude = prior.base['drift'] if prior.sources else prior.base['amplitude']
            retained_cross = (cross - rbf(points, sx, amplitude, prior.base['length'])
                              - prior.intercept_std**2)
            value = predict_from_posterior(qm, qv, retained_cross,
                                          saved['sm'], saved['sk'], saved['moments'])
            if (not torch.isfinite(value['mean']).all()
                    or not torch.isfinite(value['variance']).all()
                    or (value['variance'] < 0).any()):
                raise RuntimeError(f'AP {ap}: invalid retained propagated moments')
            for name, tensor in zip(FIELD_NAMES, (qm, qv, value['mean'], value['variance'])):
                output[name].append(tensor)
        return dict({name: torch.stack(values, dim=1) for name, values in output.items()},
                    aps=self.aps.copy(), xy_absolute=absolute.clone(),
                    units=dict(zip(FIELD_NAMES, FIELD_UNITS)))
