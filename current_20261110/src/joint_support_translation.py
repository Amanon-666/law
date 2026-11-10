"""Arbitrary-XY GP moments and the joint-support observation translator.

Collect the actual per-AP prior and complete support posterior from ONE existing
retained-GP pass. No adaptation, sampling, model fitting, Query RSS/labels, file
I/O or network hook is performed by the field collector. The context/model
below connect these fields to the existing classifier and readout; no runner
is connected or executed and this path has not been numerically validated.

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

Source supervision may evaluate at permitted source-Q coordinates;
deployment evaluates at H/S-derived candidates, never true target-Q positions.
Only the four nearest-S columns are replaced, in raw physical units; no fitted
normalization is introduced. Training the consumer remains caller work. Do not
treat this moment representation as a lossless copy of all H/S.
"""
import joblib
import numpy as np
import torch

from fields import rbf
from observations import predict_from_posterior
from observation_translation_tasks import ObservationTranslationContext
from observation_translation_model import ObservationTranslationModel, PreparedObservationTranslation


FIELD_NAMES = ('prior_mean', 'prior_variance', 'posterior_mean', 'posterior_variance')
FIELD_UNITS = ('dBm', 'dB**2', 'dBm', 'dB**2')
REPRESENTATION = 'joint_support_translation_v1'
JOINT_FEATURE_NAMES = ('posterior_mean', 'posterior_mean_update',
                       'prior_std', 'posterior_std')
JOINT_FEATURE_UNITS = ('dBm', 'dB', 'dB', 'dB')


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


class JointObservationTranslationContext(ObservationTranslationContext):
    """Same H/S features, with indices 49:53 replaced by joint GP moments.

    Keep all 43 H columns and S columns 0:6,10:12 (global/grouped statistics,
    K and coverage). S columns 6:10 are replaced in JOINT_FEATURE_NAMES order,
    in JOINT_FEATURE_UNITS, then packed as the original float32 [N,AP,55].
    The supplied completed collector MUST come from these exact legal H/S and
    matched fitted assets/state. AP binding is checked here; H-row provenance
    and S identity remain the caller's preparation contract, as for the GP.
    """

    def __init__(self, histories, support, target_floor, *, joint_support):
        super().__init__(histories, support, target_floor)
        assert isinstance(joint_support, JointSupportTranslation)
        assert np.array_equal(self.aps, joint_support.aps), 'GP and translator AP order differs'
        self.joint_support = joint_support

    def at(self, xy):
        features = super().at(xy)
        field = self.joint_support.at(xy)
        joint = torch.stack((field['posterior_mean'],
                             field['posterior_mean'] - field['prior_mean'],
                             field['prior_variance'].sqrt(),
                             field['posterior_variance'].sqrt()), dim=-1)
        features[..., 49:53] = joint.cpu().numpy().astype(np.float32)
        return features

    def source_scan_table(self, query, event):
        """Reuse original permissions, labels and weights via dynamic self.at()."""
        table = super().source_scan_table(query, event)
        table['representation'] = REPRESENTATION
        return table


class JointObservationTranslationModel(ObservationTranslationModel):
    """Original classifier/quantization/readout, bound to the joint features.

    Fit accepts only tagged source tables from the context above. Explicit
    prepare(..., joint_support=collector) only evaluates existing posterior
    fields and the trained classifier; it never fits or samples. The new
    serialized layout rejects old untagged models on load, and its nested
    model payload also prevents the old loader treating it as the old 55D
    representation. Raw predict_log_mass(X) retains the inherited low-level
    array contract; callers must supply this representation's feature arrays.
    """

    def fit(self, tables, *, seed, max_iter=200):
        tables = list(tables)
        assert all(t.get('representation') == REPRESENTATION for t in tables), 'Mixed feature representations'
        super().fit(tables, seed=seed, max_iter=max_iter)
        self.representation = REPRESENTATION
        return self

    def source_nll(self, table):
        assert table.get('representation') == REPRESENTATION, 'Wrong feature representation'
        return super().source_nll(table)

    def prepare(self, histories, support, target_floor, *, joint_support):
        context = JointObservationTranslationContext(
            histories, support, target_floor, joint_support=joint_support)
        candidates = np.unique(np.concatenate([
            *[np.asarray(h['xy'], dtype=np.float64) for h in histories.values()],
            np.asarray(support['xy'], dtype=np.float64),
        ]), axis=0)
        features = context.at(candidates)
        mass = self.predict_log_mass(features.reshape(-1, 55)).reshape(
            len(candidates), len(context.aps), len(self.cuts) + 2)
        return PreparedObservationTranslation(candidates, context.aps, self.cuts, mass)

    def save(self, path):
        joblib.dump(dict(representation=self.representation,
                         model=dict(estimator=self.estimator, cuts=self.cuts,
                                    columns=self.columns)), path)

    @classmethod
    def load(cls, path):
        saved = joblib.load(path)
        assert saved.get('representation') == REPRESENTATION, 'Wrong model representation'
        payload = saved['model']
        model = cls()
        model.representation = saved['representation']
        model.estimator, model.cuts, model.columns = (
            payload['estimator'], payload['cuts'], payload['columns'])
        assert np.array_equal(model.estimator.classes_, np.arange(len(model.cuts) + 2))
        return model
