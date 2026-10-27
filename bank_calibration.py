"""Existing calibration kernels applied directly to historical bank moments.

No re-encoding of the bank: means and pre-support variances are unchanged.
The expected-strength RBF is the existing GaussianStrengthRelation formula,
now evaluated with the bank's joint moments rather than one History object.
"""
import math

import torch

from calibration_relation import CalibrationRelation
from history_bank import BankPrior
from multi_history import GaussianStrengthRelation
from transfer import History, Prior, Transfer, rbf


class CalibratedBankPrior(BankPrior):
    def __init__(self, histories, weights, base, relation):
        # Preserve the frozen study's common policy for all-H-undetected APs.
        present = any(h is not None for h in histories)
        super().__init__(histories if present else [], weights if present else [], base)
        self.relation = relation

    def covariance(self, x, y, fx, fy):
        original = super().covariance(x, y, fx, fy)
        if not self.sources:
            return original
        base, relation = self.base, self.relation
        historical = original - rbf(x, y, base['drift'], base['length'])
        if relation.kind == 'uniform':
            change = relation.kernel(x, y, None, base)
        else:
            mx, _ = self.features(x); my, _ = self.features(y)
            vx = self.diagonal(x, fx) - base['drift'] ** 2
            vy = self.diagonal(y, fy) - base['drift'] ** 2
            difference_variance = vx[:, None] + vy[None] - 2 * historical
            if bool((difference_variance < -1e-7).any()):
                raise RuntimeError('Invalid historical Gaussian difference variance')
            v = difference_variance.clamp_min(0)
            ell2 = relation.log_strength_length.exp().square()
            strength = torch.sqrt(ell2/(ell2+v)) * torch.exp(-.5*(mx[:,None]-my[None]).square()/(ell2+v))
            local = relation.logit_local.sigmoid()
            spatial = rbf(x,y,1.,relation.log_length.exp())
            change = base['drift'] ** 2 * (local*spatial+(1-local)*strength)
        return historical + change


def load_relation(kind, path, source_base, target_base):
    cls = GaussianStrengthRelation if kind == 'strength_rbf' else CalibrationRelation
    relation = cls(kind, source_base['length']).double()
    relation.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
    with torch.no_grad():
        relation.log_length.add_(math.log(target_base['length'] / source_base['length']))
    relation.requires_grad_(False)
    return relation


@torch.no_grad()
def check():
    """Checks the changed representation and kernel's variance contract."""
    torch.set_default_dtype(torch.float64)
    base = dict(mean=-95., amplitude=16., length=5., drift=7.)
    z0 = torch.tensor([[0., 0.], [4., 1.], [7., 2.]])
    z1 = torch.tensor([[1., 3.], [6., 4.]])
    h0 = History(z0, dict(mean=torch.tensor([-70., -80., -90.]), cov=.3*rbf(z0,z0,16.,5.)), base)
    h1 = History(z1, dict(mean=torch.tensor([-75., -88.]), cov=.5*rbf(z1,z1,16.,5.)), base)
    x = torch.tensor([[2., 1.], [6., 0.], [12., 8.]])
    y = torch.tensor([[3., 4.], [14., 9.]])
    errors=[]
    for histories,weights in [([h0,h1],[.5,.5]),([h0,None],[.5,.5]),([h0],[1.])]:
        old = BankPrior(histories,weights,base)
        full=torch.cat((x,y)); original_mean,original_features=old.features(full)
        original_diagonal=old.diagonal(full,original_features)
        for kind in ['uniform','strength_rbf']:
            cls=GaussianStrengthRelation if kind=='strength_rbf' else CalibrationRelation
            relation=cls(kind,base['length']).double()
            prior=CalibratedBankPrior(histories,weights,base,relation)
            mean,features=prior.features(full)
            cov=prior.covariance(full,full,features,features)
            torch.testing.assert_close(mean,original_mean)
            torch.testing.assert_close(prior.diagonal(full,features),original_diagonal)
            torch.testing.assert_close(cov.diag(),original_diagonal,rtol=1e-6,atol=3e-5)
            if torch.linalg.eigvalsh(cov).min()<=0: raise RuntimeError('Invalid calibration covariance')
            if len(histories)==1:
                single=Prior(Transfer('identity'),h0,base,relation)
                _,single_features=single.features(full)
                error=float((cov-single.covariance(full,full,single_features,single_features)).abs().max())
                errors.append(error)
                torch.testing.assert_close(cov,single.covariance(full,full,single_features,single_features),rtol=1e-6,atol=3e-5)
    relation=GaussianStrengthRelation('strength_rbf',base['length']).double()
    empty=CalibratedBankPrior([None,None],[.5,.5],base,relation); _,ef=empty.features(x)
    torch.testing.assert_close(empty.covariance(x,x,ef,ef),rbf(x,x,base['amplitude'],base['length']))
    forward=CalibratedBankPrior([h0,h1],[.3,.7],base,relation)
    reverse=CalibratedBankPrior([h1,h0],[.7,.3],base,relation)
    fm,ff=forward.features(x);rm,rf=reverse.features(x)
    torch.testing.assert_close(fm,rm)
    torch.testing.assert_close(forward.covariance(x,x,ff,ff),reverse.covariance(x,x,rf,rf))
    return dict(max_single_history_agreement_error=max(errors), matched_pre_support_mean_and_variance=True,
        covariance_diagonal_agreement=True, positive_covariance=True,
        missing_history_and_single_source_checked=True, source_order_invariant=True)
