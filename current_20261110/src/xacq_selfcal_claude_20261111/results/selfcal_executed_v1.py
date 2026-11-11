"""History-only calibration of frozen learned candidates; empirical S unchanged."""
import copy
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.spatial import cKDTree


def fit_logistic(groups):
    # Each pseudo floor has unit total weight, regardless of receipts/AP count.
    def objective(theta):
        loss, grad = 0., np.zeros(2)
        for x, y in groups:
            z = theta[0] + theta[1]*x
            residual = expit(z)-y
            loss += np.mean(np.logaddexp(0, z)-y*z)
            grad += [residual.mean(), np.mean(residual*x)]
        return loss/len(groups), grad/len(groups)
    result = minimize(objective, [0., 1.], jac=True, method='BFGS',
                      options={'gtol': 1e-8, 'maxiter': 300})
    if not result.success and np.max(np.abs(result.jac)) > 1e-7:
        raise RuntimeError(str(result.message))
    return dict(alpha=float(result.x[0]), beta=float(result.x[1]),
                loss=float(result.fun), gradient_max=float(np.abs(result.jac).max()))


def fit_calibration(translator, histories, seed=20261111, R=3):
    if len(histories) < 2:
        return dict(alpha=0., beta=1., c=1., b=0., status='single_history_identity', floors=[])
    groups, residuals, ratios, records = [], [], [], []
    for name in sorted(histories):
        held = histories[name]
        xy, inv = np.unique(held['xy'], axis=0, return_inverse=True)
        inv = inv.ravel()
        rng = np.random.default_rng(seed)
        positions = rng.choice(len(xy), min(3, len(xy)), replace=False)
        rows = np.concatenate([rng.choice(np.flatnonzero(inv == p),
            min(R, (inv == p).sum()), replace=False) for p in positions])
        query_rows = np.flatnonzero(~np.isin(inv, positions))
        support = {key: held[key][rows] for key in ('rss', 'xy')}
        field = translator.prepare({n:h for n,h in histories.items() if n != name}, support, name)
        nh = len(field.cand_xy)-len(positions)
        distances, nearest = cKDTree(field.cand_xy[:nh]).query(held['xy'][query_rows])
        keep = distances <= 4.
        indices, nearest = query_rows[keep], nearest[keep]
        if not len(indices):
            raise ValueError(f'{name}: no non-support receipt within 4m of learned candidates')
        rss = held['rss'][indices][:, field.aps]
        detected = rss != 100
        x = logit(field.p[nearest]).ravel()
        groups.append((x, detected.ravel().astype(float)))
        residual = (rss-field.m[nearest])[detected]
        residuals.append(residual)
        ratios.append(residual**2/field.s2[nearest][detected])
        records.append(dict(floor=name, support_rows=rows.tolist(),
            support_positions=positions.tolist(), eligible_receipts=len(indices),
            rejected_distance=int((~keep).sum()), cells=len(x), detections=int(detected.sum()),
            mean_p=float(field.p[nearest].mean()), detected_fraction=float(detected.mean())))
    result = fit_logistic(groups)
    result.update(c=float(np.clip(np.concatenate(ratios).mean(), .25, 16.)),
                  b=float(np.clip(np.median(np.concatenate(residuals)), -10., 10.)),
                  status='fitted', seed=seed, pseudo_K=3, R=R, floors=records)
    return result


def apply_calibration(field, calibration, n_support_positions, mode='all'):
    assert mode in ('all', 'detection', 'intensity')
    out = copy.copy(field)
    out.p, out.m, out.s2 = field.p.copy(), field.m.copy(), field.s2.copy()
    nh = len(field.cand_xy)-n_support_positions
    if mode in ('all', 'detection'):
        out.p[:nh] = expit(calibration['alpha'] + calibration['beta']*logit(out.p[:nh]))
    if mode in ('all', 'intensity'):
        out.m[:nh] += calibration['b']
        out.s2[:nh] *= calibration['c']
    return out


def identity_check():
    # Exact balanced Bernoulli frequencies, not soft labels or random luck.
    probabilities = np.arange(1, 10)/10
    x = np.repeat(logit(probabilities), 100)
    y = np.concatenate([np.r_[np.ones(int(p*100)), np.zeros(100-int(p*100))]
                        for p in probabilities])
    fitted = fit_logistic([(x, y)])
    assert abs(fitted['alpha']) < 1e-6 and abs(fitted['beta']-1) < 1e-6
    return fitted


if __name__ == '__main__':
    print(identity_check())
