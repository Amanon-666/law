"""Synthetic check of field_posterior.condition against a direct per-AP Gaussian-process formula (no data, no model).
Also checks the degenerate limits. Must print ALL OK before any real run.
  LAW_UJI_CSV=... VT_DIR=... python check_condition.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import field_posterior as FP
from field_posterior import C

rng = np.random.default_rng(0)
# pick real S rows from the first event so det_stats sees genuine RSS; build a synthetic prior field
e = C.MAN['events'][2]; H, S, Q = C.event_arrays(e)
aps = np.flatnonzero(C.DET[S].any(0))[:7]
Xc = np.vstack([C.XY[S][::3] + rng.normal(0, 0.5, (len(S) // 3, 2)), rng.uniform(C.XY[S].min(0) - 30, C.XY[S].max(0) + 30, (40, 2))])
N, A = len(Xc), len(aps)
p0 = rng.uniform(0.05, 0.95, (N, A)); m0 = rng.uniform(-95, -50, (N, A)); s20 = rng.uniform(9, 60, (N, A))
th = (8.0, 0.6, 1.0)
p, m, s2 = FP.condition(Xc, p0, m0, s20, S, aps, th)

# direct formula for the intensity of each AP
Ps, n, nd, ybar, _ = FP.Hd.det_stats(C.RSS[S][:, aps], C.XY[S])
j = np.linalg.norm(Ps[:, None] - Xc[None], axis=2).argmin(1)
ell, c, g = th
ok = True
for a in range(A):
    det = nd[:, a] > 0
    if not det.any():
        ok &= np.allclose(m[:, a], m0[:, a]); continue
    Sx = Ps[det]; js = j[det]
    tau = lambda idx: np.sqrt(c * s20[idx, a])
    kss = np.outer(tau(js), tau(js)) * np.exp(-np.linalg.norm(Sx[:, None] - Sx[None], axis=2) / ell)
    kss += np.diag((1 - c) * s20[js, a] / nd[det, a])
    kxs = np.outer(tau(np.arange(N)), tau(js)) * np.exp(-np.linalg.norm(Xc[:, None] - Sx[None], axis=2) / ell)
    r = ybar[det, a] - m0[js, a]
    m_ref = m0[:, a] + kxs @ np.linalg.solve(kss, r)
    v_ref = (1 - c) * s20[:, a] + np.maximum(c * s20[:, a] - np.einsum('nk,kl,nl->n', kxs, np.linalg.inv(kss), kxs), 0)
    ok &= np.allclose(m[:, a], m_ref, atol=1e-3) and np.allclose(s2[:, a], v_ref, atol=1e-3)
print('intensity matches direct GP formula:', ok)
# limits
p1, m1, v1 = FP.condition(Xc, p0, m0, s20, S, aps, (8.0, 0.0, 0.0))
lim1 = np.allclose(m1, m0) and np.allclose(v1, s20) and np.allclose(p1, p0)
p2, m2, v2 = FP.condition(Xc, p0, m0, s20, S, aps, (1e-6, 0.6, 1.0))
far = np.linalg.norm(Xc[:, None] - Ps[None], axis=2).min(1) > 1.0
lim2 = np.allclose(m2[far], m0[far], atol=1e-6) and np.allclose(p2[far], p0[far], atol=1e-6)
near = (s2 <= s20 + 1e-9).all()
print('c=g=0 returns prior:', lim1, '| ell->0 leaves far cells unchanged:', lim2, '| variance never grows:', near)
print('ALL OK' if (ok and lim1 and lim2 and near) else 'FAILED')
