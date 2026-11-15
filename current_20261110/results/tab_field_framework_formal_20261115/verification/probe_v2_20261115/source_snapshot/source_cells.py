"""Original HGB source cells with a parallel, label-unit identity trace."""
import hashlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def cell_table_with_units(H, pairs, rng, draws=4, Ks=(1, 3, 5), max_rows=6000):
    # Keep the original random draws and array operations in their original order.
    C, T, V = H.C, H.T, H.V
    X, n, nd, mu, var, units = [], [], [], [], [], []
    for b, t in pairs:
        Pt, nt, ndt, mt, vt = H.STATS[(b, t)]
        Xc, F, ref = V.hist_feats(b, t)
        D = np.linalg.norm(Xc[:, None] - Pt[None], axis=2)
        j = D.argmin(1)
        ok = D.min(1) <= T.R_LABEL
        hdet = C.DET[np.flatnonzero((C.BLD == b) & (C.FLR != t))].any(0)
        for K in Ks:
            for _ in range(draws):
                s_rows = V.sample_support(b, t, K, rng, False)
                aps = np.flatnonzero(hdet | C.DET[s_rows].any(0))
                Fe = F[ok][:, aps]
                nf = Fe.shape[2]
                F2 = Fe.reshape(-1, nf)
                S2 = V.supp_feats(b, t, s_rows)[ok][:, aps].reshape(len(F2), -1)
                jj = j[ok]
                cn = np.repeat(nt[jj], len(aps))
                cnd = ndt[jj][:, aps].ravel()
                cm = mt[jj][:, aps].ravel()
                cv = vt[jj][:, aps].ravel()
                info = (np.nan_to_num(F2[:, 0:40:5], nan=T.ND).max(1) > T.ND) | (cnd > 0) | (np.nan_to_num(S2[:, 1], nan=T.ND) > T.ND)
                idx = np.flatnonzero(info | (rng.random(len(F2)) < 0.05))
                if len(idx) > max_rows:
                    idx = rng.choice(idx, max_rows, replace=False)
                X.append(np.hstack([F2[idx], S2[idx]]))
                n.append(cn[idx])
                nd.append(cnd[idx])
                mu.append(cm[idx])
                var.append(cv[idx])
                target_positions = np.repeat(jj, len(aps))[idx]
                ap_columns = np.tile(aps, len(jj))[idx]
                units.append(np.column_stack((np.full(len(idx), b), np.full(len(idx), t), target_positions, ap_columns)))
    return (np.vstack(X).astype(np.float32), *(np.concatenate(a) for a in (n, nd, mu, var)), np.vstack(units).astype(np.int32))


def array_digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def prepare_source(H, seed=1):
    pairs = [(b, t) for b in (0, 2) for t in H.V.FLOORS[b]]
    parts, parity = [], []
    for half, group in enumerate((pairs[0::2], pairs[1::2])):
        original = H.cell_table(group, np.random.default_rng([seed, half]), draws=4)
        traced = cell_table_with_units(H, group, np.random.default_rng([seed, half]), draws=4)
        for old, new in zip(original, traced[:5]):
            assert np.array_equal(old, new, equal_nan=True)
        parity.append(dict(half=half, pairs=group, rows=len(traced[0]), array_sha256=[array_digest(a) for a in traced[:5]], exact_equal=True))
        parts.append(traced)
        del original
    X, n, nd, mu, var, units = (np.concatenate([part[k] for part in parts]) for k in range(6))
    half = np.concatenate([np.full(len(part[0]), i, dtype=np.int8) for i, part in enumerate(parts)])
    unique_units, first, unit_index = np.unique(units, axis=0, return_index=True, return_inverse=True)
    unit_train = np.ones(len(unique_units), dtype=bool)
    validation_units = np.random.default_rng(seed).permutation(len(unique_units))[:int(np.ceil(len(unique_units) * .1))]
    unit_train[validation_units] = False
    train = unit_train[unit_index]
    assert X.shape[1] == 55 and set(units[:, 0]) == {0, 2}
    assert np.array_equal(nd, np.round(nd)) and np.array_equal(n, np.round(n))
    assert (nd >= 0).all() and (nd <= n).all()
    manifest = dict(seed=seed, source_buildings=[0, 2], source_floor_pairs=pairs, rows=len(X), features=55,
                    unique_label_units=len(unique_units), training_units=int(unit_train.sum()),
                    validation_units=len(validation_units), training_rows=int(train.sum()), validation_rows=int((~train).sum()),
                    detected_training_rows=int((train & (nd > 0)).sum()),
                    unit_definition=['building', 'target_floor', 'target_position_index', 'AP_column'],
                    repeated_K_draw_aliases_share_split=True, validation_fraction=.1,
                    original_cell_table_parity=parity, unit_sha256=array_digest(units),
                    unit_train_sha256=array_digest(unit_train),
                    scan_count_range=[float(n.min()), float(n.max())])
    return dict(X=X, n=n, nd=nd, mu=mu, var=var, units=units, unit_index=unit_index,
                unique_units=unique_units, first=first, half=half, train=train, manifest=manifest)


def matched_hgb_and_scale(H, data, seed=1):
    """Same original losses on the source unit-matched training rows.

    OOF labels here validate the scale interface only. Formal Tab scale targets
    must come from Tab means trained on the opposite original floor half.
    """
    X, n, nd, mu, var = (data[k] for k in ('X', 'n', 'nd', 'mu', 'var'))
    train = data['train']
    cols = np.flatnonzero(~np.isnan(X[train]).all(0))
    Xi, ni, ndi = X[train][:, cols], n[train], nd[train]
    Xd = np.vstack([Xi, Xi])
    yd = np.r_[np.ones(len(Xi)), np.zeros(len(Xi))]
    wd = np.r_[ndi, ni - ndi]
    keep = wd > 0
    det = H._Fit(HistGradientBoostingClassifier(max_iter=400, learning_rate=.06, max_leaf_nodes=63,
                 min_samples_leaf=40, l2_regularization=1., random_state=seed), Xd[keep], yd[keep], wd[keep])
    detected = train & (nd > 0)
    mean = H._Fit(H._hgb_reg(seed), X[detected][:, cols], mu[detected], nd[detected])
    oof = np.full(len(X), np.nan)
    other_means = []
    for target_half in (0, 1):
        other = train & (nd > 0) & (data['half'] != target_half)
        model = H._Fit(H._hgb_reg(seed), X[other][:, cols], mu[other], nd[other])
        own = (nd > 0) & (data['half'] == target_half)
        oof[own] = model.predict(X[own][:, cols])
        other_means.append(model)
    scale_target = np.log(np.nan_to_num(var, nan=0.) + (mu - oof) ** 2 + H.S0 ** 2)
    scale_X = np.column_stack((X, oof.astype(np.float32)))
    fitted_scale_X = np.column_stack((X[detected][:, cols], oof[detected].astype(np.float32)))
    scale = H._Fit(H._hgb_reg(seed), fitted_scale_X, scale_target[detected], nd[detected])
    return dict(cols=cols, det=det, mean=mean, scale=scale, other_means=other_means), scale_X, scale_target


def context_and_query(data, role, max_context_rows, query_rows, seed):
    """Preserve whole units and actual counts in context; hold out all aliases."""
    eligible = data['train'] & ((data['n'] > 0) if role == 'det' else (data['nd'] > 0))
    rng = np.random.default_rng(seed)
    query = rng.choice(np.flatnonzero(eligible), size=query_rows, replace=False)
    query_units = np.unique(data['unit_index'][query])
    canonical = data['first']
    context = canonical[eligible[canonical] & ~np.isin(data['unit_index'][canonical], query_units)]
    context = rng.permutation(context)
    count = data['n'][context] if role == 'det' else data['nd'][context]
    context = context[np.cumsum(count) <= max_context_rows]
    assert len(context) and not np.intersect1d(data['unit_index'][context], query_units).size
    return context, query
