"""H/S-only deployment of Claude's original five-seed hurdle assets.

Only saved plain sklearn states are loaded. No fit, CSV/global dataset access,
training-table reconstruction, GP/Tab branch or continuous_* dependency.
Original sampling, count weights, early stopping and hyperparameters belong
to the generated assets and are not replaced here. No numerical equivalence
or performance claim follows from this statically reviewed interface.

H/S dictionaries contain rss[N,A] (dBm, missing=100) and xy[N,2] in a common
registered metric frame; names have one building prefix and physical F suffix.
Query has the same A columns. AP identity need not match UJI's 520 columns.
All input columns participate in original pooled S features before selecting
the H-union-S detected APs. The 55 features, physical offsets and dtype order
match vertical_translator/vt_lib.py and hurdle.py. In particular the mean
appended to scale prediction remains float64, as in the original predictor.

The five learned H candidate distributions are averaged before empirical S
candidates are appended ONCE. H/S coordinate duplicates are retained. Reader
constants reproduce run_nine_floor_hurdle.posterior; U_DET/U_MISS are robust
score constants, not a normalized contamination-probability claim.
"""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
from scipy.special import logsumexp

from vertical_translator import translator as T


EPS, BETA = .02, 128.
U_DET, U_MISS = 1./60., .5
S0 = 3.
OFFSETS = np.array([-8., -4., 0., 4., 8.])


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _support_fields(rss, xy):
    """Original det_stats/support_candidates arithmetic on supplied S only."""
    positions, inv = np.unique(xy, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.bincount(inv, minlength=len(positions)).astype(float)
    det = (rss != 100).astype(float)
    val = np.where(rss != 100, rss, 0.)
    nd = np.zeros((len(positions), rss.shape[1]))
    s1, s2 = np.zeros_like(nd), np.zeros_like(nd)
    np.add.at(nd, inv, det)
    np.add.at(s1, inv, val)
    np.add.at(s2, inv, val**2)
    mean = np.where(nd > 0, s1/np.maximum(nd, 1), np.nan)
    variance = np.where(nd > 1, s2/np.maximum(nd, 1) - mean**2, np.nan)
    variance = np.maximum(variance, 0)
    p = (nd + .5)/(n[:, None] + 1.)
    mean = np.where(np.isnan(mean), -95., mean)
    variance = np.nan_to_num(variance, nan=0.) + S0**2 + 4.**2
    return positions, p, mean, variance


class FrozenHurdleTranslator:
    def __init__(self, generation_dir):
        """Load the completed, hash-bound original seeds 1..5; never generate."""
        root = Path(generation_dir)
        complete = json.loads((root/'complete.json').read_text())
        self.freeze = json.loads((root/'freeze.json').read_text())
        freeze_sha = _sha256(root/'freeze.json')
        assert complete['complete'] and complete['freeze_sha256'] == freeze_sha
        assert self.freeze['seeds'] == [1, 2, 3, 4, 5]
        assert self.freeze['draws'] == 4
        sources = self.freeze['source_buildings']
        assert sources and len(set(sources)) == len(sources)
        assert complete['source_buildings'] == sources
        if 'source_rows_by_floor' in self.freeze:
            assert complete['source_rows_by_floor'] == self.freeze['source_rows_by_floor']
            assert complete['source_floors'] == self.freeze['source_floors']
        feature_code = Path(T.__file__).read_bytes()
        feature_blob = hashlib.sha1(b'blob '+str(len(feature_code)).encode()+b'\0'+feature_code).hexdigest()
        assert feature_blob == self.freeze['upstream_git_blobs']['translator.py']
        records = sorted(complete['models'], key=lambda r: r['seed'])
        assert [r['seed'] for r in records] == self.freeze['seeds']
        self.models = []
        for record in records:
            path = root/record['path']
            assert _sha256(path) == record['sha256']
            model = joblib.load(path)
            assert model['kind'] == 'claude_hurdle_v1' and model['seed'] == record['seed']
            assert model['source_buildings'] == sources and model['draws'] == 4
            if 'source_rows_by_floor' in self.freeze:
                assert model['source_rows_by_floor'] == self.freeze['source_rows_by_floor']
                assert model['source_floors'] == self.freeze['source_floors']
            assert model['freeze_sha256'] == freeze_sha
            self.models.append(model)

    def prepare(self, histories, support, target_floor):
        """Prepare from matching H/S alone; no target Query or fit operation."""
        building, floor = target_floor.rsplit('F', 1)
        floor = int(floor)
        srss = np.asarray(support['rss'])
        sxy = np.asarray(support['xy'], dtype=np.float64)
        assert srss.ndim == 2 and sxy.shape == (len(srss), 2)
        width = srss.shape[1]
        assert histories and target_floor not in histories
        maps = {}
        seen = (srss != 100).any(axis=0)
        for name, scans in histories.items():
            hb, hf = name.rsplit('F', 1)
            rss, xy = np.asarray(scans['rss']), np.asarray(scans['xy'], dtype=np.float64)
            assert hb == building and int(hf) != floor
            assert rss.shape == (len(xy), width) and xy.shape[1:] == (2,)
            maps[int(hf)] = T.position_map(rss, xy)
            seen |= (rss != 100).any(axis=0)
        aps = np.flatnonzero(seen)
        assert len(aps), 'No AP detected in H or S'
        candidates = np.unique(np.concatenate([m[0] for m in maps.values()]), axis=0)
        F, ref = T.history_features(candidates, floor, maps)
        F = F.astype(np.float32)
        if len(sxy):
            sref = T.naive_ref_at(sxy, floor, maps)
            S = T.support_features(candidates, ref, sxy, srss, sref, F).astype(np.float32)
        else:
            S = np.full((len(candidates), width, 12), np.nan, dtype=np.float32)
        cells = len(candidates)*len(aps)
        features = np.hstack([F[:, aps].reshape(cells, -1), S[:, aps].reshape(cells, -1)])
        fields = []
        for model in self.models:
            X = features[:, model['cols']]
            det, mean, scale = (model[key] for key in ('det', 'mean', 'scale'))
            p = det['estimator'].predict_proba(X[:, det['cols']])[:, 1]
            m = mean['estimator'].predict(X[:, mean['cols']])
            # Intentionally retain the original float64 appended mean.
            scale_X = np.hstack([X, m[:, None]])
            s2 = np.exp(scale['estimator'].predict(scale_X[:, scale['cols']]))
            fields.append(tuple(v.reshape(len(candidates), len(aps)) for v in (p, m, s2)))
        ps, ms, vs = (np.stack([f[k] for f in fields]) for k in range(3))
        p, m, s2 = ps.mean(0), ms.mean(0), vs.mean(0) + ms.var(0)
        Ps, sp, sm, sv = _support_fields(srss[:, aps], sxy)
        xy = np.vstack([candidates, Ps])
        bounds = np.stack([np.floor(xy.min(0)/2)*2, np.ceil(xy.max(0)/2)*2])
        return PreparedFrozenHurdle(xy, aps,
            np.clip(np.vstack([p, sp]), 1e-4, 1-1e-4),
            np.vstack([m, sm]), np.vstack([s2, sv]), bounds, width)


class PreparedFrozenHurdle:
    """Original ensemble posterior-mean reader on fixed H/S candidates."""
    def __init__(self, cand_xy, aps, p, m, s2, bounds, ap_count):
        self.cand_xy, self.aps = cand_xy, aps
        self.p, self.m, self.s2 = p, m, s2
        self.bounds, self.ap_count = bounds, ap_count

    def predict(self, query_rss):
        query = np.asarray(query_rss)
        assert query.ndim == 2 and query.shape[1] == self.ap_count
        lmiss = np.log((1-EPS)*(1-self.p) + EPS*U_MISS)
        logp = np.log(self.p)
        lnorm = -.5*np.log(2*np.pi*self.s2)
        q, A = query[:, self.aps].astype(float), len(self.aps)
        L = np.empty((len(q), len(self.cand_xy)))
        for i, row in enumerate(q):
            d = row != 100
            v = row[d][None]
            per = [np.logaddexp(np.log(1-EPS) + logp[:, d] + lnorm[:, d]
                - .5*(v-o-self.m[:, d])**2/self.s2[:, d],
                np.log(EPS*U_DET)).sum(1) for o in OFFSETS]
            L[i] = (logsumexp(np.stack(per), 0) - np.log(len(OFFSETS))
                    + lmiss[:, ~d].sum(1))*BETA/A
        log_mass = L - logsumexp(L, 1, keepdims=True)
        projected = np.clip(np.exp(log_mass) @ self.cand_xy, self.bounds[0], self.bounds[1])
        return dict(projected=projected, log_position_mass=log_mass,
                    cand_xy=self.cand_xy.copy(), aps=self.aps.copy())
