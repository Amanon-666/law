"""One learned next-scan distribution on unique H-union-S positions.

This changes the previously lossy replacement premise: every support position
now supplies its n, detection count, detected-RSS sum and squared sum at that
position. The original55 history/support summaries and16 history-exposure
features remain; two partial nearest-support features are replaced by four
raw sufficient statistics. Global support summaries remain summaries, not a
lossless encoding of all spatial interactions.

No empirical-S distribution is appended after learning. The same predictor
and original reader apply at every candidate, and duplicate coordinates get
one candidate entry. Existing GP pooling can still consume this field.

Source tasks retain real acquisition groups, thin only actual scans within a
coherent history group, and expose one/three scans at S. All reserved base-S
rows are excluded from source labels. Candidate/AP construction never reads Q.
All eligible cells have the same sampling probability within each episode;
the old label-dependent negative-cell filter is removed. The original HGB
heads, scan-count losses and floor cross-fitting are unchanged.

Limits: source groups are only an empirical prior over acquisition changes;
unknown query shifts are not uniquely identified. Finite support does not
identify arbitrary fields. The old scale head remains a residual proxy, not
a proven calibrated posterior. This is a joint information-flow change, not
a single-feature ablation or a new claim for every borrowed component.
"""
import numpy as np

from candidates import acquisition_tasks_20261111 as E
from candidates.finite_observation_20261111.features import position_stats, SCHEMA as OLD_SCHEMA
from candidates.finite_observation_20261111.source import one_per_position, array_sha
from frozen_hurdle_translator import FrozenHurdleTranslator, PreparedFrozenHurdle
from vertical_translator import translator as T

RECIPE = 'unified_scan_field_77_v1'
SCHEMA = OLD_SCHEMA[:71] + ['S_nearest_n','S_nearest_nd','S_nearest_sum','S_nearest_sum_sq',
                          'S_log1p_n_eff_all','S_mean_detection_rate']
WIDTH = len(SCHEMA)


def context(histories, support, floor):
    maps = {f:T.position_map(d['rss'],d['xy']) for f,d in histories.items()}
    hs = {f:position_stats(d['rss'],d['xy']) for f,d in histories.items()}
    ss = position_stats(support['rss'],support['xy'])
    xy = np.unique(np.concatenate([*[m[0] for m in maps.values()],ss['xy']]),axis=0)
    visible = (support['rss'] != 100).any(0)
    for d in histories.values():
        visible |= (d['rss'] != 100).any(0)
    aps = np.flatnonzero(visible)
    width = support['rss'].shape[1]
    F, ref = T.history_features(xy,floor,maps)
    F = F.astype(np.float32)
    sref = T.naive_ref_at(support['xy'],floor,maps) if len(ss['xy']) else np.empty((0,width))
    S = T.support_features(xy,ref,support['xy'],support['rss'],sref,F).astype(np.float32)
    extra = []
    for offset in T.SLOTS:
        if floor+offset not in hs:
            extra.extend([np.full((len(xy),width),np.nan)]*2)
            continue
        st = hs[floor+offset]
        D = np.linalg.norm(xy[:,None]-st['xy'][None],axis=2)
        nearest = np.where(D.min(1)<=T.R_SAME,st['n'][D.argmin(1)],np.nan)
        W = D<=T.R_SMOOTH
        count = W.sum(1)
        eff = np.full(len(xy),np.nan)
        present = count>0
        eff[present] = count[present]**2/(W[present]@(1./st['n']))
        extra.extend([np.broadcast_to(np.log1p(v)[:,None],(len(xy),width)) for v in (nearest,eff)])
    if len(ss['xy']):
        D = np.linalg.norm(xy[:,None]-ss['xy'][None],axis=2)
        j = D.argmin(1)
        eff = len(ss['n'])**2/np.sum(1./ss['n'])
        extra.extend([np.broadcast_to(ss['n'][j,None],(len(xy),width)),ss['nd'][j],
                      ss['rss_sum'][j],ss['rss_square_sum'][j],
                      np.full((len(xy),width),np.log1p(eff)),
                      np.broadcast_to((ss['nd']/ss['n'][:,None]).mean(0),(len(xy),width))])
    else:
        extra.extend([np.zeros((len(xy),width))]*4+[np.full((len(xy),width),np.nan)]*2)
    X = np.concatenate([F[:,aps],S[:,aps],np.stack(extra,2)[:,aps]],axis=2).astype(np.float32)
    assert X.shape == (len(xy),len(aps),WIDTH)
    return dict(cand_xy=xy,aps=aps,features=X,ap_count=width,support_stats=ss)


class UnifiedScanField(FrozenHurdleTranslator):
    def __init__(self, directory):
        super().__init__(directory)
        assert self.freeze['recipe'] == RECIPE and self.freeze['feature_schema'] == SCHEMA

    def prepare(self, histories, support, target_floor):
        building, floor = target_floor.rsplit('F',1)
        assert all(name.rsplit('F',1)[0] == building for name in histories)
        inputs = {int(name.rsplit('F',1)[1]):value for name,value in histories.items()}
        assert int(floor) not in inputs
        c = context(inputs,support,int(floor))
        shape = (len(c['cand_xy']),len(c['aps']))
        features = c['features'].reshape(-1,WIDTH)
        fields = []
        for model in self.models:
            X = features[:,model['cols']]
            det,mean,scale = (model[n] for n in ('det','mean','scale'))
            p = det['estimator'].predict_proba(X[:,det['cols']])[:,1]
            m = mean['estimator'].predict(X[:,mean['cols']])
            sx = np.hstack([X,m[:,None]])
            v = np.exp(scale['estimator'].predict(sx[:,scale['cols']]))
            fields.append(tuple(x.reshape(shape) for x in (p,m,v)))
        ps,ms,vs = (np.stack([f[i] for f in fields]) for i in range(3))
        xy = c['cand_xy']
        bounds = np.stack([np.floor(xy.min(0)/2)*2,np.ceil(xy.max(0)/2)*2])
        return PreparedFrozenHurdle(xy,c['aps'],np.clip(ps.mean(0),1e-4,1-1e-4),
            ms.mean(0),vs.mean(0)+ms.var(0),bounds,c['ap_count'])


def task(C,H,floor,K,draw,rng):
    _, previous, data = E.task(C,H,floor,K,draw,rng)
    _,_,_,_,_,base_s,qr,hr = data
    # A real group remains coherent when its scans are thinned.
    if draw >= 2:
        hr = {f:one_per_position(C,rows,rng) for f,rows in hr.items()}
    R = int(rng.choice([1,3]))
    sr = one_per_position(C,base_s,rng) if R == 1 else base_s
    histories = {f:dict(rss=C.RSS[r],xy=C.XY[r]) for f,r in hr.items()}
    support = dict(rss=C.RSS[sr],xy=C.XY[sr])
    c = context(histories,support,floor)
    Pt = H.STATS[(0,floor)][0]
    D = np.linalg.norm(c['cand_xy'][:,None]-Pt[None],axis=2)
    Pq,n,nd,mean,var = H.det_stats(C.RSS[qr],C.XY[qr])
    index = {tuple(xy):i for i,xy in enumerate(Pq)}
    label = np.array([index.get(tuple(Pt[i]),-1) for i in D.argmin(1)])
    ok = (D.min(1)<=T.R_LABEL)&(label>=0)
    j,aps = label[ok],c['aps']
    values = (c['features'][ok].reshape(-1,WIDTH),np.repeat(n[j],len(aps)),
              *[a[j][:,aps].ravel() for a in (nd,mean,var)])
    record = dict(floor=floor,K=K,mode=E.MODES[draw],R=R,H_group=previous['H_group'],
        S_group=previous['S_group'],Q_group=previous['Q_group'],
        H_csv_rows={str(f):C.ROW_IDS[r].tolist() for f,r in hr.items()},
        base_S_csv_rows=C.ROW_IDS[base_s].tolist(),S_csv_rows=C.ROW_IDS[sr].tolist(),
        Q_csv_rows=C.ROW_IDS[qr].tolist(),candidates=len(c['cand_xy']),eligible_candidates=int(ok.sum()),
        APs=aps.tolist(),features_sha256=array_sha(c['features']))
    return values,record,(histories,support,c,base_s,sr,qr,hr)


def cell_table(C,H,records):
    def build(pairs,rng,draws=4,Ks=(1,3,5),max_rows=6000):
        assert draws == 4 and Ks == (1,3,5) and max_rows == 6000
        parts = []
        for building,floor in pairs:
            assert building == 0
            for K in Ks:
                for draw in range(draws):
                    values,record,_ = task(C,H,floor,K,draw,rng)
                    size = len(values[0])
                    selected = rng.choice(size,min(size,max_rows),replace=False)
                    parts.append(tuple(v[selected] for v in values))
                    records.append(dict(record,eligible_cells=size,selected_cells=len(selected),
                        cell_sampling='Uniform without replacement, independent of labels',
                        selected_indices_sha256=array_sha(selected)))
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(5))
    return build
