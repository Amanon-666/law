"""Real acquisition episodes for the original 55-feature HurdleField.

Mechanism fixed before fitting: keep the original field estimator, source
floor cross-fitting, cell selection/weights, and deployment reader. Change
only the joint distribution of H, S and the source labels. Each episode uses
one recorded acquisition group for S; labels come from either that group
(excluding S rows) or another group. H is either the full history or one
coherent recorded group across floors. No fabricated scans/offsets/dropouts.

Why: the completed finite-observation audit found both recorded groups mixed
in all 72 single-scan history entries and all 24 label sets. Thinning those
mixtures changes exposure, but does not present a coherent acquisition shift.
Group identities organise SOURCE data only and are never model features.

Inference remains p(scan | H,S,x) marginal over unobserved query acquisition.
The four episode modes assign equal task frequency to same/different groups;
scan-count weights remain original, so this is not group-balanced risk.
This assumes source acquisition relations are informative for future scenes.
It cannot identify arbitrary unseen per-AP shifts from sparse S, and a single
Gaussian cannot represent every multimodal acquisition mixture. Two observed
source groups are limited diversity, not a guarantee for unseen devices.

Reverse review: Q supplies only n/nd/mean/variance at the original nearest
target position; H/S alone choose candidates, APs, and every input feature.
The reference position is chosen before restricting Q, so unavailable labels
are skipped rather than reassigned. Source S rows never appear in labels.
At deployment no acquisition ID, target labels or dataset-specific switch is
required. Empty S has the original NaN support representation; unseen APs
enter only when H or S detects them. Empty H is outside this field method.

This is one training-organisation candidate, not a new deployed structure.
Success requires a frozen recipe to improve transfer while retaining existing
capability; source training loss alone is not an adoption criterion. It has
no authority to restart the separately paused four-arm Tab experiment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import socket

import numpy as np

from candidates.finite_observation_20261111.source import load_source, array_sha, CSV_SHA
from vertical_translator import translator as T


MODES = ('full_same', 'full_different', 'group_same', 'group_different')


def context(histories, support, floor):
    """Exactly the original input features; no acquisition metadata or Q."""
    maps = {f: T.position_map(d['rss'], d['xy']) for f, d in histories.items()}
    candidates = np.unique(np.concatenate([m[0] for m in maps.values()]), axis=0)
    seen = (support['rss'] != 100).any(0)
    for data in histories.values():
        seen |= (data['rss'] != 100).any(0)
    aps = np.flatnonzero(seen)
    F, ref = T.history_features(candidates, floor, maps)
    F = F.astype(np.float32)
    if len(support['xy']):
        sref = T.naive_ref_at(support['xy'], floor, maps)
        S = T.support_features(candidates, ref, support['xy'], support['rss'], sref, F)
    else:
        S = np.full((*ref.shape, 12), np.nan)
    features = np.concatenate([F[:, aps], S.astype(np.float32)[:, aps]], axis=2)
    return candidates, aps, features


def task(C, H, floor, K, draw, rng):
    """C contains one permitted source building; USER/PHONE are recorded groups."""
    groups = np.column_stack([C.USER, C.PHONE])
    target = np.flatnonzero(C.FLR == floor)
    unique = np.unique(groups[target], axis=0)
    if len(unique) < 2:
        raise ValueError('Cross-acquisition source tasks require two recorded groups')
    sg = int(rng.integers(len(unique)))
    qg = sg if draw % 2 == 0 else int(rng.choice(np.delete(np.arange(len(unique)), sg)))
    s_pool = target[(groups[target] == unique[sg]).all(1)]
    P, inverse = np.unique(C.XY[s_pool], axis=0, return_inverse=True)
    chosen = rng.choice(len(P), K, replace=False)
    srows = np.concatenate([rng.permutation(s_pool[inverse.ravel() == i])[:3] for i in chosen])
    qrows = target[(groups[target] == unique[qg]).all(1)]
    qrows = qrows[~np.isin(qrows, srows)]
    all_h = np.flatnonzero(C.FLR != floor)
    hg = None
    if draw >= 2:
        available = np.unique(groups[all_h], axis=0)
        hg = available[int(rng.integers(len(available)))]
        all_h = all_h[(groups[all_h] == hg).all(1)]
    hrows = {int(f): all_h[C.FLR[all_h] == f] for f in np.unique(C.FLR[all_h])}
    histories = {f: dict(rss=C.RSS[r], xy=C.XY[r]) for f, r in hrows.items()}
    support = dict(rss=C.RSS[srows], xy=C.XY[srows])
    candidates, aps, features = context(histories, support, floor)
    Pt = H.STATS[(0, floor)][0]
    D = np.linalg.norm(candidates[:, None] - Pt[None], axis=2)
    nearest = D.argmin(1)
    Pq, n, nd, mean, variance = H.det_stats(C.RSS[qrows], C.XY[qrows])
    lookup = {tuple(p): i for i, p in enumerate(Pq)}
    label = np.array([lookup.get(tuple(Pt[i]), -1) for i in nearest])
    ok = (D.min(1) <= T.R_LABEL) & (label >= 0)
    j = label[ok]
    table = (features[ok].reshape(-1, 55), np.repeat(n[j], len(aps)),
             *[a[j][:, aps].ravel() for a in (nd, mean, variance)])
    record = dict(floor=floor, K=K, mode=MODES[draw],
        H_group=None if hg is None else hg.tolist(),
        S_group=unique[sg].tolist(), Q_group=unique[qg].tolist(),
        H_csv_rows={str(f): C.ROW_IDS[r].tolist() for f, r in hrows.items()},
        S_csv_rows=C.ROW_IDS[srows].tolist(), Q_csv_rows=C.ROW_IDS[qrows].tolist(),
        candidates=len(candidates), eligible_candidates=int(ok.sum()), APs=aps.tolist(),
        feature_sha256=array_sha(features), label_statistics_sha256=[array_sha(v) for v in table[1:]],
        skipped_candidates_without_Q=int(((D.min(1) <= T.R_LABEL) & (label < 0)).sum()))
    return table, record, (histories, support, candidates, aps, features, srows, qrows, hrows)


def cell_table(C, H, records):
    """Drop-in replacement for original hurdle.cell_table during source fitting."""
    def build(pairs, rng, draws=4, Ks=(1, 3, 5), max_rows=6000):
        assert draws == 4 and Ks == (1, 3, 5) and max_rows == 6000
        parts = []
        for building, floor in pairs:
            assert building == 0  # this source-only pilot uses the existing B0 asset
            for K in Ks:
                for draw in range(draws):
                    values, record, _ = task(C, H, floor, K, draw, rng)
                    X, n, nd, mean, variance = values
                    info = (np.nan_to_num(X[:, 0:40:5], nan=T.ND).max(1) > T.ND) | (nd > 0) | \
                           (np.nan_to_num(X[:, 44], nan=T.ND) > T.ND)
                    selected = np.flatnonzero(info | (rng.random(len(X)) < .05))
                    if len(selected) > max_rows:
                        selected = rng.choice(selected, max_rows, replace=False)
                    parts.append(tuple(v[selected] for v in values))
                    records.append(dict(record, selected_cells=len(selected),
                                        selected_indices_sha256=array_sha(selected)))
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(5))
    return build


def check(csv_path, output):
    """No fit: verify true group separation, input permissions and deployment path."""
    C, V, H = load_source(csv_path)
    records = []
    rng = np.random.default_rng(20261111)
    groups = np.column_stack([C.USER, C.PHONE])
    for floor in V.FLOORS[0]:
        for K in (1, 3, 5):
            for draw in range(4):
                table, record, data = task(C, H, floor, K, draw, rng)
                histories, support, xy, aps, F, sr, qr, hr = data
                assert not np.intersect1d(sr, qr).size
                assert np.all(groups[sr] == record['S_group'])
                assert np.all(groups[qr] == record['Q_group'])
                assert (record['S_group'] == record['Q_group']) == (draw % 2 == 0)
                assert len(np.unique(C.XY[sr], axis=0)) == K
                assert all(f != floor and (C.FLR[r] == f).all() for f, r in hr.items())
                if draw >= 2:
                    assert np.all(groups[np.concatenate(list(hr.values()))] == record['H_group'])
                visible = np.concatenate([*hr.values(), sr])
                assert np.array_equal(aps, np.flatnonzero(C.DET[visible].any(0)))
                assert F.shape == (len(xy), len(aps), 55) and len(table[0]) > 0
                assert (table[1] > 0).all() and (table[2] <= table[1]).all()
                # Query RSS is not passed into context; replacing its observations
                # must not change any training input or visible AP identity.
                old = C.RSS[qr].copy()
                C.RSS[qr] = 100
                again = context(histories, support, floor)
                C.RSS[qr] = old
                assert np.array_equal(F, again[2], equal_nan=True)
                records.append(record)
    # The source constructor and deployment share the original empty-S convention.
    empty = dict(rss=support['rss'][:0], xy=support['xy'][:0])
    no_s = context(histories, empty, floor)
    assert np.isnan(no_s[2][:, :, 43:]).all()
    result = dict(complete=True,phase='source_task_check_only',tasks=records,
        csv_sha256=CSV_SHA,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_buildings=[0],feature_width=55,modes=list(MODES),
        inputs_exclude_query=True,group_ids_in_model_features=False,
        fits=0,localization_scores=0,adopted=False,
        remaining='Freeze source fitting and evaluation before any training; no performance claim')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(complete=True,tasks=len(records),fits=0,output=str(output))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web'
    check(args.csv,args.output)
