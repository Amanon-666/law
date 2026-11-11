"""Check then fit the single frozen unified-scan-field recipe on B0 only.

The source check resolves the actual information counterexamples and verifies
the complete H/S/Q path. It is not a localization experiment. Five seeds are
fixed in advance; trained assets use the original estimator serialization and
checkpoint at seed boundaries. No dataset-dependent reader or target tuning.
"""
import claude_hurdle_assets as A  # CPU2/offline before numerical imports
import argparse
from pathlib import Path
import socket
import time
from unittest.mock import patch

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from candidates import unified_scan_field_20261111 as U


def bindings():
    files = [Path(__file__),Path(U.__file__),Path(U.E.__file__),Path(U.T.__file__),
             A.ROOT/'candidates/finite_observation_20261111/features.py',
             A.ROOT/'candidates/finite_observation_20261111/source.py',
             A.ROOT/'frozen_hurdle_translator.py',A.PACKAGE/'hurdle.py',A.PACKAGE/'vt_lib.py']
    return {str(p.relative_to(A.ROOT)):A.digest(p) for p in files}


def check(csv_path,output):
    C,V,H = U.E.load_source(csv_path)
    rng = np.random.default_rng(20261111)
    groups = np.column_stack([C.USER,C.PHONE])
    records = []
    for f in V.FLOORS[0]:
        for K in (1,3,5):
            for draw in range(4):
                values,r,data = U.task(C,H,f,K,draw,rng)
                histories,support,c,base,sr,qr,hr = data
                assert not np.intersect1d(base,qr).size and np.isin(sr,base).all()
                assert np.all(groups[sr] == r['S_group']) and np.all(groups[qr] == r['Q_group'])
                assert (r['S_group'] == r['Q_group']) == (draw%2 == 0)
                assert all(h != f and (C.FLR[rows] == h).all() for h,rows in hr.items())
                if draw >= 2:
                    assert np.all(groups[np.concatenate(list(hr.values()))] == r['H_group'])
                    assert all(np.unique(C.XY[rows],axis=0,return_counts=True)[1].max() == 1 for rows in hr.values())
                assert np.unique(C.XY[sr],axis=0,return_counts=True)[1].tolist() == [r['R']]*K
                expected = np.unique(np.concatenate([*[d['xy'] for d in histories.values()],support['xy']]),axis=0)
                assert np.array_equal(c['cand_xy'],expected)
                visible = np.concatenate([*hr.values(),sr])
                assert np.array_equal(c['aps'],np.flatnonzero(C.DET[visible].any(0)))
                st = c['support_stats']; lookup = {tuple(x):i for i,x in enumerate(c['cand_xy'])}
                at_s = c['features'][[lookup[tuple(x)] for x in st['xy']],:,71:75]
                truth = np.stack([np.broadcast_to(st['n'][:,None],st['nd'].shape),
                                  st['nd'],st['rss_sum'],st['rss_square_sum']],axis=2)[:,c['aps']]
                assert np.array_equal(at_s,truth.astype(np.float32))
                assert values[0].shape[1] == 77 and len(values[0]) > 0
                assert (values[1] > 0).all() and (values[2] <= values[1]).all()
                records.append(r)
    assert {r['R'] for r in records} == {1,3}
    # The two exact aliases from the previous turn must now be distinguishable.
    hs = {0:dict(xy=np.array([[0.,0.],[10.,0.]]),rss=np.full((2,1),-65.))}
    features = []
    for rows in ([-45.,-45.,100.],[-55.,-65.,-75.],[-65.,-65.,-65.]):
        features.append(U.context(hs,dict(xy=np.zeros((3,2)),rss=np.array(rows)[:,None]),1)['features'])
    assert not np.array_equal(features[0],features[2],equal_nan=True)
    assert not np.array_equal(features[1],features[2],equal_nan=True)
    empty = U.context(hs,dict(xy=np.empty((0,2)),rss=np.empty((0,1))),1)
    assert np.isnan(empty['features'][:,:,43:55]).all()
    assert (empty['features'][:,:,71:75] == 0).all()
    output.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists()
    A.write(output,dict(complete=True,phase='source_information_path_check',code=bindings(),
        csv_sha256=U.E.CSV_SHA,recipe=U.RECIPE,tasks=records,
        support_statistics_preserved_at_every_support_position=True,
        both_previous_aliases_resolved=True,unique_candidate_union=True,
        fits=0,models_loaded=0,localization_scores=0))
    print(U.E.json.dumps(dict(complete=True,tasks=len(records),aliases_resolved=True,fits=0)),flush=True)


def train(csv_path,checked_path,output,resume,seconds):
    started = time.monotonic()
    checked = U.E.json.loads(checked_path.read_text())
    assert checked['complete'] and checked['code'] == bindings()
    assert A.digest(csv_path) == checked['csv_sha256'] == U.E.CSV_SHA
    C,V,H = U.E.load_source(csv_path)
    if not resume:
        output.mkdir(parents=True,exist_ok=False)
        A.write(output/'freeze.json',dict(created=A.datetime.now(A.timezone.utc).isoformat(),
            recipe=U.RECIPE,feature_schema=U.SCHEMA,seeds=A.SEEDS,draws=4,code=bindings(),
            source_buildings=[0],source_floors=sorted(V.FLOORS[0]),
            source_rows_by_floor={str(f):int((C.FLR==f).sum()) for f in V.FLOORS[0]},
            upstream_git_blobs=A.BLOBS,source_check_sha256=A.digest(checked_path),csv_sha256=U.E.CSV_SHA,
            training=dict(modes=list(U.E.MODES),R='Uniformly draw1 or3; unused reserved base-S rows stay out of Q',
                histories='Full history or one coherent group with one actual scan per recorded position',
                candidates='Unique union of H and S coordinates in training AND deployment',
                input='Original55 plus history16; nearest S n/nd/sum/squared-sum and two global exposure summaries',
                cell_sampling='Uniform at most6000 eligible cells per task, label-independent; no info+5% filter',
                labels='Real source query-group scans, nearest original target coordinate within4m; no reassignment if unavailable',
                learner='Original HurdleField heads, scan-count weights, cross-fitting and estimator settings'),
            deployment=dict(field='Same learned predictor at every H/S candidate; no empirical-S append',
                reader=dict(beta=128.,eps=.02,offsets=[-8.,-4.,0.,4.,8.]),GP_pool='Existing fixed probability_half'),
            review=dict(query_identity='Never a feature or deployment requirement',
                limits='Nearest support statistics retained; global interactions still summarized; unknown query acquisition not identified',
                counterexamples='Old detection/strength and variance aliases resolved; emptyS and singleS paths checked',
                attribution='Whole information-flow recipe, not a one-feature causal comparison'),
            evaluation=dict(first='Complete exposed strict54 retention; original and previous fixed candidates retained',
                no_target_selected_settings=True,no_automatic_adoption=True,source_only_training=True),
            resources=dict(CPU_threads=2,GPU=0,downloads=0,segment_seconds=seconds)))
    frozen = U.E.json.loads((output/'freeze.json').read_text())
    assert frozen['code'] == bindings() and frozen['source_check_sha256'] == A.digest(checked_path)
    freeze_sha = A.digest(output/'freeze.json'); models = []
    for seed in A.SEEDS:
        meta = output/f'seed_{seed}.json'
        if meta.exists():
            r = U.E.json.loads(meta.read_text())
            assert r['freeze_sha256'] == freeze_sha and A.digest(output/r['path']) == r['sha256']
            models.append(r); continue
        if time.monotonic()-started > seconds-120:
            return 2  # preserve complete seeds; explicit continuation, no timeout restart
        tick,tasks = time.monotonic(),[]
        with patch.object(H,'cell_table',U.cell_table(C,H,tasks)):
            field = H.HurdleField([0],seed=seed,draws=4)
        tp = output/f'seed_{seed}_tasks.json'; A.write(tp,dict(seed=seed,tasks=tasks,freeze_sha256=freeze_sha))
        payload = dict(kind='claude_hurdle_v1',recipe=U.RECIPE,seed=seed,draws=4,cols=field.cols,
            source_buildings=[0],source_floors=frozen['source_floors'],source_rows_by_floor=frozen['source_rows_by_floor'],
            freeze_sha256=freeze_sha,**{n:dict(estimator=getattr(field,n).est,cols=getattr(field,n).cols)
                                      for n in ('det','mean','scale')})
        path = output/f'seed_{seed}.joblib'; joblib.dump(payload,path)
        r = dict(seed=seed,path=path.name,sha256=A.digest(path),freeze_sha256=freeze_sha,
                 source_tasks=dict(path=tp.name,sha256=A.digest(tp)),elapsed_seconds=time.monotonic()-tick)
        A.write(meta,r); models.append(r); print(U.E.json.dumps(r),flush=True)
    A.write(output/'complete.json',dict(complete=True,recipe=U.RECIPE,freeze_sha256=freeze_sha,models=models,
        source_buildings=[0],source_floors=frozen['source_floors'],source_rows_by_floor=frozen['source_rows_by_floor'],
        localization_scores=0,adopted=False))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('check','train'))
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--source-check',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--seconds',type=float,default=540.)
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web' and 120 < args.seconds <= 540
    with threadpool_limits(limits=2):
        code = check(args.csv,args.output) if args.phase == 'check' else train(
            args.csv,args.source_check,args.output,args.resume,args.seconds)
    raise SystemExit(0 if code is None else code)
