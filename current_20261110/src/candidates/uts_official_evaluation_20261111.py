"""Evaluate the frozen official UTS tasks without altering their predictions.

Original A is read from its sealed cache. Two existing WKNN rules use the
same H/S and Query rows. All predictions are sealed before scorer-only XY is
opened. Report every floor, nested-K paired changes and large errors; keep
the retained-GP/full-system comparison explicitly pending.

Coverage diagnostics describe geometry and AP overlap, not an information
theoretic ceiling. Only distance outside the candidate convex hull is an
output-representation lower bound for A's posterior mean.
"""
from candidates.uts_official_20261111 import (
    ROOT, DATA, OUTPUT as A_OUTPUT, read, table, digest, now, write)
import argparse
import csv
import socket

import numpy as np
from scipy.spatial import ConvexHull
from threadpoolctl import threadpool_limits
from law_bench.task import Task
from law_bench.methods_simple import WKNN_S, WKNN_HS
from law_bench.score import score as position_score

OUTPUT = ROOT/'results/uts_official_evaluation_20261111'
A_FREEZE = '473405b52fee93de4d4f1c169af86ca9bb92457d4cbf53d5004ebff25ea87fe9'
CODE = ('candidates/uts_official_evaluation_20261111.py',
        'candidates/uts_official_20261111.py', 'law_bench/task.py',
        'law_bench/methods_simple.py', 'law_bench/score.py',
        'building_information_tab_trial.py')


def freeze():
    assert digest(A_OUTPUT/'freeze.json') == A_FREEZE
    OUTPUT.mkdir(exist_ok=False)
    write(OUTPUT/'freeze.json', dict(created=now(), A_freeze_sha256=A_FREEZE,
        code={p:digest(ROOT/p) for p in CODE},
        methods=['original_A02', 'wknn_S', 'wknn_HS'],
        metrics=dict(primary='Within-floor unique-XY balanced Euclidean error, then pooled positions',
            secondary='Scan-weighted errors and equal-floor mean; every floor retained',
            summaries=['mean','median','p90','worst_decile_mean','within5m','within10m','over20m'],
            paired_K=[[1,3],[3,5],[1,5]],
            coverage=['distance to all H positions','distance to represented +/-4-floor H',
                      'distance to S','Query detections absent from H or local H',
                      'A candidate convex-hull output lower bound']),
        interpretation=dict(single_fixed_support_draw=True,
            new_training=False, scoring_after_all_predictions=True,
            retained_GP_and_full_system='Pending; WKNN is a simple reference, not a substitute',
            acquisition_identity='Unknown; no cross-device independence claim',
            historical_scope='All other floors supplied; A directly encodes only offsets +/-1..4')))


def check():
    frozen = read(OUTPUT/'freeze.json')
    assert digest(A_OUTPUT/'freeze.json') == frozen['A_freeze_sha256'] == A_FREEZE
    for p,sha in frozen['code'].items():
        assert digest(ROOT/p) == sha
    inputs = read(A_OUTPUT/'freeze.json')
    for item in inputs['inputs']:
        assert digest(item['path']) == item['sha256']
    return frozen, inputs


def inputs_for(event, train, query):
    floor = event['floor']
    H = {}
    for f in sorted(set(train['floor'])):
        if f != floor:
            rows = train['floor'] == f
            H[int(f)] = dict(rss=train['rss'][rows], xy=train['xy'][rows],
                            acq=np.full(int(rows.sum()), -1, dtype=np.int64))
    rows = event['S_rows_zero_based']
    S = dict(rss=train['rss'][rows], xy=train['xy'][rows],
             acq=np.full(len(rows), -1, dtype=np.int64))
    return Task('uts','FEIT',floor,event['K'],0,train['aps'],H,S,
                query['rss'][event['Q_rows_zero_based']],
                dict(protocol='official-UTS',R=3,acquisition_separated=False))


def baselines():
    _, frozen = check()
    train = table(DATA/'UTS_training.csv', history=True)
    query = table(DATA/'UTS_test.csv', history=False)
    records = []
    for event in frozen['events']:
        task = inputs_for(event,train,query)
        predictions = {m.name:m.predict(task).tolist() for m in (WKNN_S(),WKNN_HS())}
        records.append(dict(event=event['event'],rows=event['Q_rows_zero_based'],
                            predictions=predictions))
    assert not (OUTPUT/'baselines.json').exists()
    write(OUTPUT/'baselines.json', dict(created=now(),complete=True,records=records,
        freeze_sha256=digest(OUTPUT/'freeze.json'),query_xy_opened=False))


def metrics(errors):
    errors = np.asarray(errors)
    count = max(1,int(np.ceil(len(errors)*.1)))
    return dict(n=len(errors),mean=float(errors.mean()),median=float(np.median(errors)),
        p90=float(np.quantile(errors,.9)),worst_decile_mean=float(np.sort(errors)[-count:].mean()),
        within5m=float(np.mean(errors<=5)),within10m=float(np.mean(errors<=10)),
        over20m=float(np.mean(errors>20)))


def hull_distance(points, candidates):
    hull = ConvexHull(np.unique(candidates,axis=0))
    vertices = hull.points[hull.vertices]
    edge = np.roll(vertices,-1,axis=0)-vertices
    delta = points[:,None]-vertices[None]
    t = np.clip((delta*edge).sum(-1)/(edge**2).sum(-1),0,1)
    distance = np.linalg.norm(delta-t[:,:,None]*edge,axis=-1).min(1)
    inside = (points@hull.equations[:,:2].T+hull.equations[:,2] <= 1e-9).all(1)
    return np.where(inside,0,distance)


def score():
    frozen, source = check()
    completed = read(A_OUTPUT/'A_complete.json')
    assert completed['complete'] and completed['freeze_sha256'] == A_FREEZE
    assert len(completed['records']) == len(source['events']) == 48
    for record in completed['records']:
        assert digest(record['path']) == record['sha256']
        for item in read(record['path'])['files']:
            assert digest(item['path']) == item['sha256']
    baseline = read(OUTPUT/'baselines.json')
    assert baseline['complete'] and baseline['freeze_sha256'] == digest(OUTPUT/'freeze.json')
    assert [r['event'] for r in baseline['records']] == [e['event'] for e in source['events']]
    # From here onwards only: scorer opens the official Query coordinates.
    with (DATA/'UTS_test.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    truth = np.array([[float(r['Pos_x']),float(r['Pos_y'])] for r in rows])
    train = table(DATA/'UTS_training.csv', history=True)
    query = table(DATA/'UTS_test.csv', history=False)
    records = []
    for event, simple in zip(source['events'],baseline['records']):
        task = inputs_for(event,train,query)
        qxy = truth[event['Q_rows_zero_based']]
        positions, groups = np.unique(qxy,axis=0,return_inverse=True)
        groups = groups.ravel()
        with np.load(A_OUTPUT/event['event']/'query.npz') as saved:
            assert saved['rows'].tolist() == simple['rows'] == event['Q_rows_zero_based']
            predictions = dict(simple['predictions'],original_A02=saved['projected'].tolist())
        with np.load(A_OUTPUT/event['event']/'field.npz') as field:
            lower_bound = hull_distance(qxy,field['cand_xy'])
        hxy = np.concatenate([h['xy'] for h in task.H.values()])
        local = [h for f,h in task.H.items() if abs(f-task.floor)<=4]
        local_xy = np.concatenate([h['xy'] for h in local])
        qdet = task.Q_rss != 100
        hseen = np.concatenate([h['rss'] for h in task.H.values()]) != 100
        local_seen = np.concatenate([h['rss'] for h in local]) != 100
        coverage = dict(
            history_distance=np.linalg.norm(qxy[:,None]-hxy[None],axis=-1).min(1).tolist(),
            local_history_distance=np.linalg.norm(qxy[:,None]-local_xy[None],axis=-1).min(1).tolist(),
            support_distance=np.linalg.norm(qxy[:,None]-task.S['xy'][None],axis=-1).min(1).tolist(),
            query_detections=qdet.sum(1).tolist(),
            detections_absent_H=(qdet & ~hseen.any(0)).sum(1).tolist(),
            detections_absent_local_H=(qdet & ~local_seen.any(0)).sum(1).tolist(),
            A_output_lower_bound=lower_bound.tolist())
        scores = {name:position_score(pred,dict(xy=qxy,groups=groups)) for name,pred in predictions.items()}
        assert np.all(np.asarray(scores['original_A02']['scan_errors'])+1e-7 >= lower_bound)
        records.append(dict(event=event['event'],floor=event['floor'],K=event['K'],
            rows=event['Q_rows_zero_based'],positions=positions.tolist(),scores=scores,coverage=coverage,
            predictions=predictions))
    summary, paired = {}, {}
    for method in frozen['methods']:
        summary[method],paired[method] = {},{}
        for k in (1,3,5):
            selected = [r for r in records if r['K']==k]
            summary[method][str(k)] = dict(
                positions=metrics([e for r in selected for e in r['scores'][method]['position_errors']]),
                scans=metrics([e for r in selected for e in r['scores'][method]['scan_errors']]),
                equal_floor_mean=float(np.mean([r['scores'][method]['mean'] for r in selected])),
                by_floor={str(r['floor']):metrics(r['scores'][method]['position_errors']) for r in selected})
        for a,b in frozen['metrics']['paired_K']:
            differences, floors = [],{}
            for floor in range(-3,13):
                ra = next(r for r in records if r['floor']==floor and r['K']==a)
                rb = next(r for r in records if r['floor']==floor and r['K']==b)
                assert ra['positions'] == rb['positions'] and ra['rows'] == rb['rows']
                delta = np.asarray(rb['scores'][method]['position_errors'])-ra['scores'][method]['position_errors']
                differences.extend(delta.tolist())
                floors[str(floor)] = float(delta.mean())
            delta = np.asarray(differences)
            paired[method][f'{a}_to_{b}'] = dict(n=len(delta),mean_change=float(delta.mean()),
                improved=int((delta < -1e-8).sum()),worsened=int((delta > 1e-8).sum()),
                improved_over5m=int((delta < -5).sum()),worsened_over5m=int((delta > 5).sum()),
                floor_mean_changes=floors)
    assert not (OUTPUT/'scores.json').exists()
    write(OUTPUT/'scores.json',dict(created=now(),complete=True,freeze_sha256=digest(OUTPUT/'freeze.json'),
        A_complete_sha256=digest(A_OUTPUT/'A_complete.json'),baseline_sha256=digest(OUTPUT/'baselines.json'),
        summary=summary,paired=paired,records=records,full_system_complete=False,
        remaining='Retained GP and fixed full-system pool; no automatic adoption'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','baselines','score'))
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web'
    with threadpool_limits(limits=2):
        globals()[args.phase]()
