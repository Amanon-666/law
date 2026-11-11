"""Fixed unified77/count checkpoint on all48 existing official UTS tasks.

This is a uniform external comparison, not resumed UTS adaptation. H/S,
coordinates, all16 floors, query rows and GP score maps are exactly the old
official protocol. Use the same five B0 models, representation, count reader
and half pool already evaluated on UJI/IPIN. No data-dependent branch,
target fitting, new-AP treatment, source retraining or parameter selection.

Forward/reverse review: published metric coordinates and known floor supply
H/S; learned fields on their unique union plus current Query mask/RSS supply
position mass; cached same-task GP supplies the fixed pool. Target XY is only
opened by score after all48 predictions seal. Count conditioning loses count
location information, and cannot cancel arbitrary AP-specific/RSS changes.
Candidate-hull/coverage limits from the old evaluator still apply; identical
candidate coordinates are checked. Three scans per S are within source R1/3.
Unknown UTS phone metadata is neither imputed nor used. The old A02 reference
uses two source buildings; that comparison is not same-source causal evidence.

Decision frozen before inference: complete all floors and K, preserve failures,
do not select a special UTS model or retry with changed settings. Two CPU
threads, existing offline data/models/GP only, bounded resumable segments.
"""
from candidates import uts_official_evaluation_20261111 as E
from candidates import uts_official_gp_20261111 as G
from candidates.unified_scan_field_20261111 import UnifiedScanField
from candidates.count_conditioned_20261111.reader import CountConditionedHurdle
from hurdle_gp_pool import fuse_hurdle_gp
import argparse
import csv
from pathlib import Path
import socket
import time

import numpy as np
from threadpoolctl import threadpool_limits

MODELS = E.ROOT/'results/unified_scan_b0_ens5_20261111'
MODEL_SHA = '62503b5c82297ffdb170e2119f3769b8594cd958419088f277523d082ccc4992'
RECIPE = 'unified_scan_field_77_v1_count'
METHODS = ('unified_count','unified_count_GP')


def binding(path):
    return dict(path=str(path),sha256=E.digest(path))


def check(output):
    frozen = E.read(output/'freeze.json')
    for name,sha in frozen['code'].items():
        assert E.digest(E.ROOT/name) == sha
    for record in frozen['inputs']:
        assert E.digest(record['path']) == record['sha256']
    return frozen


def freeze(args):
    _,tasks = G.check();E.check()
    source = E.read(MODELS/'complete.json')
    assert source['complete'] and source['freeze_sha256'] == E.digest(MODELS/'freeze.json') == MODEL_SHA
    assert source['source_buildings'] == [0]
    gp = E.read(G.OUTPUT/'predictions_complete.json')
    assert gp['complete'] and len(gp['records']) == len(tasks['events']) == 48
    paths = [MODELS/n for n in ('freeze.json','complete.json')]
    paths += [MODELS/r['path'] for r in source['models']]
    paths += [E.A_OUTPUT/'freeze.json',G.OUTPUT/'freeze.json',G.OUTPUT/'predictions_complete.json',
              E.OUTPUT/'full_scores.json',E.ROOT/'results/uji54_unified_count_20261111/complete.json',
              E.ROOT/'results/ipin2025_unified_count_20261111/scores.json']
    paths += [E.DATA/n for n in ('UTS_training.csv','UTS_test.csv')]
    for record,event in zip(gp['records'],tasks['events']):
        assert E.digest(record['path']) == record['sha256']
        saved = E.read(record['path']);assert saved['event'] == event
        paths.append(Path(record['path']))
        for item in saved['files']:
            assert E.digest(item['path']) == item['sha256']
            paths.append(Path(item['path']))
        old = E.read(E.A_OUTPUT/event['event']/'complete.json')
        paths.append(E.A_OUTPUT/event['event']/'complete.json')
        for item in old['files']:
            assert E.digest(item['path']) == item['sha256']
            paths.append(Path(item['path']))
    code = dict(E.read(MODELS/'freeze.json')['code'])
    for name in (*E.CODE,'candidates/count_conditioned_20261111/reader.py',
                 'candidates/uts_official_gp_20261111.py','hurdle_gp_pool.py',
                 str(Path(__file__).relative_to(E.ROOT))):
        code[name] = E.digest(E.ROOT/name)
    args.output.mkdir(parents=True,exist_ok=False)
    E.write(args.output/'freeze.json',dict(created=E.now(),recipe=RECIPE,code=code,
        inputs=[binding(p) for p in dict.fromkeys(paths)],events=tasks['events'],APs=tasks['APs'],
        model_freeze_sha256=MODEL_SHA,methods=METHODS,protocol=tasks['protocol'],
        review=__doc__,source_buildings=[0],reference_source_buildings=[0,2],
        adopted=False,query_xy_opened=False,resources=dict(CPU_threads=2,GPU=0,fits=0,
            GP_fits=0,downloads=0,segment_seconds=args.seconds,estimated_total_minutes=50)))


def predict(args):
    frozen = check(args.output);start = time.monotonic()
    train = E.table(E.DATA/'UTS_training.csv',history=True)
    query = E.table(E.DATA/'UTS_test.csv',history=False)
    assert train['aps'] == query['aps'] == frozen['APs']
    model = UnifiedScanField(MODELS);records=[]
    for event in frozen['events']:
        directory = args.output/event['event'];path = directory/'complete.json'
        if path.exists():
            record = E.read(path)
            assert record['freeze_sha256'] == E.digest(args.output/'freeze.json')
            assert E.digest(record['prediction']['path']) == record['prediction']['sha256']
        else:
            if time.monotonic()-start >= args.seconds:
                return 2
            tick = time.monotonic();floor=event['floor'];rows=event['S_rows_zero_based']
            histories = {f'UTSF{f}':dict(rss=train['rss'][train['floor']==f],xy=train['xy'][train['floor']==f])
                         for f in sorted(set(train['floor'])) if f != floor}
            support = dict(rss=train['rss'][rows],xy=train['xy'][rows])
            field = model.prepare(histories,support,f'UTSF{floor}')
            with np.load(E.A_OUTPUT/event['event']/'field.npz') as old:
                assert np.array_equal(field.cand_xy,np.unique(old['cand_xy'],axis=0))
                for name in ('aps','bounds','ap_count'):
                    assert np.array_equal(getattr(field,name),old[name])
            predicted = CountConditionedHurdle(field).predict(query['rss'][event['Q_rows_zero_based']])
            with np.load(G.OUTPUT/event['event']/'predictions.npz') as gp:
                assert gp['rows'].tolist() == event['Q_rows_zero_based']
                fused = fuse_hurdle_gp(field.cand_xy,predicted['log_position_mass'],
                    gp['GP_grid'],gp['GP_scores'],gp['GP_raw'],field.bounds)
                difference = float(np.max(abs(fused['predictions']['GP']-gp['GP_projected'])))
                assert difference <= 1e-8
            directory.mkdir(exist_ok=False);prediction_path=directory/'predictions.npz'
            np.savez_compressed(prediction_path,unified_count=predicted['projected'],
                unified_count_GP=fused['predictions']['probability_half'],rows=event['Q_rows_zero_based'])
            record = dict(complete=True,created=E.now(),event=event,seconds=time.monotonic()-tick,
                freeze_sha256=E.digest(args.output/'freeze.json'),prediction=binding(prediction_path),
                native_GP_max_difference=difference,candidate_AP_bounds_unchanged=True,query_xy_opened=False)
            E.write(path,record)
        records.append(binding(path))
        print(f'Sealed {len(records)}/48 {event["event"]}',flush=True)
    E.write(args.output/'predictions_complete.json',dict(complete=True,created=E.now(),
        freeze_sha256=E.digest(args.output/'freeze.json'),records=records,query_xy_opened=False))
    return 0


def score(args):
    frozen = check(args.output);sealed=E.read(args.output/'predictions_complete.json')
    assert sealed['complete'] and len(sealed['records']) == len(frozen['events']) == 48
    assert sealed['freeze_sha256'] == E.digest(args.output/'freeze.json')
    for record in sealed['records']:
        assert E.digest(record['path']) == record['sha256']
        p=E.read(record['path'])['prediction'];assert E.digest(p['path']) == p['sha256']
    base = E.read(E.OUTPUT/'full_scores.json')
    assert base['full_system_complete']
    with (E.DATA/'UTS_test.csv').open(newline='') as stream:
        truth=np.array([[float(r['Pos_x']),float(r['Pos_y'])] for r in csv.DictReader(stream)])
    for event,record in zip(frozen['events'],base['records']):
        assert record['event'] == event['event'] and record['rows'] == event['Q_rows_zero_based']
        xy=truth[record['rows']];positions,groups=np.unique(xy,axis=0,return_inverse=True)
        assert record['positions'] == positions.tolist()
        with np.load(args.output/event['event']/'predictions.npz') as p:
            for method in METHODS:
                record['scores'][method] = E.position_score(p[method],dict(xy=xy,groups=groups.ravel()))
                record['predictions'][method] = p[method].tolist()
    for method in METHODS:
        base['summary'][method],base['paired'][method]={},{}
        for k in (1,3,5):
            records=[r for r in base['records'] if r['K']==k]
            base['summary'][method][str(k)] = dict(
                positions=E.metrics([v for r in records for v in r['scores'][method]['position_errors']]),
                scans=E.metrics([v for r in records for v in r['scores'][method]['scan_errors']]),
                equal_floor_mean=float(np.mean([r['scores'][method]['mean'] for r in records])),
                by_floor={str(r['floor']):E.metrics(r['scores'][method]['position_errors']) for r in records})
        for lo,hi in ((1,3),(3,5),(1,5)):
            differences=[];floors={}
            for floor in range(-3,13):
                a,b=(next(r for r in base['records'] if r['floor']==floor and r['K']==k) for k in (lo,hi))
                delta=np.asarray(b['scores'][method]['position_errors'])-a['scores'][method]['position_errors']
                differences.extend(delta.tolist());floors[str(floor)]=float(delta.mean())
            d=np.asarray(differences)
            base['paired'][method][f'{lo}_to_{hi}'] = dict(n=len(d),mean_change=float(d.mean()),
                improved=int((d < -1e-8).sum()),worsened=int((d > 1e-8).sum()),
                improved_over5m=int((d < -5).sum()),worsened_over5m=int((d > 5).sum()),floor_mean_changes=floors)
    base.update(created=E.now(),recipe=RECIPE,freeze_sha256=E.digest(args.output/'freeze.json'),
        predictions_complete_sha256=E.digest(args.output/'predictions_complete.json'),
        original_full_scores=binding(E.OUTPUT/'full_scores.json'),source_buildings=[0],
        reference_source_buildings=[0,2],adopted=False,
        interpretation='Fixed checkpoint transfer; older A02 has more source buildings, not a same-source causal comparison')
    assert not (args.output/'scores.json').exists()
    E.write(args.output/'scores.json',base)
    return 0


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','predict','score'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=480.)
    args=parser.parse_args();args.output=args.output.resolve()
    assert socket.gethostname()=='ray-web' and 0 < args.seconds <= 540
    with threadpool_limits(limits=2):
        status=globals()[args.phase](args)
    raise SystemExit(0 if status is None else status)
