"""Official UTS inputs for the existing original five-model sources02 asset.

Keep every published floor and original XY; do not filter floors or register
coordinates by AP peaks. CNNLoc (Song et al., IEEE Access 2019), section III-E,
defines XY relative to the building's west/south sides. This published frame
is the evaluation assumption, not a guarantee inferred from RF similarity.
https://doi.org/10.1109/ACCESS.2019.2933921

H: all training scans on OTHER floors. S: K nested random recorded positions,
exactly3 real scans each. Only positions with >=3 scans enter the sampling
pool. Q: every official test row on the known floor, unchanged across K.
S selection never reads Q coordinates/RSS. Official test/S coordinate overlap
is retained, not removed with privileged Q labels. Phone metadata is unknown.

This prepares the learned component and saves reusable fields and query
probabilities for the full system. It does not score localization, fit models,
or claim a completed GP comparison. No local execution or new downloads.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import csv
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
from threadpoolctl import threadpool_limits
from building_information_tab_trial import digest, now, write
from frozen_hurdle_translator import FrozenHurdleTranslator

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/panyushuo/datasets/utsindoorloc')
MODELS = ROOT/'results/claude_hurdle_sources02_ens5_20261111'
MODEL_FREEZE = 'f852bd606eb1e1e67e5a2e965aa4f86f3da0ed3128bcc865b018fcc1562e0bd3'
AUTHOR_BLOBS = {'UTS_training.csv': '93155fe4f9448d7cd367bf31dd3dc469bd854f03',
                'UTS_test.csv': 'ac8a466f535ed0340c38d054548abbf769750c7c'}
OUTPUT = ROOT/'results/uts_official_A02_20261111'


def read(path):
    return json.loads(Path(path).read_text())


def table(path, *, history):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        aps = [c for c in reader.fieldnames if c.startswith('WAP')]
        rows = list(reader)
    result = dict(aps=aps, floor=np.array([int(r['Floor_ID']) for r in rows]),
                  rss=np.array([[float(r[c]) for c in aps] for r in rows]))
    if history:
        result['xy'] = np.array([[float(r['Pos_x']), float(r['Pos_y'])] for r in rows])
    # Query coordinates are never converted, returned or passed to a method.
    return result


def freeze(args):
    assert digest(MODELS/'freeze.json') == MODEL_FREEZE
    complete = read(MODELS/'complete.json')
    assert complete['complete'] and complete['freeze_sha256'] == MODEL_FREEZE
    assert complete['source_buildings'] == [0, 2]
    inputs = []
    for name, blob in AUTHOR_BLOBS.items():
        path = DATA/name
        raw = path.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == blob
        inputs.append(dict(path=str(path), sha256=digest(path), author_git_blob=blob))
    train = table(DATA/'UTS_training.csv', history=True)
    # Freeze only test row IDs and known-floor labels; no Query values/XY.
    with (DATA/'UTS_test.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert [c for c in reader.fieldnames if c.startswith('WAP')] == train['aps']
        qfloor = np.array([int(r['Floor_ID']) for r in reader])
    assert sorted(set(train['floor'])) == list(range(-3, 13))
    assert set(qfloor) <= set(train['floor'])
    events = []
    for floor in sorted(set(train['floor'])):
        rows = np.flatnonzero(train['floor'] == floor)
        positions, inv, count = np.unique(train['xy'][rows], axis=0, return_inverse=True, return_counts=True)
        pool = np.flatnonzero(count >= 3)
        assert len(pool) >= 5
        rng = np.random.default_rng([20261111, int(floor)+3])
        chosen = rng.permutation(pool)[:5]
        support = np.concatenate([rng.permutation(rows[inv.ravel() == i])[:3] for i in chosen])
        query = np.flatnonzero(qfloor == floor)
        for k in (1, 3, 5):
            events.append(dict(event=f'F{floor}_K{k}', floor=int(floor), K=k,
                S_rows_zero_based=support[:3*k].tolist(), Q_rows_zero_based=query.tolist(),
                eligible_S_positions=len(pool), all_target_training_positions=len(positions),
                history_rows=int((train['floor'] != floor).sum())))
    code = ['candidates/uts_official_20261111.py', 'frozen_hurdle_translator.py',
            'vertical_translator/translator.py', 'building_information_tab_trial.py']
    inputs += [dict(path=str(MODELS/name), sha256=digest(MODELS/name)) for name in ('freeze.json','complete.json')]
    inputs += [dict(path=str(MODELS/r['path']), sha256=r['sha256']) for r in complete['models']]
    args.output.mkdir(exist_ok=False)
    write(args.output/'freeze.json', dict(created=now(), inputs=inputs,
        code={p:digest(ROOT/p) for p in code}, APs=train['aps'], events=events,
        source=dict(buildings=['UJI B0','UJI B2'], five_models=True, new_training=False),
        protocol=dict(all_published_floors=list(range(-3,13)), K=[1,3,5], R=3,
            history='Every raw training row on every other floor; no target training rows outside S',
            query='Official test split; all rows, fixed across K; true XY withheld from predictor',
            coordinates='Published building west/south origin; no RF-derived transformation or floor exclusion',
            calibration='Seed-fixed recorded-position sample with>=3 real scans; not uniform over floor area',
            acquisitions='User=root and Phone_type=unknow are not identified independent acquisitions',
            status='UTS was previously studied; task-matched external evaluation, not asserted blind',
            stages='Freeze all48 tasks; save A fields/predictions; strong GP/full-system work remains; no scoring here',
            resources=dict(CPU_threads=2,GPU=0,new_fits=0,segment_seconds=args.seconds))))


def prepare(args):
    start = time.monotonic()
    frozen = read(args.output/'freeze.json')
    for p,sha in frozen['code'].items():
        assert digest(ROOT/p) == sha
    for record in frozen['inputs']:
        assert digest(record['path']) == record['sha256']
    train, query = table(DATA/'UTS_training.csv', history=True), table(DATA/'UTS_test.csv', history=False)
    assert train['aps'] == query['aps'] == frozen['APs']
    model = FrozenHurdleTranslator(MODELS)
    records = []
    for event in frozen['events']:
        directory = args.output/event['event']
        record_path = directory/'complete.json'
        if record_path.exists():
            record = read(record_path)
            assert record['freeze_sha256'] == digest(args.output/'freeze.json')
            for item in record['files']:
                assert digest(item['path']) == item['sha256']
        else:
            if time.monotonic()-start >= args.seconds:
                return 2
            floor = event['floor']
            histories = {f'UTSF{f}':dict(rss=train['rss'][train['floor']==f], xy=train['xy'][train['floor']==f])
                         for f in sorted(set(train['floor'])) if f != floor}
            rows = event['S_rows_zero_based']
            support = dict(rss=train['rss'][rows],xy=train['xy'][rows])
            field = model.prepare(histories,support,f'UTSF{floor}')
            prediction = field.predict(query['rss'][event['Q_rows_zero_based']])
            directory.mkdir(exist_ok=False)
            np.savez_compressed(directory/'field.npz', **{k:getattr(field,k) for k in
                ('cand_xy','aps','p','m','s2','bounds','ap_count')})
            np.savez_compressed(directory/'query.npz', projected=prediction['projected'],
                log_position_mass=prediction['log_position_mass'], rows=event['Q_rows_zero_based'])
            record = dict(complete=True,created=now(),event=event,freeze_sha256=digest(args.output/'freeze.json'),
                candidate_rows=len(field.cand_xy),visible_APs=len(field.aps),query_xy_opened=False,
                files=[dict(path=str(directory/name),sha256=digest(directory/name)) for name in ('field.npz','query.npz')])
            write(record_path,record)
        records.append(dict(event=event['event'],path=str(record_path),sha256=digest(record_path)))
        print(f"Sealed {len(records)}/48 {event['event']}",flush=True)
    write(args.output/'A_complete.json',dict(complete=True,created=now(),freeze_sha256=digest(args.output/'freeze.json'),
        records=records,scored=False,full_system_complete=False))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','prepare'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--seconds',type=float,default=540.)
    args = parser.parse_args()
    args.output=args.output.resolve()
    assert socket.gethostname()=='ray-web' and 0<args.seconds<=540
    with threadpool_limits(limits=2):
        code = freeze(args) if args.phase=='freeze' else prepare(args)
    raise SystemExit(0 if code is None else code)
