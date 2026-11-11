"""Unchanged full-H retained GP and original fixed pool on official UTS tasks.

Build each floor/AP history field once, then reuse it across target floors
and K. No history subsampling, target-trained hyperparameters or replacement
GP. Field fitting and prediction are bounded, resumable server-only phases.
Query XY is never loaded here. The separate evaluator owns all scoring.
"""
from candidates.uts_official_evaluation_20261111 import (
    ROOT, DATA, A_OUTPUT, A_FREEZE, read, table, digest, now, write, inputs_for)
import argparse
from pathlib import Path
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import history_fit
from uji import group_scans
from law_bench.method_retained_gp import RetainedGPMethod, history_key
from hurdle_gp_pool import fuse_hurdle_gp

OUTPUT = ROOT/'results/uts_official_GP_20261111'
PARAMETERS = ROOT/'results/paired_assets_20261106/parameters.json'
PARAM_SHA = 'f14dbededc1fb40630ec1c11a4ae49abd0f1098d01d31987141c99529ef369de'
CODE = ('candidates/uts_official_gp_20261111.py', 'candidates/uts_official_evaluation_20261111.py',
        'candidates/uts_official_20261111.py', 'law_bench/method_retained_gp.py',
        'history_fit.py','uji.py','fields.py','observations.py','adaptation.py',
        'predict.py','episodic_tasks.py','position_fusion_gp.py',
        'cached_retained_readout.py','building_information_gp_trial.py',
        'posterior.py','hurdle_gp_pool.py')


def histories(train):
    return {int(f):dict(rss=train['rss'][train['floor']==f],xy=train['xy'][train['floor']==f],
                acq=np.full(int((train['floor']==f).sum()),-1,dtype=np.int64))
            for f in sorted(set(train['floor']))}


def freeze():
    assert digest(A_OUTPUT/'freeze.json') == A_FREEZE and digest(PARAMETERS) == PARAM_SHA
    train = table(DATA/'UTS_training.csv',history=True)
    identity = SimpleNamespace(dataset='uts',building='FEIT',ap_ids=train['aps'],H=histories(train))
    fields = []
    for floor,h in identity.H.items():
        key = history_key(identity,floor,PARAM_SHA)
        fields.append(dict(floor=floor,key=key,rows=len(h['rss']),
            positions=len(np.unique(h['xy'],axis=0)),APs=np.flatnonzero((h['rss']!=100).any(0)).tolist()))
    OUTPUT.mkdir(exist_ok=False)
    write(OUTPUT/'freeze.json',dict(created=now(),A_freeze_sha256=A_FREEZE,
        parameters_sha256=PARAM_SHA,code={p:digest(ROOT/p) for p in CODE},histories=fields,
        scope='All16 original raw history floors, each field reused only in tasks where it is H',
        source='Original paired UJI B0 F0-F2 parameters; no global fitting',
        field_solver='Original full-covariance VI, max640, projected gradient<=1e-3',
        adaptation='Unchanged RetainedGPMethod, all15 H sources and real3-scan S positions',
        fusion='Original probability_half; no field recomputation or parameter selection',
        query_XY_opened=False,resource_segment_seconds=540,CPU_threads=2,GPU=0))


def check():
    frozen = read(OUTPUT/'freeze.json')
    assert digest(A_OUTPUT/'freeze.json') == frozen['A_freeze_sha256'] == A_FREEZE
    assert digest(PARAMETERS) == frozen['parameters_sha256'] == PARAM_SHA
    for p,sha in frozen['code'].items():
        assert digest(ROOT/p) == sha
    source = read(A_OUTPUT/'freeze.json')
    for item in source['inputs']:
        assert digest(item['path']) == item['sha256']
    return frozen,source


def fit_histories(seconds):
    frozen,_ = check()
    all_h = histories(table(DATA/'UTS_training.csv',history=True))
    parameters = read(PARAMETERS)
    start = time.monotonic()
    records = []
    for h in frozen['histories']:
        data = all_h[h['floor']]
        origin = data['xy'].mean(0)
        scans = group_scans(data['rss'],data['xy']-origin)
        directory = OUTPUT/'cache'/h['key']
        directory.mkdir(parents=True,exist_ok=True)
        for ap in h['APs']:
            path = directory/f'{ap}.npz'
            metadata = path.with_suffix('.json')
            if metadata.exists():
                saved = read(metadata)
                assert saved['history_key'] == h['key'] and digest(path) == saved['sha256']
            else:
                if time.monotonic()-start >= seconds:
                    return 2
                solver = {}
                original = history_fit.minimize
                def monitored(*args,**kwargs):
                    result = original(*args,**kwargs)
                    gradient = np.asarray(result.jac).copy()
                    for j,(lo,hi) in enumerate(kwargs['bounds']):
                        if lo is not None and result.x[j]<=lo+1e-10 and gradient[j]>0: gradient[j]=0
                        if hi is not None and result.x[j]>=hi-1e-10 and gradient[j]<0: gradient[j]=0
                    solver.update(success=bool(result.success),iterations=int(result.nit),
                        projected_gradient=float(np.abs(gradient).max()),message=str(result.message))
                    if not (np.isfinite(result.fun) and np.isfinite(result.x).all()
                            and np.isfinite(gradient).all() and solver['projected_gradient']<=1e-3):
                        raise RuntimeError(f'Unfinished UTS GP history F{h["floor"]}/AP{ap}: {solver}')
                    return result
                tick = time.monotonic()
                with patch.object(history_fit,'minimize',monitored):
                    field = history_fit.single_history(dict(scans=scans),ap,
                        parameters['base'],parameters['obs'],steps=640)
                np.savez_compressed(path,z=field.z.numpy(),mean=field.m.numpy(),cov=field.c.numpy(),origin=origin)
                write(metadata,dict(history_key=h['key'],ap_id=read(A_OUTPUT/'freeze.json')['APs'][ap],
                    sha256=digest(path),solver=solver,seconds=time.monotonic()-tick))
            records.append(dict(path=str(metadata),sha256=digest(metadata)))
        print(f'History F{h["floor"]}: {len(h["APs"])} fields complete',flush=True)
    write(OUTPUT/'histories_complete.json',dict(complete=True,created=now(),
        freeze_sha256=digest(OUTPUT/'freeze.json'),records=records))
    return 0


def predict(seconds,max_events):
    _,source = check()
    complete = read(OUTPUT/'histories_complete.json')
    assert complete['complete'] and complete['freeze_sha256'] == digest(OUTPUT/'freeze.json')
    for r in complete['records']:
        assert digest(r['path']) == r['sha256']
    train,query = table(DATA/'UTS_training.csv',history=True),table(DATA/'UTS_test.csv',history=False)
    gp = RetainedGPMethod(parameters_path=PARAMETERS,cache_dir=OUTPUT/'cache',fit_missing_history=False,
        provenance=dict(parameters_sha256=PARAM_SHA,training_sources=[dict(dataset='uji',building='B0')],
            evidence=str(ROOT/'results/building_information_assets_20261109/freeze.json')+'#source_calibration'))
    gp.fit_source([dict(dataset='uji',building='B0'),dict(dataset='uji',building='B2')])
    start,n = time.monotonic(),0
    records = []
    for event in source['events']:
        destination = OUTPUT/event['event']
        record_path = destination/'complete.json'
        if not record_path.exists():
            if time.monotonic()-start >= seconds or n>=max_events:
                return 2
            a = read(A_OUTPUT/event['event']/'complete.json')
            assert a['freeze_sha256'] == A_FREEZE and a['event']==event
            for item in a['files']:
                assert digest(item['path']) == item['sha256']
            task = inputs_for(event,train,query)
            tick = time.monotonic()
            result = gp.predict_scores(task)
            with np.load(A_OUTPUT/event['event']/'field.npz') as field, np.load(A_OUTPUT/event['event']/'query.npz') as aq:
                assert np.array_equal(field['aps'],result['aps'])
                assert aq['rows'].tolist()==event['Q_rows_zero_based']
                fused = fuse_hurdle_gp(field['cand_xy'],aq['log_position_mass'],
                    result['grid'],result['scores'],result['raw'],field['bounds'])
            destination.mkdir(exist_ok=False)
            np.savez_compressed(destination/'predictions.npz',GP_projected=result['projected'],
                GP_raw=result['raw'],GP_scores=result['scores'],GP_grid=result['grid'],
                probability_half=fused['predictions']['probability_half'],rows=event['Q_rows_zero_based'])
            write(destination/'diagnostics.json',gp.diagnostics)
            write(record_path,dict(complete=True,created=now(),event=event,seconds=time.monotonic()-tick,
                freeze_sha256=digest(OUTPUT/'freeze.json'),
                files=[dict(path=str(destination/f),sha256=digest(destination/f))
                       for f in ('predictions.npz','diagnostics.json')],query_xy_opened=False))
            n+=1
        record = read(record_path)
        assert record['freeze_sha256']==digest(OUTPUT/'freeze.json') and record['event']==event
        for item in record['files']:
            assert digest(item['path'])==item['sha256']
        records.append(dict(path=str(record_path),sha256=digest(record_path)))
        print('Completed GP/pool',event['event'],flush=True)
    write(OUTPUT/'predictions_complete.json',dict(complete=True,created=now(),
        freeze_sha256=digest(OUTPUT/'freeze.json'),records=records,scored=False))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','fit_histories','predict'))
    parser.add_argument('--seconds',type=float,default=540)
    parser.add_argument('--max-events',type=int,default=1)
    args = parser.parse_args()
    assert socket.gethostname()=='ray-web' and 0<args.seconds<=540 and args.max_events>0
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    with threadpool_limits(limits=2):
        code = freeze() if args.phase=='freeze' else (fit_histories(args.seconds)
            if args.phase=='fit_histories' else predict(args.seconds,args.max_events))
    raise SystemExit(0 if code is None else code)
