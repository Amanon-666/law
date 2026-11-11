"""Fit one frozen five-seed source recipe using coherent acquisition tasks.

Same original HurdleField and FrozenHurdleTranslator, including all original
estimator settings and the original raw probability reader. Source-only B0;
the separate formal four-arm Tab training remains paused. This command does
not select a recipe from target results or run any localization evaluation.
"""
import claude_hurdle_assets as A  # original CPU2/offline environment, before numpy
import argparse
from pathlib import Path
import socket
import time
from unittest.mock import patch

import joblib
from threadpoolctl import threadpool_limits

from candidates import acquisition_tasks_20261111 as E


def train(csv_path, check_path, output, resume):
    check = E.json.loads(check_path.read_text())
    assert check['complete'] and check['phase'] == 'source_task_check_only'
    assert check['code_sha256'] == A.digest(E.__file__) and len(check['tasks']) == 48
    assert A.digest(csv_path) == E.CSV_SHA == check['csv_sha256']
    C, V, H = E.load_source(csv_path)
    code = {str(Path(p).relative_to(A.ROOT)): A.digest(p) for p in
            (Path(__file__), E.__file__, A.ROOT/'frozen_hurdle_translator.py',
             A.ROOT/'candidates/finite_observation_20261111/source.py',
             A.PACKAGE/'hurdle.py', A.PACKAGE/'vt_lib.py', Path(E.T.__file__))}
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        A.write(output/'freeze.json',dict(recipe='coherent_acquisition_55_v1',
            created=A.datetime.now(A.timezone.utc).isoformat(),seeds=A.SEEDS,draws=4,
            source_buildings=[0],source_floors=sorted(V.FLOORS[0]),code=code,
            source_rows_by_floor={str(f):int((C.FLR == f).sum()) for f in V.FLOORS[0]},
            upstream_git_blobs=A.BLOBS,check_sha256=A.digest(check_path),csv_sha256=E.CSV_SHA,
            modes=list(E.MODES),features=55,estimator='Unchanged upstream HurdleField',
            changed='Coherent acquisition H/S/Q tasks; S rows excluded from labels',
            group_role='Source task organization only; never a learned input or deployment requirement',
            reader=dict(beta=128.,eps=.02,offsets=[-8.,-4.,0.,4.,8.],
                        empirical_S='Unchanged',GP_pool='Unchanged probability_half'),
            evaluation=dict(next='One complete fixed strict54 UJI retention comparison',
                manifest_sha256='bb7db6eb8ce070a400f18057af46dc68f734f5db6f60235f4c60189e62d3c496',
                baseline_sha256='03b72571ab1bf415474d5da7db5261d6594e380410a3a27b5a42a031e5d8a71c',
                scope='Exposed retention evidence, not independent transfer proof',
                metrics='Original hierarchical means and position p90, all K/floors/buildings; keep GP and original best references',
                selection='All five fixed seeds; no target-selected reader, weight or seed; no automatic adoption',
                boundary='UTS/IPIN-specific tuning stays stopped'),
            resources=dict(CPU_threads=2,GPU=0,new_downloads=0,timeout_seconds=600)))
    frozen = E.json.loads((output/'freeze.json').read_text())
    assert frozen['code'] == code and frozen['check_sha256'] == A.digest(check_path)
    frozen_sha = A.digest(output/'freeze.json')
    records = []
    for seed in A.SEEDS:
        meta = output/f'seed_{seed}.json'
        if meta.exists():
            record = E.json.loads(meta.read_text())
            assert record['freeze_sha256'] == frozen_sha
            assert A.digest(output/record['path']) == record['sha256']
            records.append(record)
            continue
        started, tasks = time.monotonic(), []
        with patch.object(H,'cell_table',E.cell_table(C,H,tasks)):
            field = H.HurdleField([0],seed=seed,draws=4)
        task_path = output/f'seed_{seed}_tasks.json'
        A.write(task_path,dict(seed=seed,tasks=tasks,freeze_sha256=frozen_sha))
        payload = dict(kind='claude_hurdle_v1',recipe=frozen['recipe'],seed=seed,draws=4,
            source_buildings=[0],source_floors=frozen['source_floors'],
            source_rows_by_floor=frozen['source_rows_by_floor'],cols=field.cols,freeze_sha256=frozen_sha,
            **{n:dict(estimator=getattr(field,n).est,cols=getattr(field,n).cols) for n in ('det','mean','scale')})
        path = output/f'seed_{seed}.joblib'
        joblib.dump(payload,path)
        record = dict(seed=seed,path=path.name,sha256=A.digest(path),freeze_sha256=frozen_sha,
            source_tasks=dict(path=task_path.name,sha256=A.digest(task_path)),seconds=time.monotonic()-started)
        A.write(meta,record)
        records.append(record)
        print(E.json.dumps(record),flush=True)
    A.write(output/'complete.json',dict(complete=True,recipe=frozen['recipe'],freeze_sha256=frozen_sha,
        source_buildings=[0],source_floors=frozen['source_floors'],
        source_rows_by_floor=frozen['source_rows_by_floor'],models=records,localization_scores=0,adopted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--source-check',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web'
    with threadpool_limits(limits=2):
        train(args.csv,args.source_check,args.output,args.resume)
