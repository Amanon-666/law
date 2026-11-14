"""Run approved matched subsets serially, with no retries or monitoring agent."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110')
SOURCE = Path('/home/panyushuo/projects/panyushuo/claude_fp_wt/current_20261110/src')
WORK = Path(__file__).resolve().parent
METHODS = ['finetune_dnn', 'maml_dnn', 'pretrain_dnn']
PLAN = [
    dict(name='uts_dl_seed0', dataset='uts', building='New IT', floors=[4, 8],
         methods=METHODS, output='paper_uts_F4_F8_dl_seed0_20261115',
         reference='paper_uts_New_IT_20261111'),
    dict(name='uts_scaml', dataset='uts', building='New IT', floors=[4, 8],
         methods=['scaml'], output='paper_uts_F4_F8_scaml_20261115',
         reference='paper_uts_New_IT_20261111'),
    dict(name='tampere_scaml', dataset='tampere', building='TIE1', floors=[0],
         methods=['scaml'], output='paper_tampere_TIE1_F0_scaml_20261115',
         reference='paper_tampere_TIE1_20261111'),
]
sys.path.insert(0, str(SOURCE))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def prepare():
    from law_bench import run
    from law_bench.data import load
    from law_bench.protocol import make_tasks

    hashes = {n: digest(SOURCE/'law_bench'/n) for n in
        ['methods_dl.py', 'run_claude.py', 'run.py', 'data.py', 'protocol.py', 'score.py', 'method_scaml.py']}
    assert hashes['methods_dl.py'] == 'fa981e883ea6dcbaa15c0b2ca7537a83091aca27afd6ea849c963dd73419c60a'
    assert hashes['run_claude.py'] == 'dff15deb8202dc722b8ac14c4cc59dec9f89adce65e39ffee361827045631613'
    assert hashes['method_scaml.py'] == 'de98aba42600750c976bbc5aa96bb9902fc2571a89dbd4695966b0bc2f4e7008'
    for name in ['data.py', 'protocol.py', 'score.py', 'run.py', 'method_scaml.py']:
        assert hashes[name] == digest(BASE/'src/law_bench'/name)
    source_buildings, source_axes = load('uji')
    source_identity = run._data_identity(source_buildings, source_axes)
    tasks = []
    for stage in PLAN:
        assert not (BASE/'results'/stage['output']).exists(), stage['output']
        reference = BASE/'results'/stage['reference']
        original = read(reference/'freeze.json')
        buildings, axes = load(stage['dataset'])
        floors, aps = buildings[stage['building']], axes[stage['building']]
        target_identity = run._data_identity({stage['building']: floors}, {stage['building']: aps})
        assert target_identity == original['target_data']
        assert source_identity == original['source_data']
        for floor in stage['floors']:
            for task, truth in make_tasks(stage['dataset'], stage['building'], floors, aps, floor,
                    budgets=(1, 3, 5), repeat=0, seed=20261111, R=3, query_positions=60):
                path = reference/'events/refi'/(task.task_id+'.json')
                event = read(path)
                assert event['status'] == 'completed'
                assert event['query_row_indices'] == truth['row_indices'].tolist()
                assert event['query_groups'] == truth['groups'].tolist()
                assert event['meta'] == task.meta
                tasks.append(dict(stage=stage['name'], task_id=task.task_id,
                    reference=str(path), reference_sha256=digest(path),
                    H_floors=sorted(task.H), H_rows={f: len(d['rss']) for f, d in task.H.items()},
                    S_rows=len(task.S['rss']), meta=task.meta,
                    query_row_indices=event['query_row_indices'], query_groups=event['query_groups']))
    freeze = dict(created_before_training_utc=datetime.now(timezone.utc).isoformat(),
        plan=PLAN, source_code_sha256=hashes, source_data=source_identity, task_bindings=tasks,
        protocol=dict(name='static-resample-v2', seed=20261111, budgets=[1, 3, 5],
            repeat=0, R=3, query_positions=60, training_seed=0),
        review=dict(
            DL='Original source-only HP selection over first two UJI buildings and first two floors, then full deployment H pretraining/meta-training and S adaptation; K1,3,5 run in original order',
            DL_representation='AP-vector xy regressor; cross-building UJI supplies HP development, deployment network training uses same-building H; FOMAML is not named MetaLoc',
            DL_scale='AP axis and coordinate scale use only H and current S; cache key includes exact AP axis and scale, so original retraining behavior is retained',
            DL_steps='Original grids, all optimization steps and CPU2 arithmetic retained; pretrain_dnn is auxiliary and has its own training cost',
            ScaML='Original CPU float64 locked BoTorch0.7.3/GPyTorch1.9, original RSS adaptation, restarts5/maxiter500; Task H source models cached only by original exact H/AP/frame identity across K',
            ScaML_assets='No prior external ScaML task results or compatible persisted source fits found; retained-GP fields are not ScaML source models',
            Query='Q RSS only enters prediction after fitting; Query labels remain with scorer; task row/group/meta bindings equal original paper',
            scope='UTS uses tower floors4..12 only; each target excludes itself and retains all other eight H floors; TIE1F0 retains all other five H floors',
            failure='No retries; budget stop preserves completed events and unfinished status, no shortened optimization result is accepted'),
        resource_policy=dict(device='cpu', threads=2, NN_budget_seconds=3600,
            combined_ScaML_budget_seconds=7200, sequential_datasets=True),
        resource_snapshot=read(WORK/'resource_snapshot.json'))
    write(WORK/'freeze.json', freeze)
    return tasks


def validate(stage, bindings):
    output = BASE/'results'/stage['output']
    records = []
    for method in stage['methods']:
        for binding in bindings:
            if binding['stage'] != stage['name']:
                continue
            path = output/'events'/method/(binding['task_id']+'.json')
            event = read(path)
            assert event['status'] == 'completed', (path, event.get('error'))
            for name in ['meta', 'query_row_indices', 'query_groups']:
                assert event[name] == binding[name], (path, name)
            records.append(dict(method=method, task_id=event['task_id'],
                seconds=event['seconds'], source_path=str(path), source_sha256=digest(path)))
    write(output/'matched_task_validation.json', dict(passed=True, records=records))
    return len(records)


def stop(process):
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    bindings = prepare()
    write(WORK/'options_seed0.json', {m: dict(seed=0) for m in METHODS})
    state = dict(status='running', start_epoch=time.time(), attempts=[], outputs=[])
    write(WORK/'batch_state.json', state)
    scaml_deadline = None
    for stage in PLAN:
        if stage['methods'] == ['scaml']:
            if not (WORK/'scaml_ready.json').exists():
                state['status'] = 'unfinished_scaml_dependencies_not_ready'
                break
            if scaml_deadline is None:
                scaml_deadline = time.monotonic()+7200
            remaining = scaml_deadline-time.monotonic()
        else:
            remaining = 3600
        if remaining <= 0:
            state['status'] = 'unfinished_scaml_budget_exhausted'
            break
        if stage['name'] == 'uts_dl_seed0':
            command = [sys.executable, '-u', str(WORK/'entry_dl.py')]
        else:
            command = [sys.executable, '-u', '-m', 'law_bench.run']
        command += ['--dataset', stage['dataset'], '--building', stage['building'],
            '--source-dataset', 'uji', '--source-buildings', 'B0', 'B1', 'B2',
            '--protocol', 'static-resample-v2', '--seed', '20261111',
            '--floors', *map(str, stage['floors']), '--budgets', '1', '3', '5',
            '--repeats', '0', '--R', '3', '--query-positions', '60',
            '--methods', *stage['methods'], '--output', str(BASE/'results'/stage['output'])]
        if stage['name'] == 'uts_dl_seed0':
            command += ['--method-options', str(WORK/'options_seed0.json')]
        start = time.time()
        with (WORK/(stage['name']+'.log')).open('w') as log:
            process = subprocess.Popen(['nice', '-n', '10', *command], cwd=SOURCE,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['active'] = dict(stage=stage['name'], pid=process.pid, start_epoch=start,
                output=str(BASE/'results'/stage['output']), command=command)
            write(WORK/'batch_state.json', state)
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                stop(process)
                state['status'] = 'unfinished_budget_exhausted'
                state['attempts'].append(dict(stage=stage['name'], pid=process.pid,
                    returncode=process.returncode, seconds=time.time()-start, unfinished=True))
                break
        attempt = dict(stage=stage['name'], pid=process.pid, returncode=returncode,
            seconds=time.time()-start)
        state['attempts'].append(attempt)
        if returncode != 0:
            state['status'] = 'failed'
            break
        try:
            attempt['validated_events'] = validate(stage, bindings)
        except (AssertionError, FileNotFoundError, ValueError) as error:
            state['status'] = 'validation_failed'
            state['error'] = repr(error)
            break
        state['outputs'].append(str(BASE/'results'/stage['output']))
        state.pop('active')
        write(WORK/'batch_state.json', state)
    else:
        state['status'] = 'completed'
    state.pop('active', None)
    state['end_epoch'] = time.time()
    state['elapsed_seconds'] = state['end_epoch']-state['start_epoch']
    write(WORK/'batch_state.json', state)
    return 0 if state['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
