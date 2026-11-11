"""Authorized serial A4 runs and lightweight result publication."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SRC = Path('/home/panyushuo/projects/panyushuo/claude_fp_wt/current_20261110/src')
REPO = SRC.parent.parent
ASSETS = SRC / 'claude_field_posterior_20261111'
RESULTS = SRC / 'results'
PYTHON = '/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python'
MAC_NET = '/home/panyushuo/projects/panyushuo/.network/mac-net'
BRANCH = 'research/claude-field-posterior-20261111'
STATE = RESULTS / 'bench_a4_execution_20261111.json'

if len(sys.argv) > 1 and sys.argv[1] == '--child':
    sys.path.insert(0, str(SRC))
    from law_bench import ours
    original_episodes = ours.episodes
    def recorded_episodes(*args, **kwargs):
        result = original_episodes(*args, **kwargs)
        print(json.dumps({'episode_cells': len(result[0]), 'feature_columns': result[0].shape[1]}), flush=True)
        return result
    ours.episodes = recorded_episodes
    from law_bench import run_claude
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    run_claude.R.main()
    sys.exit(0)

os.chdir(SRC)
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
    PYTHONPATH='/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages',
    VT_DIR=str(SRC / 'claude_hurdle_20261110'), LAW_UJI_CSV='/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv')
started = time.time()
state = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'driver_pid': os.getpid(), 'runs': []}
def save():
    STATE.write_text(json.dumps(state, indent=2) + '\n')
def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()
save()
try:
    for target, source in [('B0','B1'), ('B0','B2'), ('B1','B0'), ('B1','B2'), ('B2','B0'), ('B2','B1')]:
        if time.time() - started >= 55 * 60:
            state['status'] = 'time_limit_no_new_run'
            break
        freeze = json.loads((RESULTS / f'bench_uji_{target}_A_20261111/freeze.json').read_text())['configuration']
        output = RESULTS / f'bench_uji_{target}_1src_{source}_20261111'
        log_path = output.with_suffix('.log')
        command = [PYTHON, '-u', str(Path(__file__).resolve()), '--child', '--dataset', 'uji', '--building', target, '--source-buildings', source, '--methods', 'ours_A_1src', '--budgets', *map(str, freeze['budgets']), '--repeats', *map(str, freeze['repeats']), '--seed', str(freeze['seed']), '--R', str(freeze['R']), '--query-positions', str(freeze['query_positions']), '--output', str(output)]
        record = {'target': target, 'source': source, 'output': str(output), 'log': str(log_path), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': command}
        state['runs'].append(record)
        with log_path.open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            record['python_pid'] = process.pid
            state['status'] = 'running'
            save()
            print(json.dumps(record), flush=True)
            rc = process.wait()
        record.update(exit_code=rc, ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if rc:
            raise RuntimeError(f'{target}/{source} exit {rc}')
        events = [json.loads(p.read_text()) for p in (output/'events').rglob('*.json')]
        record['events'] = len(events)
        record['failed_events'] = [e['task_id'] for e in events if e['status'] != 'completed']
        record['unavailable'] = json.loads((output/'task_inventory.json').read_text())['unavailable']
        state['status'] = 'run_completed'
        save()
        subprocess.run([MAC_NET, 'git', 'fetch', 'origin', BRANCH], cwd=REPO, check=True)
        if git('rev-list', '--count', 'HEAD..FETCH_HEAD') != '0':
            raise RuntimeError('Collaborator advanced branch; preserve commits and stop publication')
        paths = [Path(__file__).resolve(), STATE, log_path, SRC/'law_bench/methods_claude.py', SRC/'law_bench/run_claude.py', *output.rglob('*.json')]
        for path in paths:
            subprocess.run(['git','add','-f',str(path.relative_to(REPO))], cwd=REPO, check=True)
        subprocess.run(['git','diff','--cached','--check'], cwd=REPO, check=True)
        subprocess.run(['git','commit','-m',f'Record A4 UJI {target} single source {source}'], cwd=REPO, check=True)
        record['commit'] = git('rev-parse','HEAD')
        subprocess.run([MAC_NET,'git','push','origin','HEAD:'+BRANCH], cwd=REPO, check=True)
        record['pushed'] = True
        save()
        print(json.dumps(record), flush=True)
        if record['failed_events']:
            raise RuntimeError('Failed events; stop subsequent runs')
    else:
        state['status'] = 'completed'
except Exception as error:
    state['status'] = 'stopped_error'
    state['error'] = repr(error)
    raise
finally:
    state['elapsed_seconds'] = time.time() - started
    save()
