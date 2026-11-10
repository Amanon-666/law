"""Serial authorized loaded-model benchmark, exact replay gate, per-fold lightweight commits."""
import argparse
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
STATE = RESULTS / 'bench_round3_execution_20261111.json'
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()
started = time.time()
state = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'driver_pid': os.getpid(), 'folds': []}

if args.resume:
    state = json.loads(STATE.read_text())
    state.setdefault('previous_runs', []).append({key: state[key] for key in ('driver_pid', 'status', 'elapsed_seconds')})
    state['driver_pid'] = os.getpid()
    state['resumed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()


def save():
    STATE.write_text(json.dumps(state, indent=2) + '\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def sync_commit(paths, message):
    subprocess.run([MAC_NET, 'git', 'fetch', 'origin'], cwd=REPO, check=True)
    if git('rev-list', '--count', 'HEAD..origin/' + BRANCH) != '0':
        raise RuntimeError('Collaborator advanced the research branch. Stop without merging.')
    for path in paths:
        subprocess.run(['git', 'add', '-f', str(path.relative_to(REPO))], cwd=REPO, check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=REPO, check=True)
    subprocess.run(['git', 'commit', '-m', message], cwd=REPO, check=True)
    commit = git('rev-parse', 'HEAD')
    subprocess.run([MAC_NET, 'git', 'push', 'origin', 'HEAD:' + BRANCH], cwd=REPO, check=True)
    return commit


def exact_replay(output, original):
    old = {p.relative_to(original / 'events' / 'ours_A'): json.loads(p.read_text()) for p in (original / 'events' / 'ours_A').rglob('*.json')}
    new = {p.relative_to(output / 'events' / 'ours_loaded_same'): json.loads(p.read_text()) for p in (output / 'events' / 'ours_loaded_same').rglob('*.json')}
    if old.keys() != new.keys():
        raise RuntimeError('same task coverage differs from original A')
    for key, a in old.items():
        b = new[key]
        if a['status'] != 'completed' or b['status'] != 'completed':
            raise RuntimeError(f'same replay incomplete: {key}')
        for field in ('query_row_indices', 'query_groups'):
            if a[field] != b[field]:
                raise RuntimeError(f'same {field} mismatch: {key}')
        if a['scores']['position_errors'] != b['scores']['position_errors']:
            raise RuntimeError(f'same position_errors mismatch: {key}')
    return len(old)


os.chdir(SRC)
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
    PYTHONPATH='/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages',
    VT_DIR=str(SRC / 'claude_hurdle_20261110'), LAW_UJI_CSV='/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv')
save()
try:
    for stage in ('readerablation', 'kcurve'):
        for building in ('B0', 'B1', 'B2'):
            if any(record['stage'] == stage and record['building'] == building and record.get('pushed_commit') and record.get('exit_code') == 0 and not record.get('failed_events') for record in state['folds']):
                continue
            if time.time() - started >= 55 * 60:
                state['status'] = 'time_limit_no_new_fold'
                save()
                sys.exit(0)
            original = RESULTS / f'bench_uji_{building}_A_20261111'
            freeze = json.loads((original / 'freeze.json').read_text())['configuration']
            output = RESULTS / f'bench_uji_{building}_{stage}_20261111'
            log_path = output.with_suffix('.log')
            methods = (['ours_loaded_' + tag for tag in ('same', 'no_offset', 'no_robust', 'no_temper', 'knn3')] if stage == 'readerablation' else ['ours_loaded_same', 'wknn_S', 'wknn_HS'])
            options = {m: {'model_path': str(original / 'source_models' / 'ours_A.joblib')} for m in methods if m.startswith('ours_loaded_')}
            options_path = ASSETS / f'bench_round3_{building}_{stage}_options_20261111.json'
            options_path.write_text(json.dumps(options, indent=2) + '\n')
            budgets = freeze['budgets'] if stage == 'readerablation' else [1, 3, 5, 10, 20]
            command = [PYTHON, '-u', '-m', 'law_bench.run_claude', '--dataset', 'uji', '--building', building, '--methods', *methods, '--method-options', str(options_path), '--budgets', *map(str, budgets), '--repeats', *map(str, freeze['repeats']), '--seed', str(freeze['seed']), '--R', str(freeze['R']), '--query-positions', str(freeze['query_positions']), '--output', str(output)]
            record = {'stage': stage, 'building': building, 'output': str(output), 'log': str(log_path), 'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': command}
            state['folds'].append(record)
            with log_path.open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                record['python_pid'] = process.pid
                state['status'] = 'running'
                save()
                print(json.dumps(record), flush=True)
                rc = process.wait()
            record.update(exit_code=rc, end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if rc:
                raise RuntimeError(f'{building} {stage} exit {rc}')
            if stage == 'readerablation':
                record['exact_replay_tasks'] = exact_replay(output, original)
            failures = []
            for path in (output / 'events').rglob('*.json'):
                event = json.loads(path.read_text())
                if event['status'] != 'completed':
                    failures.append({'file': str(path), 'status': event['status'], 'error': event.get('error')})
            record['failed_events'] = failures
            record['unavailable'] = json.loads((output / 'task_inventory.json').read_text())['unavailable']
            state['status'] = 'fold_completed'
            save()
            paths = [ASSETS / 'bench_round3_driver_20261111.py', options_path, STATE, log_path, SRC / 'law_bench' / 'ours.py', SRC / 'law_bench' / 'methods_claude.py', SRC / 'law_bench' / 'run_claude.py', *output.rglob('*.json')]
            record['pushed_commit'] = sync_commit(paths, f'Record round three UJI {building} {stage}')
            save()
            print(json.dumps(record), flush=True)
            if failures:
                raise RuntimeError('Failed prediction events. Stop subsequent folds.')
    state['status'] = 'completed'
except Exception as error:
    state['status'] = 'stopped_error'
    state['error'] = repr(error)
    raise
finally:
    state['elapsed_seconds'] = time.time() - started
    save()
