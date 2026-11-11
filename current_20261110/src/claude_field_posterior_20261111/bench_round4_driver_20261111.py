"""Serial authorized DL and saved-model history-floor benchmark."""
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
STATE = RESULTS / 'bench_round4_execution_20261111.json'
started = time.time()
state = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'driver_pid': os.getpid(), 'folds': []}


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


os.chdir(SRC)
sys.path.insert(0, str(SRC))
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
    PYTHONPATH='/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages',
    VT_DIR=str(SRC / 'claude_hurdle_20261110'), LAW_UJI_CSV='/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv')
if STATE.exists():
    state = json.loads(STATE.read_text())
    started = datetime.datetime.fromisoformat(state['started_utc']).timestamp()
save()
try:
    stages = [('smoke', 'B1')] + [(s, b) for s in ('dl', 'hfloors') for b in ('B0', 'B1', 'B2')] + [('legacy54_dl', b) for b in ('B1', 'B2')]
    for stage, building in stages:
        if time.time() - started >= 55*60:
            state['status'] = 'time_limit_no_new_run'
            break
        ref = RESULTS / (f'legacy54_{building}_claude_20261111' if stage == 'legacy54_dl' else f'bench_uji_{building}_20261111')
        cfg = json.loads((ref/'freeze.json').read_text())['configuration']
        name = f'legacy54_{building}_dl_20261111' if stage == 'legacy54_dl' else ('bench_round4_smoke_20261111' if stage == 'smoke' else f'bench_uji_{building}_{stage}_20261111')
        out = RESULTS/name
        log_path = out.with_suffix('.log')
        methods = ['ours_loaded_h1', 'ours_loaded_h2'] if stage == 'hfloors' else ['pretrain_dnn', 'finetune_dnn', 'maml_dnn']
        cmd = [PYTHON, '-u', '-m', 'law_bench.run_claude', '--dataset', 'uji', '--building', building, '--methods', *methods, '--budgets', *map(str, [3] if stage == 'smoke' else cfg['budgets']), '--repeats', *map(str, [0] if stage == 'smoke' else cfg['repeats']), '--seed', str(cfg['seed']), '--R', str(cfg['R']), '--output', str(out)]
        if out.exists():
            cmd += ['--resume']
        if stage == 'smoke':
            cmd += ['--floors', '0']
        if stage == 'legacy54_dl':
            cmd += ['--protocol', 'legacy54', '--manifest', cfg['manifest'], '--source-buildings', *cfg['source_buildings']]
        else:
            cmd += ['--query-positions', str(cfg['query_positions'])]
        options = None
        if stage == 'hfloors':
            options = ASSETS/f'bench_round4_{building}_hfloors_options_20261111.json'
            options.write_text(json.dumps({m: {'model_path': str(RESULTS/f'bench_uji_{building}_A_20261111/source_models/ours_A.joblib')} for m in methods}, indent=2)+'\n')
            cmd += ['--method-options', str(options)]
        record = {'stage': stage, 'building': building, 'output': str(out), 'log': str(log_path), 'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': cmd, 'code_sha256': {p.name: __import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in (SRC/'law_bench').glob('*.py')}}
        state['folds'].append(record)
        tick = time.monotonic()
        with log_path.open('w') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            record['python_pid'] = process.pid
            state['status'] = 'running'
            save()
            print(json.dumps(record), flush=True)
            rc = process.wait()
        record.update(exit_code=rc, elapsed_seconds=time.monotonic()-tick, end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if rc:
            raise RuntimeError(f'{building} {stage} exit {rc}')
        import joblib
        hp_metadata = {}
        for model_path in (out/'source_models').glob('*.joblib'):
            model = joblib.load(model_path)
            hp_metadata[model_path.stem] = {'hp': getattr(model, 'hp', None), 'hp_scores': getattr(model, 'hp_scores', None)}
        (out/'selected_hp.json').write_text(json.dumps(hp_metadata, indent=2)+'\n')
        record['selected_hp'] = hp_metadata
        failures = [str(p) for p in (out/'events').rglob('*.json') if json.loads(p.read_text())['status'] != 'completed']
        record['failed_events'] = failures
        state['status'] = 'run_completed'
        save()
        paths = [ASSETS/'bench_round4_driver_20261111.py', STATE, log_path, *out.rglob('*.json')]
        if options:
            paths.append(options)
        record['pushed_commit'] = sync_commit(paths, f'Record round four UJI {building} {stage}')
        save()
        print(json.dumps(record), flush=True)
        if failures:
            raise RuntimeError('Failed prediction events')
    else:
        state['status'] = 'completed'
except Exception as error:
    state['status'] = 'stopped_error'
    state['error'] = repr(error)
    raise
finally:
    state['elapsed_seconds'] = time.time()-started
    save()
