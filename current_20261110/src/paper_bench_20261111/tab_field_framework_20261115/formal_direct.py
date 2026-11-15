"""Complete source epochs and restartable training for direct coordinates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

PROJECTS = Path('/home/panyushuo/projects/panyushuo')
CORE = PROJECTS/'ReFi-UJI-Core-20261031'
BENCH = PROJECTS/'paper_bench_wt/current_20261110'
REQUESTED_CUDA_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
sys.path.insert(0, str(BENCH/'src'))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from law_bench import data, protocol, run, score

sys.path.insert(0, str(CORE))
import tabicl
import tabicl_history
import tabicl_history_learning
from tabicl.__about__ import __version__ as tabicl_version
from direct_coordinate import DirectCoordinateRegressor, PUBLIC_REGRESSOR_SHA256

os.environ['CUDA_VISIBLE_DEVICES'] = REQUESTED_CUDA_DEVICES
BASE_SEED = 20261111
BUDGETS = (1, 3, 5)
REPEATS = (0, 1, 2)
SAVE_EVERY = 9


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def atomic_save(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all())


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch_cpu'])
    torch.cuda.set_rng_state_all(value['torch_cuda'])


def source_tasks(buildings, axes, names, repeats, split_seed):
    episodes = []
    for building in names:
        floors = buildings[building]
        for floor in sorted(floors):
            for repeat in repeats:
                episodes.extend(protocol.make_tasks('uji', building, floors, axes[building],
                    floor, budgets=BUDGETS, repeat=repeat, seed=split_seed,
                    R=3, query_positions=60))
    return episodes


def contract(args, buildings, axes):
    training = ['B0'] if args.phase == 'development' else ['B0', 'B2']
    source = ['B0', 'B2']
    csv_path = Path(next(iter(buildings['B0'].values()))['meta']['source'])
    checkpoint = tabicl_history.CHECKPOINT
    checkpoint_sha = digest(checkpoint)
    assert checkpoint_sha == PUBLIC_REGRESSOR_SHA256
    package = Path(tabicl.__file__).parent
    code_files = [Path(__file__), Path(__file__).with_name('direct_coordinate.py'),
        Path(tabicl_history.__file__), Path(tabicl_history_learning.__file__),
        Path(data.__file__), Path(protocol.__file__), Path(score.__file__), Path(run.__file__)]
    return dict(phase=args.phase, seed=args.seed, source_epochs=args.epochs,
        role='shared_xy_coordinate_regressor', training_buildings=training,
        development_building='B2' if args.phase == 'development' else None,
        source_only=True, target_evaluation=False,
        source_data=run._data_identity({b: buildings[b] for b in source}, {b: axes[b] for b in source}),
        source_CSV=dict(path=str(csv_path), sha256=digest(csv_path)),
        source_floor_rows={b: {str(f): len(d['rss']) for f, d in buildings[b].items()} for b in source},
        checkpoint=dict(path=str(checkpoint), sha256=checkpoint_sha),
        views=1, norm_methods=['none'], recompute=True,
        precision='original CUDA fp16 autocast and GradScaler initial1024',
        parameter_training='All native learnable weights shared by both coordinate axes',
        objective='Original quantile pinball, equal axes and equal real Query positions within each task',
        optimizer=dict(name='AdamW', lr=1e-5, weight_decay=.01, clip_norm=1.),
        task_protocol=dict(name='static-resample-v2', budgets=list(BUDGETS),
            training_repeats=list(REPEATS), R=3, query_positions=60,
            order='building, physical floor, repeat, then K1/K3/K5 from one make_tasks call',
            split_seed_formula='20261111 + 100000 * training_seed + source_epoch_1_based',
            epoch_split_seeds=[BASE_SEED+100000*args.seed+epoch for epoch in range(1, args.epochs+1)],
            context='All other same-building floors and current Support, with original H/S AP mask'),
        events_per_epoch=sum(len(buildings[b]) for b in training)*len(REPEATS)*len(BUDGETS),
        source_selection=dict(candidate_epochs=[0, 1, 2, 3], building='B2',
            repeat=0, split_seed=BASE_SEED, evaluation_rng_seed=BASE_SEED,
            metric='Pool per-position mean scan errors over all B2 floors and K1/K3/K5',
            rule='Minimum MDE, with the earliest epoch selected on exact ties'),
        final_recipe='Start from the same public checkpoint and train complete B0+B2 epochs using the upper-level selected count',
        recovery=dict(checkpoint_interval_tasks=SAVE_EVERY, model_schema='config/state_dict',
            latest_contents=['model', 'optimizer', 'scaler', 'epoch', 'next_event', 'freeze_sha256', 'RNG'],
            validation_rng='Save/restore the training RNG around fixed source validation'),
        tabicl_version=tabicl_version,
        code_sha256={str(p): digest(p) for p in code_files},
        package_code_sha256={str(p.relative_to(package)): digest(p) for p in sorted(package.rglob('*.py'))})


def epoch_asset(model, output, epoch, updates, freeze_sha, frozen):
    directory = output/'checkpoints'
    directory.mkdir(exist_ok=True)
    path = directory/f'epoch_{epoch}.ckpt'
    if not path.exists():
        temporary = path.with_suffix('.tmp')
        model.save(temporary, source_updates=updates, provenance=dict(
            freeze_sha256=freeze_sha, source_epoch=epoch, seed=frozen['seed'],
            phase=frozen['phase'], training_buildings=frozen['training_buildings'],
            source_only=True, target_evaluation=False))
        temporary.replace(path)
    return dict(checkpoint=str(path.resolve()), sha256=digest(path),
        source_epoch=epoch, source_updates=updates)


def evaluate_source(model, episodes, output, epoch, asset, freeze_sha):
    directory = output/'source_validation'
    directory.mkdir(exist_ok=True)
    path = directory/f'epoch_{epoch}.json'
    if path.exists():
        result = json.loads(path.read_text())
        assert result['freeze_sha256'] == freeze_sha and result['checkpoint_sha256'] == asset['sha256']
    else:
        rng = capture_rng()
        records, errors = [], []
        started = time.monotonic()
        try:
            seed_all(BASE_SEED)
            for task, truth in episodes:
                assert task.building == 'B2'
                prediction = model.predict_task(task)
                values = score.score(prediction, truth)
                errors.extend(values['position_errors'])
                records.append(dict(task_id=task.task_id, mean=values['mean'],
                    p90=values['p90'], query_positions=values['query_positions'],
                    query_scans=values['query_scans'], position_errors=values['position_errors']))
            torch.cuda.synchronize()
            result = dict(source_epoch=epoch, source_only=True, building='B2',
                target_evaluation=False, mean_position_error=float(np.mean(errors)),
                query_position_budget_count=len(errors), tasks=len(records), task_scores=records,
                seconds=time.monotonic()-started, freeze_sha256=freeze_sha,
                checkpoint_sha256=asset['sha256'])
        finally:
            restore_rng(rng)
        write_json(path, result)
    summary = {key: result[key] for key in ['source_epoch', 'mean_position_error',
        'query_position_budget_count', 'tasks', 'seconds']}
    summary.update(path=str(path.resolve()), sha256=digest(path))
    print(json.dumps(dict(stage='source_validation_complete', **summary)), flush=True)
    return summary


def save_latest(path, model, optimizer, state, freeze_sha, frozen):
    atomic_save(path, dict(config=model.loaded.model_config_,
        state_dict={k: v.detach().cpu() for k, v in model.model.state_dict().items()},
        optimizer=optimizer.state_dict(), scaler=model.scaler.state_dict(),
        epoch=state['epoch'], next_event=state['next_event'], source_updates=state['source_updates'],
        source_validation=state['source_validation'],
        committed_training_seconds=state['committed_training_seconds'],
        freeze=frozen, freeze_sha256=freeze_sha, rng_state=capture_rng()))


def main(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    seed_all(args.seed)
    buildings, axes = data.load('uji')
    frozen = contract(args, buildings, axes)
    args.output.mkdir(parents=True, exist_ok=True)
    freeze_path, latest = args.output/'freeze.json', args.output/'latest.pt'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text()) == frozen, 'Frozen source recipe differs'
    else:
        assert not latest.exists(), 'A latest checkpoint requires its original freeze'
        write_json(freeze_path, frozen)
    freeze_sha = digest(freeze_path)
    complete_path = args.output/'complete.json'
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        assert complete['complete'] and complete['freeze_sha256'] == freeze_sha
        asset = complete['model_assets'] if args.phase == 'final' else complete['selected_model']
        assert digest(asset['checkpoint']) == asset['sha256']
        print(json.dumps(complete), flush=True)
        return

    model = DirectCoordinateRegressor(seed=args.seed, device='cuda:0', amp=True)
    optimizer = torch.optim.AdamW(model.parameters, lr=1e-5, weight_decay=.01)
    state = dict(epoch=0, next_event=0, source_updates=0,
        source_validation={}, committed_training_seconds=0.)
    validation = source_tasks(buildings, axes, ['B2'], (0,), BASE_SEED) if args.phase == 'development' else []
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        assert saved['freeze_sha256'] == freeze_sha
        assert saved['freeze'] == frozen
        assert saved['config'] == model.loaded.model_config_
        model.model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        model.scaler.load_state_dict(saved['scaler'])
        for name in state:
            state[name] = saved[name]
        assert state['source_updates'] == state['epoch']*frozen['events_per_epoch']+state['next_event']
        restore_rng(saved['rng_state'])
        del saved
    else:
        asset = epoch_asset(model, args.output, 0, 0, freeze_sha, frozen)
        if validation:
            state['source_validation']['0'] = evaluate_source(model, validation,
                args.output, 0, asset, freeze_sha)
        save_latest(latest, model, optimizer, state, freeze_sha, frozen)

    for epoch_index in range(state['epoch'], args.epochs):
        epoch = epoch_index+1
        episodes = source_tasks(buildings, axes, frozen['training_buildings'], REPEATS,
            BASE_SEED+100000*args.seed+epoch)
        assert len(episodes) == frozen['events_per_epoch']
        start_event = state['next_event']
        for index in range(start_event, len(episodes)):
            task, truth = episodes[index]
            assert task.building in frozen['training_buildings']
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = model.source_task_loss(task, truth)
            model.scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters, 1., error_if_nonfinite=True))
            model.scaler.step(optimizer)
            model.scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            state['committed_training_seconds'] += time.monotonic()-tick
            state['next_event'] = index+1
            state['source_updates'] += 1
            if state['next_event'] % SAVE_EVERY == 0:
                save_latest(latest, model, optimizer, state, freeze_sha, frozen)
                progress = dict(phase=args.phase, seed=args.seed, training_epoch=epoch,
                    next_event=state['next_event'], events_per_epoch=len(episodes),
                    source_updates=state['source_updates'], last_task_id=task.task_id,
                    last_pinball_loss=loss, gradient_norm=norm,
                    committed_training_seconds=state['committed_training_seconds'])
                write_json(args.output/'progress.json', progress)
                print(json.dumps(dict(stage='source_checkpoint_saved', **progress)), flush=True)
        asset = epoch_asset(model, args.output, epoch, state['source_updates'], freeze_sha, frozen)
        if validation:
            state['source_validation'][str(epoch)] = evaluate_source(model, validation,
                args.output, epoch, asset, freeze_sha)
        state['epoch'], state['next_event'] = epoch, 0
        save_latest(latest, model, optimizer, state, freeze_sha, frozen)

    complete = dict(complete=True, phase=args.phase, seed=args.seed, source_only=True,
        target_evaluation=False, completed_source_epochs=args.epochs,
        source_updates=state['source_updates'], freeze_sha256=freeze_sha,
        committed_training_seconds=state['committed_training_seconds'],
        invocation_wall_seconds=time.monotonic()-started)
    if args.phase == 'development':
        curve = [state['source_validation'][str(epoch)] for epoch in range(args.epochs+1)]
        chosen = min(curve, key=lambda item: (item['mean_position_error'], item['source_epoch']))
        selected = chosen['source_epoch']
        complete.update(selected_source_epoch=selected, source_validation=curve,
            selected_model=epoch_asset(model, args.output, selected,
                selected*frozen['events_per_epoch'], freeze_sha, frozen))
    else:
        asset = epoch_asset(model, args.output, args.epochs, state['source_updates'], freeze_sha, frozen)
        assets = dict(asset, seed=args.seed, training_buildings=['B0', 'B2'],
            source_only=True, target_evaluation=False, views=1, norm_methods=['none'])
        write_json(args.output/'model_assets.json', assets)
        complete.update(selected_source_epoch=args.epochs, model_assets=assets)
    write_json(complete_path, complete)
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['development', 'final'], required=True)
    parser.add_argument('--seed', type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--epochs', type=int, choices=[0, 1, 2, 3], required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.phase == 'development' and arguments.epochs != 3:
        parser.error('Development freezes complete candidate epochs 0, 1, 2 and 3')
    with threadpool_limits(limits=2):
        main(arguments)
