"""Run one explicitly selected arm of the frozen source-task comparison.

A/Aplus are ordinary dense-context training; B/C are acquisition-held-out
few-support tasks. Aplus/C share the same recorded coordinates and whole
source-observed AP-vector augmentation. All start from public weights, keep the original full-model
pinball objective and use the same 48-update, building-balanced schedule.
This entry never launches another arm, selects a checkpoint, scores protected
positions, opens B1, fits GP, or changes deployment. Final weights stay remote.
An explicit --resume continues only from the last fully saved update.
"""
import argparse
import json
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch

from history_task_protocol import allowed_rows, episodic_pack, ordinary_pack
from history_task_preflight import (
    ROOT, SOURCE, CSV, OUT as PREFLIGHT, SEED, MODEL_SEED, UPDATES,
    read, load_allowed, fit,
)
from history_task_vectors import (
    OUT as VECTORS, sample_vector_transform, apply_vector_transform,
)
from recomposition_trial import digest, now, write
from tabicl_history import CHECKPOINT, PACKAGE, state_hash
from tabicl_history_learning import EpisodicHistoryRegressor

ARMS = {'A': ('ordinary', False), 'B': ('episodic', False),
        'Aplus': ('ordinary', True), 'C': ('episodic', True)}
OUTPUT = ROOT/'results/history_task_training_20261110'


def inputs():
    """Consume the completed preflight, without recalibrating its source pool."""
    complete = read(PREFLIGHT/'complete.json')
    assert complete['complete'] and complete['weights_unchanged']
    assert complete['optimizer_steps'] == complete['held_out_RSS_parsed'] == 0
    for name in ('freeze', 'source_profile', 'schedule'):
        assert digest(PREFLIGHT/(name+'.json')) == complete[name+'_sha256']
    contract = read(PREFLIGHT/'freeze.json')['contract']
    assert (contract['seed'], contract['model_seed'], contract['updates']) == (SEED, MODEL_SEED, UPDATES)
    for name, expected in contract['code'].items():
        assert digest(ROOT/name) == expected, name
    for name, expected in contract['inputs'].items():
        assert digest(Path(name)) == expected, name
    package = {str(p.relative_to(PACKAGE)): digest(p)
               for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package == contract['package']
    vectors = read(VECTORS/'complete.json')
    assert vectors['complete'] and vectors['source_only']
    assert vectors['preflight_sha256'] == digest(PREFLIGHT/'complete.json')
    assert vectors['code_sha256'] == digest(ROOT/'history_task_vectors.py')
    for name in ('freeze', 'profile'):
        assert vectors[name+'_sha256'] == digest(VECTORS/(name+'.json'))
    assert vectors['schedule_sha256'] == digest(PREFLIGHT/'schedule.json')
    return (read(SOURCE/'catalog.json'), read(PREFLIGHT/'schedule.json')['schedule'],
            read(VECTORS/'profile.json'), contract)


def progress(out, stage, **values):
    record = dict(created=now(), stage=stage, **values)
    write(out/'progress.json', record)
    print(json.dumps(record), flush=True)


def save_update(out, model, optimizer, scaler, training, initial_hash):
    # An interrupted write leaves the preceding complete update available.
    path = out/'resume.pt.part'
    torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), training=training,
        original_state_sha256=initial_hash, freeze_sha256=digest(out/'freeze.json'),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate()), path)
    path.replace(out/'resume.pt')
    write(out/'training.json', training)


def run(args):
    started = time.monotonic()
    catalog, schedule, profile, preflight = inputs()
    recipe, augmented = ARMS[args.arm]
    out = OUTPUT/args.arm
    if args.resume:
        assert out.is_dir() and not (out/'complete.json').exists()
        freeze = read(out/'freeze.json')
        assert freeze['arm'] == args.arm
        assert freeze['runner_sha256'] == digest(Path(__file__))
        assert freeze['preflight_sha256'] == digest(PREFLIGHT/'complete.json')
        assert freeze['vector_profile_completion_sha256'] == digest(VECTORS/'complete.json')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out/'freeze.json', dict(created=now(), arm=args.arm,
            recipe=recipe, augmented=augmented, runner_sha256=digest(Path(__file__)),
            preflight_sha256=digest(PREFLIGHT/'complete.json'),
            vector_profile_completion_sha256=digest(VECTORS/'complete.json'),
            profile_sha256=digest(VECTORS/'profile.json'),
            schedule_sha256=digest(PREFLIGHT/'schedule.json'),
            source_buildings=['B0', 'B2'], held_out_building='B1',
            model_seed=MODEL_SEED, task_and_augmentation_seed=SEED, updates=UPDATES,
            augmentation='Common rigid coordinates; one whole source-observed AP vector/sign per task, S/Q shared, H RSS unchanged; scalar replaced, not added',
            objective='Original equal-axis/eight-view position-balanced pinball; one task per building, equal building gradient weights',
            optimizer=dict(name='AdamW', lr=1e-5, weight_decay=.01, clip=1.,
                initial_amp_scale=1024., retries=0, architecture_fixed=['row_interactor.tf_row.rope.freqs']),
            selection='Final step 48 only; no validation or checkpoint selection in this entry',
            planned_exposure=preflight['planned_exposure'][recipe],
            deployment='Original raw H/S Tab plus unchanged effective GP combination; no augmented target inference',
            resources=dict(gpu=1, cpu_threads=2, gpu_fraction=.8, downloads=0,
                initial_segment_seconds=args.wall_seconds, stop_boundary='Complete update',
                saved='One resumable latest update and final public-loader-compatible model, server only'),
            limits=['Source-only learning completion is not localization or transfer success',
                'Same updates and legal corpus do not mean equal compute or query exposure',
                'Existing development fold; B1 is held out of this training, not previously unseen by the project',
                'Augmentation does not prove S necessity, real future shifts, or memorization as the observed cause']))
        (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())

    permitted = allowed_rows(catalog)
    rows = load_allowed(CSV, permitted, rss=True)
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    model = EpisodicHistoryRegressor(CHECKPOINT, device='cuda:0')
    assert model.enable_recompute()
    network = model.loaded.model_
    named = dict(network.named_parameters())
    parameters = [p for p in named.values() if p.requires_grad]
    assert sum(p.numel() for p in parameters) == 28544983
    assert [n for n, p in named.items() if not p.requires_grad] == ['row_interactor.tf_row.rope.freqs']
    fixed = named['row_interactor.tf_row.rope.freqs'].detach().clone()
    initial_hash = state_hash(network)
    assert initial_hash == read(PREFLIGHT/'complete.json')['initial_state_sha256']
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    training = []
    if args.resume:
        saved = torch.load(out/'resume.pt', map_location='cpu', weights_only=False)
        assert saved['freeze_sha256'] == digest(out/'freeze.json')
        assert saved['original_state_sha256'] == initial_hash
        network.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        training = saved['training']
        assert [r['step'] for r in training] == list(range(1, len(training)+1))
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        del saved

    for index in range(len(training), UPDATES):
        if time.monotonic()-started >= args.wall_seconds:
            progress(out, 'segment_finished', arm=args.arm, completed_updates=len(training),
                seconds=time.monotonic()-started, continuation='Explicit --resume required')
            return
        tick = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        events, losses = [], []
        batch = schedule[index]
        for spec in batch:
            assert spec['loss_weight'] == 1/len(batch)
            p = (ordinary_pack(rows, catalog, spec, SEED) if recipe == 'ordinary'
                 else episodic_pack(rows, catalog, spec))
            if augmented:
                transform = sample_vector_transform(p['histories'], p['event']['floor'],
                    profile, seed=SEED, task_key=spec['task_key'])
                p = apply_vector_transform(p, transform)
            fit(model, p)
            losses.append(model.backward(p['query']['rss'], p['query']['xy'], scaler))
            event = dict(id=p['event']['id'], building=spec['building'], floor=spec['floor'],
                context_scans=model.estimators[0].n_samples_in_, APs=len(p['aps']),
                query_scans=len(p['query']['rss']), query_positions=len(p['query']['positions']))
            if augmented:
                event['augmentation'] = p['augmentation']
            events.append(event)
            del p
        scaler.unscale_(optimizer)
        for parameter in parameters:
            assert parameter.grad is not None
            parameter.grad.div_(len(batch))
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
        assert norm > 0
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() >= old_scale, 'AMP overflow; stop, no retry'
        assert int(optimizer.state[parameters[0]]['step']) == index+1
        assert named['row_interactor.tf_row.rope.freqs'].grad is None
        assert torch.equal(named['row_interactor.tf_row.rope.freqs'], fixed)
        optimizer.zero_grad(set_to_none=True)
        training.append(dict(step=index+1, events=events, losses=losses,
            mean_pinball=float(np.mean(losses)), gradient_norm=norm,
            amp_scale=scaler.get_scale(), seconds=time.monotonic()-tick,
            peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20))
        save_update(out, network, optimizer, scaler, training, initial_hash)
        progress(out, 'trained', arm=args.arm, completed_updates=len(training),
            seconds=time.monotonic()-started)

    final_hash = state_hash(network)
    assert final_hash != initial_hash
    model.save(out/'final.ckpt', source_updates=UPDATES, provenance=dict(
        arm=args.arm, freeze_sha256=digest(out/'freeze.json'),
        source_buildings=['B0', 'B2'], held_out_building='B1', source_only=True,
        protected_positions=True, augmentation=augmented, selected='Final step 48 only'))
    write(out/'complete.json', dict(complete=True, created=now(), arm=args.arm,
        source_updates=UPDATES, source_tasks=sum(len(r['events']) for r in training),
        training_seconds=sum(r['seconds'] for r in training),
        current_segment_seconds=time.monotonic()-started,
        peak_GPU_MiB=max(r['peak_GPU_MiB'] for r in training),
        query_scan_exposures=sum(e['query_scans'] for r in training for e in r['events']),
        query_position_exposures=sum(e['query_positions'] for r in training for e in r['events']),
        original_state_sha256=initial_hash, trained_state_sha256=final_hash,
        checkpoint_sha256=digest(out/'final.ckpt'), freeze_sha256=digest(out/'freeze.json'),
        training_sha256=digest(out/'training.json'), held_out_RSS_parsed=0,
        protected_RSS_parsed=0, localization_predictions=0, performance_success=None))
    (out/'resume.pt').unlink()
    progress(out, 'complete', arm=args.arm, completed_updates=UPDATES,
        seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=tuple(ARMS), required=True)
    parser.add_argument('--wall-seconds', type=float, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web' and torch.cuda.is_available()
    assert args.wall_seconds > 0
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    run(args)
