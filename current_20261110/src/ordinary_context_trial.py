"""Source-only ordinary random-context full-parameter Tab fine-tuning.

Public initialization, original raw consumer and pinball objective. Thirty
three-floor updates give 870 source-position exposures under the checked pool,
versus full24's 864; this is not equal compute or equal training distribution.
The original 27 protected source-development tasks select public/step15/step30.
This deliberately gives ordinary training broader source checkpoint selection
than full24's fixed step24. No target data or claim of sufficient optimization.

Implementation reverse path: protocol context alone fits APs/bounds/transforms;
only its disjoint source query enters backward; protected queries enter only
the unchanged source validation packs. Atomic latest.pt carries the actual
completed optimizer step and all training RNG. Validation preserves that RNG.
Segments stop at complete update/event boundaries; explicit resume only.
"""
import argparse
import gc
import json
import random
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from acquisition_tasks import load_acquisition_rows
from building_information_tab_trial import digest, now, write
from ordinary_context_protocol import make_episode
from source_episode_catalog import make_episode as catalog_episode
from tabicl_history import CHECKPOINT, PACKAGE, state_hash
from tabicl_history_learning import EpisodicHistoryRegressor
from tabicl_multitask_full_trial import events_for, fit, pack, query_weights, aggregate

ROOT = Path(__file__).resolve().parent
FULL = ROOT/'results/tabicl_multitask_full_trial_20261107_checked'
CATALOG = ROOT/'results/source_episode_catalog_20261105/catalog.json'
OUT = ROOT/'results/ordinary_context_trial_20261109'
SEED = 1150546
SPLIT_SEED = 11509163
STEPS = 30
SNAPSHOTS = (15, 30)
CODE = ('ordinary_context_trial.py', 'ordinary_context_protocol.py', 'tabicl_history.py',
        'tabicl_history_learning.py', 'tabicl_multitask_full_trial.py',
        'source_episode_catalog.py', 'acquisition_tasks.py', 'uji.py',
        'check_tabicl_history_learning.py', 'building_information_tab_trial.py')


def read(path):
    return json.loads(Path(path).read_text())


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy']); random.setstate(state['python'])


def draw_metadata(episode):
    return {key: episode[key] for key in ('draw_index', 'seed', 'context_rows', 'query_rows', 'split')}


def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    old_freeze, old_done = read(FULL/'freeze.json'), read(FULL/'complete.json')
    catalog = read(CATALOG)
    csv_path = Path(next(p for p in old_freeze['inputs'] if p.endswith('/trainingData.csv')))
    assert old_done['complete'] and digest(FULL/'freeze.json') == old_done['freeze_sha256']
    for path in (CATALOG, csv_path, CHECKPOINT):
        assert digest(path) == old_freeze['inputs'][str(path)]
    for name in ('tabicl_history.py', 'tabicl_history_learning.py', 'acquisition_tasks.py',
                 'source_episode_catalog.py', 'uji.py', 'check_tabicl_history_learning.py'):
        assert digest(ROOT/name) == old_freeze['code'][name]
    package = {str(p.relative_to(PACKAGE)): digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package == old_freeze['package']
    _, validation = events_for(catalog)
    assert validation == old_freeze['validation'] and len(validation) == 27
    reference = Path(old_freeze['initial_reuse']['reference'])
    public, full24 = read(reference/'initial.json'), read(FULL/'step_24.json')
    prediction_hashes = {}
    for spec, initial, learned in zip(validation, public, full24):
        event = catalog_episode(catalog, **spec)
        assert initial['event'] == learned['event'] == event['id']
        path = reference/f'initial_{event["id"]}.npz'
        assert digest(path) == old_freeze['inputs'][str(path)]
        prediction_hashes[str(path)] = digest(path)
        path = FULL/f'step_24_{event["id"]}.npz'
        assert digest(path) == learned['prediction_sha256']
        prediction_hashes[str(path)] = digest(path)
    assert len(public) == len(full24) == 27
    rows = load_acquisition_rows(csv_path, set(catalog['source_floors']))
    draws, permissions = [], None
    for draw in range(STEPS):
        episode = make_episode(rows, catalog, draw, SPLIT_SEED)
        draws.append(draw_metadata(episode))
        permissions = episode['permissions']
    del episode
    frozen = dict(code={name: digest(ROOT/name) for name in CODE}, package=package,
        runtime={name: version(name) for name in ('torch', 'numpy', 'scikit-learn')},
        inputs={str(p): digest(p) for p in (csv_path, CATALOG, CHECKPOINT, FULL/'freeze.json',
                 FULL/'complete.json', reference/'initial.json', FULL/'step_24.json')},
        reference_predictions=prediction_hashes, permissions=permissions, draws=draws,
        validation=validation, seed=SEED, split_seed=SPLIT_SEED, snapshots=list(SNAPSHOTS), steps=STEPS,
        optimizer=dict(name='AdamW', lr=1e-5, weight_decay=.01, clip=1., amp_scale=1024.,
                       floor_gradient_weight=1/3, trainable_parameters=28544983, fixed_rope=8),
        source_selection='Minimum equal-floor/equal-K projected MDE among public0/ordinary15/ordinary30; exact tie favors earlier step',
        full24_reference=dict(steps=24, training_seconds=829.203, query_scan_exposures=2592,
                              query_position_exposures=864, checkpoint_selection='Fixed final24'),
        forward='Original raw context, official none/power and eight views, two-axis position-balanced pinball; pooled floor/XY-block random split',
        reverse=['Thirty complete position-group splits fixed before weights update',
                 'Original training-corpus union only; protected rows and positions excluded from every training role',
                 'Query arrays never passed into context fit; APs/bounds/scalers use context only',
                 'Protected source queries select public/15/30 but never enter gradients',
                 'Single atomic latest.pt saves optimizer/scaler/model/config/RNG at each completed update',
                 'No B1/B2 data, target selection, retries, automatic extension or downloads'],
        limits=['Dense own-floor source context is not deployment K1/K3/K5',
                'Broader source checkpoint selection than full24 is explicitly allowed to strengthen the ordinary control',
                'Source development is previously exposed, not blind; thirty updates are a finite budget, not demonstrated convergence',
                '870 versus 864 position exposures does not equate repeated scans, FLOPs, context difficulty or total cost'],
        resources=dict(device='cuda:0', gpu=1, cpu_threads=2, segment_seconds=args.seconds,
                       stop_boundary='Completed training update or validation event'))
    out = args.output
    if args.resume:
        assert read(out/'freeze.json')['contract'] == frozen
        progress = read(out/'progress.json')
        assert progress['freeze_sha256'] == digest(out/'freeze.json')
        for record in progress['snapshots'].values():
            assert digest(record['path']) == record['sha256']
        for records in progress['validation'].values():
            for record in records:
                assert digest(record['path']) == record['prediction_sha256']
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out/'freeze.json', dict(created=now(), contract=frozen))
        progress = dict(freeze_sha256=digest(out/'freeze.json'), snapshots={}, validation={}, segments=[])
        write(out/'progress.json', progress)
    freeze_sha = digest(out/'freeze.json')
    if (out/'complete.json').exists():
        raise RuntimeError('Already complete; do not rerun')

    def expired():
        return time.monotonic()-started >= args.seconds

    def persist():
        write(out/'progress.json', progress)

    # Same public construction and RNG initialization as the original full24.
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = EpisodicHistoryRegressor(CHECKPOINT, device='cuda:0')
    model.enable_recompute()
    network = model.loaded.model_
    named = dict(network.named_parameters())
    parameters = [p for p in named.values() if p.requires_grad]
    assert sum(p.numel() for p in parameters) == 28544983
    assert [name for name, p in named.items() if not p.requires_grad] == ['row_interactor.tf_row.rope.freqs']
    fixed = named['row_interactor.tf_row.rope.freqs']
    assert fixed.numel() == 8
    fixed_value = fixed.detach().clone()
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    latest = out/'latest.pt'
    step, training = 0, []
    if args.resume and latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        assert saved['freeze_sha256'] == freeze_sha and saved['config'] == model.loaded.model_config_
        network.load_state_dict(saved['state_dict'], strict=True)
        assert state_hash(network) == saved['state_sha256']
        optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        step, training = saved['step'], saved['training']
        restore_rng(saved['rng'])
        del saved
    else:
        assert not progress['snapshots'] and not progress['validation']

    def validate_snapshot():
        key = str(step)
        if key not in progress['snapshots']:
            path = out/f'ordinary_step_{step}.ckpt'
            temporary = path.with_suffix('.tmp')
            model.save(temporary, source_updates=step,
                       provenance=dict(freeze_sha256=freeze_sha, source_only=True,
                                       protocol='ordinary_random_context', source_selection=True))
            temporary.replace(path)
            progress['snapshots'][key] = dict(path=str(path), sha256=digest(path),
                                             state_sha256=state_hash(network))
            persist()
        assert state_hash(network) == progress['snapshots'][key]['state_sha256']
        records = progress['validation'].setdefault(key, [])
        done = {r['event'] for r in records}
        state_before, rng_before = state_hash(network), rng_state()
        try:
            for spec in validation:
                eid = catalog_episode(catalog, **spec)['id']
                if eid in done:
                    continue
                if expired():
                    return False
                p = pack(rows, catalog, spec)
                fit(model, p)
                prediction = model.predict(p['query']['rss'])
                truth = p['query']['xy']; weights = query_weights(p)
                errors = {name: np.linalg.norm(value-truth, axis=1) for name, value in prediction.items()}
                path = out/f'step_{step}_{eid}.npz'
                np.savez_compressed(path, **prediction, truth=truth, weights=weights,
                    query_rows=p['query_rows'], aps=p['aps'], bounds=p['bounds'],
                    **{name+'_errors': value for name, value in errors.items()})
                records.append(dict(event=eid, floor=p['event']['floor'], K=p['event']['K'],
                    mde={name: float(weights@value) for name, value in errors.items()},
                    context_rows=model.estimators[0].n_samples_in_, APs=len(p['aps']),
                    path=str(path), prediction_sha256=digest(path)))
                persist()
                print(json.dumps(dict(stage='validation', step=step, completed=len(records), total=27)), flush=True)
            return True
        finally:
            assert state_hash(network) == state_before
            restore_rng(rng_before)

    complete = False
    try:
        while True:
            if step in SNAPSHOTS and not validate_snapshot():
                break
            if step == STEPS:
                complete = True
                break
            if expired():
                break
            tick = time.monotonic()
            episode = make_episode(rows, catalog, step, SPLIT_SEED)
            assert draw_metadata(episode) == draws[step]
            optimizer.zero_grad(set_to_none=True)
            losses, exposure = [], []
            for p in episode['tasks']:
                fit(model, p)
                losses.append(model.backward(p['query']['rss'], p['query']['xy'], scaler))
                exposure.append(dict(floor=p['floor'], context_rows=model.estimators[0].n_samples_in_,
                    context_positions=sum(len(h['positions']) for h in p['histories'].values())+len(p['support']['positions']),
                    APs=len(p['aps']), query_scans=len(p['query']['rss']), query_positions=len(p['query']['positions'])))
            scaler.unscale_(optimizer)
            for parameter in parameters:
                assert parameter.grad is not None
                parameter.grad.div_(3)
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
            assert norm > 0
            old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            assert scaler.get_scale() >= old_scale, 'AMP overflow; no retry'
            step += 1
            assert int(optimizer.state[parameters[0]]['step']) == step
            assert fixed.grad is None and torch.equal(fixed, fixed_value)
            seconds = time.monotonic()-tick
            optimizer.zero_grad(set_to_none=True)
            training.append(dict(step=step, losses=losses, mean_pinball=float(np.mean(losses)),
                gradient_norm=norm, amp_scale=scaler.get_scale(), seconds=seconds, exposure=exposure,
                peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20))
            atomic_save(dict(config=model.loaded.model_config_,
                state_dict={name: value.detach().cpu() for name, value in network.state_dict().items()},
                state_sha256=state_hash(network), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                rng=rng_state(), step=step, training=training, freeze_sha256=freeze_sha), latest)
            write(out/'training.json', training)
            print(json.dumps(dict(stage='training', step=step, total=STEPS,
                                  seconds=seconds, query_positions=sum(e['query_positions'] for e in exposure))), flush=True)
    except Exception as error:
        write(out/f'failure_{time.time_ns()}.json', dict(created=now(), step=step,
              error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        del optimizer, scaler, parameters, named, fixed, fixed_value, network, model
        gc.collect(); torch.cuda.empty_cache()
    elapsed = time.monotonic()-started
    progress['segments'].append(dict(ended=now(), seconds=elapsed, budget=args.seconds,
                                     budget_overrun=elapsed > args.seconds, completed_step=step))
    progress['complete'] = complete
    persist()
    if complete:
        summaries = {'0': aggregate(public, lambda r: r['mde']['projected']),
                     **{str(s): aggregate(progress['validation'][str(s)], lambda r: r['mde']['projected'])
                        for s in SNAPSHOTS}}
        chosen = min((0, *SNAPSHOTS), key=lambda s: (summaries[str(s)]['mean'], s))
        full_summary = aggregate(full24, lambda r: r['mde']['projected'])
        selected = summaries[str(chosen)]
        selected_path = str(CHECKPOINT) if chosen == 0 else progress['snapshots'][str(chosen)]['path']
        write(out/'complete.json', dict(complete=True, created=now(), freeze_sha256=freeze_sha,
            source_updates=STEPS, source_episodes=3*STEPS, selection=summaries,
            selected_step=chosen, selected_checkpoint=selected_path, selected_checkpoint_sha256=digest(selected_path),
            selected_vs_full24_gain=1-selected['mean']/full_summary['mean'], full24=full_summary,
            training_seconds=sum(r['seconds'] for r in training),
            query_scan_exposures=sum(e['query_scans'] for r in training for e in r['exposure']),
            query_position_exposures=sum(e['query_positions'] for r in training for e in r['exposure']),
            snapshots=progress['snapshots'], target_opened=False, blind_test=False,
            converged=False, broader_source_selection_than_full24=True))
    return 0 if complete else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seconds', type=float, default=900.)
    parser.add_argument('--output', type=Path, default=OUT)
    raise SystemExit(run(parser.parse_args()))
