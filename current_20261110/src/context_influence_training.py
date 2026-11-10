"""Source-only executable entry for the existing GP/Tab fixed-half system.

Implementation is not authorization to run: formal training is currently
paused. No experiment, target prediction or protected validation is launched
on import. An explicitly invoked run uses the original 24 x 3 B0 fit schedule,
frozen full24 backbone and fixed GP equations; only the v2 relation encoder is
optimized. The inherited lr/decay/AMP settings are execution defaults, not an
optimality claim or a newly authorized fair-comparison protocol.

Reverse path: catalog row permissions -> exact cached H and current S -> CPU
GP adaptation -> retained fields and bound relation graph -> Tab H/S fit ->
current legal source Q -> position-balanced complete-system MDE gradient.
The system remains .5*(projected GP + projected Tab); source error correction
need not transfer to another environment. No source-validation selection,
target labels, history refitting, sampling-policy change or automatic retry.

One consumer owns one encoder for every context. Latest.pt atomically stores
only the encoder/optimizer/scaler and training RNG after a complete update;
the large frozen backbone is referenced, never copied. Segment deadlines are
checked at complete three-floor updates, with explicit --resume required.
"""
import argparse
import hashlib
import json
import random
import socket
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

import predict as gp
from building_information_gp_trial import move_task
from building_information_tab_trial import digest, now, read_scans, write
from context_influence_consumer import InfluenceHistoryRegressor
from context_influence_features import FEATURE_NAMES, MOMENT_NAMES
from context_influence_system import InfluencePositionSystem
from ordinary_context_trial import atomic_save, rng_state, restore_rng
from position_fusion_trial import (ROOT, SOURCE, SOURCE_ASSETS, CATALOG, CSV,
    CHECKPOINT, PARAMETERS, CODE as PREPARE_CODE, canonical_context, prepare, read)
from source_episode_catalog import make_episode
from tabicl_history import PACKAGE, state_hash
from tabicl_multitask_full_trial import events_for

SEED = 1150546
STEPS = 24
CODE = tuple(dict.fromkeys((*PREPARE_CODE, 'context_influence_training.py',
    'context_influence_system.py', 'context_influence_consumer.py',
    'context_influence_features.py', 'context_influence_adapter.py',
    'context_influence_objective.py', 'tabicl_history_learning.py',
    'ordinary_context_trial.py', 'advised_source_assets.py',
    'support_recomposition.py', 'episode_queries.py')))


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_contract():
    """Only fit metadata and completion provenance; no validation arrays/records."""
    catalog, old_freeze = read(CATALOG), read(SOURCE/'freeze.json')
    done = read(SOURCE/'complete.json')
    assets_done = read(SOURCE_ASSETS/'complete.json')
    index_path = SOURCE_ASSETS/'histories.json'
    index, params = read(index_path), read(PARAMETERS)
    assert done['complete'] and assets_done['complete']
    assert digest(SOURCE/'freeze.json') == done['freeze_sha256']
    assert digest(CHECKPOINT) == done['checkpoint_sha256']['24']
    assert digest(SOURCE_ASSETS/'freeze.json') == assets_done['freeze_sha256']
    assert digest(index_path) == assets_done['histories_sha256']
    assert digest(PARAMETERS) == index['parameters_sha256'] == assets_done['source_parameters_sha256']
    for path in (CATALOG, CSV):
        assert digest(path) == old_freeze['inputs'][str(path)]
    assert catalog['source_floors'] == ['B0F0', 'B0F1', 'B0F2']
    schedule, _ = events_for(catalog)  # No validation event is materialized or read.
    assert schedule == old_freeze['schedule'] and len(schedule) == STEPS
    protected = set(catalog['protected_rows'])
    batches = []
    for batch in schedule:
        events = []
        for spec in batch:
            assert spec['phase'] == 'fit'
            event = make_episode(catalog, **spec)
            assert event['role'] == 'train' and event['K'] in (1, 3, 5)
            assert event['history_rows'] == {f: index['history_rows'][f] for f in event['histories']}
            hrows = [i for ids in event['history_rows'].values() for i in ids]
            srows, qrows = event['support_rows'], event['query_rows']
            assert not protected.intersection(hrows+srows+qrows)
            assert not set(qrows).intersection(hrows+srows)
            assert not set(srows).intersection(hrows)
            assert len(srows) == 3*event['K'] and len(qrows) == 36
            events.append(event)
        assert {e['floor'] for e in events} == set(catalog['source_floors']) and len(events) == 3
        batches.append(events)
    package = {str(p.relative_to(PACKAGE)): digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package == old_freeze['package']
    paths = (CSV, CATALOG, CATALOG.parent/'freeze.json', CATALOG.parent/'complete.json',
             CHECKPOINT, SOURCE/'complete.json', SOURCE/'freeze.json', PARAMETERS,
             index_path, SOURCE_ASSETS/'freeze.json', SOURCE_ASSETS/'complete.json')
    contract = dict(code={n: digest(ROOT/n) for n in CODE}, package=package,
        runtime={n: version(n) for n in ('torch', 'numpy', 'scikit-learn')},
        inputs={str(p): digest(p) for p in paths}, schedule=schedule, events=batches,
        seed=SEED, checkpoint=str(CHECKPOINT), feature_names=list(FEATURE_NAMES),
        moment_names=list(MOMENT_NAMES), adapter_format='gp_column_influence_v2',
        objective='Original complete-system position-balanced Euclidean MDE: .5*(GP projected + Tab projected); three floors equal per update',
        optimizer=dict(name='AdamW', owner='relation_encoder only', lr=1e-5,
            weight_decay=.01, clip=1., initial_amp_scale=1024., updates=STEPS),
        information='Only original B0 fit H/S/Q; all globally protected rows excluded from every role; no validation or B1/B2 arrays',
        GP='Exact indexed H rows and fixed paired parameters; existing CPU adapt, canonical H/S seed, unchanged retained posterior; no H fit',
        state_cache='Only this output states/{canonical_context_id}.json, bound to freeze, H/S row order, floor, K and APs',
        recompute=True, full24_state_sha256=done['trained_state_sha256'],
        resources=dict(GPUs=1, CPU_threads=2, max_gpu_fraction=.8, downloads=0),
        limits=['Entry implementation does not authorize training while the user pause remains active',
                'Inherited optimizer defaults are not tuned, optimal or a fair-control claim',
                'Source complete-system MDE can learn source GP error compensation; transfer is unproven',
                'No checkpoint selection, target prediction, additional arm or new validation',
                'Boundary stop may exceed wall budget by the current complete three-floor update',
                'Zero gradients are allowed; nonfinite gradients/AMP overflow stop without retry'])
    return json.loads(json.dumps(contract)), batches, index, params


def adapted_state(output, event, aps, task, params, freeze_sha):
    cid = canonical_context(event)
    context = {k: event[k] for k in ('floor', 'histories', 'history_rows', 'support_rows')}
    binding = dict(freeze_sha256=freeze_sha, context_id=cid, context=context,
                   K=event['K'], aps=aps.tolist())
    path = output/'states'/f'{cid}.json'
    if path.exists():
        saved = read(path)
        assert saved['binding'] == binding and json_digest(saved['state']) == saved['state_sha256']
        return cid, saved['state']
    torch.set_default_dtype(torch.float64)
    try:
        state = gp.adapt(task['cases'], params['base'], params['obs'])
    finally:
        torch.set_default_dtype(torch.float32)
    write(path, dict(binding=binding, state=state, state_sha256=json_digest(state)))
    return cid, state


def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    contract, batches, index, params = source_contract()
    output = args.output
    if args.resume:
        frozen = read(output/'freeze.json')
        assert frozen['contract'] == contract
        assert not (output/'complete.json').exists(), 'Already complete; no continuation'
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'states').mkdir()
        write(output/'freeze.json', dict(created=now(), contract=contract,
              initial_segment_wall_seconds=args.wall_seconds, formal_training_paused_at_implementation=True))
        for name in CODE:
            (output/name).write_bytes((ROOT/name).read_bytes())
    freeze_sha = digest(output/'freeze.json')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    consumer = InfluenceHistoryRegressor(CHECKPOINT, device='cuda:0')
    assert consumer.enable_recompute()
    network, encoder = consumer.loaded.model_, consumer.relation_encoder
    backbone_hash = state_hash(network)
    assert backbone_hash == contract['full24_state_sha256']
    assert not any(p.requires_grad for p in network.parameters())
    parameters = list(encoder.parameters())
    assert parameters and all(p.requires_grad for p in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    system = InfluencePositionSystem(consumer)
    latest = output/'latest.pt'
    training, step = [], 0

    def save_latest():
        assert all(p.grad is None for p in network.parameters())
        atomic_save(dict(format='context_influence_training_v2', freeze_sha256=freeze_sha,
            backbone_state_sha256=backbone_hash, config=encoder.config,
            feature_names=list(FEATURE_NAMES), moment_names=list(MOMENT_NAMES),
            encoder={k: v.detach().cpu() for k, v in encoder.state_dict().items()},
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=rng_state(),
            completed_updates=step, training=training), latest)

    if args.resume:
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        assert saved['format'] == 'context_influence_training_v2'
        assert saved['freeze_sha256'] == freeze_sha and saved['backbone_state_sha256'] == backbone_hash
        assert saved['config'] == encoder.config
        assert saved['feature_names'] == list(FEATURE_NAMES) and saved['moment_names'] == list(MOMENT_NAMES)
        encoder.load_state_dict(saved['encoder']); optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler']); restore_rng(saved['rng'])
        step, training = saved['completed_updates'], saved['training']
        assert len(training) == step and 0 <= step <= STEPS
        del saved
    else:
        save_latest()  # Normal initialization is complete before any GP/task work.
    initial_step = step
    while step < STEPS and time.monotonic()-started < args.wall_seconds:
        tick = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        records = []
        for event in batches[step]:
            # Existing GP adaptation is deterministic; posterior proposals and
            # Tab preprocessing already own explicit local seeds.
            histories, support, bounds, aps, task = prepare(event, 'source', None, index, params)
            cid, state = adapted_state(output, event, aps, task, params, freeze_sha)
            move_task(task, torch.device('cuda:0'))
            system.fit(histories, support, event['floor'], aps, bounds, SEED,
                task=task, state=state, event_id=cid, k=event['K'])
            assert consumer.relation_encoder is encoder
            assert len(consumer.estimators) == 2 and all(e.n_estimators == 8 for e in consumer.estimators)
            # No source Q measurements or labels are decoded before H/S fit.
            query = read_scans(CSV, event['query_rows'], True)
            loss = system.backward_source(query['rss'], query['xy'], scaler)
            assert all(p.grad is None for p in network.parameters())
            records.append(dict(event=event['id'], floor=event['floor'], K=event['K'],
                context_id=cid, context_rows=sum(len(h['rss']) for h in histories.values())+len(support['rss']),
                APs=len(aps), query_scans=len(query['rss']),
                query_positions=len(np.unique(query['xy'], axis=0)), system_mde=loss))
            del task, histories, support, query
        scaler.unscale_(optimizer)
        for parameter in parameters:
            parameter.grad.div_(3)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
        old_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < old_scale:
            raise FloatingPointError('AMP overflow; stopped without retry')
        step += 1
        training.append(dict(step=step, events=records, system_mde=float(np.mean([r['system_mde'] for r in records])),
            gradient_norm=norm, amp_scale=scaler.get_scale(), seconds=time.monotonic()-tick,
            peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20))
        optimizer.zero_grad(set_to_none=True)
        save_latest()
        write(output/'training.json', training)
        print(json.dumps(dict(stage='source_update', completed_updates=step,
                              system_mde=training[-1]['system_mde'])), flush=True)
    assert state_hash(network) == backbone_hash
    # latest.pt is authoritative if interrupted between its atomic save and
    # the small JSON record; resume never reconstructs training from JSON.
    write(output/'training.json', training)
    progress = dict(created=now(), freeze_sha256=freeze_sha, completed_updates=step,
        segment_updates=step-initial_step, segment_seconds=time.monotonic()-started,
        segment_wall_seconds=args.wall_seconds, training_seconds=sum(r['seconds'] for r in training),
        backbone_unchanged=True, protected_validation_read=False, target_read=False,
        automatic_resume=False, latest_sha256=digest(latest))
    write(output/'progress.json', progress)
    if step != STEPS:
        return 2
    sidecar = output/'relation_final_24.pt'
    temporary = sidecar.with_suffix('.pt.tmp')
    consumer.save(temporary, source_updates=step,
        provenance=dict(freeze_sha256=freeze_sha, source_only=True, checkpoint_selection=False,
                        objective=contract['objective'], source_episodes=3*step))
    temporary.replace(sidecar)
    write(output/'complete.json', dict(progress, complete=True, source_updates=step,
        source_episodes=3*step, sidecar=str(sidecar), sidecar_sha256=digest(sidecar),
        backbone_state_sha256=backbone_hash, encoder_state_sha256=state_hash(encoder),
        training_sha256=digest(output/'training.json'), convergence_claim=False))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=float, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert args.wall_seconds > 0 and socket.gethostname() == 'ray-web' and torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    try:
        code = run(args)
    except Exception as error:
        if (args.output/'freeze.json').exists():
            write(args.output/'failure.json', dict(created=now(), error=repr(error),
                  automatic_retry=False, recovery='Explicit resume from the last complete update only'))
        raise
    raise SystemExit(code)
