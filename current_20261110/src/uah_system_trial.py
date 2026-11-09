"""Existing GP/Tab systems on one external UAH floor and two acquisition routes.

Six support-only GP adaptations precede all evaluation. The original public,
ordinary30 and full24 networks stay fixed; their coordinate half averages
with the same retained GP are declared before scores. No historical GP fit,
network training, new probability mechanism or target model selection.
"""
import argparse
import gc
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

import predict as gp
from building_information_gp_trial import move_task
from building_information_tab_trial import digest, now, nearest3, write
from position_fusion_gp import prepare_from_scans, predict_retained
from position_fusion_trial import CODE as EXISTING_CODE
from tabicl_history import CHECKPOINT, PACKAGE, FrozenHistoryRegressor, state_hash
from uah_system_assets import (build_manifest, load_context, load_history,
                               read_query_rss, read_query_truth)

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/uah_system_20261109'
SEED = 11509159
TABS = ('public', 'ordinary', 'full24')
METHODS = ('3NN', 'GP', *TABS, *(f'GP_{m}_half' for m in TABS))
CODE = tuple(dict.fromkeys((*EXISTING_CODE, 'uah_system_trial.py', 'uah_system_assets.py')))
LIMITS = [
    'One exposed external development floor, not a blind test or broad generalization guarantee',
    'Two same-day routes and device-file labels; no claim of longitudinal or static repeated scans',
    'Positions are time-interpolated landmarks; K means K positions with one original scan each',
    'Original selected H only: 128/128/109 scans; all methods share these same historical assets',
    'Public, ordinary30, full24 and GP base/observation parameters remain UJI-derived',
    'Only GP source weights and intercept choice adapt to each UAH Support; no Query fitting',
    'Fixed half averages are mature fusion; improvement alone is not a novel historical mechanism',
    'Directions have different Query counts; aggregate equally by direction and K, then position']


def read(path):
    return json.loads(Path(path).read_text())


def initialize(args):
    manifest = build_manifest()
    events = manifest['events']
    assert len(events) == 6 and {e['K'] for e in events} == {1, 3, 5}
    source = ROOT/'results/ordinary_context_trial_20261109'
    full = ROOT/'results/tabicl_multitask_full_trial_20261107_checked'
    ordinary, trained = read(source/'complete.json'), read(full/'complete.json')
    assert ordinary['complete'] and ordinary['selected_step'] == 30 and trained['complete']
    assert digest(source/'freeze.json') == ordinary['freeze_sha256']
    assert digest(full/'freeze.json') == trained['freeze_sha256']
    checkpoints = dict(public=CHECKPOINT, ordinary=Path(ordinary['selected_checkpoint']),
                       full24=full/'full_step_24.ckpt')
    assert digest(checkpoints['ordinary']) == ordinary['selected_checkpoint_sha256']
    assert digest(checkpoints['full24']) == trained['checkpoint_sha256']['24']
    inputs = dict(manifest['inputs'])
    for p in (source/'complete.json', source/'freeze.json', full/'complete.json', full/'freeze.json',
              *checkpoints.values()):
        inputs[str(p)] = digest(p)
    contract = dict(manifest=manifest, inputs=inputs,
        code={name: digest(ROOT/name) for name in CODE},
        package={str(p.relative_to(PACKAGE)): digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))},
        checkpoints={name: str(p) for name, p in checkpoints.items()}, Tab_seed=SEED,
        methods=METHODS, primary='GP_ordinary_half versus same-information GP',
        attribution='Three unchanged Tab checkpoints, each alone and with the identical fixed GP half',
        GP='Original support-only source/intercept adaptation and retained M-PMC posterior; no numerical policy change',
        phases='All six support adaptations sealed before evaluation; per-event all predictions saved before truth',
        resources=dict(GPUs=1, CPU_threads=2, downloads=0, segment_seconds=args.seconds,
                       boundary='Complete event; explicit resume; no automatic retry'), limits=LIMITS)
    # JSON round-trip makes tuples match their persisted list representation.
    contract = json.loads(json.dumps(contract))
    out = args.output
    if args.resume:
        assert read(out/'freeze.json')['contract'] == contract
        progress = read(out/'progress.json')
        assert progress['freeze_sha256'] == digest(out/'freeze.json')
        for r in progress['states']:
            assert digest(Path(r['path'])) == r['sha256']
        for r in progress['records']:
            for key in ('prediction', 'scored'):
                assert digest(Path(r[key+'_path'])) == r[key+'_sha256']
    else:
        assert args.phase == 'adapt'
        out.mkdir(parents=True, exist_ok=False)
        write(out/'freeze.json', dict(created=now(), contract=contract))
        progress = dict(freeze_sha256=digest(out/'freeze.json'), states=[], records=[], segments=[])
        write(out/'progress.json', progress)
    return manifest, read(manifest['parameters_path']), checkpoints, progress


def summarize(records):
    assert len(records) == 6
    result = {}
    for method in METHODS:
        result[method] = dict(mean=float(np.mean([r['mde'][method] for r in records])),
            by_K={str(k): float(np.mean([r['mde'][method] for r in records if r['K'] == k])) for k in (1, 3, 5)},
            by_direction={d: float(np.mean([r['mde'][method] for r in records if r['direction'] == d]))
                          for d in sorted({r['direction'] for r in records})})
    return result


def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    manifest, params, checkpoints, progress = initialize(args)
    out = args.output
    if args.phase == 'evaluate':
        adapted = read(out/'adapted.json')
        assert adapted['complete'] and adapted['freeze_sha256'] == progress['freeze_sha256']
        assert adapted['states'] == progress['states'] and len(adapted['states']) == 6
    collection = 'states' if args.phase == 'adapt' else 'records'
    completed = {r['event'] for r in progress[collection]}
    assert len(completed) < 6, 'Do not rerun a completed phase'
    states = {r['event']: r for r in progress['states']}
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    for event in manifest['events']:
        eid = event['id']
        if eid in completed:
            continue
        if time.monotonic()-started >= args.seconds:
            break
        tick = time.monotonic()
        context = load_context(manifest, event)
        histories, support = context['histories'], context['support']
        aps, bounds, origin = context['aps'], context['bounds'], context['origin']
        task = prepare_from_scans(histories, support, params['base'], params['obs'],
            lambda f, a: load_history(manifest, f, a, origin), origin, bounds)
        assert np.array_equal(task['aps'], aps) and np.array_equal(task['bounds'], bounds)
        assert sum(len(h['rss']) for h in histories.values()) == 365
        assert len(support['rss']) == len(np.unique(support['xy'], axis=0)) == event['K']
        if args.phase == 'adapt':
            torch.set_default_dtype(torch.float64)
            try:
                state = gp.adapt(task['cases'], params['base'], params['obs'])
            finally:
                torch.set_default_dtype(torch.float32)
            path = out/f'{eid}_state.json'
            write(path, dict(event=eid, created=now(), state=state, aps=aps.tolist(), bounds=bounds.tolist()))
            record = dict(event=eid, K=event['K'], direction=event['direction'],
                          path=str(path), sha256=digest(path), seconds=time.monotonic()-tick,
                          query_rows_parsed=0)
            progress['states'].append(record)
            del task
        else:
            state = read(states[eid]['path'])
            assert state['aps'] == aps.tolist() and state['bounds'] == bounds.tolist()
            qrss = read_query_rss(manifest, event)
            move_task(task, torch.device('cuda:0'))
            prediction = predict_retained(task, state['state'], qrss, eid, event['K'])
            predictions = dict(GP=prediction['projected'])
            extra = dict(GP_raw=prediction['raw'], GP_scores=prediction['scores'],
                         grid=task['absolute_grid'])
            diagnostic = out/f'{eid}_GP_diagnostics.json'
            write(diagnostic, prediction['diagnostics'])
            del task, prediction
            gc.collect()
            torch.cuda.empty_cache()
            parts = [*histories.values(), support]
            knn = nearest3(np.concatenate([h['rss'] for h in parts]),
                           np.concatenate([h['xy'] for h in parts]), qrss, aps, bounds)
            predictions['3NN'], extra['3NN_raw'] = knn['projected'], knn['raw']
            for name, checkpoint in checkpoints.items():
                with torch.no_grad():
                    model = FrozenHistoryRegressor(checkpoint, device='cuda:0')
                    model.loaded.model_.requires_grad_(False).eval()
                    original = state_hash(model.loaded.model_)
                    model.fit(histories, support, event['floor'], aps, bounds, SEED)
                    value = model.predict(qrss)
                    predictions[name], extra[name+'_raw'] = value['projected'], value['raw']
                    predictions[f'GP_{name}_half'] = .5*(predictions['GP']+predictions[name])
                    assert state_hash(model.loaded.model_) == original
                    del model, value
                gc.collect()
                torch.cuda.empty_cache()
            prediction_path = out/f'{eid}_unscored.npz'
            np.savez_compressed(prediction_path, **predictions, **extra, aps=aps, bounds=bounds,
                                query_rows=event['query_rows'])
            # No Query coordinates enter the completed predictions above.
            truth = read_query_truth(manifest, event)
            assert not (truth[:, None] == support['xy'][None]).all(-1).any()
            positions, group, counts = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
            weights = 1/(len(positions)*counts[group])
            errors = {m: np.linalg.norm(predictions[m]-truth, axis=1) for m in METHODS}
            scored_path = out/f'{eid}_scored.npz'
            np.savez_compressed(scored_path, truth=truth, weights=weights, **errors)
            record = dict(event=eid, K=event['K'], direction=event['direction'],
                query_scans=len(truth), query_positions=len(positions),
                mde={m: float(weights@v) for m, v in errors.items()},
                prediction_path=str(prediction_path), prediction_sha256=digest(prediction_path),
                scored_path=str(scored_path), scored_sha256=digest(scored_path),
                GP_diagnostics_path=str(diagnostic), GP_diagnostics_sha256=digest(diagnostic),
                seconds=time.monotonic()-tick)
            progress['records'].append(record)
        completed.add(eid)
        write(out/'progress.json', progress)
        print(json.dumps(dict(phase=args.phase, event=eid, completed=len(completed),
                              total=6, seconds=record['seconds'])), flush=True)
    elapsed = time.monotonic()-started
    progress['segments'].append(dict(phase=args.phase, ended=now(), seconds=elapsed,
                                    budget=args.seconds, budget_overrun=elapsed > args.seconds))
    write(out/'progress.json', progress)
    complete = len(completed) == 6
    if complete and args.phase == 'adapt':
        write(out/'adapted.json', dict(complete=True, created=now(), states=progress['states'],
                                      freeze_sha256=progress['freeze_sha256'], query_rows_parsed=0))
    if complete and args.phase == 'evaluate':
        summary = summarize(progress['records'])
        write(out/'complete.json', dict(complete=True, created=now(), summary=summary,
            records=progress['records'], segments=progress['segments'],
            freeze_sha256=progress['freeze_sha256'], adapted_sha256=digest(out/'adapted.json'),
            primary_gain_vs_GP=1-summary['GP_ordinary_half']['mean']/summary['GP']['mean'],
            network_updates=0, history_fits=0, support_adaptations=6, target_selection=False,
            blind_test=False, limits=LIMITS))
    return 0 if complete else 2


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web', 'No local numerical execution'
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('adapt', 'evaluate'), required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seconds', type=float, default=900.)
    parser.add_argument('--output', type=Path, default=OUT)
    raise SystemExit(run(parser.parse_args()))
