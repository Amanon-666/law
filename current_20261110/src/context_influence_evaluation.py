"""Future fixed-v2 evaluation on the existing 54 UJI development events.

Formal training/evaluation is paused; this entry's existence does not authorize
a run. Only a completed24 source sidecar on the original frozen full24 can be
loaded. No optimizer, checkpoint choice, target adaptation fit or new data is
introduced. Baselines come from the signed, complete HS attribution table.

Forward path: original H/S rows and cached GP state -> original event-seeded
retained posterior plus v2 relations -> original raw Tab context -> Query RSS
-> new Tab/system points -> persist -> Query truth. Recomputed GP coordinates
must match their signed cache at the established1e-6 tolerance; the identical
cached GP point is used in every reported fixed-half combination. No tolerance
search, retry or alternative posterior is allowed. Truth cannot change later
events, model parameters or choices.

This measures the implemented relation path, not novelty or an equal-compute
ordinary-finetuning attribution. Existing B1/B2 positions are development data.
Only syntax/static reasoning is authorized at implementation time; v2 source
training and this full evaluation have not yet been executed.
"""
import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

from building_information_gp_trial import move_task
from building_information_tab_trial import digest, now, read_scans, read_truth, write
from context_influence_consumer import InfluenceHistoryRegressor
from context_influence_system import InfluencePositionSystem
from context_influence_training import source_contract, CODE as TRAIN_CODE
from position_fusion_trial import ROOT, CHECKPOINT, PARAMETERS, CSV, prepare, read
from tabicl_history import state_hash

RESULTS = ROOT/'results'
ASSETS = RESULTS/'building_information_assets_20261109'
BASELINES = RESULTS/'history_training_attribution_20261110'
FUSION = RESULTS/'position_fusion_20261109_continued'
GP = RESULTS/'building_information_gp_20261109'
SEED = 11509159
GP_ATOL = 1e-6
REUSED = {'GP': 'GP', 'public': 'public_HS', 'ordinary': 'ordinary_HS',
    'full24': 'full24_HS', 'GP_public_half': 'GP_public_HS_half',
    'GP_ordinary_half': 'GP_ordinary_HS_half', 'GP_full24_half': 'GP_full24_HS_half',
    '3NN': '3NN_HS'}
METHODS = (*REUSED, 'relation_Tab', 'GP_relation_half')
CODE = tuple(dict.fromkeys((*TRAIN_CODE, 'context_influence_evaluation.py',
    'history_training_attribution_trial.py', 'trained_system_validation_trial.py')))


def initialize(args):
    training = args.training_output
    done, frozen = read(training/'complete.json'), read(training/'freeze.json')
    assert done['complete'] and done['source_updates'] == done['completed_updates'] == 24
    assert done['source_episodes'] == 72 and done['backbone_unchanged']
    assert done['target_read'] is False and done['protected_validation_read'] is False
    assert done['freeze_sha256'] == digest(training/'freeze.json')
    current, _, _, _ = source_contract()
    assert current == frozen['contract']
    assert done['backbone_state_sha256'] == current['full24_state_sha256']
    sidecar = Path(done['sidecar'])
    assert digest(sidecar) == done['sidecar_sha256']
    assert digest(training/'training.json') == done['training_sha256']
    saved = torch.load(sidecar, map_location='cpu', weights_only=False)
    assert saved['format'] == 'gp_column_influence_v2' and saved['source_updates'] == 24
    assert saved['provenance']['freeze_sha256'] == done['freeze_sha256']
    assert saved['backbone_state_sha256'] == done['backbone_state_sha256']
    assert saved['feature_names'] == current['feature_names'] and saved['moment_names'] == current['moment_names']
    del saved

    manifest, index = read(ASSETS/'manifest.json'), read(ASSETS/'histories.json')
    assets_done = read(ASSETS/'complete.json')
    assert assets_done['complete'] and digest(ASSETS/'freeze.json') == assets_done['freeze_sha256']
    assert digest(ASSETS/'manifest.json') == assets_done['manifest_sha256'] == index['manifest_sha256']
    assert digest(ASSETS/'histories.json') == assets_done['histories_sha256']
    assert index['history_rows'] == manifest['history_rows']
    assert Path(manifest['csv_path']) == CSV and digest(CSV) == manifest['csv_sha256']
    assert Path(index['parameters_path']) == PARAMETERS
    assert digest(PARAMETERS) == index['parameters_sha256'] == assets_done['parameters_sha256']
    assert len(manifest['events']) == 54 and len(manifest['floors']) == 9
    assert manifest['seed'] == SEED and manifest['K'] == [1, 3, 5]
    assert manifest['buildings'] == ['B1', 'B2']
    baseline_done, gp_done, fusion_done = (read(BASELINES/'complete.json'),
        read(GP/'complete.json'), read(FUSION/'target/complete.json'))
    for folder, record in ((BASELINES, baseline_done), (GP, gp_done), (FUSION, fusion_done)):
        assert record['complete'] and digest(folder/'freeze.json') == record['freeze_sha256']
    baseline_contract = read(BASELINES/'freeze.json')['contract']
    fusion_contract = read(FUSION/'freeze.json')['contract']
    assert baseline_contract['events'] == manifest['events']
    assert baseline_contract['seed'] == SEED and baseline_contract['package'] == current['package']
    # The fusion freeze binds the original target manifest; its records own
    # GP file paths, which legitimately mix absolute and ROOT-relative paths.
    assert fusion_contract['inputs'][str(ASSETS/'manifest.json')] == digest(ASSETS/'manifest.json')
    references = {r['event']: r for r in baseline_done['records']}
    gp_records = {r['event']: r for r in fusion_done['records']}
    ids = {e['id'] for e in manifest['events']}
    assert set(references) == set(gp_records) == ids
    gp_progress = read(GP/'progress.json')
    assert gp_progress['freeze_sha256'] == gp_done['freeze_sha256']
    states, predictions = {}, {}
    for event in manifest['events']:
        eid = event['id']
        if event['K'] == 1:
            record = gp_progress['states'][f'{eid}_HS_GP_MPMC']
            path = GP/record['path']
            assert digest(path) == record['sha256']
        else:
            path = (ROOT/gp_records[eid]['prediction_path']).parent/f'{eid}_state.json'
            record = read(path)
            assert record['event'] == eid
        # K3/5 old records do not separately sign their state JSON. Bind that
        # existing file now, then require the signed GP coordinate replay.
        states[eid] = dict(path=str(path), sha256=digest(path))
        gp_path = ROOT/gp_records[eid]['gp_path']
        assert digest(gp_path) == gp_records[eid]['gp_sha256']
        predictions[str(gp_path)] = gp_records[eid]['gp_sha256']
        for key in ('prediction', 'scored'):
            path, sha = references[eid][key+'_path'], references[eid][key+'_sha256']
            assert digest(path) == sha
            predictions[path] = sha
    paths = (training/'complete.json', training/'freeze.json', training/'training.json', sidecar,
        CHECKPOINT, CSV, PARAMETERS, ASSETS/'manifest.json', ASSETS/'histories.json',
        ASSETS/'complete.json', ASSETS/'freeze.json', BASELINES/'complete.json',
        BASELINES/'freeze.json', GP/'complete.json', GP/'freeze.json', GP/'progress.json',
        FUSION/'target/complete.json', FUSION/'freeze.json')
    contract = dict(code={n: digest(ROOT/n) for n in CODE}, package=current['package'],
        inputs={str(p): digest(p) for p in paths}, events=manifest['events'], seed=SEED,
        states=states, prediction_hashes=predictions, methods=list(METHODS),
        training_freeze_sha256=done['freeze_sha256'], sidecar=str(sidecar),
        backbone_state_sha256=done['backbone_state_sha256'], encoder_state_sha256=done['encoder_state_sha256'],
        posterior_seed='multifloor/{target event_id}/K{k}/{ap}; never the source canonical context ID',
        GP_replay_atol=GP_ATOL, GP_replay_rtol=0.,
        information='Original raw H/current S, AP axis and absolute2m bounds; all54 original Query row sets',
        execution='No GP adapt/refit; recompute original retained S moments once per event; exactly one new relation-enabled Tab path',
        output='Cached GP point retained exactly for every fixed half; original baselines reused by signed row mapping',
        scoring='All new event predictions persisted before event Query truth; equal building/floor/direction/K/position and position-mean p90',
        limits=['Existing B1/B2 development evaluation, not blind or independent across K',
            'Mechanism-effect entry, not novelty or equal-compute ordinary-finetuning attribution',
            'No target selection, best checkpoint, new control training or gate',
            'Current implementation/AST does not authorize a run or establish numerical validity'],
        resources=dict(GPUs=1, CPU_threads=2, device='cuda:0', max_gpu_fraction=.8, downloads=0,
            boundary='Complete event, explicit resume only; no retry or automatic continuation'))
    contract = json.loads(json.dumps(contract))
    if args.resume:
        assert read(args.output/'freeze.json')['contract'] == contract
        progress = read(args.output/'progress.json')
        assert progress['freeze_sha256'] == digest(args.output/'freeze.json') and not progress.get('complete', False)
        for record in progress['records']:
            for key in ('prediction', 'scored'):
                assert digest(record[key+'_path']) == record[key+'_sha256']
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output/'freeze.json', dict(created=now(), contract=contract,
            formal_training_paused_at_implementation=True))
        progress = dict(freeze_sha256=digest(args.output/'freeze.json'), records=[], segments=[])
        write(args.output/'progress.json', progress)
    return manifest, index, read(PARAMETERS), done, sidecar, references, gp_records, states, baseline_done, progress


def summarize(manifest, records):
    """Same §172 weighting, with this file's explicit complete method list."""
    events = {e['id']: e for e in manifest['events']}
    samples = []
    for record in records:
        event = events[record['event']]
        with np.load(record['scored_path']) as saved:
            _, group = np.unique(saved['truth'], axis=0, return_inverse=True)
            n = int(group.max())+1
            errors = {m: np.array([saved[m][group == i].mean() for i in range(n)]) for m in METHODS}
        floors = sum(f.startswith(event['building']) for f in manifest['floors'])
        samples.append(dict(event=event, errors=errors, weight=.5/floors/2/n))

    def aggregate(selected, method):
        values = np.concatenate([s['errors'][method] for s in selected])
        weights = np.concatenate([np.full(len(s['errors'][method]), s['weight']) for s in selected])
        weights /= weights.sum()
        order = np.argsort(values)
        return dict(mean=float(weights@values), position_mean_p90=float(
            values[order][np.searchsorted(np.cumsum(weights[order]), .9)]))

    result = {}
    for method in METHODS:
        result[method] = aggregate(samples, method)
        for key, choices in (('K', (1, 3, 5)), ('floor', manifest['floors']), ('building', manifest['buildings'])):
            result[method]['by_'+key] = {str(c): aggregate(
                [s for s in samples if s['event'][key] == c], method) for c in choices}
    return result


@torch.no_grad()
def run(args):
    started = time.monotonic()
    torch.set_num_threads(2); torch.set_default_dtype(torch.float32)
    manifest, index, params, trained, sidecar, references, gp_records, states, baseline_done, progress = initialize(args)
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    consumer = InfluenceHistoryRegressor(CHECKPOINT, device='cuda:0')
    consumer.load_adapter(sidecar)
    consumer.relation_encoder.requires_grad_(False)
    consumer.relation_encoder.eval()
    backbone = state_hash(consumer.loaded.model_)
    encoder = state_hash(consumer.relation_encoder)
    assert backbone == trained['backbone_state_sha256'] and encoder == trained['encoder_state_sha256']
    system = InfluencePositionSystem(consumer)
    completed = {r['event'] for r in progress['records']}
    for event in manifest['events']:
        eid = event['id']
        if eid in completed:
            continue
        if time.monotonic()-started >= args.wall_seconds:
            break
        tick = time.monotonic()
        histories, support, bounds, aps, task = prepare(event, 'target', manifest, index, params)
        state_record = states[eid]
        assert digest(state_record['path']) == state_record['sha256']
        saved_state = read(state_record['path'])
        assert np.array_equal(saved_state['aps'], aps)
        if event['K'] == 1:
            assert saved_state['sources'] == event['histories']
            assert np.array_equal(saved_state['origin'], task['origin']) and np.array_equal(saved_state['bounds'], bounds)
        else:
            assert saved_state['event'] == eid
        move_task(task, torch.device('cuda:0'))
        system.fit(histories, support, event['floor'], aps, bounds, SEED,
            task=task, state=saved_state['state'], event_id=eid, k=event['K'])
        qrss = read_scans(CSV, event['query_rows'], False)['rss']
        new = system.predict(qrss)
        gp_path = ROOT/gp_records[eid]['gp_path']
        with np.load(gp_path) as cached:
            assert np.array_equal(cached['query_rows'], event['query_rows']) and np.array_equal(cached['aps'], aps)
            assert np.array_equal(cached['grid'], task['absolute_grid'])
            gp_gap = float(np.max(np.abs(new['GP']-cached['raw'])))
            assert gp_gap <= GP_ATOL, (eid, 'Original GP coordinate replay changed', gp_gap)
        with np.load(references[eid]['prediction_path']) as cached:
            assert np.array_equal(cached['query_rows'], event['query_rows'])
            assert np.array_equal(cached['HS_aps'], aps) and np.array_equal(cached['HS_bounds'], bounds)
            predictions = {m: cached[key].copy() for m, key in REUSED.items()}
        assert np.max(np.abs(new['GP']-predictions['GP'])) <= GP_ATOL
        predictions['relation_Tab'] = new['Tab']
        predictions['GP_relation_half'] = .5*(predictions['GP']+new['Tab'])
        path = args.output/f'{eid}_unscored.npz'
        np.savez_compressed(path, **predictions, query_rows=event['query_rows'], aps=aps, bounds=bounds)
        # No Query coordinates (including cached truth) were decoded above.
        truth = read_truth(CSV, event['query_rows'])
        _, groups, counts = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
        weights = 1/(len(counts)*counts[groups])
        errors = {m: np.linalg.norm(predictions[m]-truth, axis=1) for m in METHODS}
        with np.load(references[eid]['scored_path']) as cached:
            assert np.array_equal(truth, cached['truth']) and np.array_equal(weights, cached['weights'])
            for m, key in REUSED.items():
                assert np.allclose(errors[m], cached[key], rtol=0, atol=1e-10)
        scored = args.output/f'{eid}_scored.npz'
        np.savez_compressed(scored, truth=truth, weights=weights, **errors)
        progress['records'].append(dict(event=eid, floor=event['floor'], building=event['building'],
            direction=event['direction'], K=event['K'], APs=len(aps),
            context_rows=sum(len(h['rss']) for h in histories.values())+len(support['rss']),
            prediction_path=str(path), prediction_sha256=digest(path),
            scored_path=str(scored), scored_sha256=digest(scored), GP_replay_max_error=gp_gap,
            mde={m: float(weights@errors[m]) for m in METHODS}, seconds=time.monotonic()-tick))
        write(args.output/'progress.json', progress)
        print(json.dumps(dict(stage='target_event', event=eid, completed=len(progress['records']))), flush=True)
        del task, histories, support, qrss, new
    assert state_hash(consumer.loaded.model_) == backbone
    assert state_hash(consumer.relation_encoder) == encoder
    finished = len(progress['records']) == 54
    progress.update(complete=finished, backbone_unchanged=True, encoder_unchanged=True)
    progress['segments'].append(dict(created=now(), seconds=time.monotonic()-started,
        wall_seconds=args.wall_seconds, completed=len(progress['records']), complete=finished))
    write(args.output/'progress.json', progress)
    if not finished:
        return 2
    summary = summarize(manifest, progress['records'])
    for name, old_name in REUSED.items():
        for metric in ('mean', 'position_mean_p90'):
            assert abs(summary[name][metric]-baseline_done['summary'][old_name][metric]) <= 1e-9
    contrasts = {name: {metric: summary[name][metric]-summary['GP_relation_half'][metric]
                       for metric in ('mean', 'position_mean_p90')}
                 for name in ('GP', 'GP_public_half', 'GP_ordinary_half', 'GP_full24_half')}
    write(args.output/'complete.json', dict(progress, summary=summary, system_differences=contrasts,
        difference_sign='Reference minus relation system; positive means smaller error, not a selection rule',
        training_freeze_sha256=trained['freeze_sha256'], sidecar_sha256=trained['sidecar_sha256'],
        backbone_state_sha256=backbone, encoder_state_sha256=encoder,
        new_Tab_forwards=54, GP_adaptations=0, history_refits=0, network_updates=0,
        target_selection=False, blind_test=False, methods=list(METHODS)))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=float, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert args.wall_seconds > 0 and socket.gethostname() == 'ray-web' and torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    try:
        status = run(args)
    except Exception as error:
        if (args.output/'freeze.json').exists():
            write(args.output/'failure.json', dict(created=now(), error=repr(error), automatic_retry=False))
        raise
    raise SystemExit(status)
