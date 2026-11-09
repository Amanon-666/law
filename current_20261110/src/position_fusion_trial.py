"""One source-selected GP/full24 distribution-fusion trial; server only.

No network training. Original 27 protected B0 tasks select one lambda, then
the frozen 54-event B1/B2 protocol uses it unchanged. Existing target K1 GP
scores and Tab point predictions are reused. New GP H/S fits follow the same
VI/M-PMC path. Source/target query coordinates are read after prediction save.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import predict as gp
from building_information_gp_trial import move_task
from building_information_tab_trial import digest, now, read_scans, read_truth, write, context
from check_tabicl_history_learning import bounds_and_aps
from fields import History
from position_distribution_fusion import (LAMBDAS, BANDWIDTH_METRES, UNIFORM_MASS,
    export_tab_quantiles, tab_grid_logprob, fuse_position_scores)
from position_fusion_gp import prepare_from_scans, predict_retained
from source_episode_catalog import make_episode
from tabicl_multitask_full_trial import events_for
from tabicl_history import PACKAGE, FrozenHistoryRegressor, state_hash
from uji import group_scans, tensor

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
SOURCE = RESULTS/'tabicl_multitask_full_trial_20261107_checked'
CATALOG = RESULTS/'source_episode_catalog_20261105/catalog.json'
SOURCE_ASSETS = RESULTS/'advised_source_assets_20261109'
TARGET_ASSETS = RESULTS/'building_information_assets_20261109'
GP_CACHE = RESULTS/'building_information_gp_20261109'
TAB_CACHE = RESULTS/'building_information_tab_20261109'
PARAMETERS = RESULTS/'paired_assets_20261106/parameters.json'
OUT = RESULTS/'position_fusion_20261109'
SOURCE_REUSE = RESULTS/'position_fusion_20261109_checked'
CHECKPOINT = SOURCE/'full_step_24.ckpt'
CSV = ROOT.parent/'UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv'
CODE = ('position_fusion_trial.py', 'position_distribution_fusion.py', 'position_fusion_gp.py',
        'building_information_gp_trial.py', 'building_information_tab_trial.py',
        'tabicl_history.py', 'tabicl_multitask_full_trial.py', 'source_episode_catalog.py',
        'check_tabicl_history_learning.py', 'episodic_tasks.py', 'uji.py', 'fields.py',
        'predict.py', 'adaptation.py', 'posterior.py', 'sampling.py', 'observations.py')


def read(path):
    return json.loads(Path(path).read_text())


def canonical_context(event):
    value = {k: event[k] for k in ('floor', 'histories', 'history_rows', 'support_rows')}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def balanced_summary(records, phase):
    methods = list(records[0]['mde'])
    floors = sorted({r['floor'] for r in records})
    result = {}
    for method in methods:
        cells = {f: {str(k): float(np.mean([r['mde'][method] for r in records
                     if r['floor'] == f and r['K'] == k])) for k in (1, 3, 5)} for f in floors}
        by_floor = {f: float(np.mean(list(cells[f].values()))) for f in floors}
        if phase == 'source':
            by_k = {str(k): float(np.mean([cells[f][str(k)] for f in floors])) for k in (1, 3, 5)}
            mean = float(np.mean(list(by_floor.values())))
        else:
            by_k = {str(k): float(np.mean([np.mean([cells[f][str(k)] for f in floors
                       if f.startswith(b)]) for b in ('B1', 'B2')])) for k in (1, 3, 5)}
            mean = float(np.mean(list(by_k.values())))
        result[method] = dict(mean=mean, by_K=by_k, by_floor=by_floor, cells=cells)
    return result


def initialize(output):
    catalog = read(CATALOG)
    _, specs = events_for(catalog)
    source_events = [make_episode(catalog, **s) for s in specs]
    target_manifest = read(TARGET_ASSETS/'manifest.json')
    assert len(source_events) == 27 and len(target_manifest['events']) == 54
    assert read(SOURCE/'complete.json')['checkpoint_sha256']['24'] == digest(CHECKPOINT)
    protected = set(catalog['protected_rows'])
    for event in source_events:
        assert protected.issuperset(event['query_rows'])
        assert not protected.intersection(event['support_rows'] +
                      [i for ids in event['history_rows'].values() for i in ids])
    inputs = [CSV, CATALOG, CHECKPOINT, SOURCE/'complete.json', SOURCE/'step_24.json',
              PARAMETERS, SOURCE_ASSETS/'histories.json', SOURCE_ASSETS/'states.jsonl',
              SOURCE_ASSETS/'complete.json', TARGET_ASSETS/'manifest.json',
              TARGET_ASSETS/'histories.json', TARGET_ASSETS/'complete.json',
              GP_CACHE/'progress.json', GP_CACHE/'complete.json',
              TAB_CACHE/'progress.json', TAB_CACHE/'complete.json',
              SOURCE_REUSE/'freeze.json', SOURCE_REUSE/'source/progress.json',
              OUT/'mean_interface_check.json']
    previous = read(SOURCE_REUSE/'freeze.json')['contract']
    assert previous['source_events'] == source_events
    assert previous['lambdas'] == list(LAMBDAS)
    assert previous['bandwidth'] == BANDWIDTH_METRES and previous['uniform_mass'] == UNIFORM_MASS
    for name in CODE:
        if name != 'position_fusion_trial.py':
            assert previous['code'][name] == digest(ROOT/name)
    for path in inputs:
        if str(path) in previous['inputs']:
            assert previous['inputs'][str(path)] == digest(path)
    contract = dict(code={n: digest(ROOT/n) for n in CODE},
        inputs={str(p): digest(p) for p in inputs},
        package={str(p.relative_to(PACKAGE)): digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))},
        source_events=source_events, target_events=target_manifest['events'],
        lambdas=list(LAMBDAS), bandwidth=BANDWIDTH_METRES, uniform_mass=UNIFORM_MASS,
        seed=dict(source=1150546, target=11509159),
        replay_check=dict(mode='Record old-cache difference; compare with fresh full24 mean from the same official forward used for quantiles',
            evidence='First actual source export equals direct official predict and cache exactly. Event21 old-cache gap .02765 remains unexplained; data, weights, transforms and official package hashes match. Official memory-aware single/multi-batch output dtype differs, so bitwise cross-run equivalence is not assumed.',
            reuse='20 completed source events from unchanged method; keep original paths, hashes and failures, no rescoring or reselection'),
        selection='Minimum source equal-floor/equal-K position-balanced MDE across all 27 original protected tasks; exact ties choose smaller lambda',
        frozen_method='Unchanged retained GP full-scan scores + lambda*log(Tab marginal-product grid mass); original full24 H/S transforms and 8 views',
        geometry='Source Tab keeps its original exact H/S bounds; GP rounds those bounds outward to the original absolute 2-unit grid. Target shares the original rounded H/S bounds.',
        reverse_audit=['Field cache rows match legal H; APs and bounds only H/current S',
            'No query labels enter state adaptation, Tab fit, quantile export or fusion',
            'Full24 and GP source parameters frozen; no network or shared GP updates',
            'One scalar selected on B0 only, before any new target prediction/scoring',
            'Target K1 GP scores reused; original Tab mean and zero-lambda mean checked from real prediction output',
            'Original finite VI/M-PMC stopping policy, no retry or degraded fallback'],
        limits=['Axis independence approximation; marginals do not identify a joint position law',
            'GP and Tab share H/S; product is a predictive fusion rule, not independent Bayesian evidence',
            'Source-selected lambda has no guarantee of target non-degradation',
            'Existing development tasks, not blind or LOBO; restricted to registered XY/AP namespaces',
            'Fixed h and uniform mass are unselected practical choices; no variant sweep on failure'],
        practical_gate='Target overall gain >=3% versus GP, <=3/9 worse floor means, and K3/K5 no worse GP; also report full24 separately',
        resources=dict(gpu=1, cpu_threads=2, downloads=0, network_training_updates=0,
                       segment_seconds=900, stop_at='Complete event; explicit resume; sampler/solver failure stops'))
    if (output/'freeze.json').exists():
        assert read(output/'freeze.json')['contract'] == contract
    else:
        output.mkdir(exist_ok=False)
        write(output/'freeze.json', dict(created=now(), contract=contract))
        for name in CODE:
            (output/name).write_bytes((ROOT/name).read_bytes())
    return source_events, target_manifest


def prepare(event, phase, manifest, index, params):
    if phase == 'source':
        assert event['history_rows'] == {f: index['history_rows'][f] for f in event['histories']}
        histories = {f: group_scans(**read_scans(CSV, ids, True)) for f, ids in event['history_rows'].items()}
        support = group_scans(**read_scans(CSV, event['support_rows'], True))
        tab_bounds, aps = bounds_and_aps(histories, support)
        origin = np.zeros(2)
        bounds = np.stack((np.floor(tab_bounds[0]/2)*2, np.ceil(tab_bounds[1]/2)*2))
    else:
        histories, support, _, _, aps, bounds = context(CSV, manifest, event, 'HS')
        tab_bounds = bounds
        origin = np.concatenate([h['xy'] for h in histories.values()]).mean(0)
    fields = {(r['floor'], r['ap']): r for r in index['fields']}

    def load(floor, ap):
        record = fields[(floor, ap)]
        path = ROOT/record['path']
        assert digest(path) == record['sha256']
        with np.load(path) as saved:
            return History(tensor(saved['z']+saved['origin']-origin),
                           dict(mean=tensor(saved['mean']), cov=tensor(saved['cov'])), params['base'])

    task = prepare_from_scans(histories, support, params['base'], params['obs'], load, origin, bounds)
    assert np.array_equal(task['aps'], aps)
    return histories, support, tab_bounds, aps, task


def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    source_events, manifest = initialize(args.output)
    phase = args.phase
    events = source_events if phase == 'source' else manifest['events']
    output = args.output/phase
    output.mkdir(exist_ok=True)
    if phase == 'target':
        selection = read(args.output/'source/complete.json')
        assert selection['complete'] and selection['freeze_sha256'] == digest(args.output/'freeze.json')
        chosen = selection['selected_lambda']
    else:
        chosen = None
    progress_path = output/'progress.json'
    existed = progress_path.exists()
    progress = read(progress_path) if existed else dict(records=[], segments=[])
    if phase == 'source' and not existed:
        progress['records'] = read(SOURCE_REUSE/'source/progress.json')['records']
        assert len(progress['records']) == 20
        for record in progress['records']:
            assert digest(record['prediction_path']) == record['prediction_sha256']
        progress['continued_from'] = str(SOURCE_REUSE)
    done = {r['event'] for r in progress['records']}
    assert args.resume or not existed
    if len(done) == len(events):
        raise RuntimeError('Phase already complete; do not rerun')
    index = read((SOURCE_ASSETS if phase == 'source' else TARGET_ASSETS)/'histories.json')
    params = read(PARAMETERS)
    assert index['parameters_sha256'] == digest(PARAMETERS)
    old_source_states = {r['context_id']: r for r in
        (json.loads(line) for line in (SOURCE_ASSETS/'states.jsonl').read_text().splitlines())}
    old_source_tab = {r['event']: r for r in read(SOURCE/'step_24.json')}
    old_gp = {t['id']: t for t in read(GP_CACHE/'progress.json')['tasks']}
    old_tab = {t['id']: t for t in read(TAB_CACHE/'progress.json')['tasks']}
    model = FrozenHistoryRegressor(CHECKPOINT, device='cuda:0')
    model.loaded.model_.requires_grad_(False)
    initial_state = state_hash(model.loaded.model_)
    seed = 1150546 if phase == 'source' else 11509159
    np.random.seed(seed)
    torch.manual_seed(seed)
    first_count = len(done)
    for event in events:
        eid = event['id']
        if eid in done:
            continue
        if time.monotonic()-started >= args.seconds:
            break
        tick = time.monotonic()
        # Source states and all new target states are sealed before Q RSS.
        cached_gp = phase == 'target' and event['K'] == 1
        if cached_gp:
            histories, support, _, _, aps, bounds = context(CSV, manifest, event, 'HS')
            gp_record = next(p for p in old_gp[f'{eid}_HS_GP_MPMC']['prediction_files']
                             if p['method'] == 'GP_MPMC_retained')
            gp_path = GP_CACHE/gp_record['path']
            assert digest(gp_path) == gp_record['sha256']
            with np.load(gp_path) as saved:
                assert np.array_equal(saved['query_rows'], event['query_rows'])
                assert np.array_equal(saved['aps'], aps) and np.array_equal(saved['bounds'], bounds)
                scores, grid, gp_raw = saved['scores'].copy(), saved['grid'].copy(), saved['raw'].copy()
            task = None
        else:
            histories, support, bounds, aps, task = prepare(event, phase, manifest, index, params)
            state_path = output/f'{eid}_state.json'
            if state_path.exists():
                state = read(state_path)['state']
            else:
                cid = canonical_context(event) if phase == 'source' else None
                if cid in old_source_states:
                    record = old_source_states[cid]
                    assert digest(record['path']) == record['sha256']
                    state = read(record['path'])['state']
                else:
                    # Original GP adapter constructs an untyped jitter eye.
                    # Match its float64 run without constructing Tab in double.
                    torch.set_default_dtype(torch.float64)
                    state = gp.adapt(task['cases'], params['base'], params['obs'])
                    torch.set_default_dtype(torch.float32)
                write(state_path, dict(created=now(), state=state, event=eid, aps=aps.tolist()))
        query_rss = read_scans(CSV, event['query_rows'], False)['rss']
        if not cached_gp:
            move_task(task, torch.device('cuda:0'))
            prediction = predict_retained(task, state, query_rss, eid, event['K'])
            scores, gp_raw = prediction['scores'], prediction['raw']
            grid = task['grid'].cpu().numpy()+task['origin']
            gp_path = output/f'{eid}_gp.npz'
            np.savez_compressed(gp_path, scores=scores, raw=gp_raw, grid=grid, aps=aps,
                                query_rows=event['query_rows'])
            write(output/f'{eid}_gp_diagnostics.json', prediction['diagnostics'])
            del task, prediction
            gc.collect()
            torch.cuda.empty_cache()
        model.fit(histories, support, event['floor'], aps, bounds, seed)
        exported = export_tab_quantiles(model, query_rss)
        tab_gap = None
        if phase == 'source':
            record = old_source_tab[eid]
            old_path = SOURCE/f'step_24_{eid}.npz'
            assert digest(old_path) == record['prediction_sha256']
        elif event['K'] == 1:
            record = old_tab[f'{eid}_HS_full24']
            old_path = TAB_CACHE/record['prediction_path']
            assert digest(old_path) == record['prediction_sha256']
        else:
            old_path = None
        if old_path is not None:
            with np.load(old_path) as saved:
                assert np.array_equal(saved['query_rows'], event['query_rows'])
                tab_gap = float(np.max(np.abs(exported['mean']-saved['raw'])))
            # The control below is the official mean from THIS forward pass.
            # Preserve the old-cache gap as evidence, not a fitted tolerance.
        log_tab = tab_grid_logprob(exported['quantiles'], grid)
        fused = fuse_position_scores(scores, log_tab, grid)
        gp_gap = float(np.max(np.abs(fused['gp_mean']-gp_raw)))
        assert gp_gap <= 1e-6, ('Zero lambda changed GP', gp_gap)
        predictions = dict(GP=gp_raw, full24=exported['projected'])
        if phase == 'source':
            predictions.update({f'lambda_{value:g}': fused['predictions'][i] for i, value in enumerate(LAMBDAS)})
        else:
            predictions['fusion'] = fused['predictions'][LAMBDAS.index(chosen)]
        unscored = output/f'{eid}_unscored.npz'
        np.savez_compressed(unscored, **predictions, query_rows=event['query_rows'], aps=aps,
                            bounds=bounds, quantiles=exported['quantiles'], tab_log_mass=log_tab)
        # Labels are used only below the persisted predictions.
        truth = read_truth(CSV, event['query_rows'])
        _, groups, counts = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
        weights = 1/(len(counts)*counts[groups])
        errors = {name: np.linalg.norm(value-truth, axis=1) for name, value in predictions.items()}
        scored = output/f'{eid}_scored.npz'
        np.savez_compressed(scored, truth=truth, weights=weights, **errors)
        record = dict(event=eid, floor=event['floor'], K=event['K'],
            mde={name: float(weights@value) for name, value in errors.items()},
            tab_mean_max_gap=tab_gap, zero_lambda_max_gap=gp_gap,
            prediction_path=str(unscored), prediction_sha256=digest(unscored),
            scored_sha256=digest(scored), gp_path=str(gp_path), gp_sha256=digest(gp_path),
            seconds=time.monotonic()-tick)
        progress['records'].append(record)
        done.add(eid)
        write(progress_path, progress)
        print(json.dumps(dict(phase=phase, event=eid, completed=len(done), total=len(events),
                             seconds=record['seconds'], mde=record['mde'])), flush=True)
    assert state_hash(model.loaded.model_) == initial_state
    complete = len(done) == len(events)
    progress['segments'].append(dict(created=now(), seconds=time.monotonic()-started,
        completed=len(done)-first_count, budget=args.seconds, weights_unchanged=True))
    progress['complete'] = complete
    write(progress_path, progress)
    if complete:
        summary = balanced_summary(progress['records'], phase)
        final = dict(complete=True, created=now(), freeze_sha256=digest(args.output/'freeze.json'),
                     phase=phase, summary=summary, records=progress['records'], segments=progress['segments'])
        if phase == 'source':
            chosen = min(LAMBDAS, key=lambda value: summary[f'lambda_{value:g}']['mean'])
            final['selected_lambda'] = chosen
        else:
            final['selected_lambda'] = chosen
            a, b = summary['fusion'], summary['GP']
            worse = sum(a['by_floor'][f] > b['by_floor'][f] for f in a['by_floor'])
            gain = 1-a['mean']/b['mean']
            final['practical_gate'] = dict(relative_gain=gain, worse_floors=worse,
                passed=gain >= .03 and worse <= 3 and all(a['by_K'][k] <= b['by_K'][k] for k in ('3', '5')))
        write(output/'complete.json', final)
        print(json.dumps(dict(complete=True, phase=phase, selected_lambda=chosen, summary=summary)), flush=True)
    else:
        print(json.dumps(dict(complete=False, phase=phase, completed=len(done),
                              stop='segment_budget', explicit_resume_required=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('source', 'target'), required=True)
    parser.add_argument('--seconds', type=float, default=900.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args())
