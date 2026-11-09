"""One frozen comparison of predictive and empirical historical contexts.

Four existing K1R3 development events, 20 fits, no training/refitting/downloads.
All contexts and generated samples are frozen before opening query arrays.
"""
import ast
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import time

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch

import acquisition_bank_transfer as gp
import common_update_transport_trial as assets
from common_update_transport import build_transport
from conditional_budget_trial import history_reader
from episodic_tasks import task_inputs
from observations import scan_log_prob
from predictive_scan_context import (check_sampler, retained_fields,
    generate_context, concatenate_context)
from prior_mean_transport_trial import bind
from recomposition_trial import ROOT, OLD, PLAN, digest, write, now
from tabicl_history import FrozenHistoryRegressor, state_hash
from uji import group_scans

OUT = ROOT/'results/predictive_context_20261108'
PARENT = ROOT/'results/prior_mean_transport_20261108'
WALL_SECONDS = 900
SEED = 11508147
BASELINES = ('GP', 'GP_mean', 'retained', 'retained_mean', 'full24')


def code_inputs():
    pending, found = [Path(__file__).name], {}
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT/name
        found[name] = digest(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            pending.extend(m.split('.')[0]+'.py' for m in modules
                if (ROOT/(m.split('.')[0]+'.py')).is_file())
    return found


def progress(stage, **values):
    item = dict(created=now(), stage=stage, **values)
    write(OUT/'progress.json', item)
    print(json.dumps(item), flush=True)


def deadline(signum, frame):
    raise TimeoutError('Fixed 900-second wall budget exhausted; no retry')


@torch.no_grad()
def run():
    started = time.monotonic()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.5, 0)
    _, source, checkpoint, inputs, package = gp.source_assets()
    parent = json.loads((PARENT/'complete.json').read_text())
    assert parent['complete'] and parent['status'] == 'evaluated' and parent['model_state_unchanged']
    bind(inputs, PARENT/'complete.json')
    bind(inputs, PARENT/'freeze.json', parent['freeze_sha256'])
    frozen = json.loads((PARENT/'freeze.json').read_text())
    assert frozen['package'] == package
    for name, value in frozen['code'].items():
        assert digest(ROOT/name) == value, name
    for path, value in frozen['inputs'].items():
        bind(inputs, Path(path), value)
    events = frozen['events']
    assert [e['id'] for e in events] == [f'B1F{f}_acquisition_{a}' for f in (1, 2) for a in (0, 1)]
    assert frozen['tab_preprocessing_seed'] == assets.SEED
    sampler_check = check_sampler()
    write(OUT/'sampler_check.json', sampler_check)
    progress('selection_moment_check_passed', seconds=time.monotonic()-started)
    plan = json.loads(PLAN.read_text())
    origin = np.asarray(plan['origin'])
    meta_path = OLD/'results/decision_local_pilot_20261030/validation_0/meta.json'
    bind(inputs, meta_path)
    meta = json.loads(meta_path.read_text())
    prepared = []
    for event_index, event in enumerate(events):
        eid, floor = event['id'], event['floor']
        assert event['K'] == 1 and event['R'] == 3 and event['histories'] == ['B1F0', 'B1F3']
        old = next(r for r in parent['records'] if r['event'] == eid)
        prediction_path = PARENT/f'{eid}.npz'
        bind(inputs, prediction_path, old['prediction_sha256'])
        ids = event['support_rows']+[i for rr in event['history_rows'].values() for i in rr]
        assert not set(ids) & set(event['query_rows'])
        rows = assets.read_selected(ids)
        assert all(rows[i]['floor'] == name for name, rr in event['history_rows'].items() for i in rr)
        assert all(rows[i]['floor'] == floor for i in event['support_rows'])
        histories = {name:assets.collect(rows, rr, origin) for name, rr in event['history_rows'].items()}
        support = assets.collect(rows, event['support_rows'], origin)
        assert len(support['rss']) == 3 and len(support['positions']) == 1
        task = task_inputs(histories, support, plan['base'], plan['obs'],
            history_reader(meta, plan['base'], origin, inputs))
        selected = frozen['GP_state'][eid]
        geometry = next(g for g in frozen['geometry'] if g['event'] == eid)
        field = retained_fields(task, selected, floor)
        aps = field['aps']
        bounds = np.stack((field['grid'].min(0), field['grid'].max(0)))
        assert np.array_equal(aps, geometry['APs']) and np.array_equal(bounds, geometry['bounds'])
        # Mean output does not depend on sampled worlds. One unused world is
        # requested solely to reuse the original frozen transport implementation.
        transport = build_transport(histories, support, task, selected, worlds=1)
        real = concatenate_context(list(transport['mean_histories'].values()), floor)[floor]
        raw = concatenate_context(list(histories.values()), floor)[floor]
        count = len(real['rss'])
        assert count == len(raw['rss']) and np.array_equal(real['xy'], raw['xy'])
        generated, indices = generate_context(field, task['obs'], 2*count,
                                               real['rss'].shape[1], SEED+event_index)
        first = group_scans(generated['rss'][:count], generated['xy'][:count])
        contexts = [
            ('common_replay', {floor:real}),
            ('primary', concatenate_context([real, first], floor)),
            ('raw_augmented', concatenate_context([raw, first], floor)),
            ('GP_generated', {floor:generated}),
            ('duplicated_common', concatenate_context([real, real], floor))]
        assert all(len(h[floor]['rss']) == (count if label == 'common_replay' else 2*count)
                   for label, h in contexts)
        sample_path = OUT/f'{eid}_generated.npz'
        np.savez_compressed(sample_path, rss=generated['rss'], xy=generated['xy'],
            grid_indices=indices, aps=aps)
        # Whole matrices remain on the server; summaries disclose masks/ranges.
        values = generated['rss'][:, aps]
        detected = values != 100
        all_H_absent = np.all(raw['rss'][:, aps] == 100, axis=0)
        diagnostic = dict(event=eid, seed=SEED+event_index, original_history_rows=count,
            generated_rows=2*count, unique_generated_positions=len(generated['positions']),
            grid_points=len(field['grid']), detected_cells=int(detected.sum()),
            all_missing_scans=int((~detected).all(1).sum()),
            detected_outside_reference=int(((values[detected] < -105) | (values[detected] > 0)).sum()),
            all_H_absent_APs=aps[all_H_absent].tolist(),
            generated_detections_in_all_H_absent_APs=int(detected[:, all_H_absent].sum()),
            generated_sha256=digest(sample_path), minimum_variance=float(field['variance'].min()),
            maximum_variance=float(field['variance'].max()),
            mean_transport_worlds_used=0, unused_worlds_computed=1,
            target_role=True, real_support_rows_inserted_by_consumer=3)
        write(OUT/f'{eid}_HS_diagnostics.json', diagnostic)
        prepared.append(dict(event=event, old=old, prediction_path=prediction_path,
            support=support, contexts=contexts, count=count, field=field, obs=task['obs'],
            aps=aps, bounds=bounds, diagnostic=diagnostic,
            diagnostic_sha256=digest(OUT/f'{eid}_HS_diagnostics.json')))
        progress('HS_contexts_complete', event=eid, seconds=time.monotonic()-started)
        del task, transport, rows, histories
    code = code_inputs()
    freeze = dict(created=now(), events=events, code=code, inputs=inputs, package=package,
        parent=str(PARENT), GP_state=frozen['GP_state'], checkpoint=str(checkpoint),
        trained_state_sha256=source['trained_state_sha256'], tab_preprocessing_seed=assets.SEED,
        generation_seed=SEED, sampler_check_sha256=digest(OUT/'sampler_check.json'),
        geometry=[dict(event=p['event']['id'], APs=p['aps'].tolist(), bounds=p['bounds'].tolist(),
                       grid_points=len(p['field']['grid']), history_rows=p['count']) for p in prepared],
        HS_diagnostic_hashes={p['event']['id']:p['diagnostic_sha256'] for p in prepared},
        generated_hashes={p['event']['id']:p['diagnostic']['generated_sha256'] for p in prepared},
        distribution='Uniform original grid; independent rows/APs from retained GP Gaussian moment observation law. D/R share F; measurement noise added once.',
        primary='Single paired common-calibrated H plus N generated rows; projected full24 coordinates.',
        controls=['Raw target-role H plus same N generated rows', '2N generated rows',
                  'Common H duplicated twice', 'Single paired common H replay'],
        context_size='All four comparison arms 2N+3; common replay N+3. S inserted once.',
        adoption='Each floor >=3% better than all five GP median/mean/retained/full24 readouts; all events no worse.',
        structural='Each floor strictly better than cached four-permutation common; all events no worse.',
        attribution='Each floor strictly better than all three matched controls; all events no worse.',
        replay=dict(common_coordinate_atol=1e-4, GP_coordinate_atol=1e-6),
        weighting='Equal physical positions within event; equal directions within floor; equal floors.',
        query_isolation='All contexts, generated files and this freeze precede Q RSS/prediction arrays.',
        limitations=['Four exposed development events; no blind validation, K curve or seed selection.',
            'Gaussian moment predictive model, not exact nonGaussian S posterior or real target copula.',
            'Generated rows are not new independent target measurements or one joint GP map.',
            'GP augmentation is mature; combination and model choice do not establish novelty.',
            'One fixed generation per event; simulation error can matter. No automatic resampling.'],
        versions={n:importlib.metadata.version(n) for n in ('numpy', 'scipy', 'torch', 'scikit-learn')},
        resources=dict(cpu=2, gpu=1, max_vram_fraction=.5, hard_worker_seconds=WALL_SECONDS,
                       fits=20, downloads=0, training_updates=0, GP_refits=0, retries=0))
    write(OUT/'freeze.json', freeze)
    for name in code:
        (OUT/name).write_bytes((ROOT/name).read_bytes())
    gc.collect()
    # First query-value / cached-prediction-array access.
    query_rows = assets.read_selected([i for e in events for i in e['query_rows']])
    for p in prepared:
        event = p['event']
        query = assets.collect(query_rows, event['query_rows'], origin)
        assert all(query_rows[i]['floor'] == event['floor'] for i in event['query_rows'])
        assert len(query['positions']) == 12 and np.all(query['stats']['n'] == 3)
        assert not set(map(tuple, query['positions'])) & set(map(tuple, p['support']['positions']))
        weights = 1/(len(query['positions'])*query['stats']['n'][query['group']])
        predictions = {}
        with np.load(p['prediction_path']) as saved:
            for key, value in (('query_rows', event['query_rows']), ('truth', query['xy']),
                               ('weights', weights), ('group', query['group'])):
                assert np.array_equal(saved[key], value)
            for name in BASELINES+('old_common', 'old_common_paired', 'target_H0'):
                predictions[name] = saved[name].copy()
                mde = float(weights@np.linalg.norm(predictions[name]-query['xy'], axis=1))
                assert abs(mde-p['old']['mde'][name]) < 1e-9, (event['id'], name)
            wk = saved['WKNN_errors'].copy()
        assert abs(float(weights@wk)-p['old']['mde']['WKNN']) < 1e-9
        # Validate the newly exposed predictive-field path against the existing
        # complete retained-GP coordinate outputs; not a new GP fit/comparator.
        field = p['field']
        score = torch.zeros((len(query['rss']), len(field['grid'])))
        for column, ap in enumerate(p['aps']):
            score += scan_log_prob(torch.tensor(query['rss'][:, ap]),
                torch.tensor(field['mean'][:, column]), torch.tensor(field['variance'][:, column]), p['obs'])
        readout = gp.actions(score.log_softmax(1), torch.tensor(field['grid']))
        gp_gap = max(float(np.abs(readout[key]-predictions[name]).max())
                     for key, name in (('mean', 'retained_mean'), ('median', 'retained')))
        assert gp_gap <= 1e-6, (event['id'], gp_gap)
        p.update(query=query, weights=weights, predictions=predictions,
                 WKNN_errors=wk, GP_replay_gap=gp_gap)
        del p['field']
    torch.set_default_dtype(torch.float32)
    model = FrozenHistoryRegressor(checkpoint, device='cuda:0')
    model.loaded.model_.requires_grad_(False).eval()
    expected = source['trained_state_sha256']
    assert state_hash(model.loaded.model_) == expected
    records = []
    for p in prepared:
        event, predictions, query = p['event'], p['predictions'], p['query']
        timings = {}
        for label, h in p['contexts']:
            before = time.monotonic()
            model.fit(h, p['support'], event['floor'], p['aps'], p['bounds'], assets.SEED)
            count = (p['count'] if label == 'common_replay' else 2*p['count'])+3
            assert all(m.n_samples_in_ == count and m.model_ is model.loaded.model_ for m in model.estimators)
            predictions[label] = model.predict(query['rss'])['projected']
            timings[label] = time.monotonic()-before
            if label == 'common_replay':
                replay_gap = float(np.abs(predictions[label]-predictions['old_common_paired']).max())
                assert replay_gap <= 1e-4, (event['id'], replay_gap)
            progress('predicted', event=event['id'], context=label, seconds=time.monotonic()-started)
        errors = {name:np.linalg.norm(xy-query['xy'], axis=1) for name, xy in predictions.items()}
        errors['WKNN'] = p['WKNN_errors']
        mde = {name:float(p['weights']@error) for name, error in errors.items()}
        path = OUT/f"{event['id']}.npz"
        np.savez_compressed(path, **predictions, WKNN_errors=p['WKNN_errors'], truth=query['xy'],
            weights=p['weights'], group=query['group'], query_rows=event['query_rows'])
        records.append(dict(event=event['id'], floor=event['floor'], mde=mde,
            primary_minus={name:mde['primary']-mde[name] for name in (
                'raw_augmented', 'GP_generated', 'duplicated_common', 'old_common', 'old_common_paired')},
            per_position_errors={name:[float(error[query['group'] == j].mean())
                for j in range(len(query['positions']))] for name, error in errors.items()},
            context_seconds=timings, replay_max_coordinate_error=replay_gap,
            GP_replay_max_coordinate_error=p['GP_replay_gap'], diagnostic=p['diagnostic'],
            prediction_sha256=digest(path)))
    floor_means = {floor:{name:float(np.mean([r['mde'][name] for r in records if r['floor'] == floor]))
        for name in records[0]['mde']} for floor in ('B1F1', 'B1F2')}
    mean_gate = all(v['primary'] <= .97*v[name] for v in floor_means.values() for name in BASELINES)
    event_gate = all(r['mde']['primary'] <= r['mde'][name] for r in records for name in BASELINES)
    def increment_gate(names):
        return dict(floor_mean=all(v['primary'] < v[name] for v in floor_means.values() for name in names),
                    event=all(r['mde']['primary'] <= r['mde'][name] for r in records for name in names))
    structural = increment_gate(['old_common'])
    attribution = increment_gate(['raw_augmented', 'GP_generated', 'duplicated_common'])
    assert state_hash(model.loaded.model_) == expected
    assert all(digest(ROOT/name) == value for name, value in code.items())
    assert all(digest(Path(path)) == value for path, value in inputs.items())
    complete = dict(complete=True, status='evaluated', created=now(), seconds=time.monotonic()-started,
        records=records, floor_means=floor_means,
        equal_floor_means={name:float(np.mean([v[name] for v in floor_means.values()])) for name in records[0]['mde']},
        gate=bool(mean_gate and event_gate), floor_mean_gate=bool(mean_gate), event_gate=bool(event_gate),
        structural=structural, structural_increment_gate=all(structural.values()),
        attribution=attribution, attribution_gate=all(attribution.values()),
        fits=20, actual_fit_predict_seconds=sum(sum(r['context_seconds'].values()) for r in records),
        training_updates=0, GP_refits=0, model_state_unchanged=True, blind_test=False,
        peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20, sampler_check=sampler_check,
        freeze_sha256=digest(OUT/'freeze.json'))
    summary = json.dumps(complete, separators=(',', ':'), allow_nan=False)+'\n'
    assert len(summary.encode()) < 55000
    (OUT/'complete.json').write_text(summary)
    print(summary, end='', flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web', 'Server-only entry; do not run on the Mac'
    OUT.mkdir(exist_ok=False)
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
    try:
        run()
    except Exception as error:
        write(OUT/'complete.json', dict(complete=False, status='failed', created=now(),
            error=repr(error), automatic_resume=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
