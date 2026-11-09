"""Four fixed K1R3 events: prior-mean versus cached common-mean transport.

Only H/S contexts are rebuilt. Existing GP fits and all comparators are reused;
query arrays are opened after all contexts and the independent mean check freeze.
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
from acquisition_bank import prepare_bank_blocks
from common_transport_pairing_trial import permute_pairing, SEEDS
from conditional_budget_trial import history_reader
from episodic_tasks import task_inputs
from observations import predict_from_posterior
from prior_mean_transport import build_transport
from recomposition import Parameters, prior
from recomposition_trial import ROOT, OLD, PLAN, CSV, digest, write, now
from scalar_support import ScalarSupportIntegral
from tabicl_history import FrozenHistoryRegressor, state_hash
from target_role_transport_trial import target_role

OUT = ROOT/'results/prior_mean_transport_20261108'
CACHES = (ROOT/'results/target_role_transport_20261108',
          ROOT/'results/target_role_floor_transfer_20261108')
BANK = ROOT/'results/acquisition_bank_transfer_20261108'
WALL_SECONDS = 600
MEAN_ATOL = 1e-8
REPLAY_ATOL = 1e-4


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


def bind(inputs, path, expected=None):
    value = digest(path)
    if expected is not None:
        assert value == expected, str(path)
    if str(path) in inputs:
        assert inputs[str(path)] == value, str(path)
    inputs[str(path)] = value
    return value


def load_cache(path, inputs, package):
    complete = json.loads((path/'complete.json').read_text())
    assert complete['complete'] and complete['model_state_unchanged']
    if 'status' in complete:
        assert complete['status'] == 'evaluated'
    bind(inputs, path/'complete.json')
    bind(inputs, path/'freeze.json', complete['freeze_sha256'])
    frozen = json.loads((path/'freeze.json').read_text())
    assert frozen['package'] == package
    for name,value in frozen['code'].items():
        assert digest(ROOT/name) == value, name
    for name,value in frozen['inputs'].items():
        bind(inputs, Path(name), value)
    assert frozen['permutation_seeds'] == list(SEEDS)
    assert frozen['tab_preprocessing_seed'] == assets.SEED
    return frozen, complete


def check_mean(histories, transported, task, selected, floor):
    """Independent existing GP readout at unique H positions, before Q access.

Use bank block source means, not any means/kappa from the new transport module.
The test covers actual detected cells only; it makes no variance assertion.
"""
    positions, inverse = np.unique(np.concatenate(
        [h['positions'] for h in histories.values()]), axis=0, return_inverse=True)
    x = task['grid'].new_tensor(positions)
    reference_task = dict(task, grid=x)
    blocks = prepare_bank_blocks(reference_task,
        [int(floor.split('F')[1])-int(f.split('F')[1]) for f in task['names']])
    model = Parameters(torch.zeros(9), torch.ones(9)).requires_grad_(False)
    weight = float(selected['weight'])
    weights = x.new_tensor([weight,1-weight])
    sigma2 = float(selected['sigma'])**2
    indices, offset = {}, 0
    for name,h in histories.items():
        size = len(h['positions'])
        indices[name] = inverse[offset:offset+size][h['group']]
        offset += size
    maximum, cells, active, skipped = 0., 0, 0, []
    for ap,block in blocks.items():
        if not block['active']:
            skipped.append(int(ap))
            continue
        active += 1
        field = prior(model, block, weights)
        sk = field['sk']+sigma2
        post = ScalarSupportIntegral(field['sm'], sk, task['cases'][ap]['stats'], task['obs'], 256)
        mass = post.log_weights.exp()
        mu = mass@post.samples
        center = post.samples-mu
        posterior = dict(mean=mu, cov=(center.T*mass)@center)
        reference = predict_from_posterior(field['gm'], field['gv']+sigma2,
            field['cross']-block['rg'], field['sm'], sk, posterior)['mean'].numpy()
        for j,(name,h) in enumerate(histories.items()):
            detected = h['rss'][:,ap] != 100
            rows = indices[name][detected]
            expected = h['rss'][detected,ap]-block['gm'][j].numpy()[rows]+reference[rows]
            error = np.abs(transported[name]['rss'][detected,ap]-expected)
            if error.size:
                maximum = max(maximum, float(error.max()))
                cells += error.size
    assert maximum <= MEAN_ATOL, maximum
    return dict(passed=True, maximum_detected_cell_error=maximum, atol=MEAN_ATOL,
        detected_cells=int(cells), active_APs=active, skipped_all_absent_APs=skipped,
        unique_history_positions=len(positions), quadrature_order=256,
        reference='raw RSS - bank block gm[source] + retained predict_from_posterior.mean',
        covariance_used_only_in_reference_GP_readout=True, variance_verified=False)


def progress(stage, **values):
    row = dict(created=now(), stage=stage, **values)
    write(OUT/'progress.json', row)
    print(json.dumps(row), flush=True)


def deadline(signum, frame):
    raise TimeoutError(f'Fixed {WALL_SECONDS}-second wall budget exhausted; no retry')


@torch.no_grad()
def run():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    torch.cuda.set_per_process_memory_fraction(.5, 0)
    _, source, checkpoint, inputs, package = gp.source_assets()
    bank_complete = json.loads((BANK/'complete.json').read_text())
    assert bank_complete['complete'] and bank_complete['status'] == 'evaluated'
    bind(inputs, BANK/'complete.json')
    bind(inputs, BANK/'freeze.json', bank_complete['freeze_sha256'])
    bank_freeze = json.loads((BANK/'freeze.json').read_text())
    cached = [load_cache(path, inputs, package) for path in CACHES]
    plan = json.loads(PLAN.read_text())
    origin = np.asarray(plan['origin'])
    meta_path = OLD/'results/decision_local_pilot_20261030/validation_0/meta.json'
    bind(inputs, meta_path)
    meta = json.loads(meta_path.read_text())
    prepared, events = [], []
    for floor_index,(cache,(frozen,complete)) in enumerate(zip(CACHES,cached)):
        floor = f'B1F{floor_index+1}'
        assert [e['id'] for e in frozen['events']] == [f'{floor}_acquisition_{j}' for j in (0,1)]
        for event in frozen['events']:
            eid = event['id']
            assert event['floor'] == floor and event['K'] == 1 and event['R'] == 3
            assert event['histories'] == ['B1F0','B1F3']
            history_parent = next(e for e in plan['events'] if e['id'] ==
                event.get('history_parent_event', event.get('parent_event')))
            assert event['history_rows'] == history_parent['history_rows']
            assert event['histories'] == history_parent['histories']
            selected = frozen['GP_state'][eid]
            fit_record = (bank_complete if floor_index == 0 else complete)['numerical'][eid]
            assert fit_record['GP_fit']['passed'] and selected == fit_record['GP_fit']['selected']
            assert fit_record['GP']['passed'] and fit_record['retained']['passed']
            old_record = next(r for r in complete['records'] if r['event'] == eid)
            prediction_path = cache/f'{eid}.npz'
            bind(inputs, prediction_path, old_record['prediction_sha256'])
            if floor_index == 0:
                original = next(e for e in bank_freeze['events'] if e['id'] == eid)
                assert event == original
                geometry = next(g for g in bank_freeze['geometry'] if g['event'] == eid)
                original_record = next(r for r in bank_complete['records'] if r['event'] == eid)
                gp_path = BANK/f'{eid}_GP_256.npz'
                # Legacy complete did not hash this individual GP NPZ. Bind it
                # now and check rows + original accepted MDE after Q is opened.
                bind(inputs, gp_path)
                retained_path = BANK/f'{eid}_retained_256.npz'
                bind(inputs, retained_path)
            else:
                geometry = next(g for g in frozen['geometry'] if g['event'] == eid)
                original_record, gp_path, retained_path = None, None, None
                for path,value in old_record['artifact_hashes'].items():
                    bind(inputs, Path(path), value)
            ids = event['support_rows']+[i for rr in event['history_rows'].values() for i in rr]
            assert not set(ids).intersection(event['query_rows'])
            rows = assets.read_selected(ids)
            assert all(rows[i]['floor'] == name for name,rr in event['history_rows'].items() for i in rr)
            assert all(rows[i]['floor'] == floor for i in event['support_rows'])
            histories = {name:assets.collect(rows, rr, origin) for name,rr in event['history_rows'].items()}
            support = assets.collect(rows, event['support_rows'], origin)
            assert len(support['positions']) == 1 and len(support['rss']) == 3
            task = task_inputs(histories, support, plan['base'], plan['obs'],
                history_reader(meta, plan['base'], origin, inputs))
            assert task['names'] == event['histories']
            aps = np.array(list(task['cases']))
            bounds = np.stack([task['grid'].min(0).values.numpy(), task['grid'].max(0).values.numpy()])
            assert np.array_equal(aps, geometry['APs']) and np.array_equal(bounds, geometry['bounds'])
            assert len(task['grid']) == geometry['grid_points']
            transport = build_transport(histories, support, task, selected)
            validation = check_mean(histories, transport['mean_histories'], task, selected, floor)
            contexts = [('full24_replay',histories)]
            paired, role = target_role(transport['mean_histories'], support, floor, aps)
            contexts.append(('prior_paired',paired))
            invariant = dict(prior_paired=role)
            for seed in SEEDS:
                shuffled, permutation = permute_pairing(transport['mean_histories'], aps, seed)
                mapped, role = target_role(shuffled, support, floor, aps)
                label = f'prior_seed_{seed}'
                contexts.append((label,mapped))
                invariant[label] = dict(role, permutation=permutation)
            assert len(contexts) == 6
            diagnostic_path = OUT/f'{eid}_HS_diagnostics.json'
            write(diagnostic_path, dict(transport=transport['diagnostics'], independent_mean_check=validation,
                                       context_invariants=invariant))
            transport_summary = {key:transport['diagnostics'][key] for key in (
                'common_update','applied_displacement','common_minus_prior_field',
                'common_minus_prior_applied','mean_range')}
            prepared.append(dict(event=event, cache=cache, old_record=old_record,
                prediction_path=prediction_path, gp_path=gp_path, retained_path=retained_path, original_record=original_record,
                histories=histories, support=support, aps=aps, bounds=bounds, contexts=contexts,
                selected=selected, validation=validation, transport_summary=transport_summary,
                diagnostic_sha256=digest(diagnostic_path), source_row_spans=role['source_row_spans']))
            events.append(event)
            del task, transport, rows
            progress('HS_contexts_and_independent_mean_check_complete', event=eid,
                     seconds=time.monotonic()-started)
    code = code_inputs()
    freeze = dict(created=now(), events=events, inputs=inputs, code=code, package=package,
        cache_paths=[str(p) for p in CACHES], origin=origin.tolist(), base=plan['base'], obs=plan['obs'],
        history_manifest=str(meta_path), GP_state={p['event']['id']:p['selected'] for p in prepared},
        geometry=[dict(event=p['event']['id'], APs=p['aps'].tolist(), bounds=p['bounds'].tolist(),
                       source_row_spans=p['source_row_spans']) for p in prepared],
        H_S_checks={p['event']['id']:p['validation'] for p in prepared},
        H_S_diagnostic_hashes={p['event']['id']:p['diagnostic_sha256'] for p in prepared},
        checkpoint=str(checkpoint), trained_state_sha256=source['trained_state_sha256'],
        permutation_seeds=list(SEEDS), tab_preprocessing_seed=assets.SEED,
        prior_formula='M0-m_j0+kappa; fixed H-only empirical residual', random_worlds_generated=0,
        primary='Mean projected coordinates of four fixed prior target-role shuffle contexts',
        paired_control='Unshuffled prior target-role, cost control only; never selected as primary',
        replay=dict(events='Each of all four original full24 contexts', coordinate_max_atol=REPLAY_ATOL),
        mean_validation=dict(atol=MEAN_ATOL, scope='Detected cells against independent retained GP mean; no variance test'),
        adoption='Each floor mean >=3% better than GP/GP_mean/retained/retained_mean/full24, and all four events no worse than all five fixed complete readouts.',
        attribution='Separate: each floor mean strictly better than matched target H0, and all four events no worse.',
        structural_increment='Separate: each floor mean strictly better than old common primary, and all four events no worse.',
        weighting='Positions equal within event, two events equal within floor, floors equal; seeds not independent experiments.',
        query_isolation='All four H/S contexts, independent mean checks and this freeze precede Q/NPZ array access.',
        legacy_GP_provenance='B1F1 GP and retained NPZ newly bound or verified against an inherited input hash. Old complete did not itself sign these NPZs; rows and original mean/median MDE checked after freeze.',
        source_scope='Existing full24 task fine-tuning on B0; no claim about public pretraining overlap.',
        limitations=['Four exposed K1 development events; no new data, blind test, K curve or model selection.',
            'Fixed empirical residual transfer is assumed; no recovery of real target scans or coverage.',
            'No new GP fit, source learning, target training, or 8-world generation.'],
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','scikit-learn')},
        resources=dict(cpu=2, gpu=1, max_vram_fraction=.5, hard_worker_seconds=WALL_SECONDS,
                       fits_per_event=6, downloads=0, new_training_updates=0, retries=0))
    assert all(digest(ROOT/name) == value for name,value in code.items())
    write(OUT/'freeze.json', freeze)
    for name in code:
        (OUT/name).write_bytes((ROOT/name).read_bytes())
    gc.collect()
    # First access to current query RSS or any cached prediction arrays.
    query_rows = assets.read_selected([i for e in events for i in e['query_rows']])
    for p in prepared:
        event, old = p['event'], p['old_record']
        query = assets.collect(query_rows, event['query_rows'], origin)
        assert all(query_rows[i]['floor'] == event['floor'] for i in event['query_rows'])
        assert len(query['positions']) == 12 and np.all(query['stats']['n'] == 3)
        assert not set(map(tuple, query['positions'])) & set(map(tuple, p['support']['positions']))
        weights = 1/(len(query['positions'])*query['stats']['n'][query['group']])
        mapping = dict(old_common='primary', target_H0='target_H0_ensemble', old_common_paired='target_paired_H1')
        for seed in SEEDS:
            mapping[f'old_seed_{seed}'] = f'target_H1_seed_{seed}'
        if event['floor'] == 'B1F1':
            mapping.update(full24='source_full24', retained='source_retained')
        else:
            mapping.update(full24='full24', retained='retained', GP='GP',
                           GP_mean='GP_mean', retained_mean='retained_mean')
        predictions = {}
        with np.load(p['prediction_path']) as saved:
            for key,value in (('query_rows',event['query_rows']), ('truth',query['xy']),
                              ('weights',weights), ('group',query['group'])):
                assert np.array_equal(saved[key], value)
            for name,key in mapping.items():
                predictions[name] = saved[key].copy()
                mde = float(weights@np.linalg.norm(predictions[name]-query['xy'], axis=1))
                assert abs(mde-old['mde'][key]) < 1e-9, (event['id'],name)
            if event['floor'] == 'B1F2':
                wk_errors = saved['WKNN_errors'].copy()
        if event['floor'] == 'B1F1':
            with np.load(p['gp_path']) as saved:
                assert np.array_equal(saved['query_rows'], event['query_rows'])
                predictions['GP'] = saved['median'].copy()
                predictions['GP_mean'] = saved['mean'].copy()
            with np.load(p['retained_path']) as saved:
                assert np.array_equal(saved['query_rows'], event['query_rows'])
                predictions['retained_mean'] = saved['mean'].copy()
            for name in ('GP','retained'):
                mean_mde = float(weights@np.linalg.norm(predictions[name+'_mean']-query['xy'], axis=1))
                assert abs(mean_mde-p['original_record']['metrics'][name]['mean']) < 1e-9
            mde = float(weights@np.linalg.norm(predictions['GP']-query['xy'], axis=1))
            assert abs(mde-p['original_record']['metrics']['GP']['median']) < 1e-9
            wk_errors = np.asarray(p['original_record']['metrics']['WKNN']['errors'])
        assert abs(float(weights@wk_errors)-old['mde']['WKNN']) < 1e-9
        p.update(query=query, weights=weights, predictions=predictions, WKNN_errors=wk_errors)
    torch.set_default_dtype(torch.float32)
    model = FrozenHistoryRegressor(checkpoint, device='cuda:0')
    model.loaded.model_.requires_grad_(False).eval()
    expected = source['trained_state_sha256']
    assert state_hash(model.loaded.model_) == expected
    records, actual_fit_seconds = [], 0.
    for p in prepared:
        event, query, weights = p['event'], p['query'], p['weights']
        eid, predictions = event['id'], p['predictions']
        context_seconds = {}
        for label,h in p['contexts']:
            before = time.monotonic()
            model.fit(h, p['support'], event['floor'], p['aps'], p['bounds'], assets.SEED)
            count = sum(len(hh['rss']) for hh in p['histories'].values())+3
            assert all(m.n_samples_in_ == count and m.model_ is model.loaded.model_ for m in model.estimators)
            predictions[label] = model.predict(query['rss'])['projected']
            context_seconds[label] = time.monotonic()-before
            actual_fit_seconds += context_seconds[label]
            if label == 'full24_replay':
                replay = float(np.abs(predictions[label]-predictions['full24']).max())
                assert replay <= REPLAY_ATOL, (eid,replay)
            progress('predicted', event=eid, context=label, seconds=time.monotonic()-started)
        predictions['primary'] = np.mean([predictions[f'prior_seed_{s}'] for s in SEEDS], axis=0)
        errors = {name:np.linalg.norm(xy-query['xy'], axis=1) for name,xy in predictions.items()}
        errors['WKNN'] = p['WKNN_errors']
        mde = {name:float(weights@error) for name,error in errors.items()}
        new_values = [mde[f'prior_seed_{s}'] for s in SEEDS]
        old_values = [mde[f'old_seed_{s}'] for s in SEEDS]
        differences = [a-b for a,b in zip(new_values,old_values)]
        contrasts = dict(primary_minus_old_common=mde['primary']-mde['old_common'],
            primary_minus_target_H0=mde['primary']-mde['target_H0'],
            paired_minus_old_common_paired=mde['prior_paired']-mde['old_common_paired'],
            per_seed_prior_minus_old=differences,
            prior_single_mean=float(np.mean(new_values)), prior_single_range=[min(new_values),max(new_values)],
            old_single_mean=float(np.mean(old_values)), old_single_range=[min(old_values),max(old_values)],
            single_mean_prior_minus_old=float(np.mean(differences)), improved_seeds=sum(d < 0 for d in differences),
            shuffle_single_mean_minus_prior_paired=float(np.mean(new_values))-mde['prior_paired'],
            shuffle_ensemble_minus_prior_paired=mde['primary']-mde['prior_paired'])
        path = OUT/f'{eid}.npz'
        np.savez_compressed(path, **predictions, WKNN_errors=errors['WKNN'], truth=query['xy'],
                            weights=weights, group=query['group'], query_rows=event['query_rows'])
        # Full coordinate/error arrays stay server-side; compact per-position
        # summaries include every fixed single permutation, without seed selection.
        position_names = ['primary','prior_paired','GP','GP_mean','retained','retained_mean','full24','target_H0','old_common','WKNN']
        position_names += [f'{family}_seed_{s}' for family in ('prior','old') for s in SEEDS]
        records.append(dict(event=eid, floor=event['floor'], mde=mde, contrasts=contrasts,
            per_position_errors={name:[float(errors[name][query['group']==j].mean()) for j in range(len(query['positions']))]
                                 for name in position_names},
            replay_max_coordinate_error=replay, context_seconds=context_seconds,
            primary_seconds=sum(context_seconds[f'prior_seed_{s}'] for s in SEEDS),
            transport=p['transport_summary'], mean_check=p['validation'], prediction_sha256=digest(path)))
    floor_means = {floor:{name:float(np.mean([r['mde'][name] for r in records if r['floor'] == floor]))
                         for name in records[0]['mde']} for floor in ('B1F1','B1F2')}
    means = {name:float(np.mean([v[name] for v in floor_means.values()])) for name in records[0]['mde']}
    per_floor = {}
    for floor,values in floor_means.items():
        rr = [r for r in records if r['floor'] == floor]
        per_floor[floor] = dict(relative_gains={n:1-values['primary']/values[n]
            for n in ('GP','GP_mean','retained','retained_mean','full24','WKNN','target_H0','old_common')},
            primary_minus_old_common=values['primary']-values['old_common'],
            per_seed_prior_minus_old=[float(np.mean([r['contrasts']['per_seed_prior_minus_old'][i] for r in rr]))
                                      for i in range(len(SEEDS))],
            single_mean_prior_minus_old=float(np.mean([r['contrasts']['single_mean_prior_minus_old'] for r in rr])))
    mean_gate = all(v['primary'] <= .97*v[n] for v in floor_means.values() for n in ('GP','GP_mean','retained','retained_mean','full24'))
    direction_gate = all(r['mde']['primary'] <= r['mde'][n] for r in records for n in ('GP','GP_mean','retained','retained_mean','full24'))
    attribution_mean = all(v['primary'] < v['target_H0'] for v in floor_means.values())
    attribution_direction = all(r['mde']['primary'] <= r['mde']['target_H0'] for r in records)
    structural_mean = all(v['primary'] < v['old_common'] for v in floor_means.values())
    structural_direction = all(r['mde']['primary'] <= r['mde']['old_common'] for r in records)
    assert state_hash(model.loaded.model_) == expected
    assert all(digest(ROOT/name) == value for name,value in code.items())
    assert all(digest(Path(path)) == value for path,value in inputs.items())
    complete = dict(complete=True, status='evaluated', created=now(), seconds=time.monotonic()-started,
        records=records, floor_means=floor_means, equal_floor_means=means, per_floor=per_floor,
        gate=bool(mean_gate and direction_gate), floor_mean_gate=bool(mean_gate), event_gate=bool(direction_gate),
        attribution_gate=bool(attribution_mean and attribution_direction),
        attribution_floor_mean_gate=bool(attribution_mean), attribution_event_gate=bool(attribution_direction),
        structural_increment_gate=bool(structural_mean and structural_direction),
        structural_floor_mean_gate=bool(structural_mean), structural_event_gate=bool(structural_direction),
        actual_fit_predict_seconds=actual_fit_seconds, fits=24, model_state_unchanged=True,
        GP_refits=0, training_updates=0, random_worlds_generated=0, blind_test=False,
        peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20, freeze_sha256=digest(OUT/'freeze.json'))
    summary = json.dumps(complete, separators=(',', ':'), allow_nan=False)+'\n'
    assert len(summary.encode()) <= 50000
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
            error=repr(error), scoring=False, automatic_resume=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
