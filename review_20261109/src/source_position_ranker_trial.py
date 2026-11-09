"""Learn GP-relative location decisions from real source acquisition tasks."""
import ast
from collections import Counter
import json
import os
from pathlib import Path
import signal
import socket
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')

import numpy as np
import torch
import xgboost as xgb

import acquisition_bank_transfer as gp
import common_update_transport_trial as assets
from acquisition_bank import prepare_bank_blocks
from conditional_budget_trial import history_reader
from episodic_tasks import task_inputs
from history_position_ranking import candidate_groups, PARAMS, ROUNDS
from observations import scan_log_prob
from predictive_scan_context import retained_fields
from prior_mean_transport_trial import bind
from recomposition import Parameters
from recomposition_trial import ROOT, OLD, PLAN, CSV, digest, write, now
from source_gp_boundary import complete_boundary
from source_position_ranker import (SEED, MAX_APS, descriptor, candidate_features,
    fit_shared, score_shared, make_features)

OUT = ROOT/'results/source_position_ranker_fixed_20261109'
FAILED = ROOT/'results/source_position_ranker_20261109'
PARENT = ROOT/'results/history_position_ranking_fixed_20261109'
FIELDS = ROOT/'results/forest_observation_20261109'
SOURCE = ROOT/'results/acquisition_transfer_structure_20261105'
SOURCE_GP = ROOT/'results/scalar_gp_reference_trial_20261105'
WALL_SECONDS = 600
BASELINES = ('GP', 'GP_mean', 'retained', 'retained_mean', 'full24')
ARMS = ('primary', 'no_offset', 'gp_features')


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
    row = dict(created=now(), stage=stage, **values)
    write(OUT/'progress.json', row)
    print(json.dumps(row), flush=True)


def deadline(signum, frame):
    raise TimeoutError('Fixed 600-second source-ranking budget exhausted; no retry')


def context(event, origin):
    ids = event['support_rows']+[i for rr in event['history_rows'].values() for i in rr]
    assert not set(ids) & set(event['query_rows'])
    rows = assets.read_selected(ids)
    assert all(rows[i]['floor'] == h for h, rr in event['history_rows'].items() for i in rr)
    assert all(rows[i]['floor'] == event['floor'] for i in event['support_rows'])
    histories = {h:assets.collect(rows, rr, origin) for h, rr in event['history_rows'].items()}
    support = assets.collect(rows, event['support_rows'], origin)
    assert len(histories) == 2 and len(support['positions']) == 1 and len(support['rss']) == 3
    return histories, support


@torch.no_grad()
def run():
    started = time.monotonic()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    code = code_inputs()
    for name in code:
        (OUT/name).write_bytes((ROOT/name).read_bytes())
    write(OUT/'preparation_freeze.json',dict(created=now(),code=code,prior_failed_run=str(FAILED),target_query_opened=False))
    _, _, _, inputs, package = gp.source_assets()
    bind(inputs,FAILED/'complete.json')
    bind(inputs,FAILED/'source_position_ranker.py','f269b1f5541e48e4c5e2ff70bbb684a046402e874be16e9693231880c65ea06e')
    bind(inputs,FAILED/'source_position_ranker_trial.py','a39e18d2833fa260f83fea5c0c1833be014e4386f5362fd1a56231b8516bbf1c')
    complete = json.loads((PARENT/'complete.json').read_text())
    assert complete['complete'] and complete['status'] == 'evaluated'
    bind(inputs, PARENT/'complete.json')
    bind(inputs, PARENT/'freeze.json', complete['freeze_sha256'])
    frozen = json.loads((PARENT/'freeze.json').read_text())
    assert frozen['package'] == package
    for path, value in frozen['inputs'].items():
        bind(inputs, Path(path), value)
    field_done = json.loads((FIELDS/'complete.json').read_text())
    bind(inputs, FIELDS/'freeze.json', field_done['freeze_sha256'])
    ff = json.loads((FIELDS/'freeze.json').read_text())
    assert ff['events'] == frozen['events'] and ff['GP_state'] == frozen['GP_state']
    assert xgb.__version__ == '3.1.3' and not xgb.build_info()['USE_CUDA']
    bind(inputs, Path(xgb.__file__))
    bind(inputs, Path(xgb.__file__).parent/'lib/libxgboost.so')
    plan = json.loads(PLAN.read_text())
    origin = np.asarray(plan['origin'])
    source_manifest = SOURCE/'source_tasks.json'
    parent = ROOT/'results/acquisition_retention_trial_20261105/freeze.json'
    pf = json.loads(parent.read_text())
    bind(inputs, parent)
    bind(inputs, source_manifest, pf['inputs'][str(source_manifest)])
    manifest = json.loads(source_manifest.read_text())
    sources = [f['events'][0] for f in manifest['families']]
    counts = Counter(e['floor'] for e in sources)
    assert counts == {'B0F0':4, 'B0F1':2, 'B0F2':4}
    parents = {e['id']:e for e in plan['events']}
    cached_freeze = json.loads((SOURCE_GP/'freeze.json').read_text())
    cached_result = json.loads((SOURCE_GP/'complete.json').read_text())
    assert cached_result['complete'] and cached_result['fit_gate']
    bind(inputs, SOURCE_GP/'freeze.json')
    bind(inputs, SOURCE_GP/'complete.json')
    source_data, source_states = [], {}
    model = Parameters(torch.zeros(9), torch.ones(9)).requires_grad_(False)
    # Source H/S are fixed independently of source Q labels.
    for event in sources:
        assert event['K'] == 1 and event['R'] == 3
        assert event['support_acquisition'][:2] != event['query_acquisition'][:2]
        assert event['history_rows'] == parents[event['parent_event']]['history_rows']
        histories, support = context(event, origin)
        meta = OLD/f"results/decision_local_pilot_20261030/{event['parent_event']}/meta.json"
        bind(inputs, meta)
        old_field = FAILED/f"{event['id']}_HS.npz"
        old_fit = FAILED/f"{event['id']}_GP_fit.json"
        cached_event = cached_freeze['event']
        same_cached = all(event[k] == cached_event[k] for k in ('floor', 'support_rows', 'history_rows'))
        reuse = old_field.exists()
        if reuse:
            bind(inputs, old_field)
            if old_fit.exists():
                bind(inputs, old_fit)
                fitted = json.loads(old_fit.read_text())
                assert fitted['passed']
                selected = fitted['selected']
            else:
                assert same_cached
                selected = cached_result['selected']
            with np.load(old_field) as saved:
                field = {k:saved[k].copy() for k in ('mean','variance','grid','aps')}
                desc = {k[len('descriptor_'):]:saved[k].copy() for k in saved.files if k.startswith('descriptor_')}
            assert np.array_equal(desc['support_position'],support['positions'][0])
        else:
            task = task_inputs(histories, support, plan['base'], plan['obs'],
                              history_reader(json.loads(meta.read_text()), plan['base'], origin, inputs))
            blocks = prepare_bank_blocks(task,
                [int(event['floor'].split('F')[1])-int(h.split('F')[1]) for h in task['names']])
            stats = {a:c['stats'] for a,c in task['cases'].items()}
            if old_fit.exists():
                bind(inputs, old_fit)
                fitted = json.loads(old_fit.read_text())
            else:
                fitted = gp.fit_gp(blocks,stats,plan['obs'],model,task['grid'])
            fitted = complete_boundary(fitted,blocks,stats,plan['obs'])
            write(OUT/f"{event['id']}_GP_fit.json",fitted)
            assert fitted['passed'],event['id']
            selected = fitted['selected']
            field = retained_fields(task,selected,event['floor'])
            desc = descriptor(histories,support,event['floor'],plan['base'],field)
            del task,blocks
        path = OUT/f"{event['id']}_HS.npz"
        np.savez(path, **field, **{'descriptor_'+k:v for k,v in desc.items()})
        source_states[event['id']] = dict(selected=selected, reused=reuse, field_sha256=digest(path))
        source_data.append(dict(event=event, field=field, desc=desc, support=support))
        progress('source_HS_prepared', event=event['id'], seconds=time.monotonic()-started)
        del histories
    target_data = []
    for event in frozen['events']:
        eid = event['id']
        old = next(r for r in complete['records'] if r['event'] == eid)
        prediction_path = PARENT/f'{eid}.npz'
        bind(inputs, prediction_path, old['prediction_sha256'])
        path = FIELDS/f'{eid}_HS_proxy.npz'
        bind(inputs, path, ff['HS_proxies'][eid])
        with np.load(path) as saved:
            field = {k:saved[k].copy() for k in ('mean', 'variance', 'grid', 'aps')}
        histories, support = context(event, origin)
        desc = descriptor(histories, support, event['floor'], plan['base'], field)
        target_data.append(dict(event=event, field=field, desc=desc, old=old, prediction_path=prediction_path))
    write(OUT/'HS_ready.json', dict(created=now(), source_states=source_states, target_query_opened=False))
    source_rows = assets.read_selected([i for e in sources for i in e['query_rows']])
    chunks, simple_chunks, offsets, group_weights, training_records = [], [], [], [], []
    canonical_check = None
    for index, p in enumerate(source_data):
        event, field, desc = p['event'], p['field'], p['desc']
        query = assets.collect(source_rows, event['query_rows'], origin)
        assert all(source_rows[i]['floor'] == event['floor'] for i in event['query_rows'])
        assert len(query['positions']) == 12 and np.all(query['stats']['n'] == 3)
        assert not set(map(tuple, query['positions'])) & set(map(tuple, p['support']['positions']))
        distance2 = np.square(query['xy'][:, None, :]-field['grid'][None]).sum(2)
        positive = distance2.argmin(1)
        candidates = candidate_groups(positive, len(field['grid']), SEED+index)
        rss = query['rss'][:, field['aps']]
        x, simple, offset = candidate_features(rss, candidates, field, plan['obs'], desc)
        if canonical_check is None:
            # Actual source data; one check catches AP identity leaking through ordering.
            perm = np.arange(len(field['aps']))[::-1]
            perm_field = dict(field, mean=field['mean'][:, perm], variance=field['variance'][:, perm], aps=field['aps'][perm])
            perm_desc = dict(desc, at_grid=desc['at_grid'][:, perm], at_support=desc['at_support'][perm])
            xx, ss, oo = candidate_features(rss[:1, perm], candidates[:1], perm_field, plan['obs'], perm_desc)
            assert np.array_equal(x[:5], xx) and np.array_equal(simple[:5], ss)
            assert np.allclose(offset[:5], oo, rtol=0, atol=1e-5)
            canonical_check = dict(passed=True, rows=5, kind='Real source AP reverse permutation; both representations invariant')
        chunks.append(x); simple_chunks.append(simple); offsets.append(offset)
        group_weights.extend([1/counts[event['floor']]]*len(rss))
        training_records.append(dict(event=event['id'], rows=event['query_rows'], candidates=candidates.tolist(),
            grid_snap_max=float(np.sqrt(distance2[np.arange(len(rss)), positive]).max()),
            outside=int(((query['xy']<field['grid'].min(0)) | (query['xy']>field['grid'].max(0))).any(1).sum())))
    del source_data, source_rows
    full, simple, offset = np.concatenate(chunks), np.concatenate(simple_chunks), np.concatenate(offsets)
    del chunks, simple_chunks, offsets
    weights = np.asarray(group_weights, dtype=np.float32)
    weights /= weights.mean()
    training_path = OUT/'source_training.npz'
    np.savez(training_path, full=full, simple=simple, offset=offset, group_weights=weights)
    code = code_inputs()
    freeze = dict(created=now(), code=code, inputs=inputs, package=package,
        source_events=sources, source_states=source_states, target_events=frozen['events'],
        prior_failed_run=str(FAILED), source_GP_fix='Only complete an inward-descent selected endpoint bracket; orders, tolerances, valid cached states unchanged.',
        training_records=training_records, training_sha256=digest(training_path),
        params=dict(PARAMS, seed=SEED), rounds=ROUNDS, MAX_APS=MAX_APS,
        representation='All AP tokens: H at S(16), local S(5), H at x(16), D/residual/GP detection(3); lexicographic sort, validity bit and 520 slots.',
        globals='Raw GP score, displacement from S divided by fixed kernel length, AP count, 73 H/S common descriptors.',
        objective='Real source Q nearest-grid positive vs four fixed uniform alternatives; complete scans, equal floors then acquisition families.',
        controls=['Same features without fixed GP offset', 'GP-only anonymous triples and geometry, own independent sort, same real source supervision'],
        readout='Original full-grid softmax; geometric median primary, mean secondary.',
        adoption='Both floors >=3% better than all five GP/full24 readouts; every target event no worse.',
        mechanism='Each floor strictly better and each event no worse than each matched control and prior local ranker, reported separately.',
        query_isolation='All H/S fields before source Q construction; all 3 rankers saved before any target Q access.',
        limitations=['Old full-H source tasks, not protected-catalog source validation; only B0 supervised, B1 already exposed development.',
            'Source Q may have contributed to prior legal source nuisance calibration or other source-task histories.',
            'Anonymous selected-field multiset is not complete H joint information or preservation of local AP-specific gains.',
            'Source slots ordered by floor; coordinates/APs aligned only within each building.',
            'Grid labels snapped without using true positions to enlarge H/S geometry.',
            'Learner and representation jointly changed versus section152; no pure loss ablation or novelty guarantee.',
            'Source S adaptation uses the existing fixed GP policy; no new historical GP fitting or base-model tuning.',
            'No hyperparameter search, automatic retry, target fitting or large download.'],
        canonical_check=canonical_check, resources=dict(cpu=2,gpu=0,seconds=WALL_SECONDS,ranker_fits=3,downloads=0))
    write(OUT/'freeze.json', freeze)
    for name in code:
        (OUT/name).write_bytes((ROOT/name).read_bytes())
    models, model_hashes, fit_seconds = {}, {}, {}
    for name in ARMS:
        before = time.monotonic()
        models[name] = fit_shared(simple if name == 'gp_features' else full, offset, weights, name != 'no_offset')
        path = OUT/f'{name}.ubj'
        models[name].save_model(path)
        model_hashes[name] = digest(path)
        fit_seconds[name] = time.monotonic()-before
        progress('shared_ranker_fitted', arm=name, seconds=time.monotonic()-started)
    write(OUT/'models.json', dict(created=now(), models=model_hashes, target_query_opened=False))
    del full, simple
    query_rows = assets.read_selected([i for e in frozen['events'] for i in e['query_rows']])
    records = []
    for p in target_data:
        event, field = p['event'], p['field']
        query = assets.collect(query_rows, event['query_rows'], origin)
        weights = 1/(len(query['positions'])*query['stats']['n'][query['group']])
        predictions = {}
        with np.load(p['prediction_path']) as saved:
            for key,value in (('query_rows',event['query_rows']),('truth',query['xy']),('weights',weights),('group',query['group'])):
                assert np.array_equal(saved[key], value)
            for name in BASELINES+('old_common','primary','primary_mean'):
                key = {'primary':'old_local_rank','primary_mean':'old_local_rank_mean'}.get(name,name)
                predictions[key] = saved[name].copy()
                assert abs(float(weights@np.linalg.norm(predictions[key]-query['xy'],axis=1))-p['old']['mde'][name]) < 1e-9
            wknn = saved['WKNN_errors'].copy()
        rss = query['rss'][:, field['aps']]
        score = torch.zeros((len(rss),len(field['grid'])))
        for column in range(len(field['aps'])):
            score += scan_log_prob(torch.tensor(rss[:,column]), torch.tensor(field['mean'][:,column]),
                                   torch.tensor(field['variance'][:,column]), plan['obs'])
        grid = torch.tensor(field['grid'])
        base = gp.actions(score.log_softmax(1),grid)
        replay = max(float(np.abs(base[key]-predictions[name]).max())
                     for key,name in (('mean','retained_mean'),('median','retained')))
        assert replay <= 1e-6, replay
        before = time.monotonic()
        residuals = score_shared(models,rss,score.numpy(),field,plan['obs'],p['desc'])
        for name,residual in residuals.items():
            result = gp.actions((torch.tensor(residual)+(score if name!='no_offset' else 0.)).log_softmax(1),grid)
            predictions[name],predictions[name+'_mean'] = result['median'],result['mean']
        elapsed = time.monotonic()-before
        errors = {name:np.linalg.norm(xy-query['xy'],axis=1) for name,xy in predictions.items()}
        errors['WKNN'] = wknn
        path = OUT/f"{event['id']}.npz"
        np.savez_compressed(path,**predictions,WKNN_errors=wknn,truth=query['xy'],weights=weights,
                            group=query['group'],query_rows=event['query_rows'])
        records.append(dict(event=event['id'],floor=event['floor'],mde={n:float(weights@e) for n,e in errors.items()},
            per_position_errors={n:[float(e[query['group']==j].mean()) for j in range(len(query['positions']))] for n,e in errors.items()},
            predict_seconds=elapsed,GP_replay_max_coordinate_error=replay,prediction_sha256=digest(path)))
        progress('event_evaluated',event=event['id'],seconds=time.monotonic()-started)
    floors = {f:{n:float(np.mean([r['mde'][n] for r in records if r['floor']==f])) for n in records[0]['mde']} for f in ('B1F1','B1F2')}
    def increment(name):
        return all(v['primary']<v[name] for v in floors.values()) and all(r['mde']['primary']<=r['mde'][name] for r in records)
    mean_gate = all(v['primary']<=.97*v[n] for v in floors.values() for n in BASELINES)
    event_gate = all(r['mde']['primary']<=r['mde'][n] for r in records for n in BASELINES)
    assert all(digest(ROOT/n)==h for n,h in code.items())
    assert all(digest(Path(p))==h for p,h in inputs.items())
    assert digest(training_path)==freeze['training_sha256']
    assert all(digest(OUT/f'{n}.ubj')==h for n,h in model_hashes.items())
    result = dict(complete=True,status='evaluated',created=now(),seconds=time.monotonic()-started,
        records=records,floor_means=floors,gate=bool(mean_gate and event_gate),floor_mean_gate=bool(mean_gate),event_gate=bool(event_gate),
        mechanism_gates={n:bool(increment(n)) for n in ARMS[1:]},old_local_rank_gate=bool(increment('old_local_rank')),
        structural_increment_gate=bool(increment('old_common')),fit_seconds=fit_seconds,shared_ranker_fits=3,
        source_tasks=len(sources),source_query_scans=sum(len(e['query_rows']) for e in sources),
        reused_source_S_states=sum(v['reused'] for v in source_states.values()),historical_GP_refits=0,target_ranker_fits=0,
        canonical_check=canonical_check,freeze_sha256=digest(OUT/'freeze.json'),models_sha256=digest(OUT/'models.json'))
    text = json.dumps(result,separators=(',',':'),allow_nan=False)+'\n'
    assert len(text.encode())<50000
    (OUT/'complete.json').write_text(text)
    print(text,end='',flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web','Server-only entry'
    OUT.mkdir(exist_ok=False)
    signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,WALL_SECONDS)
    try:
        run()
    except Exception as error:
        write(OUT/'complete.json',dict(complete=False,status='failed',created=now(),error=repr(error)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
