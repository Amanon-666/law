"""Score the frozen source-selected A/B/C/D models on the original B1 tasks."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import time

REQUESTED_CUDA_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from law_bench import run
from law_bench.data import load
from law_bench.protocol import make_tasks
from law_bench.score import score
from frozen_hurdle_translator import FrozenHurdleTranslator
from hurdle_gp_pool import fuse_hurdle_gp

from direct_coordinate import DirectCoordinateRegressor
from field_predictor import NativeObservationField

os.environ['CUDA_VISIBLE_DEVICES'] = REQUESTED_CUDA_DEVICES
BASE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110')
ORIGINAL = BASE/'results/paper_uji_B1_20261111'
GP_REFERENCE = BASE/'results/paper_uji_B1_ext_20261111'


def matched(record, task, truth):
    assert record['status'] == 'completed' and record['task_id'] == task.task_id
    assert record['query_row_indices'] == truth['row_indices'].tolist()
    assert record['query_groups'] == truth['groups'].tolist()
    assert record['meta'] == task.meta


def read_gp(work, task, truth):
    path = work/'gp_surfaces'/(task.task_id+'.npz')
    metadata = json.loads(path.with_suffix('.json').read_text())
    binding = metadata['binding']
    reference = GP_REFERENCE/'events/refi_GP'/(task.task_id+'.json')
    assert binding['reference_event_sha256'] == run._sha(reference)
    assert binding['task_id'] == task.task_id and binding['meta'] == task.meta
    assert binding['query_rows'] == truth['row_indices'].tolist()
    assert binding['query_groups'] == truth['groups'].tolist()
    assert metadata['new_history_fields'] == 0 and run._sha(path) == metadata['sha256']
    with np.load(path) as arrays:
        gp = {name: arrays[name] for name in arrays.files}
    return gp, dict(path=str(path), sha256=metadata['sha256'],
        metadata_sha256=run._sha(path.with_suffix('.json')))


def event(output, method, seed, task, truth, prediction, provenance, seconds):
    path = output/'events'/method/('reference' if seed is None else f'seed_{seed}')/(task.task_id+'.json')
    if path.exists():
        record = json.loads(path.read_text())
        matched(record, task, truth)
        assert record['provenance'] == provenance and record['training_seed'] == seed
    else:
        # Ground-truth coordinates are used only after the deployment prediction.
        record = dict(task_id=task.task_id, method=method, training_seed=seed,
            dataset=task.dataset, building=task.building, floor=task.floor,
            repeat=task.repeat, K=task.K, meta=task.meta, status='completed',
            prediction=np.asarray(prediction).tolist(), scores=score(prediction, truth),
            query_row_indices=truth['row_indices'].tolist(), query_groups=truth['groups'].tolist(),
            provenance=provenance, seconds=seconds)
        run._write(path, record)
    return record


def completed_pair(output, arm, seed, task, truth, provenance):
    records = []
    for suffix in ('field', 'complete'):
        path = output/'events'/f'{arm}_{suffix}'/f'seed_{seed}'/(task.task_id+'.json')
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        matched(record, task, truth)
        assert record['provenance'] == provenance and record['training_seed'] == seed
        records.append(record)
    return records


def statistics(records):
    errors = np.array([e for r in records for e in r['scores']['position_errors']])
    return dict(tasks=len(records), position_budget_seed_observations=len(errors),
        mean=float(errors.mean()), p90=float(np.quantile(errors, .9)))


def summarize(records, output, frozen):
    methods = sorted({r['method'] for r in records})
    summary, rows = {}, []
    for method in methods:
        selected = [r for r in records if r['method'] == method]
        seeds = sorted({r['training_seed'] for r in selected}, key=lambda s: -1 if s is None else s)
        by_seed = {}
        for seed in seeds:
            subset = [r for r in selected if r['training_seed'] == seed]
            by_seed[str(seed)] = dict(overall=statistics(subset), by_K={})
            for K in (None, 1, 3, 5):
                group = subset if K is None else [r for r in subset if r['K'] == K]
                values = statistics(group)
                if K is not None:
                    by_seed[str(seed)]['by_K'][str(K)] = values
                rows.append(dict(method=method, training_seed=seed, K='all' if K is None else K, **values))
        summary[method] = dict(by_training_seed=by_seed, pooled_over_training_seeds=statistics(selected),
            pooled_by_K={str(K):statistics([r for r in selected if r['K']==K]) for K in (1,3,5)})
        if seeds != [None]:
            means = [by_seed[str(seed)]['overall']['mean'] for seed in seeds]
            summary[method]['training_seed_mean_error'] = dict(values=means,
                mean=float(np.mean(means)), sample_std=float(np.std(means,ddof=1)))
    csv_path = output/'comparison.csv'
    with csv_path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run._write(output/'comparison.json',dict(complete=True,freeze_sha256=run._sha(output/'freeze.json'),
        methods=summary,
        metric='Pool original per-position mean scan errors. Repeated K budgets, split repeats and training seeds retain their identities and are not independent locations. P90 is a pooled quantile, never an average of task P90.',
        interpretation=frozen['comparison_scope'],
        timing_scope=dict(event_seconds='Recorded new field preparation/readout or direct deployment only. Shared GP surfaces are read from the checksum-bound cache and their original adaptation is not repeated. Reference events retain original task seconds.',
            source_training='See each source stage training.json/complete.json and formal driver stage exit records. Stage wall time includes source selection, OOF labels and serialization.',
            shared_GP='gp_surfaces metadata records score-surface recomputation with original saved adaptation state and full-H fields. It is not an online adaptation timing.'),
        records=[dict(method=r['method'],training_seed=r['training_seed'],task_id=r['task_id'],
            path=str((output/'events'/r['method']/('reference' if r['training_seed'] is None else f"seed_{r['training_seed']}")/(r['task_id']+'.json')).relative_to(output)),
            sha256=run._sha(output/'events'/r['method']/('reference' if r['training_seed'] is None else f"seed_{r['training_seed']}")/(r['task_id']+'.json'))) for r in records]))


def main(args):
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    work, output = args.work, args.work/'target_B1'
    output.mkdir(parents=True, exist_ok=True)
    selection_path, assets_path = work/'selected_epochs.json', work/'model_manifest.json'
    selection = json.loads(selection_path.read_text())
    assets = json.loads(assets_path.read_text())
    assert selection['target_evaluation_used'] is False and assets['source_selection_sha256'] == run._sha(selection_path)
    assert json.loads((work/'gp_surfaces/complete.json').read_text())['complete']
    for arm in ('B','C','D'):
        assert sorted(assets[arm]) == ['1','2','3']
        for record in assets[arm].values():
            assert run._sha(record['path']) == record['sha256']
    context_matches = {}
    for seed in (1,2,3):
        trained = json.loads(Path(assets['B'][str(seed)]['path']).read_text())
        frozen_public = json.loads(Path(assets['D'][str(seed)]['path']).read_text())
        context_matches[str(seed)] = {}
        for role in ('det','mean','scale'):
            with np.load(trained['contexts'][role]['path']) as b, np.load(frozen_public['contexts'][role]['path']) as d:
                assert np.array_equal(b['source_row_indices'],d['source_row_indices'])
                assert np.array_equal(b['source_units'],d['source_units'])
                context_matches[str(seed)][role] = dict(identical_source_units_and_aliases=True,
                    units=len(b['source_units']),count_rows=len(b['y']))
    buildings, axes = load('uji')
    original = json.loads((ORIGINAL/'freeze.json').read_text())
    assert run._data_identity({b:buildings[b] for b in ('B0','B2')},{b:axes[b] for b in ('B0','B2')}) == original['source_data']
    assert run._data_identity({'B1':buildings['B1']},{'B1':axes['B1']}) == original['target_data']
    frozen = dict(source_selection_sha256=run._sha(selection_path), model_manifest_sha256=run._sha(assets_path),
        gp_cache_complete_sha256=run._sha(work/'gp_surfaces/complete.json'),
        original_task_freeze_sha256=run._sha(ORIGINAL/'freeze.json'),
        B_D_source_context_matches=context_matches,
        protocol=original['configuration'], source_data=original['source_data'], target_data=original['target_data'],
        code_sha256={name:run._sha(Path(__file__).parent/name) for name in ('evaluate_final.py','field_predictor.py','native_tab.py','direct_coordinate.py')},
        comparison_scope=dict(A_B='Matched original 55/56 features, scan-count objectives, source rows per seed, candidate construction and fixed reader/GP. Different estimator families and pretraining.',
            B_C='Complete field-learning pipeline versus ordinary episodic direct-coordinate fine-tuning with the same public regressor family, legal source buildings and H/S. Representation, supervision and inference all differ. C has one shared backbone, B has three task-specific native backbones.',
            B_D='Train all native field weights versus frozen public weights. Same source context unit/alias selections per seed; each version uses its own legal opposite-half Tab means for scale labels.',
            reference='Original five-HGB-seed complete and field predictions and native GP remain separate internal references.',
            exposed_development='B1 has prior project results. This is frozen target evaluation after B0-to-B2 source selection, not a never-exposed external benchmark.'))
    freeze_path = output/'freeze.json'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text()) == frozen
    else:
        run._write(freeze_path,frozen)
    episodes = []
    config = original['configuration']
    for floor in sorted(buildings['B1']):
        for repeat in (0,1,2):
            episodes.extend(make_tasks('uji','B1',buildings['B1'],axes['B1'],floor,
                budgets=(1,3,5),repeat=repeat,seed=config['seed'],R=config['R'],query_positions=config['query_positions']))
    assert len(episodes) == 36
    assert {t.task_id for t,_ in episodes} == {p.stem for p in (GP_REFERENCE/'events/refi_GP').glob('*.json')}
    original_model_path = ORIGINAL/'source_models/refi.joblib'
    source_info = json.loads(original_model_path.with_suffix('.json').read_text())
    assert source_info['source_buildings'] == ['B0','B2'] and run._sha(original_model_path) == source_info['sha256']
    original_model = joblib.load(original_model_path)
    assert original_model.sources == ['B0','B2'] and original_model.provenance['data'] == original['source_data']
    assert original_model.provenance['seeds'] == [1,2,3,4,5] and len(original_model.translator.models) == 5
    records = []
    for task, truth in episodes:
        gp, gp_binding = read_gp(work, task, truth)
        records.append(event(output,'GP_reference',None,task,truth,gp['projected'],gp_binding,None))
        for method, directory, name in (
            ('adopted_5seed_field', ORIGINAL, 'refi'),
            ('adopted_5seed_complete', GP_REFERENCE, 'refi_GP')):
            path = directory/'events'/name/(task.task_id+'.json')
            old = json.loads(path.read_text())
            matched(old,task,truth)
            records.append(event(output,method,None,task,truth,old['prediction'],
                dict(original_event=str(path),sha256=run._sha(path)),old['seconds']))
    for seed in (1,2,3):
        for arm in ('A','B','D'):
            torch.manual_seed(seed)
            if arm == 'D' and assets['D'][str(seed)]['path'] == assets['B'][str(seed)]['path']:
                for task,truth in episodes:
                    for suffix in ('field','complete'):
                        path = output/'events'/f'B_{suffix}'/f'seed_{seed}'/(task.task_id+'.json')
                        previous = json.loads(path.read_text())
                        matched(previous,task,truth)
                        provenance = dict(previous['provenance'],source_model=assets['D'][str(seed)],
                            prediction_reused_from=dict(path=str(path),sha256=run._sha(path)))
                        records.append(event(output,f'D_{suffix}',seed,task,truth,previous['prediction'],provenance,None))
                continue
            if arm == 'A':
                model = FrozenHurdleTranslator.__new__(FrozenHurdleTranslator)
                model.models = [original_model.translator.models[seed-1]]
                model_binding = dict(source_joblib=str(original_model_path),sha256=source_info['sha256'],member_seed=seed)
            else:
                asset = assets[arm][str(seed)]
                model = NativeObservationField.from_assets(asset['path'])
                model_binding = asset
            for task, truth in episodes:
                gp, gp_binding = read_gp(work,task,truth)
                provenance = dict(source_model=model_binding,shared_GP=gp_binding,
                    source_selection_sha256=run._sha(selection_path),pool='Original fixed probability_half')
                existing = completed_pair(output,arm,seed,task,truth,provenance)
                if existing is not None:
                    records.extend(existing)
                    continue
                tick = time.monotonic()
                prepared = (model.prepare({f'B1F{f}':h for f,h in sorted(task.H.items())},task.S,f'B1F{task.floor}')
                    if arm == 'A' else model.prepare(task))
                field = prepared.predict(task.Q_rss)
                field_seconds = time.monotonic()-tick
                assert np.array_equal(field['aps'],gp['aps']) and np.array_equal(prepared.bounds,gp['bounds'])
                tick = time.monotonic()
                complete = fuse_hurdle_gp(field['cand_xy'],field['log_position_mass'],
                    gp['grid'],gp['scores'],gp['raw'],prepared.bounds)['predictions']['probability_half']
                complete_seconds = field_seconds+time.monotonic()-tick
                records.append(event(output,f'{arm}_field',seed,task,truth,field['projected'],provenance,field_seconds))
                records.append(event(output,f'{arm}_complete',seed,task,truth,complete,provenance,complete_seconds))
                print(json.dumps(dict(stage='target_scored',arm=arm,seed=seed,task_id=task.task_id)),flush=True)
            if arm != 'A':
                model.close()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        asset = assets['C'][str(seed)]
        direct_asset = json.loads(Path(asset['path']).read_text())
        assert run._sha(direct_asset['checkpoint']) == direct_asset['sha256']
        model = DirectCoordinateRegressor(checkpoint=Path(direct_asset['checkpoint']),seed=seed,amp=True)
        provenance = dict(source_model=asset,source_selection_sha256=run._sha(selection_path),
            GP='Independent direct-coordinate readout')
        for task, truth in episodes:
            path = output/'events/C_direct'/f'seed_{seed}'/(task.task_id+'.json')
            if path.exists():
                previous = json.loads(path.read_text())
                matched(previous,task,truth)
                assert previous['provenance'] == provenance and previous['training_seed'] == seed
                records.append(previous)
                continue
            tick = time.monotonic()
            prediction = model.predict_task(task)
            seconds = time.monotonic()-tick
            records.append(event(output,'C_direct',seed,task,truth,prediction,
                provenance,seconds))
            print(json.dumps(dict(stage='target_scored',arm='C',seed=seed,task_id=task.task_id)),flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    assert len(records) == 3*36+3*(6+1)*36
    summarize(records,output,frozen)
    run._write(output/'complete.json',dict(complete=True,events=len(records),tasks=36,
        training_seeds=[1,2,3],comparison_sha256=run._sha(output/'comparison.json'),
        comparison_CSV_sha256=run._sha(output/'comparison.csv'),freeze_sha256=run._sha(freeze_path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    with threadpool_limits(limits=2):
        main(parser.parse_args())
