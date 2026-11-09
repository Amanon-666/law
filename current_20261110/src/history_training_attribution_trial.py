"""Complete frozen H versus HS training attribution with nine new forwards.

Ordinary30 sees H only, once per floor. Public/full24 H predictions and all
54 HS system predictions are signed-cache reuses. GP always remains GP_HS:
its half with Tab_H removes S only from the Tab branch, not the whole system.
H->HS also changes the legal AP catalogue, bounds and original preprocessing;
the contrast does not isolate a causal label-calibration mechanism. No training,
GP fitting, model selection, confidence intervals or new prediction formula.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from building_information_tab_trial import context, digest, now, read_scans, read_truth, tasks, write
from tabicl_history import PACKAGE, FrozenHistoryRegressor, state_hash

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
ASSETS = RESULTS/'building_information_assets_20261109'
OLD = RESULTS/'building_information_tab_20261109'
SOURCE = RESULTS/'ordinary_context_trial_20261109'
REFERENCE = RESULTS/'trained_system_validation_20261110'
OUT = RESULTS/'history_training_attribution_20261110'
SEED = 11509159
MODELS = ('public', 'ordinary', 'full24')
REUSED = {'GP': 'GP', '3NN_HS': '3NN',
          **{f'{m}_HS': m for m in MODELS},
          **{f'GP_{m}_HS_half': f'GP_{m}_half' for m in MODELS}}
METHODS = (*REUSED, *(f'{m}_H' for m in MODELS), *(f'GP_{m}_H_half' for m in MODELS))
CODE = ('history_training_attribution_trial.py', 'building_information_tab_trial.py',
        'building_information_protocol.py', 'tabicl_history.py', 'trained_system_validation_trial.py')


def read(path):
    return json.loads(Path(path).read_text())


def initialize(args):
    manifest = read(ASSETS/'manifest.json')
    assert read(ASSETS/'complete.json')['manifest_sha256'] == digest(ASSETS/'manifest.json')
    assert manifest['seed'] == SEED and manifest['K'] == [1, 3, 5]
    assert len(manifest['events']) == 54 and len(manifest['floors']) == 9
    csv = Path(manifest['csv_path'])
    assert digest(csv) == manifest['csv_sha256']
    events = {e['id']: e for e in manifest['events']}
    plan = [t for t in tasks(manifest, 5) if t['information'] == 'H' and t['method'] == 'public']
    assert len(plan) == 9
    for t in plan:
        group = [events[eid] for eid in t['events']]
        assert len(group) == 6 and len({tuple(e['histories']) for e in group}) == 1
        for direction in ('AB', 'BA'):
            nested = sorted((e for e in group if e['direction'] == direction), key=lambda e: e['K'])
            assert [e['K'] for e in nested] == [1, 3, 5]
            assert nested[0]['query_rows'] == nested[1]['query_rows'] == nested[2]['query_rows']
    source = read(SOURCE/'complete.json')
    assert source['complete'] and source['selected_step'] == 30
    assert source['freeze_sha256'] == digest(SOURCE/'freeze.json')
    checkpoint = Path(source['selected_checkpoint'])
    assert digest(checkpoint) == source['selected_checkpoint_sha256']
    old = read(OLD/'complete.json')
    old_contract = read(OLD/'freeze.json')['contract']
    assert old['complete'] and old['freeze_sha256'] == digest(OLD/'freeze.json')
    old_tasks = {t['id']: t for t in read(OLD/'progress.json')['tasks']}
    reference = read(REFERENCE/'complete.json')
    reference_contract = read(REFERENCE/'freeze.json')['contract']
    assert reference['complete'] and reference['freeze_sha256'] == digest(REFERENCE/'freeze.json')
    assert reference_contract['events'] == manifest['events'] and reference_contract['seed'] == SEED
    assert reference_contract['source_selection']['checkpoint'] == str(checkpoint)
    assert old_contract['manifest_sha256'] == digest(ASSETS/'manifest.json')
    assert old_contract['csv_sha256'] == digest(csv) and old_contract['seed'] == SEED
    references = {r['event']: r for r in reference['records']}
    assert set(references) == set(events)
    hashes = {}
    cached = {}
    for floor in manifest['floors']:
        cached[floor] = {m: old_tasks[f'{floor}_H_{m}'] for m in ('public', 'full24')}
        for record in cached[floor].values():
            path = OLD/record['prediction_path']
            assert digest(path) == record['prediction_sha256']
            hashes[str(path)] = record['prediction_sha256']
    for record in references.values():
        for key in ('prediction', 'scored'):
            assert digest(record[key+'_path']) == record[key+'_sha256']
            hashes[record[key+'_path']] = record[key+'_sha256']
    package = {str(p.relative_to(PACKAGE)): digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package == old_contract['package'] == reference_contract['package']
    for name in ('tabicl_history.py', 'building_information_tab_trial.py'):
        assert digest(ROOT/name) == old_contract['code'][name] == reference_contract['code'][name]
    inputs = [csv, checkpoint, ASSETS/'manifest.json', ASSETS/'complete.json',
        SOURCE/'complete.json', SOURCE/'freeze.json', OLD/'complete.json', OLD/'progress.json',
        OLD/'freeze.json', REFERENCE/'complete.json', REFERENCE/'freeze.json']
    frozen = dict(code={n: digest(ROOT/n) for n in CODE}, package=package,
        inputs={str(p): digest(p) for p in inputs}, prediction_hashes=hashes,
        events=manifest['events'], plan=plan, seed=SEED, checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint), methods=list(METHODS),
        source_selection=reference_contract['source_selection'],
        new_work='Nine ordinary30 H-only fit/predict calls; no other model forward, training, GP computation or changed input',
        permission='H-only APs/bounds/origin/scalers from H only; empty S. Exact query-row mapping verified across AB/BA and all three K.',
        scoring='Seal all nine H forwards and all 54 combined prediction files before reading any query truth; equal building/floor/direction/K/position',
        contrasts='H->HS gain = H error minus HS error; training gain = reference-model error minus trained-model error. Positive means improvement. p90 differences compare marginal weighted quantiles, not a quantile of paired differences.',
        limits=['GP_HS remains fixed and reads S in every half: Tab-side S ablation, not a whole-system H-only predictor',
            'H->HS includes S-induced AP catalogue, coordinate bounds and original preprocessing changes; not isolated label calibration or a causal mechanism',
            'Same-position same-information comparisons of frozen algorithms; no weak S-only difference-in-differences innovation claim',
            'Existing development buildings, not blind/LOBO; repeated cross-K positions are not independent observations',
            'Descriptive means/p90 only; no selection gate, confidence interval, model/weight selection or parameter sweep'],
        resources=dict(gpu=1, cpu_threads=2, device='cuda:0', seconds=args.seconds,
            boundary='Complete floor forward; explicit resume, no automatic retries or extension'))
    if args.resume:
        assert read(args.output/'freeze.json')['contract'] == frozen
        progress = read(args.output/'progress.json')
        assert progress['freeze_sha256'] == digest(args.output/'freeze.json')
        assert not progress.get('complete', False), 'Already complete'
        for r in progress['floors']:
            assert digest(r['path']) == r['sha256']
    else:
        args.output.mkdir(exist_ok=False)
        write(args.output/'freeze.json', dict(created=now(), contract=frozen))
        progress = dict(freeze_sha256=digest(args.output/'freeze.json'), floors=[], segments=[])
        write(args.output/'progress.json', progress)
    return manifest, csv, plan, checkpoint, cached, references, old, reference, progress


def mapped(saved, row_ids):
    rows = saved['query_rows'].tolist()
    assert len(rows) == len(set(rows))
    index = {row: i for i, row in enumerate(rows)}
    return saved['projected'][[index[row] for row in row_ids]]


def finish(args, manifest, csv, cached, references, old, reference, progress):
    assert len(progress['floors']) == 9
    floor_records = {r['floor']: r for r in progress['floors']}
    predictions = []
    # First persist every combined prediction, with no label reads in this loop.
    for event in manifest['events']:
        eid, floor = event['id'], event['floor']
        with np.load(references[eid]['prediction_path']) as saved:
            assert np.array_equal(saved['query_rows'], event['query_rows'])
            values = {m: saved[name].copy() for m, name in REUSED.items()}
            hs_aps, hs_bounds = saved['aps'].copy(), saved['bounds'].copy()
        with np.load(floor_records[floor]['path']) as saved:
            values['ordinary_H'] = mapped(saved, event['query_rows'])
            h_aps, h_bounds = saved['aps'].copy(), saved['bounds'].copy()
        for m in ('public', 'full24'):
            with np.load(OLD/cached[floor][m]['prediction_path']) as saved:
                assert np.array_equal(saved['aps'], h_aps) and np.array_equal(saved['bounds'], h_bounds)
                values[f'{m}_H'] = mapped(saved, event['query_rows'])
        for m in MODELS:
            values[f'GP_{m}_H_half'] = .5*(values['GP']+values[f'{m}_H'])
        path = args.output/f'{eid}_unscored.npz'
        np.savez_compressed(path, **values, query_rows=event['query_rows'],
            H_aps=h_aps, H_bounds=h_bounds, HS_aps=hs_aps, HS_bounds=hs_bounds)
        predictions.append(dict(event=eid, prediction_path=str(path), prediction_sha256=digest(path)))
    samples, records = [], []
    for event, record in zip(manifest['events'], predictions):
        truth = read_truth(csv, event['query_rows'])
        _, group, count = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
        weights = 1/(len(count)*count[group])
        with np.load(record['prediction_path']) as saved:
            errors = {m: np.linalg.norm(saved[m]-truth, axis=1) for m in METHODS}
        with np.load(references[event['id']]['scored_path']) as saved:
            assert np.array_equal(truth, saved['truth']) and np.array_equal(weights, saved['weights'])
            for m, name in REUSED.items():
                assert np.allclose(errors[m], saved[name], rtol=0, atol=1e-10)
        if event['K'] == 1:
            for m in ('public', 'full24'):
                old_record = next(r for r in cached[event['floor']][m]['records'] if r['event'] == event['id'])
                assert abs(float(weights@errors[f'{m}_H'])-old_record['mde']) <= 1e-9
        path = args.output/f'{event["id"]}_scored.npz'
        np.savez_compressed(path, truth=truth, weights=weights, **errors)
        records.append(dict(record, scored_path=str(path), scored_sha256=digest(path)))
        floor_count = sum(f.startswith(event['building']) for f in manifest['floors'])
        samples.append(dict(event=event, weight=.5/floor_count/2/len(count),
            errors={m: np.array([e[group == i].mean() for i in range(len(count))]) for m, e in errors.items()}))

    def aggregate(subset, m):
        v = np.concatenate([s['errors'][m] for s in subset])
        w = np.concatenate([np.full(len(s['errors'][m]), s['weight']) for s in subset])
        w /= w.sum()
        order = np.argsort(v)
        return dict(mean=float(w@v), position_mean_p90=float(v[order][np.searchsorted(np.cumsum(w[order]), .9)]))

    def summary(m):
        result = aggregate(samples, m)
        for key, choices in (('K', (1, 3, 5)), ('floor', manifest['floors']), ('building', manifest['buildings'])):
            result['by_'+key] = {str(c): aggregate([s for s in samples if s['event'][key] == c], m) for c in choices}
        return result

    summaries = {m: summary(m) for m in METHODS}
    for m, name in REUSED.items():
        assert abs(summaries[m]['mean']-reference['summary'][name]['mean']) <= 1e-9
        assert abs(summaries[m]['position_mean_p90']-reference['summary'][name]['position_mean_p90']) <= 1e-9
        for k in ('1', '3', '5'):
            assert abs(summaries[m]['by_K'][k]['mean']-reference['summary'][name]['by_K'][k]['mean']) <= 1e-9
    for m in ('public', 'full24'):
        assert abs(summaries[f'{m}_H']['by_K']['1']['mean']-old['summary']['1'][f'{m}_H']['equal_building']) <= 1e-9

    def difference(a, b):
        return {key: difference(a[key], b[key]) if isinstance(a[key], dict) else a[key]-b[key] for key in a}

    gains = {}
    for m in MODELS:
        for prefix, suffix in (('', ''), ('GP_', '_half')):
            gains[f'{prefix}{m}_H_to_HS{suffix}'] = difference(
                summaries[f'{prefix}{m}_H{suffix}'], summaries[f'{prefix}{m}_HS{suffix}'])
    for baseline, learned in (('public', 'ordinary'), ('public', 'full24'), ('ordinary', 'full24')):
        for info in ('H', 'HS'):
            for prefix, suffix in (('', ''), ('GP_', '_half')):
                gains[f'{prefix}{learned}_vs_{baseline}_{info}{suffix}'] = difference(
                    summaries[f'{prefix}{baseline}_{info}{suffix}'], summaries[f'{prefix}{learned}_{info}{suffix}'])
    progress['complete'] = True
    write(args.output/'progress.json', progress)
    write(args.output/'complete.json', dict(complete=True, created=now(), freeze_sha256=progress['freeze_sha256'],
        summary=summaries, gains=gains, gain_sign='Reference minus candidate; positive is improvement; p90 difference is not paired-difference p90',
        records=records, floor_forwards=progress['floors'], segments=progress['segments'],
        model_state_before=progress['model_state_before'], model_state_after=progress['model_state_after'],
        training_updates=0, GP_fits=0, new_forwards=9, target_selection=False, blind_test=False,
        limits='GP always uses HS; H->HS also changes legal APs/bounds/preprocessing. Across-K repetitions are not independent samples.'))
    print(json.dumps(dict(complete=True, means={m: summaries[m]['mean'] for m in METHODS})), flush=True)


@torch.no_grad()
def run(args):
    started = time.monotonic()
    manifest, csv, plan, checkpoint, cached, references, old, reference, progress = initialize(args)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    events = {e['id']: e for e in manifest['events']}
    done = {r['floor'] for r in progress['floors']}
    initial_count = len(done)
    model = None
    for task in plan:
        event = events[task['events'][0]]
        floor = event['floor']
        if floor in done:
            continue
        if time.monotonic()-started >= args.seconds:
            break
        tick = time.monotonic()
        histories, support, rss, _, aps, bounds = context(csv, manifest, event, 'H')
        row_ids = list(dict.fromkeys(row for eid in task['events'] for row in events[eid]['query_rows']))
        for record in cached[floor].values():
            with np.load(OLD/record['prediction_path']) as saved:
                assert set(saved['query_rows'].tolist()) == set(row_ids)
                assert np.array_equal(saved['aps'], aps) and np.array_equal(saved['bounds'], bounds)
        if model is None:
            model = FrozenHistoryRegressor(checkpoint, device='cuda:0')
            model.loaded.model_.requires_grad_(False)
            before = state_hash(model.loaded.model_)
            if 'model_state_before' in progress:
                assert progress['model_state_before'] == before
            progress['model_state_before'] = before
        model.fit(histories, support, floor, aps, bounds, SEED)
        qrss = read_scans(csv, row_ids, False)['rss']
        predictions = model.predict(qrss)
        path = args.output/f'{floor}_ordinary_H_unscored.npz'
        np.savez_compressed(path, **predictions, query_rows=row_ids, aps=aps, bounds=bounds)
        after = state_hash(model.loaded.model_)
        assert after == before
        progress['model_state_after'] = after
        progress['floors'].append(dict(floor=floor, path=str(path), sha256=digest(path),
            context_rows=len(rss), query_rows=len(row_ids), APs=len(aps), seconds=time.monotonic()-tick))
        write(args.output/'progress.json', progress)
        done.add(floor)
        print(json.dumps(dict(floor=floor, sealed=len(done), total=9, truth_read=False)), flush=True)
    progress['segments'].append(dict(created=now(), seconds=time.monotonic()-started,
        budget=args.seconds, completed=len(done)-initial_count))
    write(args.output/'progress.json', progress)
    if len(done) != 9:
        print(json.dumps(dict(complete=False, stop='segment_budget', explicit_resume_required=True)), flush=True)
        return 2
    finish(args, manifest, csv, cached, references, old, reference, progress)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=450.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    raise SystemExit(run(parser.parse_args()))
