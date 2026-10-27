"""Full-scan localization of frozen source residual probes (CPU, offline).

Reverse audit before scoring: H/S determine the AP set, grid, w and sigma.
Only H creates grid Phi. The SAME g changes cached S and grid means through
learning.prediction, with the original full fixed proposal. AP log scores are
summed, never averaged; Q coordinates enter only the final scoring below.
Grid covariance is GxK (K=5), never GxG. A failed ESS stops that model.
"""
import argparse
from datetime import datetime
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'operator_prior_v1'))
os.environ['LAW_ROOT'] = str(ROOT)

import numpy as np
import torch

from residual_probe.prepare import InterceptBankPrior, history_features, data_module
import residual_probe.prepare as preparation_module
data_module()  # Resolve the same loader selected by the frozen preparation.
from residual_probe.paired_features import features as paired_features
from inference import observation_log_prob
from run_multi_history import digest

SOURCE = ROOT / 'results/source_residual_probe_v1_20261026'
PREPARATION, PREPARED = SOURCE / 'preparation', SOURCE / 'prepared'
GRID, OUT = SOURCE / 'grid_fields_v1', SOURCE / 'localization_v2'
KINDS = ('content', 'floor', 'paired')
BLOCK = 2048


def now():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False)+'\n')
    temp.replace(path)


def log(**record):
    print(json.dumps(dict(time=now(), **record), allow_nan=False), flush=True)


def manifest():
    prepared = read(PREPARED / 'manifest.json')
    if not prepared['complete'] or len(prepared['events']) != 24:
        raise RuntimeError('Expected all 24 frozen prepared source episodes')
    return prepared


def make_grid(definition):
    low, high = np.array(definition['lower']), np.array(definition['upper'])
    nx, ny = definition['shape_xy']; spacing = definition['spacing']
    axes = [low[0]+spacing*np.arange(nx), low[1]+spacing*np.arange(ny)]
    np.testing.assert_allclose([axes[0][-1], axes[1][-1]], high, atol=1e-10, rtol=0)
    grid = np.stack(np.meshgrid(*axes, indexing='xy'), -1).reshape(-1, 2)
    assert len(grid) == definition['points']
    # indexing='xy' makes array shape [ny,nx]; C flatten makes X fastest.
    np.testing.assert_array_equal(grid[:nx, 0], axes[0])
    np.testing.assert_array_equal(grid[:nx, 1], np.full(nx, low[1]))
    return torch.from_numpy(grid).double()


@lru_cache(maxsize=1)
def learning():
    path = OUT / 'code/learning.py'
    if not path.exists():
        path = ROOT / 'residual_probe/learning.py'
    spec = importlib.util.spec_from_file_location('_frozen_localization_learning', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def budget():
    events = read(PREPARATION / 'manifest.json')['events']
    values = [dict(id=e['id'], APs=len(e['AP_indices']), grid=e['grid']['points']) for e in events]
    # qm, qv, cross Gx5 and Phi Gx10: exactly 17 float64 values per grid/AP.
    matrix_bytes = sum(e['APs']*e['grid']*17*8 for e in values)
    score_bytes = sum(4*12*e['grid']*8 for e in values)
    return dict(events=values, APs=sum(e['APs'] for e in values),
                grid_matrix_uncompressed_bytes=matrix_bytes,
                four_arm_scan_score_uncompressed_bytes=score_bytes,
                available_bytes=shutil.disk_usage(SOURCE).free,
                layout='Per AP lossless NPZ; Gx5 cross covariance and Gx10 Phi; no GxG',
                proposal_copies=0, grid_block_rows=BLOCK, model_arms=4,
                transfer='Computation and large caches stay on ray-web; no download')


@torch.no_grad()
def grid_arrays(raw, prior, points, base):
    _, sf = prior.features(raw['sx'])
    chunks = []
    for x in points.split(BLOCK):
        mean, f = prior.features(x)
        chunks.append(dict(qm=mean, qv=prior.diagonal(x, f),
            cross=prior.covariance(x, raw['sx'], f, sf),
            phi_q=torch.cat((history_features(raw, x, base), paired_features(raw, x, base)), 1)))
    return {key: torch.cat([part[key] for part in chunks]) for key in chunks[0]}


def as_grid_case(case, arrays):
    return dict(case, **arrays)


@torch.no_grad()
def check():
    if (OUT / 'check.json').exists():
        raise FileExistsError(OUT / 'check.json')
    estimate = budget(); log(stage='budget', **estimate)
    if estimate['grid_matrix_uncompressed_bytes'] + estimate['four_arm_scan_score_uncompressed_bytes'] > estimate['available_bytes']:
        raise RuntimeError('Grid/score budget exceeds free disk')
    event = manifest()['events'][0]
    cases, meta = data_module().load_probe_episode(event['id'], PREPARATION)
    points = make_grid(meta['grid'])
    ap = next(a for a, c in cases.items() if any(h is not None for h in c['histories']))
    raw = cases[ap]; path = PREPARED / event['id'] / f'ap{ap}.npz'
    case = learning().load_case(path, SOURCE / 'paired_features')
    prior = InterceptBankPrior(raw['histories'], event['weights'], meta['base'], event['intercept_std'])
    arrays = grid_arrays(raw, prior, points, meta['base'])
    torch.testing.assert_close(case['phi_s'], torch.cat((history_features(raw, raw['sx'], meta['base']),
                                paired_features(raw, raw['sx'], meta['base'])), 1), atol=1e-12, rtol=1e-12)
    grid_case = as_grid_case(case, arrays)
    theta = torch.zeros(10)
    zero, ess = learning().prediction(theta, grid_case, meta['obs'], list(range(10)))
    indices = torch.linspace(0, len(points)-1, min(8, len(points))).long()
    from source_weight_mixture import field_from_prior as legacy_field
    with np.load(path, allow_pickle=False) as stored:
        info = json.loads(str(stored['proposal_info']))
        original_logw = torch.from_numpy(stored['logw']).double()
        seed = int(stored['seed'])
    with patch('source_weight_mixture.sample_reference', return_value=(case['samples'], original_logw, info)):
        old = legacy_field(prior, meta['obs'], raw['sx'], raw['stats'], points[indices], seed, defensive_fraction=.1)
    errors = {}
    for key in ['mean', 'variance']:
        torch.testing.assert_close(zero[key][indices], old[key], atol=1e-7, rtol=1e-8)
        errors[key] = float((zero[key][indices]-old[key]).abs().max())
    fake_rss = torch.tensor([[-70.], [100.]])
    left = observation_log_prob(fake_rss, zero['mean'][indices], zero['variance'][indices], meta['obs'])
    right = observation_log_prob(fake_rss, old['mean'], old['variance'], meta['obs'])
    torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-8)
    errors['scan_log_prob'] = float((left-right).abs().max())
    # Nonzero paired g checks that both S and every candidate mean use the
    # same H-only feature rule, including the all-J denominator/missing H.
    from learnable_profile.core import field_from_prior
    theta = torch.linspace(-.01, .01, 10)
    def correction(x):
        phi = torch.cat((history_features(raw, x, meta['base']), paired_features(raw, x, meta['base'])), 1)
        return 20*(phi @ theta)
    generic = field_from_prior(prior, meta['obs'], raw['sx'], raw['stats'], points[indices], correction,
                               {k: case[k] for k in ['samples', 'log_proposal']})
    value, nonzero_ess = learning().prediction(theta, grid_case, meta['obs'], list(range(10)))
    nonzero_errors = {}
    for key in ['mean', 'variance']:
        torch.testing.assert_close(value[key][indices], generic[key], atol=1e-7, rtol=1e-8)
        nonzero_errors[key] = float((value[key][indices]-generic[key]).abs().max())
    record = dict(completed=now(), event=event['id'], AP=ap, grid_points=len(points),
        zero_g_errors=errors, paired_g_errors=nonzero_errors, min_ESS=min(float(ess), float(nonzero_ess)),
        original_flatten_description=meta['grid']['flatten_order'],
        actual_flatten='meshgrid(indexing=xy), array[ny,nx], C order: x varies fastest',
        disk_budget=estimate, localization_sha256=digest(Path(__file__)),
        learning_sha256=digest(ROOT/'residual_probe/learning.py'),
        prepared_case_sha256=digest(path), actual_Q_scored=False, training_updates=0)
    write(OUT / 'check.json', record); log(stage='checked', **record)


def freeze():
    if (OUT / 'freeze.json').exists():
        raise FileExistsError(OUT / 'freeze.json')
    check_record = read(OUT / 'check.json')
    assert digest(Path(__file__)) == check_record['localization_sha256']
    assert digest(ROOT/'residual_probe/learning.py') == check_record['learning_sha256']
    data_module().verify_freeze(PREPARATION)
    files = [Path(__file__), ROOT/'residual_probe/prepare.py', ROOT/'residual_probe/paired_features.py',
             Path(data_module().__file__), ROOT/'inference.py', ROOT/'support_integral.py',
             ROOT/'history_bank.py', ROOT/'transfer.py', Path(preparation_module.operator_prior.__file__),
             ROOT/'residual_probe/adaptive_proposal.py', ROOT/'residual_probe/learning.py', ROOT/'reference_pmc.py']
    metadata = [SOURCE/folder/name for folder, names in {
        'preparation': ['manifest.json', 'freeze.json'],
        'prepared': ['manifest.json', 'parameters.json', 'freeze.json', 'summary.json'],
        'paired_features': ['freeze.json', 'summary.json'],
    }.items() for name in names]
    if not read(SOURCE/'paired_features/summary.json')['complete']:
        raise RuntimeError('Paired feature preparation incomplete')
    (OUT/'code').mkdir(exist_ok=False)
    shutil.copy2(ROOT/'residual_probe/learning.py', OUT/'code/learning.py')
    shutil.copy2(Path(__file__), OUT/'code/localization.py')
    write(OUT/'freeze.json', dict(created=now(), check_sha256=digest(OUT/'check.json'),
        code_sha256={str(p): digest(p) for p in files},
        input_metadata_sha256={str(p): digest(p) for p in metadata},
        frozen_learning_sha256=digest(OUT/'code/learning.py'),
        learning_policy='Use this exact prediction/load_case snapshot; training driver may evolve separately',
        source_scope='24 source-development episodes, K5/R1/Q12; g-only leave-building-out; fixed B0 base/obs',
        grid='Frozen H union S bounds and spacing; meshgrid xy/C, x fastest; Q never expands grid',
        APs='All frozen H/S eligible APs, sum scan log likelihood; no division by AP count',
        readout='Uniform grid prior; existing posterior mean of original candidate coordinates',
        NLL='Position NLL at nearest grid point, plus unnormalized scan NLL there; Q labels only for scoring',
        adaptation='Reuse S-only w/sigma. For learning-v2 models only, ESS<1000 triggers one H/S/current-theta-only proposal refresh with unchanged M-PMC settings; original prepared/grid caches unchanged.',
        failure='After one refresh, ESS<1000 still stops the affected model; no model fallback',
        model_selection='Score zero and each completed held-out content/floor/paired model; no hyperparameter/checkpoint selection',
        cache='Per AP qm/qv/Gx5 cross/Phi10, float64 lossless; no proposal copies or GxG covariance',
        reverse_audit=__doc__, disk_budget=check_record['disk_budget']))
    log(stage='frozen')


def verify():
    frozen = read(OUT/'freeze.json')
    for group in ['code_sha256', 'input_metadata_sha256']:
        for path, sha in frozen[group].items():
            if digest(Path(path)) != sha:
                raise RuntimeError(f'Frozen localization dependency changed: {path}')
    assert digest(OUT/'code/learning.py') == frozen['frozen_learning_sha256']
    return frozen


def cached_grid(event, ap, raw, prior, points, base, prepared_path, grid_sha):
    path = GRID/event['id']/f'ap{ap}.npz'
    prepared_sha = digest(prepared_path)
    if not path.exists():
        arrays = grid_arrays(raw, prior, points, base)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.stem+'.tmp.npz')
        np.savez_compressed(temp, **{k: v.numpy() for k, v in arrays.items()},
                            prepared_sha256=np.array(prepared_sha), grid_sha256=np.array(grid_sha))
        temp.replace(path)
        return arrays
    with np.load(path, allow_pickle=False) as stored:
        assert str(stored['prepared_sha256']) == prepared_sha
        assert str(stored['grid_sha256']) == grid_sha
        return {k: torch.from_numpy(stored[k]).double() for k in ['qm', 'qv', 'cross', 'phi_q']}


@torch.no_grad()
def score_event(event, model, destination):
    cases, meta = data_module().load_probe_episode(event['id'], PREPARATION)
    assert list(cases) == event['APs'] == meta['AP_indices']
    assert meta['adaptation']['weights'] == event['weights']
    points = make_grid(meta['grid']); grid_file = GRID/event['id']/'grid.npy'
    grid_file.parent.mkdir(parents=True, exist_ok=True)
    if grid_file.exists():
        np.testing.assert_array_equal(np.load(grid_file, allow_pickle=False), points.numpy())
    else:
        np.save(grid_file, points.numpy())
    grid_sha = digest(grid_file)
    first = next(iter(cases.values())); scans = len(first['rss'])
    scores = torch.zeros((scans, len(points))); effective = []
    theta, ids = torch.tensor(model['theta']).double(), model['feature_indices']
    predictor = learning().prediction
    if model.get('numerical_policy') == 'S_only_ESS_refresh':
        from residual_probe.adaptive_proposal import AdaptivePrediction
        predictor = AdaptivePrediction(f'localization-v2/{model["sha256"]}/{event["id"]}', destination/'proposal_refreshes.jsonl')
    for index, (ap, raw) in enumerate(cases.items()):
        path = PREPARED/event['id']/f'ap{ap}.npz'
        case = learning().load_case(path, SOURCE/'paired_features')
        prior = InterceptBankPrior(raw['histories'], event['weights'], meta['base'], event['intercept_std'])
        arrays = cached_grid(event, ap, raw, prior, points, meta['base'], path, grid_sha)
        try:
            value, ess = predictor(theta, as_grid_case(case, arrays), meta['obs'], ids)
        except RuntimeError as error:
            raise RuntimeError(f'{event["id"]} AP{ap}: {error}') from error
        scores += observation_log_prob(case['rss'][:, None], value['mean'][None], value['variance'][None], meta['obs'])
        effective.append(float(ess))
        if index % 100 == 0:
            log(stage='APs', model=model['kind'], event=event['id'], complete=index+1, total=len(cases))
    assert len(effective) == len(event['APs'])
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Nonfinite full-scan grid score')
    logp = torch.log_softmax(scores, 1); predictions = logp.exp() @ points
    # Q true coordinates are used for the following metrics only.
    truth = first['qx'][first['group']]; weights = first['query_weights']
    distances = (truth[:, None]-points[None]).square().sum(-1)
    nearest_distance, nearest = distances.min(1); rows = torch.arange(scans)
    errors = (predictions-truth).norm(dim=1)
    result = dict(event=event['id'], building=event['building'], kind=model['kind'],
        coordinate_unit=event['coordinate_unit'], position_kind=event['position_kind'],
        J=event['J'], APs=len(cases), query_scans=scans, grid_points=len(points),
        mde=float(weights @ errors), median=float(np.median(errors.numpy())),
        p90=float(np.quantile(errors.numpy(), .9)),
        position_nll=float(-weights @ logp[rows, nearest]),
        scan_nll_nearest_grid=float(-weights @ scores[rows, nearest]),
        min_ESS=min(effective), mean_nearest_grid_distance=float(weights @ nearest_distance.sqrt()),
        queries_outside_grid=int(((truth < points.min(0).values)|(truth > points.max(0).values)).any(1).sum()),
        per_query_errors=errors.tolist(), query_weights=weights.tolist(),
        model_sha256=model['sha256'], grid_sha256=grid_sha, completed=now())
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination/'predictions.npz', predictions=predictions.numpy(),
        query_xy=truth.numpy(), query_weights=weights.numpy(), errors=errors.numpy(),
        scores=scores.numpy(), AP_indices=np.array(event['APs']),
        query_scan_ids=np.array(meta['query_scan_ids']), origin=np.array(meta['origin']))
    write(destination/'result.json', result)
    log(stage='event_complete', **{k:v for k,v in result.items() if k not in ['per_query_errors','query_weights']})
    return result


def score_model(model, events, directory):
    if (directory/'status.json').exists() and read(directory/'status.json')['status'] in ['complete', 'failed_ESS']:
        assert read(directory/'status.json')['model_sha256'] == model['sha256']
        return
    directory.mkdir(parents=True, exist_ok=True)
    write(directory/'model.json', model)
    completed = []
    for event in events:
        dest = directory/event['id']
        if (dest/'result.json').exists():
            result = read(dest/'result.json')
            assert result['model_sha256'] == model['sha256']
        else:
            try:
                result = score_event(event, model, dest)
            except RuntimeError as error:
                if 'insufficient ESS' not in str(error):
                    raise
                write(directory/'status.json', dict(status='failed_ESS', error=str(error),
                    failed_event=event['id'], completed_events=completed, model_sha256=model['sha256'], time=now()))
                log(stage='model_stopped_ESS', kind=model['kind'], error=str(error))
                return
        completed.append(result['event'])
        write(directory/'status.json', dict(status='running', completed_events=completed,
                                           model_sha256=model['sha256'], time=now()))
    write(directory/'status.json', dict(status='complete', completed_events=completed,
                                       model_sha256=model['sha256'], time=now()))


def zero():
    verify(); events = manifest()['events']
    model = dict(kind='zero', theta=[0.]*10, feature_indices=list(range(10)), sha256='all-zero-fixed')
    score_model(model, events, OUT/'zero')
    summarize()


def reuse_zero():
    """Reuse completed identical zero-g results, not another performance run."""
    import ast
    verify()
    old = SOURCE/'localization_v1'
    assert read(old/'zero/status.json')['status'] == 'complete'
    def prediction_ast(path):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body
                             if isinstance(n, ast.FunctionDef) and n.name == 'prediction'))
    assert prediction_ast(old/'code/learning.py') == prediction_ast(OUT/'code/learning.py')
    previous, current = read(old/'freeze.json'), read(OUT/'freeze.json')
    assert previous['input_metadata_sha256'] == current['input_metadata_sha256']
    assert len(list((old/'zero').glob('*/result.json'))) == 24
    shutil.copytree(old/'zero', OUT/'zero')
    write(OUT/'zero_reuse.json', dict(source=str(old/'zero'),
        source_status_sha256=digest(old/'zero/status.json'), events=24,
        reason='Identical prediction AST, H/S/grid/input metadata; all-zero model needs no adaptive refresh'))
    summarize()


def models(heldout=None, kind=None):
    verify(); events = manifest()['events']; pending = []
    if not (OUT/'zero/status.json').exists() or read(OUT/'zero/status.json')['status'] != 'complete':
        raise RuntimeError('Complete all 24 zero-baseline events before scoring learned models')
    buildings = sorted({e['building'] for e in events})
    for building in buildings:
        if heldout is not None and building != heldout:
            continue
        for arm in KINDS:
            if kind is not None and arm != kind:
                continue
            path = SOURCE/'learning_v2'/building/arm/'model.json'
            if not path.exists():
                pending.append(f'{building}/{arm}'); continue
            model = read(path)
            assert model['kind'] == arm and model['steps'] == 512
            assert model['feature_indices'] == learning().FEATURES[arm]
            assert sorted(model['training_buildings']) == [b for b in buildings if b != building]
            model.update(sha256=digest(path), original_model=str(path), heldout=building)
            score_model(model, [e for e in events if e['building'] == building], OUT/building/arm)
            summarize()
    log(stage='models_checked', pending=pending)


def summarize():
    records = [read(p) for p in sorted(OUT.glob('zero/*/result.json'))]
    records += [read(p) for p in sorted(OUT.glob('*/*/*/result.json')) if 'code' not in p.parts]
    rows = []
    for building in sorted({r['building'] for r in records}):
        for kind in ('zero', *KINDS):
            group = [r for r in records if r['building'] == building and r['kind'] == kind]
            if not group:
                continue
            units = {r['coordinate_unit'] for r in group}; assert len(units) == 1
            rows.append(dict(building=building, kind=kind, events=len(group), coordinate_unit=units.pop(),
                mde=float(np.mean([r['mde'] for r in group])),
                median=float(np.median([e for r in group for e in r['per_query_errors']])),
                position_nll=float(np.mean([r['position_nll'] for r in group])),
                scan_nll_nearest_grid=float(np.mean([r['scan_nll_nearest_grid'] for r in group])),
                min_ESS=min(r['min_ESS'] for r in group)))
    statuses = {str(p.parent.relative_to(OUT)): read(p) for p in OUT.rglob('status.json')}
    write(OUT/'summary.json', dict(updated=now(), buildings=rows, events=records, statuses=statuses,
        aggregation='Equal events within each building; median pools equal-sized 12-scan events. No cross-unit overall MDE.',
        scope='Source-development g-only leave-building-out; fixed B0 base/obs and all original H/S permissions'))


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical localization stays on ray-web')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['budget','check','freeze','zero','reuse_zero','models','summary'])
    parser.add_argument('--heldout'); parser.add_argument('--kind', choices=KINDS)
    args = parser.parse_args()
    if args.stage == 'budget':
        log(stage='budget', **budget())
    elif args.stage == 'models':
        models(args.heldout, args.kind)
    elif args.stage == 'summary':
        summarize()
    else:
        globals()[args.stage]()
