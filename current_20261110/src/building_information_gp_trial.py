"""K1 H/HS GP comparison on the frozen nine-floor information protocol.

HS reuses arbitrary-source VI adaptation and the existing M-PMC moment
readout. H uses declared uniform source weights and zero intercept, not a
support-selected or claimed optimal historical fusion. No new GP algorithms.
"""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import predict as gp
from episodic_tasks import task_inputs
from fields import History, InterceptBankPrior, rbf
from observations import observation_log_prob, predict_from_posterior
from posterior import ImportanceSupport, proposal as original_proposal
from uji import group_scans, tensor
from building_information_tab_trial import digest, now, read_scans, score_saved, write

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT/'results/building_information_assets_20261109'
OUT = ROOT/'results/building_information_gp_20261109'
SEED = 11509159


def task_plan(manifest):
    tasks = []
    for floor in manifest['floors']:
        events = [e for e in manifest['events'] if e['floor'] == floor and e['K'] == 1]
        tasks.append(dict(id=f'{floor}_H_GP_uniform', method='GP_uniform', information='H',
                          events=[e['id'] for e in events]))
        tasks.extend(dict(id=f'{e["id"]}_HS_GP_MPMC', method='GP_MPMC', information='HS',
                          events=[e['id']]) for e in events)
    return tasks


def prepare(csv_path, manifest, records, event, info, base, obs):
    """Read legal H/S, translate cached fields, and round absolute grid first."""
    names = event['histories']
    assert event['floor'] not in names
    assert names == [f for f in manifest['floors'] if f.startswith(event['building']) and f != event['floor']]
    raw = {f: read_scans(csv_path, manifest['history_rows'][f], True) for f in names}
    origin = np.concatenate([h['xy'] for h in raw.values()]).mean(0)
    histories = {f: group_scans(h['rss'], h['xy']-origin) for f, h in raw.items()}
    detected = {f: (h['rss'] != 100).any(0) for f, h in histories.items()}
    seen = np.logical_or.reduce(list(detected.values()))
    geometry = [h['xy'] for h in raw.values()]
    loaded = {}

    def load(floor, ap):
        assert floor in names and floor != event['floor'] and detected[floor][ap]
        key = floor, ap
        if key not in loaded:
            record = records[key]  # Missing detected-AP asset is an error, never None.
            path = ROOT/record['path']
            assert digest(path) == record['sha256']
            with np.load(path) as saved:
                loaded[key] = History(tensor(saved['z']+saved['origin']-origin),
                                      dict(mean=tensor(saved['mean']), cov=tensor(saved['cov'])), base)
        return loaded[key]

    if info == 'HS':
        s = read_scans(csv_path, event['support_rows'], True)
        support = group_scans(s['rss'], s['xy']-origin)
        assert len(support['positions']) == 1 and len(support['rss']) == 3
        geometry.append(s['xy'])
        task = task_inputs(histories, support, base, obs, load)
    else:
        cases = {int(ap): dict(histories=[load(f, int(ap)) if detected[f][ap] else None for f in names])
                 for ap in np.flatnonzero(seen)}
        task = dict(cases=cases, names=names, base=base, obs=obs,
                    positions=[tensor(h['positions']) for h in histories.values()])
    xy = np.concatenate(geometry)
    bounds = np.stack((np.floor(xy.min(0)/2)*2, np.ceil(xy.max(0)/2)*2))
    axes = [np.arange(bounds[0, d], bounds[1, d]+1., 2.) for d in range(2)]
    absolute_grid = np.stack(np.meshgrid(*axes, indexing='xy'), -1).reshape(-1, 2)
    # task_inputs rounds local coordinates; replace only its grid with the
    # frozen absolute-rounding convention shared with the Tab information arm.
    task['grid'] = tensor(absolute_grid-origin)
    task.update(origin=origin, bounds=bounds, aps=np.asarray(list(task['cases'])))
    return task


def move_task(task, device):
    """Move already-adapted fields; no optimizer is moved onto CUDA."""
    task['grid'] = task['grid'].to(device)
    task['positions'] = [p.to(device) for p in task['positions']]
    for case in task['cases'].values():
        if 'sx' in case:
            case['sx'] = case['sx'].to(device)
            case['stats'] = {k: v.to(device) for k, v in case['stats'].items()}
        case['histories'] = [None if h is None else History(h.z.to(device),
            dict(mean=h.m.to(device), cov=h.c.to(device)), task['base']) for h in case['histories']]


def cpu_proposal(case, obs, theta, seed):
    """Original pypmc/scipy backend requires CPU .numpy(); preserve its policy."""
    device = case['sm'].device
    cpu = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in case.items()}
    numerical, diagnostics = original_proposal(cpu, obs, theta.detach().cpu(), seed)
    return {key: value.to(device) for key, value in numerical.items()}, diagnostics


@torch.no_grad()
def predict_task(task, state, rss, spec):
    points = task['grid']
    qrss = torch.as_tensor(rss[:, task['aps']], dtype=torch.float64, device=points.device)
    diagnostics = []
    if spec['information'] == 'HS':
        scores_by_method = {method: points.new_zeros((len(qrss), len(points)))
                            for method in ('GP_MPMC_shared', 'GP_MPMC_retained')}
        base, obs = task['base'], task['obs']
        sigma = state['sigma']['intercept_std']
        for column, (ap, case) in enumerate(task['cases'].items()):
            prior = InterceptBankPrior(case['histories'], state['weights'], base, sigma)
            sm, sf = prior.features(case['sx'])
            qm, qf = prior.features(points)
            sk = prior.covariance(case['sx'], case['sx'], sf, sf)
            qv = prior.diagonal(points, qf)
            cross = prior.covariance(points, case['sx'], qf, sf)
            prepared = dict(sm=sm, sk=sk, phi_s=sm.new_zeros((1, 10)), **case['stats'])
            seed = int.from_bytes(hashlib.sha256(
                f'multifloor/{spec["events"][0]}/K1/{ap}'.encode()).digest()[:4], 'little')
            numerical, info = cpu_proposal(prepared, obs, sm.new_zeros(10), seed)
            posterior = ImportanceSupport(sm, sk, case['stats'], obs,
                                          numerical['samples'], numerical['log_proposal'])
            weights = posterior.log_weights.exp()
            ess = float(weights.square().sum().reciprocal())
            if ess < 1000:
                raise RuntimeError(f'AP {ap}: ESS {ess}')
            mean = weights@posterior.samples
            centered = posterior.samples-mean
            moments = dict(mean=mean, cov=(centered.T*weights)@centered)
            amplitude = base['drift'] if any(h is not None for h in case['histories']) else base['amplitude']
            retained_cross = cross-rbf(points, case['sx'], amplitude, base['length'])-sigma**2
            for method, used_cross in (('GP_MPMC_shared', cross), ('GP_MPMC_retained', retained_cross)):
                value = predict_from_posterior(qm, qv, used_cross, sm, sk, moments)
                if (not torch.isfinite(value['mean']).all() or not torch.isfinite(value['variance']).all()
                        or (value['variance'] < 0).any()):
                    raise RuntimeError(f'AP {ap} {method}: invalid propagated moments')
                scores_by_method[method] += observation_log_prob(
                    qrss[:, column, None], value['mean'][None], value['variance'][None], obs)
            diagnostics.append(dict(AP=ap, ESS=ess, sampler=info['sampler'],
                                    shared_support_posterior=True))
    else:
        scores = points.new_zeros((len(qrss), len(points)))
        for column, case in enumerate(task['cases'].values()):
            prior = InterceptBankPrior(case['histories'], state['weights'], task['base'], 0.)
            mean, features = prior.features(points)
            variance = prior.diagonal(points, features)
            if not torch.isfinite(variance).all() or (variance < 0).any():
                raise RuntimeError('Invalid H-only predictive variance')
            scores += observation_log_prob(qrss[:, column, None], mean[None], variance[None], task['obs'])
        if not torch.isfinite(scores).all():
            raise RuntimeError('Invalid H-only complete scan scores')
        scores_by_method = {'GP_uniform': scores}
    predictions = {}
    for method, scores in scores_by_method.items():
        if not torch.isfinite(scores).all():
            raise RuntimeError(f'Invalid complete scan scores: {method}')
        raw = (scores.softmax(1)@points).cpu().numpy()+task['origin']
        predictions[method] = dict(raw=raw, projected=np.clip(raw, task['bounds'][0], task['bounds'][1]),
                                   scores=scores.cpu().numpy())
    return predictions, diagnostics


def aggregate(records):
    summaries = {}
    for method in ('GP_uniform', 'GP_MPMC_shared', 'GP_MPMC_retained'):
        selected = [r for r in records if r['method'] == method]
        assert len(selected) == 18
        floors = {f: float(np.mean([r['mde'] for r in selected if r['floor'] == f]))
                  for f in sorted({r['floor'] for r in selected})}
        buildings = {b: float(np.mean([v for f, v in floors.items() if f.startswith(b)])) for b in ('B1', 'B2')}
        summaries[method] = dict(equal_building=float(np.mean(list(buildings.values()))),
                               by_building=buildings, by_floor=floors)
    return summaries


def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    manifest_path, index_path = ASSETS/'manifest.json', ASSETS/'histories.json'
    manifest = json.loads(manifest_path.read_text())
    index = json.loads(index_path.read_text())
    complete = json.loads((ASSETS/'complete.json').read_text())
    assert complete['complete'] and digest(index_path) == complete['histories_sha256']
    assert digest(manifest_path) == complete['manifest_sha256'] == index['manifest_sha256']
    assert digest(ASSETS/'freeze.json') == complete['freeze_sha256']
    assert manifest['seed'] == SEED and len(manifest['events']) == 54 and len(manifest['floors']) == 9
    assert index['history_rows'] == manifest['history_rows']
    csv_path, parameters_path = Path(manifest['csv_path']), Path(index['parameters_path'])
    assert digest(csv_path) == manifest['csv_sha256']
    assert digest(parameters_path) == index['parameters_sha256'] == complete['parameters_sha256']
    parameters = json.loads(parameters_path.read_text())
    source_freeze = json.loads((parameters_path.parent/'freeze.json').read_text())
    assert source_freeze['source_floors'] == ['B0F0', 'B0F1', 'B0F2'] and parameters['updates'] == 32
    records = {(r['floor'], r['ap']): r for r in index['fields']}
    assert len(records) == len(index['fields'])
    for r in records.values():
        assert digest(ROOT/r['path']) == r['sha256']
    plan = task_plan(manifest)
    event_map = {e['id']: e for e in manifest['events']}
    code = ('building_information_gp_trial.py', 'building_information_tab_trial.py',
            'building_information_protocol.py', 'episodic_tasks.py', 'predict.py',
            'adaptation.py', 'fields.py', 'posterior.py', 'sampling.py', 'observations.py', 'uji.py', 'tabicl_history.py')
    frozen = dict(seed=SEED, K=1, plan=plan, device='cuda:0', cpu_threads=2,
        segment_seconds=args.seconds,
        code={name: digest(ROOT/name) for name in code},
        inputs={str(p): digest(p) for p in (manifest_path, index_path, ASSETS/'freeze.json',
                ASSETS/'complete.json', csv_path, parameters_path, parameters_path.parent/'freeze.json')},
        fields={r['path']: r['sha256'] for r in records.values()},
        forward=dict(H='Uniform all-source InterceptBankPrior, sigma=0, independent AP complete observation likelihood, uniform-grid posterior mean',
                     HS='Original arbitrary-J VI source weights and intercept adaptation; one M-PMC posterior per AP shared by two complete-scan moment readouts',
                     primary='GP_MPMC_retained, fixed before Q because AB/BA use different acquisitions; GP_MPMC_shared is the matched control, never Q-selected',
                     retained='cross_retained=cross_shared-rbf(grid,S,drift if any H else amplitude,length)-sigma^2; SS and prior grid marginal variance unchanged',
                     equivalence='Existing AcquisitionBank independent_residual=True semantics; all 3/4 sources and the same S fit retained',
                     purpose='Same-information mature GP comparison, no learned shared parameter updates'),
        reverse_audit=['Own target-floor history is never loaded; all 3/4 allowed sources retained',
                       'H-only AP/domain/weights depend only on H; HS only on H and current S',
                       'NPZ z+cache_origin-event_origin restores shared coordinates',
                       'Absolute H or HS bounds rounded to 2-unit lattice before origin subtraction',
                       'CPU adapt and original CPU proposal; only proposal tensors cross device boundary',
                       'Each AP has one proposal, support posterior and moments; only query-support cross differs between shared and retained',
                       'Query RSS only after state saved; unscored predictions saved before reading query truth'],
        limits=['B0-only finite 32-update source calibration, no global-optimum guarantee',
                'Development evaluation, not blind and not three-fold LOBO',
                'HS is existing arbitrary-source VI/M-PMC, not the older two-source profile/scalar retained-GP algorithm',
                'H uniform fusion is a declared simple comparator, not source weights selected without data',
                'Existing proposal ESS and two-scramble policy retained; no new retries or degraded fallback',
                'Task-boundary budget may overrun by one in-flight task; explicit resume required'])
    output = args.output
    if args.resume:
        assert json.loads((output/'freeze.json').read_text())['contract'] == frozen
        progress = json.loads((output/'progress.json').read_text())
        assert progress['freeze_sha256'] == digest(output/'freeze.json')
        for state in progress['states'].values():
            assert digest(output/state['path']) == state['sha256']
        for result in progress['tasks']:
            for prediction in result['prediction_files']:
                assert digest(output/prediction['path']) == prediction['sha256']
            for record in result['records']:
                assert digest(output/record['path']) == record['sha256']
    else:
        output.mkdir(parents=True, exist_ok=False)
        write(output/'freeze.json', dict(created=now(), contract=frozen))
        progress = dict(freeze_sha256=digest(output/'freeze.json'), states={}, tasks=[], segments=[])
        write(output/'progress.json', progress)
    done = {r['id'] for r in progress['tasks']}
    for spec in plan:
        if spec['id'] in done:
            continue
        if time.monotonic()-started >= args.seconds:
            break
        tick = time.monotonic()
        try:
            event = event_map[spec['events'][0]]
            task = prepare(csv_path, manifest, records, event, spec['information'], parameters['base'], parameters['obs'])
            if spec['id'] in progress['states']:
                state = json.loads((output/progress['states'][spec['id']]['path']).read_text())['state']
            else:
                state = (gp.adapt(task['cases'], task['base'], task['obs']) if spec['information'] == 'HS'
                         else dict(weights=[1/len(task['names'])]*len(task['names']), trials=[],
                                   sigma=dict(intercept_std=0., selection='fixed_without_S')))
                path = output/f'{spec["id"]}_state.json'
                write(path, dict(state=state, origin=task['origin'].tolist(), bounds=task['bounds'].tolist(),
                                 aps=task['aps'].tolist(), sources=task['names']))
                progress['states'][spec['id']] = dict(path=path.name, sha256=digest(path))
                write(output/'progress.json', progress)
            move_task(task, torch.device('cuda:0'))
            query_ids = list(dict.fromkeys(i for eid in spec['events'] for i in event_map[eid]['query_rows']))
            rss = read_scans(csv_path, query_ids, False)['rss']
            predictions, diagnostics = predict_task(task, state, rss, spec)
            prediction_files, scored = [], []
            # Persist both frozen readouts before either opens query truth.
            for method, prediction in predictions.items():
                path = output/f'{spec["id"]}_{method}_unscored.npz'
                np.savez_compressed(path, **prediction, query_rows=query_ids,
                                    aps=task['aps'], bounds=task['bounds'], origin=task['origin'],
                                    grid=task['grid'].cpu().numpy()+task['origin'])
                prediction_files.append(dict(method=method, path=path.name, sha256=digest(path)))
            for saved in prediction_files:
                readout = dict(spec, method=saved['method'])
                sha, records_scored = score_saved(csv_path, output, readout, output/saved['path'], event_map)
                assert sha == saved['sha256']
                scored.extend(records_scored)
            progress['tasks'].append(dict(id=spec['id'], seconds=time.monotonic()-tick,
                prediction_files=prediction_files, diagnostics=diagnostics,
                grid_points=len(task['grid']), APs=len(task['aps']), records=scored))
            done.add(spec['id'])
            write(output/'progress.json', progress)
            print(json.dumps(dict(task=spec['id'], completed=len(done), total=len(plan),
                                  elapsed_seconds=time.monotonic()-started)), flush=True)
        except Exception as error:
            failure = dict(created=now(), task=spec['id'], error_type=type(error).__name__,
                           error=str(error), traceback=traceback.format_exc(),
                           freeze_sha256=digest(output/'freeze.json'), completed=len(done))
            write(output/f'failure_{time.time_ns()}.json', failure)
            raise
    elapsed = time.monotonic()-started
    finished = len(done) == len(plan)
    progress['complete'] = finished
    progress['segments'].append(dict(ended=now(), wall_seconds=elapsed, budget_seconds=args.seconds,
                                     completed=len(done), complete=finished, budget_overrun=elapsed > args.seconds))
    write(output/'progress.json', progress)
    if finished:
        scored = [r for task in progress['tasks'] for r in task['records']]
        assert len(scored) == 54
        write(output/'complete.json', dict(complete=True, created=now(), K=1,
              freeze_sha256=digest(output/'freeze.json'), records=scored, summary=aggregate(scored),
              primary='GP_MPMC_retained', matched_control='GP_MPMC_shared',
              segments=progress['segments'], shared_parameter_updates=0, history_refits=0,
              support_adaptations=18, blind_test=False, LOBO=False))
    else:
        print(json.dumps(dict(complete=False, stop='segment_budget', completed=len(done),
                              total=len(plan), explicit_resume_required=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args())
