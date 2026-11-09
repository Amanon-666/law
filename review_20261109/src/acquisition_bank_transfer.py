"""One frozen B1F1 K1R3 comparison; no training, downloads, or automatic resume.

The caller enforces a 2400-second timeout. All input
and code provenance is frozen before any query prediction. Large predictions
and code snapshots stay on the server; complete.json is the small result.
"""
import ast
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch

from acquisition_bank import AcquisitionBank, prepare_bank_blocks
from acquisition_bank_trial import move_task
from acquisition_tasks import load_acquisition_rows, episode_scans
from adaptation import SIGMA
from conditional_budget_trial import history_reader
from episodic_tasks import task_inputs
from location_decision import geometric_median
from observations import predict_from_posterior, scan_log_prob
from recomposition import Parameters, prior
from recomposition_multibudget_natural import wknn
from recomposition_trial import ROOT, OLD, PLAN, CSV, digest, write, now
from scalar_gp_adaptation import evidence_at, fit_profile
from scalar_support import ScalarSupportIntegral
from tabicl_history import CHECKPOINT, PACKAGE, FrozenHistoryRegressor, state_hash

OUT = ROOT/'results/acquisition_bank_transfer_20261108'
TARGET = ROOT/'results/acquisition_retention_transfer_20261105'
REFERENCE = ROOT/'results/acquisition_bank_conditioned_trial_20261105'
SOURCE = ROOT/'results/tabicl_multitask_full_trial_20261107_checked'
PUBLIC = ROOT/'results/tabicl_history_transfer_20261105'
SEED = 1150541
WALL_SECONDS = 2400
ORDERS = (64, 128)
CONFIGS = [[1., 0.], [.5, .5], [0., 1.]]


def code_inputs():
    """Hash and snapshot the transitive local Python imports of this entry."""
    pending = [Path(__file__).name]; found = {}
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


def source_assets():
    reference = json.loads((REFERENCE/'freeze.json').read_text())
    inputs = dict(reference['inputs'])
    for path, expected in inputs.items():
        assert digest(Path(path)) == expected, path
    hyper = reference['hyper']
    assert hyper['sigma'] == 10. and hyper['offsets'] == [-6.414849124155285, 0., 6.414849124155285]
    assert reference['source_weights'] == CONFIGS
    source = json.loads((SOURCE/'complete.json').read_text())
    assert source == json.loads((SOURCE/'final.json').read_text()) and source['complete']
    assert source['formal_step'] == source['source_updates'] == 24 and source['source_episodes'] == 72
    assert source['source']['source_passed'] and source['target_opened'] is False
    assert digest(SOURCE/'freeze.json') == source['freeze_sha256']
    checkpoint = SOURCE/'full_step_24.ckpt'
    assert digest(checkpoint) == source['checkpoint_sha256']['24']
    sf = json.loads((SOURCE/'freeze.json').read_text())
    public = json.loads((PUBLIC/'freeze.json').read_text())
    for path in (CSV, PLAN, CHECKPOINT):
        value = digest(path)
        assert value == sf['inputs'][str(path)] == public['inputs'][str(path)]
        inputs[str(path)] = value
    package = {str(p.relative_to(PACKAGE)): digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package and package == sf['package'] == public['package']
    assert digest(TARGET/'freeze.json') == public['inputs'][str(TARGET/'freeze.json')]
    assert public['seed'] == SEED
    for path in (REFERENCE/'freeze.json', SOURCE/'complete.json', SOURCE/'final.json',
                 SOURCE/'freeze.json', checkpoint, PUBLIC/'freeze.json', TARGET/'freeze.json'):
        inputs[str(path)] = digest(path)
    return hyper, source, checkpoint, inputs, package


def actions(log_probability, grid):
    logp = log_probability.detach().cpu().numpy()
    points = grid.cpu().numpy()
    mass = np.exp(logp)
    solved = [geometric_median(points, w) for w in mass]
    gap = max(float(g) for _, g in solved)
    assert gap <= .002, gap
    return dict(log_probability=logp, mean=mass@points,
                median=np.stack([a for a, _ in solved]), median_gap=gap)


def convergence(low, high, joint=False):
    result = dict(query_tv=float(np.abs(np.exp(low['log_probability'])-
                                       np.exp(high['log_probability'])).sum(1).max()/2),
                  mean_change=float(np.linalg.norm(low['mean']-high['mean'], axis=1).max()),
                  median_change=float(np.linalg.norm(low['median']-high['median'], axis=1).max()),
                  median_gap=max(low['median_gap'], high['median_gap']))
    if joint:
        result['support_joint_tv'] = float(np.abs(low['joint_mass']-high['joint_mass']).sum()/2)
    result['passed'] = (result['query_tv'] <= .005 and result['mean_change'] <= .1
                        and result['median_change'] <= .1 and result['median_gap'] <= .002
                        and (not joint or result['support_joint_tv'] <= .005))
    return result


def metrics(prediction, query):
    weights = 1/(len(query['positions'])*query['stats']['n'][query['group']])
    return {key: float(weights@np.linalg.norm(value-query['xy'], axis=1))
            for key, value in prediction.items() if key in ('mean', 'median', 'raw', 'projected')}


def fit_gp(blocks, stats, obs, model, grid):
    aps = list(blocks)
    ss = {k: np.array([float(stats[a][k][0]) for a in aps]) for k in ('n', 'k', 's1', 's2')}
    means = np.stack([b['sm'][:, 0].cpu().numpy() if b['active'] else np.full(2, b['base_mean'])
                     for b in blocks.values()])
    variances = np.stack([b['ss'][:, 0, 0, 0].cpu().numpy() if b['active'] else np.zeros(2)
                         for b in blocks.values()])
    residual = np.array([float(b['rs'][0, 0]) for b in blocks.values()])
    audit = []
    for w, sigma in ((0., 0.), (.5, 10.), (1., 20.)):
        vector = evidence_at(w, sigma, means, variances, residual, ss, obs, 128)
        scalar = []
        for a, block in blocks.items():
            p = prior(model, block, grid.new_tensor([w, 1-w]))
            post = ScalarSupportIntegral(p['sm'], p['sk']+sigma**2, stats[a], obs, 128)
            scalar.append(float(post.log_detection_evidence-post.rss['loss']))
        gap = float(np.abs(vector-np.asarray(scalar)).max())
        assert gap <= 1e-8, gap
        audit.append(dict(weight=w, sigma=sigma, maximum_evidence_gap=gap))
    fits = {o: fit_profile(means, variances, residual, ss, obs, SIGMA, o) for o in (128, 256)}
    low, high = fits[128]['selected'], fits[256]['selected']
    evidence = {o: float(evidence_at(high['weight'], high['sigma'], means, variances,
                                    residual, ss, obs, o).sum()) for o in (128, 256)}
    passed = (all(f['selected']['success'] and f['selected']['projected_derivative'] <= 1e-3
                  for f in fits.values()) and low['sigma'] == high['sigma']
              and abs(low['weight']-high['weight']) <= .005 and abs(evidence[128]-evidence[256]) <= .01)
    return dict(passed=passed, selected=high, evidence=evidence, audit=audit, fits=fits)


def gp_predictions(blocks, stats, obs, model, grid, rss, selected, order):
    """Both old moment-field readouts share exactly the same fitted S posterior."""
    scores = {name: grid.new_zeros((len(rss), len(grid))) for name in ('GP', 'retained')}
    w = grid.new_tensor([selected['weight'], 1-selected['weight']])
    sigma2 = selected['sigma']**2
    for column, (ap, block) in enumerate(blocks.items()):
        p = prior(model, block, w)
        sk, gv = p['sk']+sigma2, p['gv']+sigma2
        post = ScalarSupportIntegral(p['sm'], sk, stats[ap], obs, order)
        mass = post.log_weights.exp()
        mean = mass@post.samples
        centered = post.samples-mean
        posterior = dict(mean=mean, cov=(centered.T*mass)@centered)
        for name, cross in (('GP', p['cross']+sigma2), ('retained', p['cross']-block['rg'])):
            field = predict_from_posterior(p['gm'], gv, cross, p['sm'], sk, posterior)
            assert bool(torch.isfinite(field['variance']).all() and (field['variance'] >= 0).all())
            scores[name] += scan_log_prob(rss[:, column], field['mean'], field['variance'], obs)
    return {name: actions(value.log_softmax(1), grid) for name, value in scores.items()}


@torch.no_grad()
def run():
    started = time.monotonic()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.5, 0)
    hyper, source, checkpoint, inputs, package = source_assets()
    parent = json.loads((TARGET/'freeze.json').read_text())
    assert [e['id'] for e in parent['events']] == ['B1F1_acquisition_0', 'B1F1_acquisition_1']
    plan = json.loads(PLAN.read_text()); origin = np.asarray(plan['origin'])
    rows = load_acquisition_rows(CSV, {'B1F0', 'B1F1', 'B1F3'})
    prepared = []; events = []
    for old in parent['events']:
        assert old['K'] == 5 and old['R'] == 3 and old['floor'] == 'B1F1' and old['role'] == 'validation'
        assert old['histories'] == ['B1F0', 'B1F3']
        original = next(e for e in plan['events'] if e['id'] == old['parent_event'])
        assert old['history_rows'] == original['history_rows'] and old['histories'] == original['histories']
        e = dict(old, K=1, support_rows=old['support_rows'][:3]); events.append(e)
        assert len(set(e['support_rows'])) == 3 and len(set(e['query_rows'])) == 36
        assert len({rows[i]['xy'] for i in e['support_rows']}) == 1
        assert not {rows[i]['xy'] for i in old['support_rows']} & {rows[i]['xy'] for i in e['query_rows']}
        assert e['support_acquisition'][:2] != e['query_acquisition'][:2]
        for key in ('support', 'query'):
            assert all(list(rows[i]['acquisition']) == e[key+'_acquisition'] for i in e[key+'_rows'])
            assert all(rows[i]['floor'] == e['floor'] for i in e[key+'_rows'])
        assert all(rows[i]['floor'] == floor for floor, ids in e['history_rows'].items() for i in ids)
        packed = episode_scans(rows, e, origin)
        assert len(packed['support']['positions']) == 1 and len(packed['query']['positions']) == 12
        assert np.all(packed['query']['stats']['n'] == 3)
        meta_path = OLD/f"results/decision_local_pilot_20261030/{e['parent_event']}/meta.json"
        inputs[str(meta_path)] = digest(meta_path)
        task = task_inputs(packed['histories'], packed['support'], plan['base'], plan['obs'],
                           history_reader(json.loads(meta_path.read_text()), plan['base'], origin, inputs))
        assert task['names'] == e['histories']
        prepared.append(dict(event=e, packed=packed, task=task, aps=np.array(list(task['cases'])),
                             bounds=np.stack([task['grid'].min(0).values.numpy(), task['grid'].max(0).values.numpy()])))
    del rows
    code = code_inputs()
    freeze = dict(created=now(), events=events, seed=SEED, inputs=inputs, code=code, package=package,
        versions={n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'torch', 'scikit-learn')},
        hyper=hyper, source_weights=CONFIGS, independent_residual=True, same_acquisition=False,
        theta='identity, frozen; no training', GP_sigmas=SIGMA, GP_orders=[128, 256], bank_orders=list(ORDERS),
        GP_fit='101 weights per sigma, all local brackets polished, endpoints retained; S evidence only',
        GP_gate=dict(same_sigma=True, weight_change=.005, evidence_change=.01, projected_derivative=.001),
        numerical_gate=dict(query_tv=.005, support_joint_tv=.005, mean_change=.1, median_change=.1, median_gap=.002),
        readout='GP/bank geometric median primary, mean secondary; Tab native projected mean; balanced positions/directions',
        criterion='candidate mean MDE <= .97 times each GP/retained/full24/WKNN; each direction <= retained/full24; zero attribution only',
        checkpoints={str(CHECKPOINT): inputs[str(CHECKPOINT)], str(checkpoint): inputs[str(checkpoint)]},
        geometry=[dict(event=p['event']['id'], APs=p['aps'].tolist(), grid_points=len(p['task']['grid']),
                       bounds=p['bounds'].tolist(), context_rows=sum(len(h['rss']) for h in p['packed']['histories'].values())+3)
                  for p in prepared],
        limitations=['Previously exposed B1 development floor; not blind validation or a K curve',
                     'Finite source prior is assumed, no new training or physical-bias identification',
                     'Dense-profile refinement is not a global optimality certificate'],
        resources=dict(cpu=2, gpu=1, max_vram_fraction=.5, hard_worker_seconds=WALL_SECONDS, downloads=0,
                       retries=0, new_training_updates=0, target_updates=0))
    write(OUT/'freeze.json', freeze)
    for name in code: (OUT/name).write_bytes((ROOT/name).read_bytes())
    records = []; numerical = {}; timings = {}

    def finish(status, **extra):
        result = dict(complete=True, status=status, scoring=status=='evaluated', created=now(),
                      seconds=time.monotonic()-started, numerical=numerical, timing=timings,
                      freeze_sha256=digest(OUT/'freeze.json'), automatic_resume=False,
                      peak_vram_mib=torch.cuda.max_memory_allocated()/2**20, **extra)
        write(OUT/'complete.json', result)
        print(json.dumps(result), flush=True)

    for p in prepared:
        e, packed, task = p['event'], p['packed'], p.pop('task')
        eid = e['id']; before = time.monotonic()
        move_task(task, torch.device('cuda:0'))
        grid = task['grid']; aps = p['aps']
        blocks = prepare_bank_blocks(task, [1-int(f.split('F')[1]) for f in task['names']])
        stats = {a: c['stats'] for a, c in task['cases'].items()}
        model = Parameters(torch.zeros(9), torch.ones(9)).to(grid.device).requires_grad_(False)
        fitted = fit_gp(blocks, stats, task['obs'], model, grid)
        write(OUT/f'{eid}_fits.json', fitted)
        numerical[eid] = dict(GP_fit={k: v for k, v in fitted.items() if k != 'fits'})
        if not fitted['passed']:
            finish('numerical_failed', failed_event=eid, failed_gate='GP_fit'); return
        rss = grid.new_tensor(packed['query']['rss'][:, aps])
        predictions = {}
        for name, offsets in (('zero', [0.]), ('candidate', hyper['offsets'])):
            both = {}
            for order in ORDERS:
                bank = AcquisitionBank(model, blocks, stats, grid, task['obs'], grid.new_tensor(CONFIGS),
                    grid.new_full((3,), hyper['sigma']), grid.new_zeros(3), grid.new_tensor(offsets),
                    grid.new_zeros(len(offsets)), independent_residual=True, scalar_order=order, query_conditioned=True)
                result = bank.locate(rss, same_acquisition=False, chunk_size=256)
                both[order] = dict(actions(result['log_probability'], grid), joint_mass=bank.log_joint_mass.exp().cpu().numpy())
                np.savez_compressed(OUT/f'{eid}_{name}_{order}.npz', **both[order], query_rows=e['query_rows'])
                del bank, result
            check = convergence(both[64], both[128], joint=True)
            numerical[eid][name] = check
            if not check['passed']:
                finish('numerical_failed', failed_event=eid, failed_gate=name); return
            predictions[name] = both[128]
        low = gp_predictions(blocks, stats, task['obs'], model, grid, rss, fitted['selected'], 128)
        high = gp_predictions(blocks, stats, task['obs'], model, grid, rss, fitted['selected'], 256)
        for name in high:
            check = convergence(low[name], high[name]); numerical[eid][name] = check
            if not check['passed']:
                finish('numerical_failed', failed_event=eid, failed_gate=name); return
            predictions[name] = high[name]
            np.savez_compressed(OUT/f'{eid}_{name}_256.npz', **high[name], query_rows=e['query_rows'])
        record = dict(event=eid, metrics={name: metrics(value, packed['query']) for name, value in predictions.items()})
        record['metrics']['WKNN'] = wknn(packed['support'], packed['query'], list(packed['histories'].values()),
                                       aps, torch.arange(1), grid.cpu())
        records.append(record); timings[eid+'_GP_bank'] = time.monotonic()-before
        write(OUT/'numerical.json', numerical)
        print(json.dumps(dict(stage='GP_bank_numerically_accepted', event=eid, seconds=time.monotonic()-started)), flush=True)
        del task, grid, blocks, stats, model, rss, low, high, predictions, both
        gc.collect(); torch.cuda.empty_cache()

    # Official checkpoint construction must keep its original float32 default.
    torch.set_default_dtype(torch.float32)
    for name, path, expected in (('public', CHECKPOINT, source['original_state_sha256']),
                                  ('full24', checkpoint, source['trained_state_sha256'])):
        before = time.monotonic()
        model = FrozenHistoryRegressor(path, device='cuda:0')
        model.loaded.model_.requires_grad_(False).eval()
        assert state_hash(model.loaded.model_) == expected
        for p, record in zip(prepared, records):
            pack = p['packed']
            model.fit(pack['histories'], pack['support'], p['event']['floor'], p['aps'], p['bounds'], SEED)
            count = sum(len(h['rss']) for h in pack['histories'].values())+3
            assert all(m.n_samples_in_ == count and m.model_ is model.loaded.model_ for m in model.estimators)
            prediction = model.predict(pack['query']['rss'])
            np.savez_compressed(OUT/f'{p["event"]["id"]}_{name}.npz', **prediction, query_rows=p['event']['query_rows'])
            record['metrics'][name] = metrics(prediction, pack['query'])
        assert state_hash(model.loaded.model_) == expected
        timings[name] = time.monotonic()-before
        del model; gc.collect(); torch.cuda.empty_cache()
    assert all(digest(ROOT/n) == h for n, h in code.items())
    means = {name: float(np.mean([r['metrics'][name][key] for r in records]))
             for name, key in (('candidate', 'median'), ('zero', 'median'), ('GP', 'median'),
                               ('retained', 'median'), ('WKNN', 'mde'), ('public', 'projected'), ('full24', 'projected'))}
    gains = {n: 1-means['candidate']/means[n] for n in ('GP', 'retained', 'full24', 'WKNN')}
    mean_gate = all(means['candidate'] <= .97*means[n] for n in gains)
    direction_gate = all(r['metrics']['candidate']['median'] <= r['metrics'][n][key]
                         for r in records for n, key in (('retained', 'median'), ('full24', 'projected')))
    finish('evaluated', records=records, means=means, relative_gains=gains,
           secondary_coordinate_mean_MDE={n: float(np.mean([r['metrics'][n]['mean'] for r in records]))
                                          for n in ('candidate', 'zero', 'GP', 'retained')},
           mean_gate=mean_gate, direction_gate=direction_gate, retain_configuration=mean_gate and direction_gate,
           candidate_gain_over_zero=1-means['candidate']/means['zero'], source_updates=24,
           new_training_updates=0, target_updates=0, blind_test=False, K_curve=False)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web', 'Server-only entry; do not run on the Mac'
    OUT.mkdir(exist_ok=False)
    try:
        run()
    except Exception as error:
        write(OUT/'complete.json', dict(complete=False, status='failed', created=now(),
            error=repr(error), scoring=False, automatic_resume=False))
        raise
