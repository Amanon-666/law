"""One existing B0F1 acquisition direction, K1; bounded no-training decision."""
import argparse
import hashlib
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

from acquisition_bank import AcquisitionBank, prepare_bank_blocks
from acquisition_tasks import load_acquisition_rows, episode_scans
from conditional_budget_trial import history_reader
from episodic_tasks import task_inputs
from fields import History
from recomposition import Parameters
from recomposition_multibudget_natural import wknn
from recomposition_trial import ROOT, OLD, PLAN, CSV, digest, write, now

OUT = ROOT/'results/acquisition_bank_trial_20261105'
PARENT = ROOT/'results/acquisition_retention_trial_20261105'
NATURAL = ROOT/'results/recomposition_multibudget_natural_20261104'
CALIBRATION = ROOT/'results/acquisition_calibration_probe_20261105'
ORDERS = (32, 64)
QUERY_CONDITIONED = False
REFERENCE = None


def source_prior(plan):
    floors = {e['id']: e['floor'] for e in plan['events']}
    used = {}; sigmas = []
    for r in json.loads((NATURAL/'progress.json').read_text())['records']:
        if r['role'] != 'train' or floors[r['id']] == 'B0F1':
            continue
        p = NATURAL/r['path']; assert digest(p) == r['sha256']
        used[str(p)] = r['sha256']
        payload = torch.load(p, weights_only=False)
        sigmas.append(float(payload['states'][1]['intercept']))
    pairs = {}
    for r in json.loads((CALIBRATION/'records.json').read_text()):
        if r['floor'] == 'B0F1':
            continue
        pair = tuple(sorted(r['family'].split('_to_')))
        pairs.setdefault(pair, r['offset'])
    tau = float(np.sqrt(np.mean(np.square(list(pairs.values())))/2))
    sigma = float(np.median(sigmas))
    return dict(sigma=sigma, tau=tau, other_floor_sigmas=sigmas, distinct_pairs=len(pairs),
                offsets=[-np.sqrt(1.5)*tau, 0., np.sqrt(1.5)*tau]), used


def move_task(task, device):
    task['grid'] = task['grid'].to(device)
    task['positions'] = [x.to(device) for x in task['positions']]
    for case in task['cases'].values():
        case['sx'] = case['sx'].to(device)
        case['stats'] = {k: v.to(device) for k, v in case['stats'].items()}
        case['histories'] = [None if h is None else History(h.z.to(device),
            dict(mean=h.m.to(device), cov=h.c.to(device)), task['base']) for h in case['histories']]


def score_prediction(xy, query):
    errors = np.linalg.norm(xy-query['xy'], axis=1)
    weights = 1/(len(query['positions'])*query['stats']['n'][query['group']])
    return dict(mde=float(weights@errors), errors=errors.tolist())


@torch.no_grad()
def run():
    OUT.mkdir(exist_ok=False); started = time.monotonic()
    structure = ROOT/('results/conditioned_scalar_structure_20261105/attempt_2' if QUERY_CONDITIONED
                      else 'results/scalar_support_structure_20261105')
    assert json.loads((structure/'complete.json').read_text())['complete']
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.075)
    plan = json.loads(PLAN.read_text())
    parent = json.loads((PARENT/'freeze.json').read_text())
    e = dict(next(e for e in parent['events'] if e['id'] == 'acquisition_4_K5'))
    e.update(K=1, support_rows=e['support_rows'][:3], query_rows=e['evaluation_rows'])
    assert e['floor'] == 'B0F1' and e['role'] == 'train'
    hyper, hyper_inputs = source_prior(plan)
    names = ['acquisition_bank_trial.py', 'acquisition_bank.py', 'acquisition_offset.py', 'scalar_support.py',
             'recomposition.py', 'fields.py', 'posterior.py', 'observations.py', 'sampling.py',
             'episodic_tasks.py', 'acquisition_tasks.py', 'conditional_budget_trial.py',
             'recomposition_multibudget_natural.py', 'recomposition_trial.py', 'uji.py']
    inputs = {str(p): digest(p) for p in [PLAN, CSV, PARENT/'freeze.json', CALIBRATION/'records.json', NATURAL/'progress.json']}
    inputs.update(hyper_inputs)
    assert inputs[str(CSV)] == parent['inputs'][str(CSV)]
    if REFERENCE is not None:
        reference = json.loads((REFERENCE/'freeze.json').read_text())
        assert inputs == reference['inputs'] and e == reference['event'] and hyper == reference['hyper']
        numerical_changes = {'acquisition_bank_trial.py', 'acquisition_offset.py', 'scalar_support.py'}
        assert all(digest(ROOT/n) == h for n, h in reference['code'].items() if n not in numerical_changes)
    write(OUT/'freeze.json', dict(created=now(), event=e, hyper=hyper, inputs=inputs,
        code={n: digest(ROOT/n) for n in names}, source_weights=[[1., 0.], [.5, .5], [0., 1.]],
        orders=list(ORDERS), same_acquisition=False, theta='identity; no parameter updates',
        numerical_rule='condition on query RSS before detection integral' if QUERY_CONDITIONED else 'direct full query likelihood',
        reference=None if REFERENCE is None else dict(path=str(REFERENCE), freeze_sha256=digest(REFERENCE/'freeze.json')),
        structure=dict(path=str(structure), complete_sha256=digest(structure/'complete.json')),
        gate=f'{ORDERS} max query TV and support joint TV <=.005; max XY change <=.1 U. Fixed order{ORDERS[-1]} for scoring.',
        comparison='offset vs zero-offset identical finite configurations, sigma and full posterior; same-H/S WKNN',
        scope='One known source-development direction, K1R3, all 12 evaluation positions/36 scans. B0 global nuisance fitted previously.',
        limitations=['No strong-GP optimality comparison', 'No learning or cross-environment claim',
                    'Finite offset prior assumes centered independent acquisition population; unverified physical interpretation'],
        resources=dict(cpu=2, gpu=1, vram_fraction=.075, seconds=600, downloads=0)))
    for n in names: (OUT/n).write_bytes((ROOT/n).read_bytes())
    rows = load_acquisition_rows(CSV, {'B0F0', 'B0F1', 'B0F2'})
    assert len({rows[i]['xy'] for i in e['support_rows']}) == 1
    assert all(list(rows[i]['acquisition']) == e['support_acquisition'] for i in e['support_rows'])
    assert all(list(rows[i]['acquisition']) == e['query_acquisition'] for i in e['query_rows'])
    assert not {rows[i]['xy'] for i in e['support_rows']} & {rows[i]['xy'] for i in e['query_rows']}
    origin = np.asarray(plan['origin']); packed = episode_scans(rows, e, origin)
    meta = json.loads((OLD/f"results/decision_local_pilot_20261030/{e['parent_event']}/meta.json").read_text())
    used = {}
    task = task_inputs(packed['histories'], packed['support'], plan['base'], plan['obs'],
                       history_reader(meta, plan['base'], origin, used))
    write(OUT/'history_inputs.json', used)
    move_task(task, device)
    blocks = prepare_bank_blocks(task, [int(e['floor'].split('F')[1])-int(f.split('F')[1]) for f in e['histories']])
    stats = {a: c['stats'] for a, c in task['cases'].items()}
    grid = task['grid']; aps = list(blocks)
    model = Parameters(torch.zeros(9), torch.ones(9)).to(device).requires_grad_(False)
    weights = grid.new_tensor([[1., 0.], [.5, .5], [0., 1.]])
    sigma = grid.new_full((3,), hyper['sigma']); cm = grid.new_zeros(3)
    rss = grid.new_tensor(packed['query']['rss'][:, aps])
    write(OUT/'geometry.json', dict(APs=aps, grid_points=len(grid), support_rows=e['support_rows'],
        query_rows=e['query_rows'], query_positions=len(packed['query']['positions']), seconds=time.monotonic()-started))
    print(json.dumps(dict(stage='prepared', APs=len(aps), grid=len(grid), seconds=time.monotonic()-started)), flush=True)
    saved = {}
    for name, offsets in [('zero', [0.]), ('offset', hyper['offsets'])]:
        saved[name] = {}
        for order in ORDERS:
            before = time.monotonic()
            bank = AcquisitionBank(model, blocks, stats, grid, task['obs'], weights, sigma, cm,
                                   grid.new_tensor(offsets), grid.new_zeros(len(offsets)), scalar_order=order,
                                   query_conditioned=QUERY_CONDITIONED)
            result = bank.locate(rss, chunk_size=256)
            arrays = dict(xy=result['xy'].cpu().numpy(), log_probability=result['log_probability'].cpu().numpy(),
                          joint_mass=bank.log_joint_mass.exp().cpu().numpy(), truth=packed['query']['xy'],
                          rows=np.asarray(e['query_rows']))
            np.savez_compressed(OUT/f'{name}_{order}.npz', **arrays)
            saved[name][order] = arrays
            print(json.dumps(dict(stage='predicted', method=name, order=order, seconds=time.monotonic()-before,
                                  total=time.monotonic()-started)), flush=True)
            del bank, result
    convergence = {}
    for name, both in saved.items():
        a, b = (both[o] for o in ORDERS)
        convergence[name] = dict(query_tv=float(np.abs(np.exp(a['log_probability'])-np.exp(b['log_probability'])).sum(1).max()/2),
            support_tv=float(np.abs(a['joint_mass']-b['joint_mass']).sum()/2),
            maximum_xy_change=float(np.linalg.norm(a['xy']-b['xy'], axis=1).max()))
    passed = all(v['query_tv'] <= .005 and v['support_tv'] <= .005 and v['maximum_xy_change'] <= .1 for v in convergence.values())
    write(OUT/'convergence.json', dict(passed=passed, methods=convergence))
    if not passed:
        write(OUT/'complete.json', dict(complete=True, numerical_gate=False, convergence=convergence,
                                       scoring=False, seconds=time.monotonic()-started,
                                       peak_vram_mib=torch.cuda.max_memory_allocated()/2**20))
        return
    metrics = {name: score_prediction(both[ORDERS[-1]]['xy'], packed['query']) for name, both in saved.items()}
    metrics['WKNN'] = wknn(packed['support'], packed['query'], list(packed['histories'].values()),
                          np.asarray(aps), torch.arange(1), grid.cpu())
    result = dict(complete=True, created=now(), numerical_gate=True, convergence=convergence, metrics=metrics,
        relative_gain=1-metrics['offset']['mde']/metrics['zero']['mde'],
        useful=all(metrics['offset']['mde'] <= .97*metrics[n]['mde'] for n in ('zero', 'WKNN')),
        seconds=time.monotonic()-started, peak_vram_mib=torch.cuda.max_memory_allocated()/2**20,
        training_steps=0, independent_environment=False)
    frozen = json.loads((OUT/'freeze.json').read_text())
    assert all(digest(ROOT/n) == h for n, h in frozen['code'].items())
    write(OUT/'complete.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    parser = argparse.ArgumentParser()
    parser.add_argument('--rule', choices=('direct', 'conditioned'), default='direct')
    args = parser.parse_args()
    if args.rule == 'conditioned':
        REFERENCE = OUT
        OUT = ROOT/'results/acquisition_bank_conditioned_trial_20261105'
        QUERY_CONDITIONED = True
        ORDERS = (64, 128)
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    try:
        run()
    except Exception as error:
        if OUT.exists(): write(OUT/'failure.json', dict(created=now(), error=repr(error)))
        raise
