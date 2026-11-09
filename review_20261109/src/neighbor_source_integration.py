"""One real source family: all-AP runtime and gradient, no optimizer."""
import json
import socket
import time
import numpy as np
import torch
from recomposition_trial import ROOT, OLD, PLAN, CSV, digest, write, now, subset
from source_context_replay import PREP, FIXED, OUT as REPLAY
from episodic_tasks import task_inputs
from uji import load_uji, load_history
from neighbor_recomposition import NeighborParameters
from neighbor_stream import evaluate_stream
from check_neighbor_recomposition import checks

OUT = ROOT/'results/neighbor_source_integration_20261105'


def run():
    OUT.mkdir(exist_ok=False); started = time.monotonic()
    previous = json.loads((PREP/'freeze.json').read_text())
    assert digest(PLAN) == previous['plan_sha256'] and digest(CSV) == previous['csv_sha256']
    plan = json.loads(PLAN.read_text())
    family = json.loads((PREP/'tasks.json').read_text())['family']
    event = next(e for e in family if e['K'] == 5 and len(e['histories']) == 2)
    assert event['role'] == 'train' and event['floor'] == 'B0F1' and event['histories'] == ['B0F0', 'B0F2']
    metadata = OLD/'results/decision_local_pilot_20261030/train_1/meta.json'
    assert digest(metadata) == previous['history_metadata_sha256']
    checkpoint = FIXED/'initial.pt'
    assert digest(checkpoint) == json.loads((REPLAY/'freeze.json').read_text())['checkpoints']['identity']
    write(OUT/'freeze.json', dict(created=now(), event=event['id'],
        code={n:digest(ROOT/n) for n in ('neighbor_recomposition.py', 'neighbor_stream.py',
            'neighbor_source_integration.py', 'check_neighbor_recomposition.py', 'recomposition.py',
            'posterior.py', 'observations.py', 'episodic_tasks.py', 'uji.py')},
        inputs={str(p):digest(p) for p in (CSV, PLAN, PREP/'tasks.json', metadata,
            REPLAY/'K5_adaptation.json', REPLAY/'K5_metrics.json', REPLAY/'history_inputs.json', checkpoint)},
        phases=['zero model identity replay', 'theta zero, neighbor linspace(-.05,.05,10), full forward/backward'],
        thresholds=dict(identity_error=1e-6, RSS_MiB=4096, wall_seconds=600),
        threads=2, training_updates=0, target_evaluations=0, downloads=0,
        limitations='Previously exposed source labels. Nonzero parameters are predeclared perturbations, not trained performance.'))
    write(OUT/'algebra.json', checks())
    floors = load_uji(CSV); origin = np.array(plan['origin'])
    support = subset(floors[event['floor']], event['support_rows'], origin)
    query = subset(floors[event['floor']], event['query_rows'], origin)
    assert not (support['positions'][:, None] == query['positions'][None]).all(-1).any()
    histories = {f:subset(floors[f], event['history_rows'][f], origin) for f in event['histories']}
    meta = json.loads(metadata.read_text())
    entries = {(r['floor'], r['ap']):r for r in meta['history_records']}
    previous_inputs = json.loads((REPLAY/'history_inputs.json').read_text()); used = {}
    def history(floor, ap):
        record = entries[(floor, ap)]
        p = (OLD/record['reuse_path'].split('/Law-Probabilistic-Transfer-20261023/', 1)[1]
             if 'reuse_path' in record else OLD/f'results/decision_local_pilot_20261030/histories/{floor}/ap{ap}.npz')
        sha = digest(p)
        assert sha == record['sha256'] == previous_inputs[str(p)]
        used[str(p)] = sha
        with np.load(p) as a: shift = tuple(a['origin']-origin)
        return load_history(str(p), shift, tuple(plan['base'].items()))
    task = task_inputs(histories, support, plan['base'], plan['obs'], history)
    task['state'] = json.loads((REPLAY/'K5_adaptation.json').read_text())
    write(OUT/'history_inputs.json', used)
    assert len(task['cases']) == 178 and len(task['grid']) == 3355
    saved = torch.load(checkpoint, weights_only=True)['model']
    model = NeighborParameters(saved['center'], saved['scale'])
    assert bool((saved['theta'] == 0).all())
    delta = [int(event['floor'].split('F')[1])-int(f.split('F')[1]) for f in event['histories']]
    def progress(row):
        print(json.dumps(row), flush=True)
        write(OUT/'progress.json', dict(created=now(), **row))
    initial = {k:v.clone() for k,v in model.state_dict().items()}
    identity, proposals = evaluate_stream(model, task, delta, event['id'], query, progress=progress)
    reference = json.loads((REPLAY/'K5_metrics.json').read_text())['identity']
    error = max(abs(identity['mde']-reference['mde']),
                max(abs(a-b) for a,b in zip(identity['errors'], reference['errors'])))
    write(OUT/'identity.json', dict(**identity, previous_error_gap=error))
    assert error < 1e-6
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in initial.items())
    with torch.no_grad(): model.neighbor.copy_(torch.linspace(-.05, .05, 10))
    initial = {k:v.clone() for k,v in model.state_dict().items()}
    model.zero_grad()
    perturbed, proposals = evaluate_stream(model, task, delta, event['id'], query,
                                           backward=True, proposals=proposals, progress=progress)
    gradients = {name:p.grad.tolist() for name,p in model.named_parameters()}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in initial.items())
    assert float(model.neighbor.grad.norm()) > 0
    write(OUT/'perturbation.json', dict(**perturbed, gradients=gradients,
        interpretation='Predeclared nonzero path verification, not a learned method score.'))
    write(OUT/'complete.json', dict(complete=True, created=now(), seconds=time.monotonic()-started,
        event=event['id'], APs=len(task['cases']), grid_points=len(task['grid']),
        identity_mde=identity['mde'], identity_error_gap=error,
        identity_seconds=identity['seconds'], nonzero_forward_backward_seconds=perturbed['seconds'],
        nonzero_forward_seconds=perturbed['forward_seconds'],
        peak_rss_MiB=perturbed['peak_rss_MiB'], neighbor_gradient_norm=float(model.neighbor.grad.norm()),
        training_updates=0, target_evaluations=0, large_caches_written=0,
        scope='Single known source family, full scans; implementation/cost evidence only.'))
    print((OUT/'complete.json').read_text(), flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    try: run()
    except Exception as error:
        if OUT.exists(): write(OUT/'failure.json', dict(created=now(), error=repr(error)))
        raise
