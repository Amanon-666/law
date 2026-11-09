"""One fixed real task: CPU/GPU equivalence and synchronized runtime."""
import json
import socket
import time
import numpy as np
import torch
from recomposition_trial import ROOT, OLD, PLAN, CSV, digest, write, now, subset
from source_context_replay import PREP, FIXED, OUT as REPLAY
from neighbor_source_integration import OUT as CPU
from episodic_tasks import task_inputs
from uji import load_uji, load_history
from neighbor_recomposition import NeighborParameters
from neighbor_stream import prepare_cpu_proposals, task_to_device, evaluate_stream

OUT = ROOT/'results/neighbor_gpu_check_20261105'


def source_task():
    """Reuse the exact already-scored source task and H/S adaptation."""
    previous = json.loads((CPU/'freeze.json').read_text())
    assert all(digest(p) == h for p,h in previous['inputs'].items())
    for name in ('neighbor_recomposition.py', 'recomposition.py', 'posterior.py', 'observations.py', 'uji.py'):
        assert digest(ROOT/name) == previous['code'][name]
    plan = json.loads(PLAN.read_text())
    family = json.loads((PREP/'tasks.json').read_text())['family']
    event = next(e for e in family if e['id'] == previous['event'])
    assert event['role'] == 'train' and event['floor'] == 'B0F1'
    floors = load_uji(CSV); origin = np.array(plan['origin'])
    support = subset(floors[event['floor']], event['support_rows'], origin)
    query = subset(floors[event['floor']], event['query_rows'], origin)
    assert not (support['positions'][:, None] == query['positions'][None]).all(-1).any()
    histories = {f:subset(floors[f], event['history_rows'][f], origin) for f in event['histories']}
    meta = json.loads((OLD/'results/decision_local_pilot_20261030/train_1/meta.json').read_text())
    entries = {(r['floor'], r['ap']):r for r in meta['history_records']}
    old_inputs = json.loads((CPU/'history_inputs.json').read_text())
    def history(floor, ap):
        record = entries[(floor, ap)]
        path = (OLD/record['reuse_path'].split('/Law-Probabilistic-Transfer-20261023/', 1)[1]
                if 'reuse_path' in record else OLD/f'results/decision_local_pilot_20261030/histories/{floor}/ap{ap}.npz')
        assert digest(path) == old_inputs[str(path)] == record['sha256']
        with np.load(path) as array: shift = tuple(array['origin']-origin)
        return load_history(str(path), shift, tuple(plan['base'].items()))
    task = task_inputs(histories, support, plan['base'], plan['obs'], history)
    task['state'] = json.loads((REPLAY/'K5_adaptation.json').read_text())
    delta = [int(event['floor'].split('F')[1])-int(f.split('F')[1]) for f in event['histories']]
    saved = torch.load(FIXED/'initial.pt', weights_only=True)['model']
    model = NeighborParameters(saved['center'], saved['scale'])
    return model, task, event, delta, query


def metric_gap(new, old):
    return max(abs(new['nll']-old['nll']), abs(new['mde']-old['mde']),
               max(abs(a-b) for a,b in zip(new['errors'], old['errors'])))


def run():
    OUT.mkdir(exist_ok=False); started = time.monotonic()
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    write(OUT/'freeze.json', dict(created=now(),
        code={n:digest(ROOT/n) for n in ('neighbor_gpu_check.py', 'neighbor_stream.py',
            'neighbor_recomposition.py', 'posterior.py', 'observations.py', 'fields.py', 'sampling.py')},
        cpu_reference={str(p):digest(p) for p in (CPU/'freeze.json', CPU/'complete.json',
            CPU/'identity.json', CPU/'perturbation.json')},
        device=torch.cuda.get_device_name(device), torch_version=torch.__version__,
        precision='float64', proposal_policy='Original seeded CPU M-PMC; exact tensors copied to GPU.',
        grid_chunk='full per-AP grid', threads=2, wall_seconds=300,
        error_tolerance=1e-6, speed_threshold='GPU nonzero forward/backward <= CPU / 3',
        training_updates=0, target_evaluations=0, downloads=0))
    model, task, event, delta, query = source_task()
    assert len(task['cases']) == 178 and len(task['grid']) == 3355
    t = time.monotonic()
    proposals = prepare_cpu_proposals(model, task, delta, event['id'])
    proposal_seconds = time.monotonic()-t
    print(json.dumps(dict(stage='CPU_proposals_complete', seconds=proposal_seconds)), flush=True)
    task = task_to_device(task, device); model = model.to(device)
    proposals = {ap:None if p is None else tuple(x.to(device) for x in p) for ap,p in proposals.items()}
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    def progress(row):
        print(json.dumps(row), flush=True)
        write(OUT/'progress.json', dict(created=now(), **row))
    initial = {k:v.clone() for k,v in model.state_dict().items()}
    identity, proposals = evaluate_stream(model, task, delta, event['id'], query,
                                          proposals=proposals, progress=progress, chunk=len(task['grid']))
    identity_gap = metric_gap(identity, json.loads((CPU/'identity.json').read_text()))
    write(OUT/'identity.json', dict(**identity, CPU_error_gap=identity_gap))
    assert identity_gap < 1e-6
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in initial.items())
    with torch.no_grad(): model.neighbor.copy_(torch.linspace(-.05, .05, 10, device=device))
    model.zero_grad()
    initial = {k:v.clone() for k,v in model.state_dict().items()}
    perturbed, _ = evaluate_stream(model, task, delta, event['id'], query, backward=True,
                                  proposals=proposals, progress=progress, chunk=len(task['grid']))
    reference = json.loads((CPU/'perturbation.json').read_text())
    perturbation_gap = metric_gap(perturbed, reference)
    gradients = {name:p.grad.detach().cpu().tolist() for name,p in model.named_parameters()}
    gradient_gap = max(abs(x-y) for name,g in gradients.items() for x,y in zip(g, reference['gradients'][name]))
    assert perturbation_gap < 1e-6 and gradient_gap < 1e-6
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in initial.items())
    write(OUT/'perturbation.json', dict(**perturbed, gradients=gradients,
        CPU_metric_gap=perturbation_gap, CPU_gradient_gap=gradient_gap))
    speedup = reference['seconds']/perturbed['seconds']
    write(OUT/'complete.json', dict(complete=True, created=now(), seconds=time.monotonic()-started,
        event=event['id'], APs=len(task['cases']), grid_points=len(task['grid']),
        CPU_proposal_seconds=proposal_seconds, identity_GPU_seconds=identity['seconds'],
        nonzero_GPU_forward_seconds=perturbed['forward_seconds'], nonzero_GPU_forward_backward_seconds=perturbed['seconds'],
        nonzero_CPU_forward_backward_seconds=reference['seconds'], speedup=speedup,
        meets_runtime_rule=speedup >= 3, identity_gap=identity_gap,
        nonzero_metric_gap=perturbation_gap, gradient_gap=gradient_gap,
        GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20,
        GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved(device)/2**20,
        CPU_peak_rss_MiB=perturbed['peak_rss_MiB'], training_updates=0, target_evaluations=0,
        scope='Fixed source task and model; same predictions/gradient, runtime only. No learned localization gain.'))
    print((OUT/'complete.json').read_text(), flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    try: run()
    except Exception as error:
        if OUT.exists(): write(OUT/'failure.json', dict(created=now(), error=repr(error)))
        raise
