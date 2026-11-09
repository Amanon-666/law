"""One UJI-only 24-step comparison: relational versus radial neighbor rules."""
import gc
import json
import socket
import time
from collections import Counter
import numpy as np
import torch
from adaptation import SIGMA
from recomposition_trial import ROOT, OLD, PLAN, CSV, digest, write, now, subset
from conditional_budget_trial import history_reader
from episodic_tasks import task_inputs
from uji import load_uji, tensor, ap_stats
from neighbor_recomposition import NeighborParameters
from neighbor_stream import task_to_device, evaluate_stream, prepare_cpu_proposals
from neighbor_gpu_check import source_task
from source_context_replay import FIXED, ROTATED

OUT = ROOT/'results/neighbor_learning_uji_20261105'
PREP = ROOT/'results/recomposition_prepared_20261103'
GPU_CHECK = ROOT/'results/neighbor_gpu_check_20261105'


def load_original(event, row, plan, floors, used):
    """Read existing state/proposals without retaining old dense grid blocks."""
    path = PREP/row['path']
    assert digest(path) == row['sha256']
    inputs = json.loads((PREP/(event['id']+'_inputs.json')).read_text())
    assert inputs['event'] == event
    old = torch.load(path, weights_only=False, mmap=True)
    assert old['base'] == plan['base'] and old['obs'] == plan['obs']
    origin = np.array(plan['origin'])
    support = subset(floors[event['floor']], event['support_rows'], origin)
    query = subset(floors[event['floor']], event['query_rows'], origin)
    assert not (support['positions'][:, None] == query['positions'][None]).all(-1).any()
    histories = {f:subset(floors[f], event['history_rows'][f], origin) for f in event['histories']}
    metadata = json.loads((OLD/f"results/decision_local_pilot_20261030/{event['id']}/meta.json").read_text())
    event_used = {}
    reader = history_reader(metadata, plan['base'], origin, event_used)
    task = task_inputs(histories, support, plan['base'], plan['obs'], reader)
    assert all(inputs['history_sha256'][path] == sha for path,sha in event_used.items())
    used.update(event_used)
    assert list(task['cases']) == list(old['cases']) and torch.equal(task['grid'], old['grid'])
    assert torch.equal(tensor(query['xy']), old['truth'])
    first = next(iter(old['cases'].values()))['blocks']
    amplitude = plan['base']['drift'] if first['active'] else plan['base']['amplitude']
    v = float(first['rv'][0])-amplitude**2
    intercept = min(SIGMA, key=lambda x:abs(x*x-v))
    assert abs(intercept**2-v) < 1e-7
    for ap, case in old['cases'].items():
        assert all(torch.equal(value, case['stats'][key]) for key,value in ap_stats(support, ap).items())
        assert torch.equal(tensor(query['rss'][:, ap]), case['rss'])
        b = case['blocks']; a = plan['base']['drift'] if b['active'] else plan['base']['amplitude']
        assert float((b['rv']-a*a-intercept**2).abs().max()) < 1e-7
    task['state'] = dict(weights=old['weights'].tolist(), sigma=dict(intercept_std=intercept))
    proposals = {ap:c['proposal'] for ap,c in old['cases'].items()}
    delta = [int(event['floor'].split('F')[1])-int(f.split('F')[1]) for f in event['histories']]
    return task, proposals, delta, query


def gpu_proposals(proposals):
    return {ap:None if p is None else tuple(x.cuda() for x in p) for ap,p in proposals.items()}


def score(model, item, query=None, backward=False):
    task, proposals, delta, oldq, ident = item
    metric, _ = evaluate_stream(model, task, delta, ident, oldq if query is None else query,
                               backward=backward, proposals=proposals, chunk=len(task['grid']))
    return metric


def gain(value, reference):
    return 100*(reference-value)/reference


def run():
    OUT.mkdir(exist_ok=False); started = time.monotonic()
    checked = json.loads((GPU_CHECK/'freeze.json').read_text())
    assert json.loads((GPU_CHECK/'complete.json').read_text())['meets_runtime_rule']
    for name,h in checked['code'].items(): assert digest(ROOT/name) == h
    plan = json.loads(PLAN.read_text()); events = {e['id']:e for e in plan['events']}
    manifest = json.loads((PREP/'complete.json').read_text())
    rows = {r['id']:r for r in manifest['episodes']}
    batches = json.loads((ROTATED/'queries.json').read_text())['batches'][:24]
    assert Counter(b['event'] for b in batches) == {f'train_{i}':4 for i in range(6)}
    previous = json.loads((ROTATED/'freeze.json').read_text())
    assert digest(CSV) == previous['csv_sha256'] and digest(PLAN) == previous['plan_sha256']
    inputs = [PLAN, CSV, PREP/'complete.json', FIXED/'initial.pt', FIXED/'source_initial.json',
              FIXED/'complete.json', ROTATED/'queries.json', GPU_CHECK/'complete.json']
    inputs += [PREP/(f'train_{i}_inputs.json') for i in range(6)]
    write(OUT/'freeze.json', dict(created=now(),
        code={n:digest(ROOT/n) for n in ('neighbor_learning_trial.py', 'neighbor_stream.py',
            'neighbor_recomposition.py', 'neighbor_gpu_check.py', 'conditional_budget_trial.py',
            'episodic_tasks.py', 'recomposition.py', 'observations.py', 'posterior.py', 'uji.py')},
        inputs={str(p):digest(p) for p in inputs}, original_episodes=manifest['episodes'], schedule=batches,
        steps_per_arm=24, arms=dict(full='22 effective parameters', radial='Original12 + neighbor[4]; other9 exactly zero'),
        optimizer='Adam', learning_rate=.001, regularizer='Unchanged .001 quadratic penalty',
        source_gate='New B0F1 S, K5 only: full improves >=3% vs identity and >=1% vs radial.',
        development_gate='Only after source gate. B1 two K5 draws both beat identity; means improve >=3% vs identity and >=1% vs radial.',
        checkpoint='Final24 only; no hyperparameter/seed search or step extension.',
        numerical='Existing same CPU proposals copied to GPU; ESS<1000 stops, no refresh.',
        scope='UJI K5 only; six B0 source tasks, previously exposed source labels and B1 development.',
        wall_seconds=900, threads=2, device='cuda:0', downloads=0, large_cache_writes=0))
    floors = load_uji(CSV); origin = np.array(plan['origin'])
    saved = torch.load(FIXED/'initial.pt', weights_only=True)['model']
    identity = NeighborParameters(saved['center'], saved['scale']).cuda()
    models = {name:NeighborParameters(saved['center'], saved['scale']).cuda() for name in ('radial', 'full')}
    forbidden = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8, 9], device='cuda')
    tasks, used, audits = {}, {}, {}
    old_initial = json.loads((FIXED/'source_initial.json').read_text())
    for ident in (f'train_{i}' for i in range(6)):
        assert events[ident]['role'] == 'train'
        task, proposals, delta, query = load_original(events[ident], rows[ident], plan, floors, used)
        tasks[ident] = (task_to_device(task, 'cuda'), gpu_proposals(proposals), delta, query, ident)
        del task, proposals; gc.collect()
        metric = score(identity, tasks[ident])
        ref = old_initial[ident]
        error = max(abs(metric['nll']-ref['nll']), max(abs(a-b) for a,b in zip(metric['errors'], ref['errors'])))
        assert error < 1e-6, (ident, error)
        audits[ident] = dict(identity_error=error, APs=len(tasks[ident][0]['cases']), seconds=metric['seconds'])
        print(json.dumps(dict(stage='prepared', event=ident, seconds=time.monotonic()-started)), flush=True)
    queries = []
    for b in batches:
        e = events[b['event']]
        q = subset(floors[e['floor']], b['query_rows'], origin)
        s = subset(floors[e['floor']], e['support_rows'], origin)
        assert len(q['positions']) == 12 and q['stats']['n'].tolist() == [3.]*12
        assert not (q['positions'][:, None] == s['positions'][None]).all(-1).any()
        queries.append(q)
    write(OUT/'pretraining_audit.json', dict(created=now(), source=audits,
        all_queries_exclude_support_positions=True, training_updates=0))
    write(OUT/'history_inputs.json', used)
    optimizers = {n:torch.optim.Adam(m.parameters(), lr=.001) for n,m in models.items()}
    completed = {n:0 for n in models}
    with (OUT/'train.jsonl').open('w') as log:
        for b,q in zip(batches, queries):
            for name,model in models.items():
                optimizer = optimizers[name]; optimizer.zero_grad(set_to_none=True)
                try:
                    metric = score(model, tasks[b['event']], q, backward=True)
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    if name == 'radial': model.neighbor.grad[forbidden] = 0
                    optimizer.step()
                    if name == 'radial': assert bool((model.neighbor[forbidden] == 0).all())
                except Exception:
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        completed_steps=completed[name]), OUT/(name+'_failure.pt'))
                    raise
                completed[name] += 1
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                completed_steps=completed[name]), OUT/(name+'_state.pt'))
                row = dict(step=completed[name], arm=name, event=b['event'], **metric)
                log.write(json.dumps(row)+'\n'); log.flush()
                write(OUT/'progress.json', dict(created=now(), completed=completed, seconds=time.monotonic()-started))
            if b['step'] % 3 == 0:
                print(json.dumps(dict(stage='training', completed=completed, seconds=time.monotonic()-started)), flush=True)
    assert completed == dict(radial=24, full=24)
    # Free source proposal maps before opening the held support configuration.
    del tasks; gc.collect(); torch.cuda.empty_cache()
    cpu_zero, task, event, delta, query = source_task()
    proposals = prepare_cpu_proposals(cpu_zero, task, delta, event['id'])
    item = (task_to_device(task, 'cuda'), gpu_proposals(proposals), delta, query, event['id'])
    del task, proposals, cpu_zero
    source = {n:score(m, item) for n,m in dict(identity=identity, **models).items()}
    source_pass = gain(source['full']['mde'], source['identity']['mde']) >= 3 and gain(source['full']['mde'], source['radial']['mde']) >= 1
    write(OUT/'source_gate.json', dict(metrics=source, passed=bool(source_pass),
        gain_identity_pct=gain(source['full']['mde'], source['identity']['mde']),
        gain_radial_pct=gain(source['full']['mde'], source['radial']['mde'])))
    del item; gc.collect(); torch.cuda.empty_cache()
    development = []; development_pass = False
    if source_pass:
        reference = json.loads((FIXED/'complete.json').read_text())['development_by_event']
        for ident in ('validation_0', 'validation_1'):
            task, proposals, delta, query = load_original(events[ident], rows[ident], plan, floors, used)
            item = (task_to_device(task, 'cuda'), gpu_proposals(proposals), delta, query, ident)
            del task, proposals
            metrics = {n:score(m, item) for n,m in dict(identity=identity, **models).items()}
            assert abs(metrics['identity']['mde']-reference[ident]['identity']) < 1e-6
            development.append(dict(event=ident, metrics=metrics))
            del item; gc.collect(); torch.cuda.empty_cache()
        means = {n:float(np.mean([d['metrics'][n]['mde'] for d in development])) for n in ('identity', 'radial', 'full')}
        development_pass = (all(d['metrics']['full']['mde'] < d['metrics']['identity']['mde'] for d in development)
                            and gain(means['full'], means['identity']) >= 3 and gain(means['full'], means['radial']) >= 1)
    write(OUT/'history_inputs.json', used)
    write(OUT/'complete.json', dict(complete=True, created=now(), seconds=time.monotonic()-started,
        completed=completed, source_mde={n:m['mde'] for n,m in source.items()}, source_gate_passed=bool(source_pass),
        development=development, development_gate_passed=bool(development_pass), target_evaluations=len(development),
        GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        scope='Bounded K5 screen, not converged training or independent generalization. Source-gate failure skips B1.'))
    print((OUT/'complete.json').read_text(), flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    try: run()
    except Exception as error:
        if OUT.exists(): write(OUT/'failure.json', dict(created=now(), error=repr(error)))
        raise
