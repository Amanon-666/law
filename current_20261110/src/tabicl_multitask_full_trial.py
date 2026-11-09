"""Frozen 24-update full-parameter H+S learning on the existing B0 catalog.

No target evaluation or continuation. Step 12 is a recorded learning-curve
point; only step 24 is the formal comparison. Run on ray-web, never locally.
"""
import argparse
import json
import signal
import socket
import time
from collections import Counter
from importlib.metadata import version

import numpy as np
import torch

from acquisition_tasks import load_acquisition_rows, episode_scans
from check_tabicl_history_learning import bounds_and_aps
from source_episode_catalog import make_episode
from tabicl_history import CHECKPOINT, PACKAGE, state_hash
from tabicl_history_learning import EpisodicHistoryRegressor
from recomposition_trial import ROOT, PLAN, CSV, digest, write, now

OUT = ROOT/'results/tabicl_multitask_full_trial_20261107_checked'
CATALOG = ROOT/'results/source_episode_catalog_20261105'
REFERENCE = ROOT/'results/tabicl_balanced_source_trial_20261105'
VERIFIED = ROOT/'results/tabicl_history_source_trial_20261105'
SEED = 1150546
STEPS = 24
PREDICTION_ATOL = 1e-6


def events_for(catalog):
    by_floor = {f:[e['id'] for e in catalog['families'] if e['floor']==f]
                for f in catalog['source_floors']}
    assert sorted(map(len, by_floor.values())) == [2, 3, 4]
    schedule = []
    for step in range(STEPS):
        local = step % 12
        draw = local//6 + (0 if step < 12 else 3)
        batch = []
        for ids in by_floor.values():
            index = (local+local//3 if len(ids)==3 else local) % len(ids)
            batch.append(dict(family_id=ids[index], draw_index=draw,
                              k=(1, 3, 5)[local % 3], phase='fit'))
        schedule.append(batch)
    validation = [dict(family_id=e['id'], draw_index=2, k=k, phase='validation')
                  for e in catalog['families'] for k in (1, 3, 5)]
    for start in (0, 12):
        for ids in by_floor.values():
            selected = [s for batch in schedule[start:start+12] for s in batch
                        if s['family_id'] in ids]
            assert Counter(s['k'] for s in selected) == {1:4, 3:4, 5:4}
            assert set(Counter(s['family_id'] for s in selected).values()) == {12//len(ids)}
    assert len(validation)==27 and all(s['draw_index']!=2 for b in schedule for s in b)
    return schedule, validation


def pack(rows, catalog, spec):
    event = make_episode(catalog, **spec)
    protected = set(catalog['protected_rows'])
    inputs = event['support_rows']+[i for ids in event['history_rows'].values() for i in ids]
    assert not protected.intersection(inputs)
    assert all((i in protected)==(event['role']=='source_validation') for i in event['query_rows'])
    p = episode_scans(rows, event, np.zeros(2))
    bounds, aps = bounds_and_aps(p['histories'], p['support'])
    return dict(p, event=event, query_rows=event['query_rows'], bounds=bounds, aps=aps)


def fit(model, p):
    model.fit(p['histories'], p['support'], p['event']['floor'], p['aps'], p['bounds'], SEED)
    count = sum(len(h['rss']) for h in p['histories'].values())+len(p['support']['rss'])
    assert len(model.estimators)==2
    assert all(m.n_samples_in_==count and m.model_ is model.loaded.model_
               and m.n_estimators==8 for m in model.estimators)


def query_weights(p):
    q = p['query']
    return 1/(len(q['positions'])*q['stats']['n'][q['group']])


def aggregate(records, value):
    floors = sorted({r['floor'] for r in records})
    cells = {f:{str(k):float(np.mean([value(r) for r in records if r['floor']==f and r['K']==k]))
                for k in (1, 3, 5)} for f in floors}
    by_floor = {f:float(np.mean(list(v.values()))) for f, v in cells.items()}
    by_k = {str(k):float(np.mean([cells[f][str(k)] for f in floors])) for k in (1, 3, 5)}
    return dict(mean=float(np.mean(list(by_floor.values()))), by_floor=by_floor, by_K=by_k,
                cells=cells, K_monotonic=by_k['1']>=by_k['3']>=by_k['5'])


def progress(stage, **values):
    record = dict(created=now(), stage=stage, **values)
    write(OUT/'progress.json', record)
    print(json.dumps(record), flush=True)


@torch.no_grad()
def score(model, p, label):
    fit(model, p)
    q = p['query']; event = p['event']
    predictions = model.predict(q['rss'])
    weights = query_weights(p)
    errors = {k:np.linalg.norm(v-q['xy'], axis=1) for k, v in predictions.items()}
    path = OUT/f'{label}_{event["id"]}.npz'
    np.savez(path, **predictions, truth=q['xy'], weights=weights, query_rows=p['query_rows'],
             **{k+'_errors':v for k, v in errors.items()})
    return dict(event=event['id'], floor=event['floor'], K=event['K'],
                mde={k:float(weights@v) for k, v in errors.items()},
                context_rows=model.estimators[0].n_samples_in_, APs=len(p['aps']),
                prediction_sha256=digest(path))


def check_cached_records(rows, catalog, specs, records, label):
    assert len(records)==len(specs)==27
    for spec, record in zip(specs, records):
        p = pack(rows, catalog, spec); event = p['event']; q = p['query']
        assert (record['event'], record['floor'], record['K']) == (event['id'], event['floor'], event['K'])
        assert record['context_rows']==sum(len(h['rss']) for h in p['histories'].values())+len(p['support']['rss'])
        assert record['APs']==len(p['aps'])
        with np.load(REFERENCE/f'{label}_{event["id"]}.npz') as cached:
            assert np.array_equal(cached['truth'], q['xy'])
            assert np.array_equal(cached['query_rows'], p['query_rows'])
            assert np.array_equal(cached['weights'], query_weights(p))
            for key in ('raw', 'projected'):
                errors = np.linalg.norm(cached[key]-q['xy'], axis=1)
                assert np.allclose(errors, cached[key+'_errors'], rtol=0, atol=1e-10)
                assert abs(float(query_weights(p)@errors)-record['mde'][key])<1e-10
        if label=='initial':
            assert abs(float(query_weights(p)@np.asarray(record['WKNN']['errors']))-record['WKNN']['mde'])<1e-10


def comparison(records, initial, limited):
    current = aggregate(records, lambda r:r['mde']['projected'])
    controls = dict(public=aggregate(initial, lambda r:r['mde']['projected']),
                    limited_step12=aggregate(limited, lambda r:r['mde']['projected']),
                    WKNN=aggregate(initial, lambda r:r['WKNN']['mde']))
    gains = {name:dict(mean=1-current['mean']/base['mean'],
                      by_floor={f:1-v/base['by_floor'][f] for f, v in current['by_floor'].items()},
                      by_K={k:1-v/base['by_K'][k] for k, v in current['by_K'].items()})
             for name, base in controls.items()}
    source_pass = bool(gains['public']['mean']>=.03
                       and all(v>=0 for v in gains['public']['by_floor'].values())
                       and all(v>=0 for v in gains['public']['by_K'].values()))
    return dict(full=current, controls=controls, relative_gains=gains, source_passed=source_pass,
                WKNN_no_floor_worse=all(v>=0 for v in gains['WKNN']['by_floor'].values()),
                limited_comparison='Descriptive: old run has 12 restricted updates at lr .001; not a capacity-only ablation')


def run(wall_seconds):
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    catalog = json.loads((CATALOG/'catalog.json').read_text())
    prior = json.loads((REFERENCE/'freeze.json').read_text())
    prior_complete = json.loads((REFERENCE/'complete.json').read_text())
    assert prior_complete['complete'] and prior['seed']==SEED
    assert json.loads((CATALOG/'complete.json').read_text())['complete']
    assert json.loads((VERIFIED/'complete.json').read_text())['complete']
    precision = json.loads((VERIFIED/'precision.json').read_text())
    assert precision['loss_difference']<1e-6 and precision['gradient_max_difference']<1e-5
    schedule, specs = events_for(catalog)
    assert schedule[:12]==prior['schedule'] and specs==prior['validation']
    names = ['tabicl_multitask_full_trial.py', 'source_episode_catalog.py', 'tabicl_history.py',
             'tabicl_history_learning.py', 'check_tabicl_history_learning.py', 'acquisition_tasks.py',
             'uji.py', 'recomposition_trial.py', 'recomposition_multibudget_natural.py']
    for name in names[1:]:
        assert digest(ROOT/name)==prior['code'][name], name
    verified_freeze = json.loads((VERIFIED/'freeze.json').read_text())
    for name in ('tabicl_history.py', 'tabicl_history_learning.py'):
        assert digest(ROOT/name)==verified_freeze['code'][name]
    package = {str(p.relative_to(PACKAGE)):digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package and package==prior['package']==verified_freeze['package']
    for path in (CATALOG/'catalog.json', CATALOG/'freeze.json', CSV, PLAN, CHECKPOINT):
        assert digest(path)==prior['inputs'][str(path)], str(path)
    initial = json.loads((REFERENCE/'initial.json').read_text())
    limited = json.loads((REFERENCE/'trained.json').read_text())
    assert len(initial)==len(limited)==len(specs)==27
    paths = [CSV, PLAN, CHECKPOINT, CATALOG/'catalog.json', CATALOG/'freeze.json', CATALOG/'complete.json',
             REFERENCE/'freeze.json', REFERENCE/'complete.json', REFERENCE/'initial.json', REFERENCE/'trained.json',
             VERIFIED/'freeze.json', VERIFIED/'complete.json', VERIFIED/'precision.json']
    paths += [REFERENCE/f'{label}_{make_episode(catalog, **spec)["id"]}.npz'
              for label in ('initial', 'trained') for spec in specs]
    freeze = dict(created=now(), seed=SEED, code={n:digest(ROOT/n) for n in names},
                  inputs={str(p):digest(p) for p in paths}, package=package,
                  runtime={n:version(n) for n in ('torch', 'numpy', 'scikit-learn')},
                  schedule=schedule, validation=specs,
                  optimizer=dict(name='AdamW', lr=1e-5, weight_decay=.01, clip=1.,
                                 joint_updates=STEPS, episodes_per_update=3, initial_amp_scale=1024., retries=0),
                  objective='Equal-floor gradient average; both coordinate axes and all 8 official views per episode',
                  selection='Steps 12 and 24 recorded; step 24 alone is formal. No best checkpoint, continuation or target evaluation.',
                  criterion='Public-initial relative mean gain >=3%; no floor or K mean worsens. WKNN and restricted learning reported separately.',
                  initial_reuse=dict(reference=str(REFERENCE), check_event=make_episode(catalog, **specs[0])['id'],
                                     prediction_absolute_tolerance_m=PREDICTION_ATOL, relative_tolerance=0.),
                  boundaries=['Previously viewed B0 source positions; not blind or cross-building evidence',
                              'Pinball objective differs from Euclidean MDE; no exact AP/rotation equivariance or K monotonicity',
                              'All historical rows allowed by the protected-position catalog; no history/AP pruning',
                              'Direct conditional predictor, not an implementation of ReFi spatial transforms',
                              'Restricted comparator differs in trainable parameters, steps and learning rate'],
                  resources=dict(cpu=2, gpu=1, wall_seconds=wall_seconds, downloads=0,
                                 max_gpu_fraction=.8, checkpoints='Steps 12/24 stay on server'))
    write(OUT/'freeze.json', freeze)
    for name in names:(OUT/name).write_bytes((ROOT/name).read_bytes())
    rows = load_acquisition_rows(CSV, set(catalog['source_floors']))
    check_cached_records(rows, catalog, specs, initial, 'initial')
    check_cached_records(rows, catalog, specs, limited, 'trained')
    assert aggregate(initial, lambda r:r['mde']['projected'])==prior_complete['source']['initial']
    assert aggregate(limited, lambda r:r['mde']['projected'])==prior_complete['source']['trained']
    model = EpisodicHistoryRegressor(device='cuda:0')
    assert model.enable_recompute()
    all_parameters = dict(model.loaded.model_.named_parameters())
    parameters = [p for p in all_parameters.values() if p.requires_grad]
    # The released model stores eight fixed RoPE frequencies as a Parameter.
    # Preserve its architecture; "full" means all intended learnable weights.
    fixed_names = [name for name, p in all_parameters.items() if not p.requires_grad]
    assert fixed_names == ['row_interactor.tf_row.rope.freqs']
    fixed_frequencies = all_parameters[fixed_names[0]].detach().clone()
    assert sum(p.numel() for p in all_parameters.values()) == 28544991
    assert sum(p.numel() for p in parameters) == 28544983
    write(OUT/'parameters.json', dict(total=28544991, trainable=28544983,
                                    architecture_fixed=fixed_names, fixed_count=8))
    original = state_hash(model.loaded.model_)
    assert original==prior_complete['original_state_sha256']
    first = pack(rows, catalog, specs[0])
    check = score(model, first, 'initial_check')
    with np.load(OUT/f'initial_check_{check["event"]}.npz') as new, np.load(REFERENCE/f'initial_{check["event"]}.npz') as old:
        differences = {k:float(np.max(np.abs(new[k]-old[k]))) for k in ('raw', 'projected')}
    assert max(differences.values())<=PREDICTION_ATOL, differences
    assert state_hash(model.loaded.model_)==original
    write(OUT/'initial_reuse.json', dict(verified=True, events=27, seed=SEED,
                                       checked_event=check, max_prediction_difference_m=differences,
                                       original_state_sha256=original))
    write(OUT/'initial.json', initial); write(OUT/'limited_reference.json', limited)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    training = []; curve = []; checkpoint_hashes = {}
    torch.cuda.reset_peak_memory_stats()
    for step, batch in enumerate(schedule, 1):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        losses = []; events = []
        for spec in batch:
            p = pack(rows, catalog, spec); fit(model, p)
            losses.append(model.backward(p['query']['rss'], p['query']['xy'], scaler))
            events.append(dict(id=p['event']['id'], context_rows=model.estimators[0].n_samples_in_,
                               APs=len(p['aps']), query_rows=len(p['query_rows'])))
        scaler.unscale_(optimizer)
        for parameter in parameters:
            assert parameter.grad is not None
            parameter.grad.div_(3)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
        assert norm>0
        old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
        assert scaler.get_scale()>=old_scale, 'AMP overflow; stopped without retry'
        assert int(optimizer.state[parameters[0]]['step'])==step
        assert all_parameters[fixed_names[0]].grad is None
        assert torch.equal(all_parameters[fixed_names[0]], fixed_frequencies)
        training.append(dict(step=step, events=events, losses=losses, mean_pinball=float(np.mean(losses)),
                             gradient_norm=norm, amp_scale=scaler.get_scale(), seconds=time.monotonic()-tick,
                             peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20))
        write(OUT/'training.json', training)
        progress('trained', completed_updates=step, completed_episodes=3*step, seconds=time.monotonic()-started)
        if step in (12, 24):
            optimizer.zero_grad(set_to_none=True)
            current_hash = state_hash(model.loaded.model_)
            checkpoint = OUT/f'full_step_{step}.ckpt'
            model.save(checkpoint, source_updates=step,
                       provenance=dict(freeze_sha256=digest(OUT/'freeze.json'), source_floors=catalog['source_floors'],
                                       source_only=True, protected_positions=True, converged=False))
            checkpoint_hashes[str(step)] = digest(checkpoint)
            write(OUT/'checkpoints.json', checkpoint_hashes)
            restored = EpisodicHistoryRegressor(checkpoint, device='cuda:0')
            assert state_hash(restored.loaded.model_)==current_hash
            label = f'step_{step}'; records = []
            for spec in specs:
                records.append(score(restored, pack(rows, catalog, spec), label))
                write(OUT/f'{label}.json', records)
                progress('validation', step=step, completed=len(records), total=27,
                         completed_updates=step, seconds=time.monotonic()-started)
            assert state_hash(restored.loaded.model_)==current_hash
            del restored
            torch.cuda.empty_cache()
            summary = comparison(records, initial, limited)
            curve.append(dict(step=step, formal=step==24, result=summary,
                              state_sha256=current_hash, checkpoint_sha256=checkpoint_hashes[str(step)]))
            write(OUT/'learning_curve.json', curve)
    assert len(training)==24 and [r['step'] for r in curve]==[12, 24]
    final = dict(complete=True, created=now(), seconds=time.monotonic()-started,
                 source_updates=24, source_episodes=72, formal_step=24, source=curve[-1]['result'],
                 original_state_sha256=original, trained_state_sha256=state_hash(model.loaded.model_),
                 checkpoint_sha256=checkpoint_hashes, peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20,
                 target_opened=False, target_updates=0, converged=False, blind_test=False,
                 freeze_sha256=digest(OUT/'freeze.json'), training_sha256=digest(OUT/'training.json'))
    assert final['trained_state_sha256']==curve[-1]['state_sha256']!=original
    write(OUT/'final.json', final); write(OUT/'complete.json', final)
    progress('complete', completed_updates=24, completed_episodes=72, seconds=time.monotonic()-started)


def deadline(signum, frame):
    raise TimeoutError('Frozen total wall-time budget exhausted; no automatic continuation')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wall-seconds', type=int, choices=(1800,), default=1800)
    args = parser.parse_args()
    assert args.wall_seconds>0 and socket.gethostname()=='ray-web' and torch.cuda.is_available()
    assert not OUT.exists(), 'Fresh frozen run only; no implicit resume'
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.8, 0)
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.wall_seconds)
    try:
        run(args.wall_seconds)
    except Exception as error:
        if OUT.exists():
            write(OUT/'failure.json', dict(created=now(), complete=False, error=repr(error),
                                          timeout=isinstance(error, TimeoutError), automatic_resume=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
