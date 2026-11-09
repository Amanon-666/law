"""Two fixed B1 development predictions, gated by completed B0 step 24.

Reuses existing tasks and frozen comparator predictions. No training, baseline
rerun, target-driven selection, or automatic extension. Server execution only.
"""
import json
import signal
import socket
import time

import torch

import tabicl_context_matching_trial as helper
from tabicl_history import CHECKPOINT, PACKAGE, state_hash
from tabicl_history_learning import EpisodicHistoryRegressor
from recomposition_trial import ROOT, PLAN, CSV, digest, write, now

SOURCE = ROOT/'results/tabicl_multitask_full_trial_20261107_checked'
OUT = ROOT/'results/tabicl_multitask_transfer_20261107'
TARGET = ROOT/'results/acquisition_retention_transfer_20261105'
PUBLIC = ROOT/'results/tabicl_history_transfer_20261105'
BALANCED = ROOT/'results/tabicl_balanced_source_trial_20261105'
SEED = 1150541
WALL_SECONDS = 120


@torch.no_grad()
def run():
    started = time.monotonic()
    # The source gate precedes output creation and every target-file read.
    source = json.loads((SOURCE/'complete.json').read_text())
    final = json.loads((SOURCE/'final.json').read_text())
    source_freeze = json.loads((SOURCE/'freeze.json').read_text())
    assert source==final and source['complete']
    assert source['formal_step']==24 and source['source_updates']==24 and source['source_episodes']==72
    assert source['source']['source_passed'] and source['target_opened'] is False
    assert digest(SOURCE/'freeze.json')==source['freeze_sha256']
    checkpoint = SOURCE/'full_step_24.ckpt'
    checkpoint_hash = digest(checkpoint)
    assert checkpoint_hash==source['checkpoint_sha256']['24']
    OUT.mkdir(exist_ok=False)

    balanced = json.loads((BALANCED/'freeze.json').read_text())
    assert digest(BALANCED/'freeze.json')==source_freeze['inputs'][str(BALANCED/'freeze.json')]
    target_freeze = json.loads((TARGET/'freeze.json').read_text())
    public_freeze = json.loads((PUBLIC/'freeze.json').read_text())
    public = json.loads((PUBLIC/'complete.json').read_text())
    public_records = json.loads((PUBLIC/'frozen.json').read_text())
    assert public['complete'] and json.loads((TARGET/'complete.json').read_text())['complete']
    events = target_freeze['events']
    assert events==public_freeze['events'] and len(events)==2
    assert [e['id'] for e in events]==['B1F1_acquisition_0', 'B1F1_acquisition_1']
    assert all(e['floor']=='B1F1' and e['role']=='validation' and e['K']==5 and e['R']==3
               and e['histories']==['B1F0', 'B1F3'] for e in events)
    assert public_records==public['events']['frozen']
    assert [r['event'] for r in public_records]==[e['id'] for e in events]
    assert helper.SEED==SEED==public_freeze['seed']
    assert helper.TARGET==TARGET and helper.FULL_TARGET==PUBLIC

    names = ['tabicl_multitask_transfer.py', 'tabicl_context_matching_trial.py',
             'tabicl_context_matching.py', 'check_tabicl_context_matching.py',
             'tabicl_history.py', 'tabicl_history_learning.py', 'check_tabicl_history_learning.py',
             'acquisition_tasks.py', 'uji.py', 'recomposition_trial.py']
    for name in names[1:]:
        assert digest(ROOT/name)==balanced['code'][name], name
        if name in source_freeze['code']:
            assert digest(ROOT/name)==source_freeze['code'][name], name
        if name in public_freeze['code']:
            assert digest(ROOT/name)==public_freeze['code'][name], name
    package = {str(p.relative_to(PACKAGE)):digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package and package==source_freeze['package']==public_freeze['package']==balanced['package']
    for path in (CSV, PLAN, CHECKPOINT):
        assert digest(path)==source_freeze['inputs'][str(path)]==public_freeze['inputs'][str(path)]
    target_paths = [TARGET/'freeze.json', TARGET/'evaluation.json']
    target_paths += [TARGET/f'{e["id"]}_episode.pt' for e in events]
    for path in target_paths:
        assert digest(path)==public_freeze['inputs'][str(path)], str(path)
    paths = [SOURCE/'complete.json', SOURCE/'final.json', SOURCE/'freeze.json', checkpoint,
             BALANCED/'freeze.json', CSV, PLAN, CHECKPOINT, TARGET/'complete.json',
             PUBLIC/'freeze.json', PUBLIC/'complete.json', PUBLIC/'frozen.json', *target_paths]
    paths += [PUBLIC/f'frozen_{e["id"]}.npz' for e in events]
    write(OUT/'freeze.json', dict(created=now(), seed=SEED, events=events,
        code={n:digest(ROOT/n) for n in names}, inputs={str(p):digest(p) for p in paths}, package=package,
        checkpoint=dict(path=str(checkpoint), sha256=checkpoint_hash, source_updates=24,
                        state_sha256=source['trained_state_sha256'], source_gate_passed=True),
        new_training_updates=0, target_updates=0,
        readout='Native mean projected to the existing H/S GP-grid rectangle; raw retained; position-balanced MDE, directions equally weighted',
        reuse='Existing public TabICL predictions and GP/retained GP/WKNN results; no baseline prediction or GP fitting',
        criterion='Both directions no worse than public and mean gain >=3%; base replacement additionally >=3% mean gain against GP/retained/WKNN',
        selection='Fixed completed source step 24 only; no alternative checkpoint, retry, training or additional target',
        limits='Previously exposed B1F1 development tasks; no blind, unseen-device, K-curve or ReFi spatial-transform claim',
        resources=dict(cpu=2, gpu=1, wall_seconds=WALL_SECONDS, max_gpu_fraction=.5, downloads=0)))
    for name in names:(OUT/name).write_bytes((ROOT/name).read_bytes())

    # Only now may the helper parse target RSS and reconstruct the old tasks.
    prepared, reference = helper.prepare(TARGET, False)
    model = EpisodicHistoryRegressor(checkpoint, device='cuda:0')
    model.enable_recompute()
    model.loaded.model_.requires_grad_(False).eval()
    before = state_hash(model.loaded.model_)
    assert before==source['trained_state_sha256']
    assert all(not p.requires_grad for p in model.loaded.model_.parameters())
    helper.OUT = OUT
    records = helper.evaluate(model, prepared, 'target', PUBLIC)
    assert state_hash(model.loaded.model_)==before
    assert len(records)==2
    for record, old in zip(records, public_records):
        assert record['event']==old['event']
        assert record['context_rows']==old['context_rows']==2331 and record['APs']==old['APs']==173
        assert abs(record['frozen']-old['mde']['projected'])<1e-10
    result = helper.summary(records, reference, 'retained')
    result['means']['full_model'] = result['means'].pop('matching')
    directions = [dict(event=r['event'], full_model=r['mde'], public=r['frozen'],
                       GP=ref['GP']['mde']['median'], retained=ref['retained']['mde']['median'],
                       WKNN=ref['WKNN']['mde']) for r, ref in zip(records, reference)]
    complete = dict(complete=True, created=now(), seconds=time.monotonic()-started,
        summary=result, directions=directions, records=records,
        source_checkpoint=str(checkpoint), source_checkpoint_sha256=checkpoint_hash,
        source_final_sha256=digest(SOURCE/'final.json'), source_state_sha256=before,
        model_state_unchanged=True, source_updates=24, new_training_updates=0, target_updates=0,
        target_tasks=2, public_repredicted=False, GP_refitted=False, blind_test=False,
        converged=False, freeze_sha256=digest(OUT/'freeze.json'),
        predictions={e['id']:digest(OUT/f'target_{e["id"]}.npz') for e in events},
        peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20)
    write(OUT/'complete.json', complete)
    print(json.dumps(complete), flush=True)


def deadline(signum, frame):
    raise TimeoutError('Frozen 120-second transfer budget exhausted; no automatic extension')


if __name__=='__main__':
    assert socket.gethostname()=='ray-web' and torch.cuda.is_available()
    assert not OUT.exists(), 'Fresh evaluation only; no implicit resume'
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.5, 0)
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
    try:
        run()
    except Exception as error:
        if OUT.exists():
            write(OUT/'failure.json', dict(created=now(), complete=False, error=repr(error),
                                           timeout=isinstance(error, TimeoutError), automatic_resume=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
