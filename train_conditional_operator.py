"""Matched bounded continuation: prior versus few-S-conditioned scan learning.

Both arms start at the same existing checkpoint and consume the exact same
source tasks/AP samples. No target query or target checkpoint selection.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from conditional_operator_learning import (OUT, ROOT, SOURCE, parameters,
    episode, proposal_for, loss_for)
from operator_prior import OperatorNet
from run_multi_history import digest, write

UPDATES, APS_PER_UPDATE, LR = 128, 4, 1e-4


def freeze():
    audit = json.loads((OUT / 'audit.json').read_text())
    assert audit['completed']
    prepared = json.loads((OUT / 'freeze.json').read_text())
    for section in ['code_sha256', 'inputs']:
        for path, expected in prepared[section].items():
            assert digest(Path(path)) == expected, path
    events = json.loads((OUT / 'manifest.json').read_text())['events']
    rng = np.random.default_rng(261029)
    schedule = []
    # Two complete shuffled traversals, not selection of easy source episodes.
    for cycle in range(2):
        for index in rng.permutation(len(events)):
            meta = events[index]
            fit = json.loads((OUT / meta['id'] / 'adaptation.json').read_text())
            schedule.append(dict(event=meta['id'], aps=rng.choice(fit['APs'], APS_PER_UPDATE, replace=False).tolist()))
    assert len(schedule) == UPDATES
    files = [Path(__file__), ROOT / 'conditional_operator_learning.py']
    paths = [OUT / 'manifest.json', OUT / 'audit.json', SOURCE / 'operator.pt']
    paths += [OUT / e['id'] / 'adaptation.json' for e in events]
    record = dict(created_unix=time.time(), hypothesis='Matching the few-position Support update during source learning improves use of the same historical corpus beyond continuing unconditional fitting',
        arms=['prior', 'conditional'], initialization=str(SOURCE / 'operator.pt'),
        updates=UPDATES, aps_per_update=APS_PER_UPDATE, learning_rate=LR, optimizer='Adam',
        schedule=schedule, source_positions_per_task=dict(S=5, Q=12), scans_per_position=3,
        reduction='Equal episode weight and uniform AP samples after full-AP S-only adaptation; same within-Q position/scans weights',
        numerical='Fresh detached M-PMC proposal per AP/update, 0.1 defensive component, ESS>=1000; self-normalized importance gradient',
        target_scope='No target evaluation in this training command. Final checkpoint only; B1/Tampere/GIA outcomes do not select updates or learning rate.',
        evaluation_needed='Compare frozen initial, prior continuation and conditional continuation under the SAME g-free S weights and localization backend; prior v1 refit results alone are not this control.',
        limitations='Conditional scan NLL is not distance loss. Fixed S-derived weights remain point estimates. Two source buildings; history count is variable but coordinate/AP correspondence is still required.',
        code_sha256={str(p): digest(p) for p in files}, inputs={str(p): digest(p) for p in paths})
    with (OUT / 'training_freeze.json').open('x') as f:
        json.dump(record, f, indent=2)
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)


def train(arm):
    record = json.loads((OUT / 'training_freeze.json').read_text())
    prepared = json.loads((OUT / 'freeze.json').read_text())
    for source in [prepared, record]:
        for section in ['code_sha256', 'inputs']:
            for path, expected in source[section].items():
                assert digest(Path(path)) == expected, path
    dest = OUT / arm; dest.mkdir(exist_ok=False)
    base, obs = parameters()
    net = OperatorNet().double()
    net.load_state_dict(torch.load(SOURCE / 'operator.pt', weights_only=True))
    optimizer = torch.optim.Adam(net.parameters(), lr=record['learning_rate'])
    events = {e['id']: e for e in json.loads((OUT / 'manifest.json').read_text())['events']}
    started = time.perf_counter()
    with (dest / 'training.jsonl').open('x') as stream:
        for step, entry in enumerate(record['schedule']):
            cases = episode(events[entry['event']])
            fit = json.loads((OUT / entry['event'] / 'adaptation.json').read_text())
            assert list(cases) == fit['APs']
            optimizer.zero_grad(); values, effective = [], []
            for index, ap in enumerate(entry['aps']):
                case = cases[ap]
                if arm == 'conditional':
                    samples, logq, info = proposal_for(net, case, fit['weights'], base, obs, 261029 + step*4 + index)
                    value, posterior = loss_for(net, case, fit['weights'], base, obs, samples, logq)
                    ess = float((1 / posterior.log_weights.exp().square().sum()).detach())
                    if ess < 1000:
                        raise RuntimeError(f'Insufficient differentiable posterior ESS: {ess}')
                    effective.append(ess)
                else:
                    value, _ = loss_for(net, case, fit['weights'], base, obs)
                if not torch.isfinite(value):
                    raise RuntimeError('Nonfinite source objective')
                if value.requires_grad:  # S-only new AP uses the unchanged local GP.
                    (value / len(entry['aps'])).backward()
                values.append(value.item())
            gradients = [p.grad for p in net.parameters() if p.grad is not None]
            if not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError('Nonfinite source gradient')
            norm = float(torch.sqrt(sum(g.square().sum() for g in gradients))) if gradients else 0.
            optimizer.step()
            row = dict(step=step, event=entry['event'], aps=entry['aps'], loss=float(np.mean(values)),
                min_ess=min(effective) if effective else None, gradient_norm=norm, seconds=time.perf_counter()-started)
            stream.write(json.dumps(row)+'\n'); stream.flush()
            if step % 8 == 0 or step == UPDATES-1:
                print(json.dumps(dict(arm=arm, **row)), flush=True)
    torch.save(net.state_dict(), dest / 'operator.pt')
    write(dest / 'completed.json', dict(completed=True, arm=arm, updates=UPDATES,
        seconds=time.perf_counter()-started, checkpoint_sha256=digest(dest / 'operator.pt'),
        freeze_sha256=digest(OUT / 'training_freeze.json'), target_evaluations=0))


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze', 'train'])
    parser.add_argument('--arm', choices=['prior', 'conditional'])
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    else:
        train(args.arm)
