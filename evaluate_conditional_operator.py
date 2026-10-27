"""Matched K5 development readout for initial/prior/conditional checkpoints.

All three use identical g-free Support weights. Reuses the audited evaluation
backend and exact mature-baseline fields; preserves all previous experiments.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

import evaluate_operator_paired as evaluation
from conditional_operator_learning import OUT as TRAINING, SOURCE
from run_multi_history import digest, write

OUT = TRAINING / 'evaluation'
BASELINE = evaluation.ROOT / 'results/uji_operator_paired_v2_20261026'
ARMS = ['initial', 'prior', 'conditional']


def checkpoint_root(arm):
    return SOURCE if arm == 'initial' else TRAINING / arm


def prepare():
    for arm in ['prior', 'conditional']:
        assert json.loads((TRAINING / arm / 'completed.json').read_text())['completed']
    old = json.loads((BASELINE / 'freeze.json').read_text())
    # Baseline reuse is valid only with the same numerical/model backend.
    for name in ['history_bank.py', 'reference_pmc.py', 'source_weight_mixture.py', 'inference.py', 'transfer.py']:
        path = evaluation.ROOT / name
        assert digest(path) == old['code_sha256'][str(path)], name
    files = [Path(__file__), evaluation.ROOT / 'evaluate_operator_paired.py',
             *[Path(p) for p in old['code_sha256'] if not p.endswith('evaluate_operator_paired.py')]]
    inputs = [Path(p) for p in old['inputs'] if not '/uji_operator_prior_v1_' in p]
    inputs += [TRAINING / 'training_freeze.json', TRAINING / 'audit.json']
    inputs += [checkpoint_root(a) / 'operator.pt' for a in ARMS]
    inputs += sorted(BASELINE.glob('B*/*.npz')) + sorted(BASELINE.glob('B*/*.json'))
    OUT.mkdir(exist_ok=False)
    record = dict(created_unix=time.time(), purpose=__doc__,
        comparison='Conditional versus prior continuation and frozen initial. Same checkpoint initialization, source episodes, AP update schedule and step count; only training conditioning changes between continued arms.',
        adaptation='Reuse mature full-AP S-only g-free weights in EVERY arm; network weights frozen. Same H/S/Q/AP/search-region access.',
        scope='Six old B1 K5R3 development events; J2 localization only. Variable-J source training does not itself prove variable-J target accuracy.',
        criterion='Support for conditional continuation requires lower mean and >=4/6 improved events versus BOTH prior and initial, separately for internal and official; otherwise partial/unsupported. No retuning from these results.',
        history_contribution='All arms and mature baseline retain the same historical fingerprints. This contrast tests how training uses them, not the total value of history.',
        code_sha256={str(p): digest(p) for p in files}, inputs={str(p): digest(p) for p in inputs})
    write(OUT / 'freeze.json', record)
    (OUT / 'code').mkdir()
    for p in files:
        shutil.copy2(p, OUT / 'code' / p.name)
    for arm in ARMS:
        dest = OUT / arm; dest.mkdir()
        write(dest / 'freeze.json', dict(record, checkpoint=digest(checkpoint_root(arm) / 'operator.pt'), arm=arm))


def run():
    frozen = json.loads((OUT / 'freeze.json').read_text())
    for section in ['code_sha256', 'inputs']:
        for path, expected in frozen[section].items():
            assert digest(Path(path)) == expected, path
    events = json.loads((evaluation.OLD / 'manifest.json').read_text())['events']
    for event in events:
        for arm in ARMS:
            evaluation.TRAIN = checkpoint_root(arm); evaluation.OUT = OUT / arm
            dest = evaluation.OUT / event['id']; dest.mkdir(exist_ok=True)
            if not (dest / 'adaptation.json').exists():
                original = evaluation.OLD / event['id'] / 'adaptation.json'
                weights = json.loads(original.read_text())['weights']
                # The existing backend reads this declared adaptation record;
                # it must not refit weights with the candidate's g/s.
                write(dest / 'adaptation.json', dict(weights=weights, baseline_weights=weights,
                    policy='Frozen g-free shared S-only fit, as in source episodes', source=str(original), source_sha256=digest(original)))
            print(json.dumps(dict(stage='event', arm=arm, event=event['id'])), flush=True)
            evaluation.event(event['id'], baseline_directory=BASELINE)
    summarize()


def summarize():
    events = json.loads((evaluation.OLD / 'manifest.json').read_text())['events']; rows = []
    for event in events:
        result = {a: json.loads((OUT / a / event['id'] / 'completed.json').read_text()) for a in ARMS}
        for split in ['internal', 'official']:
            def score(arm, method):
                return next(r['mde'] for r in result[arm]['results'] if r['split'] == split and r['method'] == method)
            baseline = score('initial', 'adapted_bank')
            assert all(score(a, 'adapted_bank') == baseline for a in ARMS)
            rows.append(dict(event=event['id'], floor=event['floor'], split=split, baseline=baseline,
                             **{a: score(a, 'operator_bank') for a in ARMS}))
    summary = {}
    for split in ['internal', 'official']:
        subset = [r for r in rows if r['split'] == split]
        comparisons = {}
        for reference in ['initial', 'prior', 'baseline']:
            delta = [r['conditional'] - r[reference] for r in subset]
            comparisons[reference] = dict(mean_change=float(np.mean(delta)), improved_events=sum(d < 0 for d in delta))
        summary[split] = dict(mean={a: float(np.mean([r[a] for r in subset])) for a in ['baseline', *ARMS]},
                              conditional_versus=comparisons)
    supported = all(summary[s]['conditional_versus'][r]['mean_change'] < 0 and
                    summary[s]['conditional_versus'][r]['improved_events'] >= 4
                    for s in ['internal', 'official'] for r in ['initial', 'prior'])
    write(OUT / 'summary.json', dict(summary=summary, records=rows, development_criterion_met=supported,
        limits='Previously viewed B1 development floors, K5R3, two histories at deployment; not independent confirmation or proof of novelty.'))
    print(json.dumps(dict(summary=summary, development_criterion_met=supported)), flush=True)


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'run', 'summarize'])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'summarize': summarize}[args.stage]()
