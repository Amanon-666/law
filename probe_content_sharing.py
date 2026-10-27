"""Source-cache diagnostic, not training or a localization experiment.

Before running: at the already fitted scalar pooling probability, differentiate
the existing complete-scan conditional NLL with respect to one logit per AP.
Opposite signs are necessary local evidence for heterogeneous sharing, not proof
that H predicts it. Repeated tasks on the same floor are not independent domains.
No target evaluation files are read and no parameters are updated.
"""
import hashlib
import json
from pathlib import Path
import shutil
import socket
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

from partial_source_pooling import partial_ap_log_prob, scan_log_prob
from source_weight_mixture import mixture_log_prob


def score(logits, z, lp, mass, weights):
    ls, li = F.logsigmoid(-logits)[:, None], F.logsigmoid(logits)[:, None]
    independent_z = torch.logsumexp(z + mass.log(), 1)
    evidence = torch.logaddexp(ls + z, li + independent_z[:, None])
    gamma = torch.log_softmax(mass.log() + evidence.sum(0), 0)
    shared = ls + z - evidence
    local = li + independent_z[:, None] - evidence
    independent = torch.log_softmax(z + mass.log(), 1)
    joint = torch.zeros_like(lp[0])
    for a in range(len(z)):
        joint = joint + partial_ap_log_prob(lp[a], shared[a], local[a], independent[a])
    return -(weights * mixture_log_prob(joint, gamma)).sum()


def main():
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    source = root / 'results/uji_partial_pooling_full_scan_v1_20261026'
    out = root / 'results/content_sharing_probe_20261026'
    out.mkdir(exist_ok=False)
    freeze = json.loads((source / 'freeze.json').read_text())
    events = json.loads((source / 'source_manifest.json').read_text())['events']
    lam = json.loads((source / 'source_fit.json').read_text())['unpooled_probability']
    paths = [Path(__file__), root / 'partial_source_pooling.py', root / 'source_weight_mixture.py',
             source / 'freeze.json', source / 'source_manifest.json', source / 'source_fit.json']
    paths += [source / 'source' / f'task{e["id"]:02}' / 'fields.npz' for e in events]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    registration = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose=__doc__, fitted_scalar=lam, parameter_updates=0,
        interpretation='Mixed stable gradients motivate a content-predictability check only; no performance or novelty claim',
        source_sha256={str(p): digest(p) for p in paths})
    (out / 'freeze.json').write_text(json.dumps(registration, indent=2))
    shutil.copy2(__file__, out / Path(__file__).name)
    mass = torch.tensor(freeze['prior_mass'])
    records, differences = [], []
    grouped = {}
    for event in events:
        task = event['id']
        with np.load(source / 'source' / f'task{task:02}' / 'fields.npz') as a:
            z, lp, weights = [torch.tensor(a[k]) for k in ['log_z', 'log_prob', 'weights']]
            aps = a['ap_columns'].tolist()
        logits = torch.full((len(z),), float(np.log(lam / (1 - lam))), requires_grad=True)
        loss = score(logits, z, lp, mass, weights)
        old = -(weights * scan_log_prob(z, lp, mass, lam)).sum()
        torch.testing.assert_close(loss, old, rtol=1e-11, atol=1e-11)
        gradient, = torch.autograd.grad(loss, logits)
        if task == events[0]['id']:
            for i in torch.topk(gradient.abs(), min(3, len(z))).indices.tolist():
                step = torch.zeros_like(logits); step[i] = 1e-4
                fd = (score(logits.detach()+step, z, lp, mass, weights)
                      - score(logits.detach()-step, z, lp, mass, weights)) / 2e-4
                torch.testing.assert_close(fd, gradient[i], rtol=1e-4, atol=1e-7)
                differences.append(float(abs(fd-gradient[i])))
        g = gradient.detach().numpy()
        for ap, value in zip(aps, g):
            grouped.setdefault((event['floor'], tuple(sorted(event['histories'])), ap), []).append(float(value))
        records.append(dict(task=task, floor=event['floor'], APs=aps, gradient=g.tolist(),
            loss=float(loss.detach()), scalar_replay_difference=float(abs(loss-old).detach()),
            positive_mass=float(g[g>0].sum()), negative_mass=float(-g[g<0].sum())))
    stability = []
    for (floor, histories, ap), values in grouped.items():
        if len(values) < 2:
            continue
        v = np.array(values)
        # Magnitude-weighted agreement: zero means cancellation, one no sign reversal.
        stability.append(dict(floor=floor, histories=histories, ap=ap, repeats=len(v),
            absolute_mass=float(abs(v).sum()), signed_sum=float(v.sum()),
            agreement=float(abs(v.sum())/abs(v).sum()) if abs(v).sum() else None))
    total = sum(r['absolute_mass'] for r in stability)
    summary = dict(completed=True, tasks=len(records), floors=sorted({e['floor'] for e in events}),
        positive_gradient_mass=sum(r['positive_mass'] for r in records),
        negative_gradient_mass=sum(r['negative_mass'] for r in records),
        repeated_AP_floor_groups=len(stability),
        magnitude_weighted_repeat_agreement=sum(abs(r['signed_sum']) for r in stability)/total if total else None,
        max_finite_difference_error=max(differences), parameter_updates=0)
    (out / 'results.json').write_text(json.dumps(dict(summary=summary, tasks=records, repeats=stability), indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
