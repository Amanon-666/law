"""Exact reuse of query-independent missing-AP likelihoods.

For a fixed prepared H/S field, sum_a log p(r_a|x) equals
sum_a log p(missing_a|x) + sum_detected_a [log p(r_a|x)-log p(missing_a|x)].
Only summation order changes. No AP removal, likelihood tempering, posterior
approximation, model fitting or protocol change. Each prepared dictionary is
immutable apart from this cache; a new H/S context gets a new dictionary.

The optional prediction entry reuses the original frozen stream runner. Its
explicit execution record is written before Query access and bound into the
completed prediction manifest. Existing frozen preparation code is untouched.
"""
import argparse
import socket
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from observations import observation_log_prob


@torch.no_grad()
def score_cached_fields(prepared, qrss):
    points = prepared['grid']
    if '_missing_log_likelihood' not in prepared:
        missing = torch.stack([observation_log_prob(
            100., prepared['mean'][j], prepared['variance'][j], prepared['obs'])
            for j in range(len(prepared['aps']))])
        total = points.new_zeros(len(points))
        for row in missing:
            total += row
        prepared['_missing_log_likelihood'] = missing
        prepared['_all_missing_scores'] = total
    rss = torch.as_tensor(np.asarray(qrss)[:, prepared['aps']],
                          dtype=torch.float64, device=points.device)
    scores = prepared['_all_missing_scores'].expand(len(rss), -1).clone()
    detected = rss != 100
    for column in torch.nonzero(detected.any(0), as_tuple=False).flatten().tolist():
        rows = torch.nonzero(detected[:, column], as_tuple=False).flatten()
        likelihood = observation_log_prob(rss[rows, column, None],
            prepared['mean'][column][None], prepared['variance'][column][None], prepared['obs'])
        scores[rows] += likelihood - prepared['_missing_log_likelihood'][column][None]
    if not torch.isfinite(scores).all():
        raise RuntimeError('Invalid retained complete scan scores')
    raw = (scores.softmax(1) @ points).cpu().numpy() + prepared['origin']
    return dict(scores=scores.cpu().numpy(), raw=raw,
        projected=np.clip(raw, prepared['bounds'][0], prepared['bounds'][1]),
        diagnostics=prepared['diagnostics'])


def run(args):
    import ipin2025_confirmation as trial
    trial.verify_freeze(args.output)
    context = trial.read(args.output/'contexts_complete.json')
    assert context['complete']
    execution_path = args.output/'cached_readout_execution.json'
    binding = dict(source=trial.binding(Path(__file__).resolve()),
        preparation_freeze_sha256=trial.digest(args.output/'freeze.json'),
        contexts_complete_sha256=trial.digest(args.output/'contexts_complete.json'),
        identity='all missing sum + detected likelihood differences; original observation_log_prob',
        model_updates=0, parameters_changed=False, device=args.device)
    if execution_path.exists():
        assert trial.read(execution_path)['binding'] == binding
    else:
        assert not (args.output/'predictions_progress.json').exists()
        trial.write(execution_path, dict(created=trial.now(), binding=binding,
            query_accessed_by_this_entry=False, GT_opened=False))
    with patch.object(trial, 'score_retained_fields', score_cached_fields):
        code = trial.predict_streams(args)
    if code == 0:
        path = args.output/'predictions.json'
        manifest = trial.read(path)
        manifest['readout_execution'] = trial.binding(execution_path)
        trial.write(path, manifest)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=780.)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 780
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    raise SystemExit(run(args))
