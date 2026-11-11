"""Two exact information counterexamples before replacing the empirical-S path.

For a one-position support set and fixed H, original55 encodes each AP's
sentinel-filled scan mean. Detection frequency and detected strength can
trade off while every feature stays identical. Finite75 additionally exposes
counts/detection rates, but still cannot distinguish equal-mean supports with
different within-position variance. Any deterministic model on these features
has identical output for each colliding pair, irrespective of retraining.

Therefore neither representation can losslessly replace the empirical-S
statistics already used by the reader. This rejects direct replacement using
only the current learned inputs, not unified probability fields in general.
The required n, nd, sum and sum-of-squares already exist in raw H/S and in
finite_observation.position_stats. They are sufficient for a working iid
Bernoulli/Gaussian hurdle likelihood at one position; that model identity is
not an independence guarantee for real repeated scans or a transfer result.

No source or target files, model weights, localization score or fitting are
used. Synthetic cases demonstrate input information loss only.
"""
import claude_hurdle_assets  # CPU2/offline environment before numpy
import argparse
import hashlib
import json
from pathlib import Path
import socket

import numpy as np

from candidates.acquisition_tasks_20261111 import context
from candidates.finite_observation_20261111.features import build_context
from frozen_hurdle_translator import _support_fields


def check(output):
    histories = {0:dict(xy=np.array([[0.,0.],[10.,0.],[20.,0.]]),
                       rss=np.full((3,1),-65.))}
    cases = {
        'detection_strength_alias': ([-45.,-45.,100.],[-65.,-65.,-65.]),
        'variance_alias': ([-55.,-65.,-75.],[-65.,-65.,-65.]),
    }
    results = {}
    for name, rows in cases.items():
        supports = [dict(xy=np.zeros((3,2)),rss=np.array(r)[:,None]) for r in rows]
        original = [context(histories,s,1) for s in supports]
        expanded = [build_context(histories,s,1) for s in supports]
        same55 = np.array_equal(original[0][2],original[1][2],equal_nan=True)
        same75 = np.array_equal(expanded[0]['features'],expanded[1]['features'],equal_nan=True)
        direct = [_support_fields(s['rss'],s['xy']) for s in supports]
        assert same55
        assert same75 == (name == 'variance_alias')
        assert any(not np.array_equal(direct[0][i],direct[1][i]) for i in (1,2,3))
        results[name] = dict(scans=list(rows),same_original55=same55,same_finite75=same75,
            original55_max_finite_difference=float(np.nanmax(abs(original[0][2]-original[1][2]))),
            empirical=[dict(detection_probability=d[1].ravel().tolist(),
                            detected_mean=d[2].ravel().tolist(),variance=d[3].ravel().tolist()) for d in direct],
            available_statistics=[{k:v.tolist() for k,v in e['support_stats'].items()} for e in expanded])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(dict(complete=True,phase='synthetic_information_counterexample',cases=results,
            conclusion='Do not replace empirical S with current55/75 learned inputs while claiming to preserve its observation information',
            recovery='Use the already available raw sufficient statistics in a unified conditioning/learning path',
            fits=0,models_loaded=0,dataset_files_read=0,localization_scores=0,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(complete=True,cases=results)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web'
    check(args.output)
