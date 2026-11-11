"""Same unified77/count recipe on strict54, reusing the complete UJI runner.

No new training, threshold, pool weight or dataset-specific reader. Source
models and every H/S/query row stay fixed. The old raw coordinates computed
inside the count reader must reproduce the completed unified-raw baseline.
All54 predictions precede scoring, as in the parent runner. This implements
the retention comparison; preparing this file does not imply it was run.
"""
from candidates import unified_scan_retention_20261111 as U
from candidates.count_conditioned_20261111.reader import CountConditionedHurdle
import argparse
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

R = U.R
RECIPE = 'unified_scan_field_77_v1_count'


def run(models,output):
    baseline_path = R.ROOT/'results/uji54_unified_scan_20261111/complete.json'
    baseline = R.read(baseline_path)
    assert baseline['complete'] and len(baseline['records']) == 54
    events = R.read(R.ASSETS/'manifest.json')['events']
    records = {r['event']:r for r in baseline['records']}
    prepared,checks = [],{}
    real_write = R.write

    class Reader(CountConditionedHurdle):
        def predict(self,query):
            result = super().predict(query)
            event = prepared[-1];record = records[event]
            assert R.digest(record['prediction_path']) == record['prediction_sha256']
            with np.load(record['prediction_path']) as old:
                difference = float(np.max(abs(result['original_projected']-old['hurdle'])))
                assert difference <= 1e-8 and np.array_equal(old['cand_xy'],self.cand_xy)
            checks[event] = difference
            return result

    class Translator(U.UnifiedScanField):
        def prepare(self,histories,support,target_floor):
            event = events[len(prepared)]
            assert event['floor'] == target_floor
            prepared.append(event['id'])
            return Reader(super().prepare(histories,support,target_floor))

    def write(path,value):
        if path == output/'freeze.json':
            value['recipe'] = RECIPE
            value['inputs'][str(baseline_path)] = R.digest(baseline_path)
            value['contract'].update(
                changed='Only unified-raw -> existing exact count-conditioned reader',
                fixed='Same frozen77 source models, H/S/Q, unique candidates, GP, half pool and metrics',
                question='Does the same count-conditioned unified recipe retain UJI gains?',
                review=__doc__)
            for p in (Path(__file__),R.ROOT/'candidates/count_conditioned_20261111/reader.py'):
                value['code'][str(p.relative_to(R.ROOT))] = R.digest(p)
        elif path == output/'complete.json':
            assert len(checks) == 54
            value.update(recipe=RECIPE,unified_raw_parity=checks,
                unified_raw_reference=dict(path=str(baseline_path),sha256=R.digest(baseline_path)),
                unified_raw_comparison=R.compare_summaries(baseline['summary'],value['summary']))
        real_write(path,value)

    with patch.object(U,'UnifiedScanField',Translator),patch.object(R,'write',write):
        U.run(models,output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web'
    with threadpool_limits(limits=2):
        run(args.models.resolve(),args.output.resolve())
