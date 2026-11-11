"""One complete strict54 comparison of the unified next-scan field.

All old H/S/Q, GP caches, original probability reader, five-model averaging,
half pool and hierarchical metrics are reused. The field uses unique H-union-S
positions and learned distributions everywhere; no empirical-S append. No query
coordinates enter prediction. All54 predictions are saved before scoring.
This exposed UJI comparison asks whether source task organisation preserves
existing ability; it is not independent transfer evidence or automatic adoption.
"""
import hurdle_gp_pool_trial as R  # CPU2 and offline environment before numpy
import argparse
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from candidates.unified_scan_field_20261111 import UnifiedScanField, RECIPE


def run(models, output):
    started = time.monotonic()
    baseline_path = R.ROOT/'results/hurdle_gp_pool_20261110/complete.json'
    assert R.digest(baseline_path) == R.BASELINE_SHA
    baseline = R.read(baseline_path)
    assert baseline['complete']
    manifest = R.read(R.ASSETS/'manifest.json')
    assert R.digest(R.ASSETS/'manifest.json') == R.MANIFEST_SHA
    assert len(manifest['events']) == 54
    frozen = R.read(models/'freeze.json')
    source = R.read(models/'complete.json')
    assert frozen['recipe'] == source['recipe'] == RECIPE
    assert frozen['code']['candidates/unified_scan_field_20261111.py'] == R.digest(R.ROOT/'candidates/unified_scan_field_20261111.py')
    assert source['complete'] and source['freeze_sha256'] == R.digest(models/'freeze.json')
    gp = R.read(R.GP_RECORDS)
    assert gp['complete']
    gp_records = {r['event']:r for r in gp['records']}
    base_records = {r['event']:r for r in baseline['records']}
    csv = Path(manifest['csv_path'])
    assert R.digest(csv) == manifest['csv_sha256']
    inputs = [baseline_path, R.ASSETS/'manifest.json', R.GP_RECORDS,
              models/'freeze.json', models/'complete.json']
    output.mkdir(parents=True,exist_ok=False)
    R.write(output/'freeze.json',dict(created=R.now(),recipe=frozen['recipe'],
        source_training=frozen,inputs={str(p):R.digest(p) for p in inputs},
        code={n:R.digest(R.ROOT/n) for n in (*R.CODE,'candidates/unified_scan_field_20261111.py','candidates/finite_observation_20261111/features.py',str(Path(__file__).relative_to(R.ROOT)))},
        contract=dict(events=manifest['events'],methods=list(R.METHODS),
            changed='Unified learned distributions at unique H/S positions, complete local S statistics, coherent finite-observation source tasks and uniform cell sampling',
            fixed='Original reader, source estimator settings and five seeds, H/S/Q information, GP scores, half pool and metrics',
            question='Does the complete unified information-flow recipe improve the existing system without a separate empirical-S path?',
            target_tuning=False,adopted=False,source_buildings=[0],
            evaluation_status='Previously exposed strict54; retention evidence only',
            ordering='All prediction files saved before query coordinates are used in scoring',
            resources=dict(CPU_threads=2,GPU=0,new_fits=0,timeout_seconds=600))))
    os.environ['LAW_UJI_CSV'] = str(csv)
    sys.path.insert(0,str(R.PACKAGE))
    import core as C
    assert C.MAN['events'] == manifest['events']
    model = UnifiedScanField(models)
    records = []
    for event in manifest['events']:
        def scans(ids):
            rows = np.asarray(ids,dtype=int)-1
            return dict(rss=C.RSS[rows],xy=C.XY[rows])
        histories = {n:scans(manifest['history_rows'][n]) for n in event['histories']}
        field = model.prepare(histories,scans(event['support_rows']),event['floor'])
        old = base_records[event['id']]
        assert R.digest(old['prediction_path']) == old['prediction_sha256']
        with np.load(old['prediction_path']) as saved:
            assert np.array_equal(saved['query_rows'],event['query_rows'])
            assert np.array_equal(saved['aps'],field.aps)
            assert np.array_equal(np.unique(saved['cand_xy'],axis=0),field.cand_xy)
        prediction = field.predict(C.RSS[np.asarray(event['query_rows'])-1])
        gr = gp_records[event['id']]
        assert R.digest(gr['gp_path']) == gr['gp_sha256']
        with np.load(gr['gp_path']) as g:
            assert np.array_equal(g['query_rows'],event['query_rows'])
            assert np.array_equal(g['aps'],field.aps)
            fused = R.fuse_hurdle_gp(field.cand_xy,prediction['log_position_mass'],
                                    g['grid'],g['scores'],g['raw'],field.bounds)
        pred = fused['predictions']
        pred['hurdle'] = prediction['projected']
        with np.load(old['prediction_path']) as saved:
            assert np.allclose(pred['GP'],saved['GP'],rtol=0,atol=1e-8)
            # GP_candidates is intentionally recomputed on the deduplicated set.
        path = output/f"{event['id']}_unscored.npz"
        np.savez_compressed(path,**pred,query_rows=event['query_rows'],aps=field.aps,cand_xy=field.cand_xy)
        records.append(dict(event=event['id'],floor=event['floor'],K=event['K'],
            prediction_path=str(path),prediction_sha256=R.digest(path)))
        print(f'Predicted {len(records)}/54',flush=True)
    R.write(output/'predictions_complete.json',dict(complete=True,records=records))
    for event, record in zip(manifest['events'],records):
        truth = C.XY[np.asarray(event['query_rows'])-1]
        with np.load(record['prediction_path']) as pred:
            errors = {n:np.linalg.norm(pred[n]-truth,axis=1) for n in R.METHODS}
        path = output/f"{event['id']}_scored.npz"
        np.savez_compressed(path,truth=truth,**errors)
        record.update(scored_path=str(path),scored_sha256=R.digest(path))
    summary = R.summarize(manifest,records)
    R.write(output/'complete.json',dict(complete=True,created=R.now(),seconds=time.monotonic()-started,
        recipe=frozen['recipe'],freeze_sha256=R.digest(output/'freeze.json'),records=records,summary=summary,
        baseline_sha256=R.BASELINE_SHA,baseline_summary=baseline['summary'],
        comparison=R.compare_summaries(baseline['summary'],summary),adopted=False,
        source_fits=0,GP_fits=0,local_adaptation_search=False))
    print('Complete: all54 predictions and scores',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web'
    with threadpool_limits(limits=2):
        run(args.models.resolve(),args.output.resolve())
