"""Post-hoc original A field/readout distinction on the saved F2 K1 22 points."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import datetime
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from frozen_hurdle_translator import PreparedFrozenHurdle, BETA, EPS, OFFSETS
from ipin2025_observation_data import read_query_stream, rss_matrix, match_prediction_times, REF_LAT, REF_LON

started = time.monotonic()
assert socket.gethostname() == 'ray-web'
BASE = ROOT/'results/ipin2025_confirmation_20261110'
inputs = {}

def bind(path, expected=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:
        assert digest == expected
    inputs[str(path)] = digest
    return path

def read(path, expected=None):
    return json.loads(bind(path, expected).read_text())

scores = read(BASE/'scores.json', '50716b22d18af368b1ffbbe9c48f345d24a1dac5335b5792f0d8b216a387ef04')
manifest = read(BASE/'predictions.json', '10fad9325ec3b0ee08837624ee7e0f4f67ce32feb64fb9499b30e7c17a297026')
contexts = read(BASE/'contexts_complete.json', 'de03679c06d5b3f22b7b5d67c057e78971beb5ce2391cf2289af6fb21370a1f8')
record = next(r for r in contexts['contexts'] if r['floor'] == 2 and r['K'] == 1)
context = read(record['path'], record['sha256'])
field = bind(context['hurdle']['path'], context['hurdle']['sha256'])
with np.load(field, allow_pickle=False) as cached:
    h = PreparedFrozenHurdle(**{k: cached[k].copy() for k in ('cand_xy','aps','p','m','s2','bounds','ap_count')})
points = [r for r in scores['points'] if r['floor'] == 2 and r['K'] == 1]
assert len(points) == 22 and all(r['covered'] for r in points)
rows = []
centroid = h.cand_xy.mean(0)
for log in manifest['logs']:
    selected = [r for r in points if r['log_id'] == log['id']]
    stream = read_query_stream(bind(log['query_path'], log['query_sha256']))
    prediction = next(r for r in manifest['predictions'] if r['floor'] == 2 and r['K'] == 1 and r['log_id'] == log['id'])
    with np.load(bind(prediction['path'], prediction['sha256']), allow_pickle=False) as cached:
        times, receipts, old_xy = (cached[k].copy() for k in ('times','snapshot_times','hurdle'))
    assert np.array_equal(times, stream['prediction_times'])
    indices = np.array([r['prediction_index'] for r in selected])
    assert np.array_equal(indices, match_prediction_times(times, [r['time'] for r in selected]))
    snapshot_indices = stream['snapshot_indices'][indices]
    query_records = [stream['snapshots'][i] for i in snapshot_indices]
    assert np.array_equal(receipts[indices], [r['app_timestamp'] for r in query_records])
    result = h.predict(rss_matrix(query_records, context['ap_ids']))
    assert np.allclose(result['projected'], old_xy[indices], rtol=0, atol=1e-9)
    gt = np.loadtxt(bind(log['gt_path'], log['gt_sha256']), delimiter=',', ndmin=2)
    for j, point in enumerate(selected):
        truth = gt[point['gt_row']]
        assert truth[0] == point['time'] and truth[3] == 2
        xy = np.array([(truth[1]-REF_LON)*111320*np.cos(np.radians(REF_LAT)), (truth[2]-REF_LAT)*111320])
        logmass = result['log_position_mass'][j]
        mass = np.exp(logmass)
        distances = np.linalg.norm(h.cand_xy-xy, axis=1)
        nearest = np.flatnonzero(np.isclose(distances, distances.min(), rtol=0, atol=1e-10))
        ranks = 1 + np.sum(logmass[:,None] > logmass[nearest][None,:], axis=0)
        mode = int(np.argmax(logmass))
        mean_error = float(np.linalg.norm(result['projected'][j]-xy))
        assert abs(mean_error-point['errors']['hurdle']) <= 1e-8
        rows.append(dict(log_id=log['id'],gt_row=point['gt_row'],GT_time=point['time'],prediction_index=point['prediction_index'],
            tick_time=float(times[indices[j]]),receipt_time=float(receipts[indices[j]]),snapshot_index=int(snapshot_indices[j]),
            truth_xy=xy.tolist(),mean_xy=result['projected'][j].tolist(),mode_xy=h.cand_xy[mode].tolist(),
            mean_error=mean_error,mode_error=float(distances[mode]),centroid_error=float(np.linalg.norm(centroid-xy)),
            nearest_distance=float(distances.min()),nearest_ties=len(nearest),nearest_rank_best=int(ranks.min()),
            nearest_rank_worst=int(ranks.max()),nearest_mass=float(mass[nearest].sum()),
            nearest_log_score_gap=float(logmass[mode]-logmass[nearest].max()),
            max_mass=float(mass.max()),top10_mass=float(np.sort(mass)[-10:].sum()),
            entropy=float(-(mass*logmass).sum()),effective_candidates=float(1/(mass@mass))))

def summary(name):
    values = np.array([r[name] for r in rows])
    return dict(mean=float(values.mean()),median=float(np.median(values)),minimum=float(values.min()),
                maximum=float(values.max()),p90=float(np.quantile(values,.9,method='inverted_cdf')))

assert abs(summary('mean_error')['mean']-86.93037455044409) < 1e-8
code = {str(ROOT/n): hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
        ('frozen_hurdle_translator.py','ipin2025_observation_data.py','ipin2025_score.py')}
code[str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
output = dict(complete=True,created=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-started,
    status='Post-hoc mechanism diagnostic only, not a method performance trial',floor=2,K=1,points=22,
    candidate_count=len(h.cand_xy),ap_count=int(h.ap_count),scored_APs=len(h.aps),centroid_xy=centroid.tolist(),
    reader=dict(BETA=BETA,EPS=EPS,OFFSETS=OFFSETS.tolist(),score='Original temperature-scaled log scores, normalized by candidate logsumexp',
        rank='1 plus number of candidates with strictly greater original log score, original candidate multiplicities retained',
        nearest='All tied geometrically nearest candidates, best/worst score ranks retained'),
    summary={name:summary(name) for name in ('mean_error','mode_error','centroid_error','nearest_distance','nearest_rank_best',
        'nearest_rank_worst','nearest_log_score_gap','max_mass','top10_mass','entropy','effective_candidates')},
    inputs=inputs,code=code,rows=rows,model_loads=0,fits=0,HS_preparations=0)
Path(__file__).with_name('result.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:output[k] for k in ('seconds','candidate_count','summary')},indent=2))
