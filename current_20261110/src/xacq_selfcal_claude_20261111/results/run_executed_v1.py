"""Frozen A self-calibration: UJI retention then exposed IPIN mechanism check."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
import sys
from pathlib import Path
ROOT = Path('/home/panyushuo/projects/panyushuo/ReFi-UJI-Core-20261031')
sys.path.insert(0, str(ROOT))
import json
import time
import hashlib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from selfcal import fit_calibration, apply_calibration, identity_check
from frozen_hurdle_translator import FrozenHurdleTranslator, PreparedFrozenHurdle
import ipin2025_confirmation as I
from cached_retained_readout import score_cached_fields
from hurdle_gp_pool import fuse_hurdle_gp
from ipin2025_score import score

OUT = Path(__file__).resolve().parent/'results'
MODES = ('baseline', 'all', 'detection', 'intensity')
KEYS = ('cand_xy', 'aps', 'p', 'm', 's2', 'bounds', 'ap_count')

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def field_load(path):
    with np.load(path) as a:
        return PreparedFrozenHurdle(*(a[key] for key in KEYS))

def aggregate(samples, mode):
    values = np.concatenate([s[mode] for s in samples])
    weights = np.concatenate([np.full(len(s[mode]), s['weight']) for s in samples])
    weights /= weights.sum()
    order = np.argsort(values)
    return dict(mean=float(weights@values), p90=float(values[order[np.searchsorted(np.cumsum(weights[order]), .9)]]))

def uji(model):
    manpath = ROOT/'results/building_information_assets_20261109/manifest.json'
    assert digest(manpath) == 'bb7db6eb8ce070a400f18057af46dc68f734f5db6f60235f4c60189e62d3c496'
    man = read(manpath)
    assert digest(man['csv_path']) == man['csv_sha256']
    data = pd.read_csv(man['csv_path'])
    rss = data[[c for c in data if c.startswith('WAP')]].to_numpy(float)
    xy = data[['LONGITUDE', 'LATITUDE']].to_numpy(float)
    def arrays(ids):
        rows = np.asarray(ids)-1
        return dict(rss=rss[rows], xy=xy[rows])
    cache = read(ROOT/'results/uji54_count_conditioned_20261111/complete.json')
    fields = {r['event']:r for r in cache['fields']}
    old = {r['event']:r for r in read(ROOT/'results/hurdle_gp_pool_20261110/complete.json')['records']}
    samples, calibrations = [], {}
    for event in man['events']:
        floor = event['floor']
        if floor not in calibrations:
            histories = {n:arrays(man['history_rows'][n]) for n in event['histories']}
            calibrations[floor] = fit_calibration(model, histories, R=3)
            write(OUT/'uji_calibrations.json', calibrations)
        record = fields[event['id']]
        assert digest(record['path']) == record['sha256']
        field = field_load(record['path'])
        query = arrays(event['query_rows'])
        ns = len(np.unique(arrays(event['support_rows'])['xy'], axis=0))
        predictions = {mode:(field if mode == 'baseline' else apply_calibration(field, calibrations[floor], ns, mode)).predict(query['rss'])['projected']
                       for mode in ('baseline', 'all')}
        oldpath = ROOT/old[event['id']]['prediction_path']
        assert digest(oldpath) == old[event['id']]['prediction_sha256']
        with np.load(oldpath) as saved:
            assert np.max(np.abs(predictions['baseline']-saved['hurdle'])) < 1e-8
        _, group = np.unique(query['xy'], axis=0, return_inverse=True)
        group = group.ravel()
        count = np.bincount(group)
        sample = dict(event=event, weight=.5/sum(f.startswith(event['building']) for f in man['floors'])/2/3/len(count))
        for mode in predictions:
            error = np.linalg.norm(predictions[mode]-query['xy'], axis=1)
            sample[mode] = np.bincount(group, weights=error)/count
        samples.append(sample)
        np.savez_compressed(OUT/f"uji_{event['id']}.npz", **predictions, truth=query['xy'], query_rows=event['query_rows'])
        print('UJI', len(samples), event['id'], flush=True)
    result = {mode:dict(overall=aggregate(samples, mode),
        by_K={str(k):aggregate([s for s in samples if s['event']['K']==k], mode) for k in (1,3,5)},
        by_floor={f:aggregate([s for s in samples if s['event']['floor']==f], mode) for f in man['floors']}) for mode in ('baseline','all')}
    assert abs(result['baseline']['overall']['mean']-9.4325) < .00005
    result['P1'] = dict(delta=result['all']['overall']['mean']-result['baseline']['overall']['mean'],
        beta_in_range=all(.7<=c['beta']<=1.3 for c in calibrations.values()))
    write(OUT/'uji_summary.json', result)

def history_only(manifest, floor):
    # Deliberately do not call context(): its S-derived AP-axis is unavailable to calibration.
    raw = {}
    for name, record in manifest['histories'].items():
        if int(name)==floor:
            continue
        assert digest(record['path'])==record['sha256']
        with np.load(record['path']) as a:
            raw['IPIN2025_B2025F'+name] = {k:a[k] for k in ('rss','xy','ap_ids')}
    aps = sorted(set().union(*(set(h['ap_ids']) for h in raw.values())))
    columns = {ap:i for i,ap in enumerate(aps)}
    result = {}
    for name,h in raw.items():
        rss = np.full((len(h['rss']),len(aps)),100.)
        for j,ap in enumerate(h['ap_ids']):
            rss[:,columns[ap]]=h['rss'][:,j]
        result[name]=dict(xy=h['xy'],rss=rss)
    return result

def ipin(model):
    base = ROOT/'results/ipin2025_confirmation_20261110'
    manifest = read(I.ASSETS/'manifest.json')
    calibrations = {str(f):fit_calibration(model, history_only(manifest,f), R=1) for f in range(2,7)}
    write(OUT/'ipin_calibrations.json', calibrations)
    contexts = read(base/'contexts_complete.json')
    freeze = read(base/'freeze.json')
    oldpred = read(base/'predictions.json')
    oldmap = {(r['floor'],r['K'],r['log_id']):r for r in oldpred['predictions']}
    records = {mode:[] for mode in MODES}
    for mode in MODES:
        directory = OUT/'ipin'/mode
        write(directory/'freeze.json', dict(freeze, selfcal_status='exposed post-hoc mechanism test', selfcal_mode=mode))
        write(directory/'contexts_complete.json', contexts)
    streams = {log['id']:I.read_query_stream(log['query_path']) for log in freeze['logs']}
    for record in contexts['contexts']:
        field, gp, aps = I.load_prepared(record, torch.device('cpu'))
        # Original IPIN has R=1 and exactly K distinct S positions.
        variants = {m:field if m=='baseline' else apply_calibration(field,calibrations[str(record['floor'])],record['K'],m) for m in MODES}
        for log,stream in streams.items():
            used = stream['used_snapshot_indices']
            query = I.rss_matrix([stream['snapshots'][i] for i in used],aps)
            predictions = {m:{method:[] for method in I.METHODS} for m in MODES}
            for begin in range(0,len(query),16):
                q = query[begin:begin+16]
                g = score_cached_fields(gp,q)
                for mode,h in variants.items():
                    hp=h.predict(q)
                    fused=fuse_hurdle_gp(h.cand_xy,hp['log_position_mass'],gp['grid'].cpu().numpy(),g['scores'],g['projected'],h.bounds)
                    for method in I.METHODS:
                        predictions[mode][method].append(fused['predictions'][method])
            expand=np.searchsorted(used,stream['snapshot_indices'])
            prior=oldmap[(record['floor'],record['K'],log)]
            assert digest(prior['path'])==prior['sha256']
            with np.load(prior['path']) as old:
                for mode in MODES:
                    pred={m:np.concatenate(p)[expand] for m,p in predictions[mode].items()}
                    for method in (I.METHODS if mode=='baseline' else ('GP','GP_candidates')):
                        assert np.max(np.abs(pred[method]-old[method]))<1e-7,(mode,record['event'],method)
                    assert np.array_equal(old['times'],stream['prediction_times'])
                    path=OUT/'ipin'/mode/f"{record['event']}_log{log}.npz"
                    np.savez_compressed(path,times=old['times'],snapshot_times=old['snapshot_times'],**pred)
                    records[mode].append(dict(prior,path=str(path),sha256=digest(path)))
                    write(path.parent/'progress.json',records[mode])
            print('IPIN',len(records['baseline']),record['event'],log,flush=True)
    # Seal every mode and every stream before any GT read.
    for mode in MODES:
        directory=OUT/'ipin'/mode
        write(directory/'predictions.json',dict(oldpred,predictions=records[mode],freeze_sha256=digest(directory/'freeze.json'),contexts_complete_sha256=digest(directory/'contexts_complete.json')))
    summaries={}
    for mode in MODES:
        directory=OUT/'ipin'/mode
        result=score(directory/'predictions.json')
        write(directory/'scores.json',result)
        summaries[mode]={key:value for key,value in result.items() if key!='points'}
    for k,expected in zip(('1','3','5'),(37.2106,37.1091,36.0987)):
        assert abs(summaries['baseline']['by_K'][k]['methods']['hurdle']['mean']-expected)<.00005
    write(OUT/'ipin_summary.json',summaries)

if __name__=='__main__':
    import socket
    assert socket.gethostname()=='ray-web'
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    OUT.mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        write(OUT/'identity_check.json',identity_check())
        model=FrozenHurdleTranslator(I.SOURCE)
        assert model.freeze['source_buildings']==[0] or model.freeze['source_buildings']==['B0']
        if sys.argv[1]=='uji': uji(model)
        elif sys.argv[1]=='ipin': ipin(model)
        else: raise ValueError(sys.argv[1])
