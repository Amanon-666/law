"""Paper extras: existing five-seed caches only; no source fitting."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
from . import run as entry
from .refi import ReFi
from frozen_hurdle_translator import FrozenHurdleTranslator
from law_bench.data import load
from law_bench.protocol import make_tasks
from law_bench.score import aggregate
from law_bench import ours
from vertical_translator import translator as T

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'results'
OUT = RESULTS/'paper_extras_20261111'
BUDGETS = [1, 3, 5, 10, 20]
METHODS = ['wknn_HS', 'gp_histmean', 'refi']

def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))

def csvwrite(name, rows):
    with (OUT/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def cache_only(self, sources):
    self.sources = [s['building'] for s in sources]
    identity = entry.run._data_identity({s['building']: s['floors'] for s in sources},
                                      {s['building']: s['ap_ids'] for s in sources})
    from . import refi
    src = ROOT/'src'
    code = {str(p.relative_to(src)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(refi.__file__), src/'frozen_hurdle_translator.py',
                      *sorted((src/'claude_hurdle_20261110').glob('*.py'))]}
    expected = dict(sources=self.sources, data=identity, code=code, seeds=[1,2,3,4,5],
                    draws=4, reader=dict(beta=128, eps=.02, offsets=True),
                    evaluation_status='exposed_development; reader selected on UJI B0')
    key = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    cache = self.cache_dir/key
    assert json.loads((cache/'provenance.json').read_text()) == expected
    paths = [cache/f'seed_{s}.joblib' for s in range(1,6)]
    assert all(p.is_file() for p in paths), 'Missing frozen seed: training forbidden'
    self.translator = FrozenHurdleTranslator.__new__(FrozenHurdleTranslator)
    self.translator.models = [joblib.load(p) for p in paths]
    self.provenance = expected
    print(json.dumps(dict(cache_only=True, sources=self.sources, cache=str(cache))), flush=True)
    return self

ReFi.fit_source = cache_only

def curves():
    for b in ('B0', 'B1', 'B2'):
        output = RESULTS/f'paper_uji_{b}_kcurve_20261111'
        args = argparse.Namespace(dataset='uji', building=b, protocol='static-resample-v2',
            manifest=None, root=None, source_dataset=None, source_root=None, source_buildings=None,
            floors=None, budgets=BUDGETS, repeats=[0,1,2], seed=None, R=None, query_positions=None,
            methods=METHODS, method_options=Path(__file__).with_name('options.json'),
            output=output, resume=output.exists())
        entry.run.evaluate(args)

def summarize():
    merged, pairs, tables = {}, [], []
    for method in METHODS:
        records = []
        for b in ('B0','B1','B2'):
            root = RESULTS/f'paper_uji_{b}_kcurve_20261111'
            rs = [json.loads(p.read_text()) for p in (root/'events'/method).rglob('*.json')]
            records.extend(rs + json.loads((root/'task_inventory.json').read_text())['unavailable'])
            index = {(r['floor'],r['repeat'],r['K']):r for r in rs}
            for (f,r,k), a in index.items():
                if k != 1: continue
                z = index[(f,r,5)]
                assert a['status'] == z['status'] == 'completed'
                assert a['query_row_indices'] == z['query_row_indices']
                assert a['query_groups'] == z['query_groups']
                for g,(e1,e5) in enumerate(zip(a['scores']['position_errors'], z['scores']['position_errors'])):
                    pairs.append(dict(method=method,building=b,floor=f,repeat=r,query_group=g,
                                      error_K1=e1,error_K5=e5,improved=int(e5<e1),delta=e1-e5))
        merged[method] = aggregate(records)
        for k,v in merged[method].items(): tables.append(dict(method=method,K=k,**v))
    csvwrite('kcurve_merged.csv', tables)
    csvwrite('k1_k5_paired_positions.csv', pairs)
    improvement = {m:dict(paired_positions=len(p:=[r for r in pairs if r['method']==m]),
                        improved=sum(r['improved'] for r in p),
                        fraction=float(np.mean([r['improved'] for r in p]))) for m in METHODS}
    write(OUT/'kcurve_merged.json', dict(by_method=merged, improvement=improvement,
        weighting='pooled position-repeat errors across floors/buildings; p90 is pooled quantile',
        pairing='identical query rows and groups within floor/repeat; repeats counted separately'))

def visual_and_timing():
    buildings, aps = load('uji')
    options = json.loads(Path(__file__).with_name('options.json').read_text())
    rng = np.random.default_rng(20261111)
    samples, selections, deployment = [], [], None
    for b, floors in buildings.items():
        model = ReFi(**options['refi']).fit_source([dict(dataset='uji',building=s,floors=d,ap_ids=aps[s])
                                                  for s,d in buildings.items() if s!=b])
        for f in sorted(floors):
            # Match original paper_uji K=3/r0, including its max budget=5 sampling.
            task, truth = make_tasks('uji',b,floors,aps[b],f,budgets=(1,3,5),repeat=0)[1]
            tick=time.perf_counter()
            prepared=model.translator.prepare({f'{b}F{hf}':d for hf,d in task.H.items()},task.S,f'{b}F{f}')
            prep_s=time.perf_counter()-tick
            if b=='B2' and f==4:
                tick=time.perf_counter(); prepared.predict(task.Q_rss[:1]); scan_s=time.perf_counter()-tick
                deployment=dict(building=b,floor=f,K=3,repeat=0,prepare_seconds=prep_s,
                                one_scan_seconds=scan_s,model_load_excluded=True,candidates=len(prepared.cand_xy))
            maps=ours.make_maps(task.H)
            n=len(np.unique(np.concatenate([v[0] for v in maps.values()]),axis=0))
            xy=prepared.cand_xy[:n]; a=prepared.aps
            copied=ours.naive_field(maps,f,a,xy,tau2=0.,sig2=0.)
            _,ref=T.history_features(xy,f,{k:v[:3] for k,v in maps.items()})
            P,N,nd,mu,_=ours.det_stats(floors[f]['rss'],floors[f]['xy'])
            rate=nd/N[:,None]
            D=np.linalg.norm(xy[:,None]-P[None],axis=2); j=D.argmin(1); dist=D.min(1)
            kind='edge' if f in (min(floors),max(floors)) else 'middle'
            # Keep a uniform global sample via independent random priorities (all modeled H cells).
            priority=rng.random((n,len(a))); flat=np.argsort(priority.ravel())[:20000]
            for ix in flat:
                row,col=divmod(int(ix),len(a)); ap=int(a[col]); valid=bool(dist[row]<=T.R_LABEL)
                p=float(prepared.p[row,col]); m=float(prepared.m[row,col])
                samples.append(dict(priority=float(priority[row,col]),building=b,floor=f,floor_type=kind,
                    x=float(xy[row,0]),y=float(xy[row,1]),ap=aps[b][ap],
                    history_strength=float(ref[row,ap]),refi_strength=p*m+(1-p)*T.ND,
                    true_strength=float(rate[j[row],ap]*np.nan_to_num(mu[j[row],ap],nan=0.)+(1-rate[j[row],ap])*T.ND) if valid else '',
                    truth_distance_m=float(dist[row]),truth_matched=valid,refi_p=p,refi_m=m,
                    true_p=float(rate[j[row],ap]) if valid else '',
                    true_m=float(mu[j[row],ap]) if valid and nd[j[row],ap]>0 else ''))
            if b=='B2' and f in (2,4):
                counts=(nd>0).sum(0)
                hist_counts=np.array([sum((v[2][:,ap]>0).sum() for v in maps.values()) for ap in range(len(aps[b]))])
                ap=sorted(a.tolist(),key=lambda ap:(-counts[ap],-hist_counts[ap],aps[b][ap]))[0]
                col=int(np.flatnonzero(a==ap)[0])
                selections.append(dict(building=b,floor=f,ap=aps[b][ap],target_detected_positions=int(counts[ap]),
                    history_detected_positions=int(hist_counts[ap]),rule='max target detected-position count; ties max history detected-position count, then AP name',
                    source_buildings=model.sources,K=3,repeat=0))
                csvwrite(f'{b}F{f}_candidates.csv',[dict(x=float(x[0]),y=float(x[1]),ap=aps[b][ap],
                    candidate_kind='history' if i<n else 'support',refi_p=float(prepared.p[i,col]),refi_m=float(prepared.m[i,col]),
                    copied_p=float(copied['p'][i,col]) if i<n else '',copied_m=float(copied['m'][i,col]) if i<n else '')
                    for i,x in enumerate(prepared.cand_xy)])
                csvwrite(f'{b}F{f}_truth.csv',[dict(x=float(x[0]),y=float(x[1]),ap=aps[b][ap],scans=int(N[i]),
                    detections=int(nd[i,ap]),true_p=float(rate[i,ap]),true_m=float(mu[i,ap]) if nd[i,ap]>0 else '') for i,x in enumerate(P)])
            print(json.dumps(dict(visual_floor=f'{b}F{f}',prepare_seconds=prep_s)),flush=True)
    samples=sorted(samples,key=lambda r:r['priority'])[:20000]
    for r in samples: del r['priority']
    csvwrite('history_refi_truth_20000.csv',samples)
    write(OUT/'visual_metadata.json',dict(selections=selections,rows=len(samples),seed=20261111,
        population='all history-candidate x H/S-detected AP cells, all 13 UJI floors; support candidates excluded',
        truth='nearest recorded target position within R_LABEL=4m; unmatched truth blank; exact distances exported',
        intensity='encoded dBm: p*m+(1-p)*(-105); history_strength is original translator history ref',
        protocol='original paper_uji budgets 1,3,5; K3/r0; target truth used only after prepare',
        scope='exposed development, not independent validation'))
    timing=[]
    for method in ('wknn_S','wknn_HS','gp_S','gp_histmean','refi'):
        seconds=[]; fits=[]
        for b in buildings:
            root=RESULTS/f'paper_uji_{b}_20261111'
            records=[json.loads(p.read_text()) for p in (root/'events'/method).rglob('*.json')]
            seconds.extend(r['seconds'] for r in records if r['status']=='completed')
            fits.append(json.loads((root/'source_models'/f'{method}.json').read_text())['seconds'])
        timing.append(dict(method=method,completed_tasks=len(seconds),mean_task_seconds=float(np.mean(seconds)),
                           mean_source_fit_stage_seconds=float(np.mean(fits)),total_source_fit_stage_seconds=sum(fits)))
    csvwrite('timing.csv',timing)
    write(OUT/'timing.json',dict(methods=timing,deployment=deployment,
        caveat='Original source-fit stage includes fit/load and serialization; not pure optimizer time. Concurrent batch load affects wall times. Single-scan timing is one observation.'))

def finalize_exports():
    """Complete copied fields at appended S candidates and expose original seed timing."""
    buildings, aps = load('uji')
    for f in (2,4):
        path=OUT/f'B2F{f}_candidates.csv'
        rows=list(csv.DictReader(path.open()))
        task,_=make_tasks('uji','B2',buildings['B2'],aps['B2'],f,budgets=(1,3,5),repeat=0)[1]
        ap=aps['B2'].index(rows[0]['ap'])
        xy=np.array([[float(r['x']),float(r['y'])] for r in rows])
        field=ours.naive_field(ours.make_maps(task.H),f,np.array([ap]),xy,tau2=0.,sig2=0.)
        for i,r in enumerate(rows):
            r.update(copied_p=float(field['p'][i,0]),copied_m=float(field['m'][i,0]))
        csvwrite(path.name,rows)
    timing=json.loads((OUT/'timing.json').read_text())
    seed_records=[]
    for b in ('B0','B1','B2'):
        for line in (RESULTS/f'paper_uji_{b}_20261111.log').read_text().splitlines():
            if line.startswith('{"refi_seed":'):
                seed_records.append(dict(heldout_building=b,**json.loads(line)))
    write(OUT/'original_refi_seed_timing.json',seed_records)
    totals={b:sum(r['seconds'] for r in seed_records if r['heldout_building']==b and r['state']=='trained')
            for b in ('B0','B1','B2')}
    timing['original_refi_training_seconds_by_fold']=totals
    timing['original_refi_mean_training_seconds']=float(np.mean(list(totals.values())))
    timing['training_timer_scope']='Original refi.py seed timers: field training plus joblib export; no new training in extras.'
    write(OUT/'timing.json',timing)

if __name__=='__main__':
    assert os.uname().sysname=='Linux'
    os.nice(10)
    OUT.mkdir(exist_ok=True)
    visual_and_timing()
    curves()
    summarize()
    finalize_exports()
    write(OUT/'complete.json',dict(complete=True,pid=os.getpid(),finished=time.time()))
