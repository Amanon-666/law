"""Continue ONLY unfinished selected support objectives; preserve first budget.

This is a numerical follow-up after inspecting optimization statuses. Query
scores neither select cases nor determine continuation or stopping.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

import run_tcms_history_bank as study
import score_tcms_area as scoring
from data import group_scans
from run_multi_history import digest, write
from tcms_stack_bridge import SourcePair, fit


ORIGINAL = study.OUT
CONTINUED = ORIGINAL / 'continued_support_fit'


def prepare():
    CONTINUED.mkdir(exist_ok=False)
    write(CONTINUED/'freeze.json',dict(created_unix=time.time(),post_hoc=True,
        reason='Only 9 of 24 selected coordinate models reported convergence under the original five-start budget',
        selection='Continue every selected model whose optimizer_converged is false; no Query criterion',
        initialization='Original lowest support group-LOO loss parameters',
        objective='Same whole-position LOO loss, bounds, kernel, labels and source set',
        extra_budget='One continuation, max 200 optimizer iterations / 1000 function calls; no new random starts',
        area_rule='Same previously specified H+S rectangle projection for all methods; raw results also retained',
        source_freeze_sha256=digest(ORIGINAL/'freeze.json'),
        code_sha256={str(p):digest(p) for p in [Path(__file__),Path(study.__file__),Path(scoring.__file__)]}))
    shutil.copy2(ORIGINAL/'manifest.json',CONTINUED/'manifest.json')
    shutil.copy2(__file__,CONTINUED/'continuation_code.py')


def run(variant):
    study.OUT=CONTINUED
    for meta in json.loads((CONTINUED/'manifest.json').read_text())['events']:
        name=meta['id']; destination=CONTINUED/variant/name;destination.mkdir(parents=True,exist_ok=True)
        if (destination/'completed.json').exists(): continue
        sources=study.VARIANTS[variant]
        floors=study.load_raw('trainingData.csv',set(sources+[meta['floor']]+study.SOURCES))
        f=floors[meta['floor']];lookup={int(r):i for i,r in enumerate(f['row_ids'])}
        sr=np.array([lookup[r] for r in meta['support_rows']]);aps=meta['ap_columns']
        support=group_scans(f['rss'][sr],f['xy'][sr])
        known=np.concatenate([floors[s]['xy'] for s in study.SOURCES]);center,scale=known.mean(0),known.std(0)
        device=torch.device('cuda:0')
        def features(rss): return torch.tensor((np.where(rss[:,aps]==100,-110,rss[:,aps])+110)/110.,device=device)
        sx=features(support['rss']);hx=[features(floors[s]['rss']) for s in sources]
        for axis in range(2):
            path=destination/f'axis{axis}.json'
            if path.exists(): continue
            old_path=ORIGINAL/variant/name/f'axis{axis}.json';old=json.loads(old_path.read_text())
            chosen=old['diagnostics']['runs'][old['diagnostics']['chosen_restart']]
            if chosen['optimizer_converged']:
                shutil.copy2(old_path,path);continue
            ys=[torch.tensor((floors[s]['xy'][:,axis]-center[axis])/scale[axis],device=device) for s in sources]
            yt=torch.tensor((support['xy'][:,axis]-center[axis])/scale[axis],device=device)
            pairs=[SourcePair(x,sx,y,yt,support['group']) for x,y in zip(hx,ys)]
            for pair,parameters in zip(pairs,old['parameters']):
                # Reuse the existing optimizer with the actual saved starting point.
                pair.initial=lambda rng,p=np.array(parameters):p.copy()
            started=time.perf_counter();p,w,d=fit(pairs,261026+axis,restarts=1,evaluations=1000)
            write(path,dict(parameters=p.tolist(),weights=w.tolist(),diagnostics=d,
                seconds=time.perf_counter()-started,source_scan_counts=[len(x) for x in hx],target_scans=len(sx),
                continuation_of=str(old_path),old_support_objective=chosen['objective']))
            study.log(stage='continued_fit',event=name,variant=variant,axis=axis,
                before=chosen['objective'],after=d['runs'][0]['objective'],converged=d['runs'][0]['optimizer_converged'])
        study.event(name,variant)


if __name__=='__main__':
    if socket.gethostname()!='ray-web': raise RuntimeError('Server only')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','summarize']);p.add_argument('--variant',choices=list(study.VARIANTS))
    args=p.parse_args()
    if args.stage=='prepare': prepare()
    else:
        original_freeze=json.loads((ORIGINAL/'freeze.json').read_text())
        for path,sha in original_freeze['code_sha256'].items():
            if digest(Path(path))!=sha: raise RuntimeError(f'Original frozen code changed: {path}')
        frozen=json.loads((CONTINUED/'freeze.json').read_text())
        for path,sha in frozen['code_sha256'].items():
            if digest(Path(path))!=sha: raise RuntimeError(f'Continuation code changed: {path}')
        if args.stage=='run': run(args.variant)
        else:
            study.OUT=CONTINUED;study.summarize()
            scoring.OUT=CONTINUED;scoring.main()
