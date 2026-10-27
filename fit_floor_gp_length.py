"""Source-only finite-grid fit of a conventional floor correlation length."""
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from data import group_scans
from floor_gp import floor_kernel
from inference import infer
from run import ap_stats, tensor
from run_multi_history import digest, write
from run_operator_prior import TRAIN_FLOORS
from uji_probability_study import load_raw, DATA

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/floor_gp_source_fit_20261026'
LENGTHS = [.1, .25, .5, 1., 2., 4., 8., 1e6]


def main():
    OUT.mkdir(exist_ok=False)
    parameter_path = ROOT / 'results/uji_history_bank_v1_20261026/parameters.json'
    parameter = json.loads(parameter_path.read_text()); base, obs = parameter['base'], parameter['obs']
    raw = load_raw('trainingData.csv', {f'B{b}F{f}' for b, fl in TRAIN_FLOORS.items() for f in fl})
    rng = np.random.default_rng(261030); cases = []; manifest = []
    for building, floors in TRAIN_FLOORS.items():
        records = []; row_ids = {}; origin = np.concatenate([raw[f'B{building}F{f}']['positions'] for f in floors]).mean(0)
        for floor in floors:
            name = f'B{building}F{floor}'; data = raw[name]
            groups = rng.permutation(len(data['positions']))[:32]
            indices = np.flatnonzero(np.isin(data['group'], groups))
            row_ids[name] = data['row_ids'][indices].tolist()
            records.append(dict(floor=floor, scans=group_scans(data['rss'][indices], data['xy'][indices]-origin)))
        visible = (np.concatenate([r['scans']['rss'] for r in records]) != 100).any(0)
        aps = sorted(rng.choice(np.flatnonzero(visible), 16, replace=False).tolist())
        xyz = torch.cat([torch.cat((tensor(r['scans']['positions']),
            torch.full((len(r['scans']['positions']),1),float(r['floor']))),1) for r in records])
        for ap in aps:
            per_floor = [ap_stats(r['scans'],ap) for r in records]
            stats = {k:torch.cat([s[k] for s in per_floor]) for k in per_floor[0]}
            cases.append(dict(building=building, AP=ap, xyz=xyz, stats=stats))
        manifest.append(dict(building=building, row_ids=row_ids, APs=aps, positions=len(xyz), origin=origin.tolist()))
    files = [Path(__file__), ROOT/'floor_gp.py', ROOT/'inference.py',ROOT/'data.py', ROOT/'uji_probability_study.py']
    write(OUT/'freeze.json',dict(created_unix=time.time(),scope=__doc__,manifest=manifest,
        lengths=LENGTHS, units='floor index', VI_max_iterations=640, base=base, obs=obs,
        selection='Mean negative joint ELBO per scan, equal weight per AP/building; no target query/target support used',
        limits='Finite grid, variational approximation, source subset; independent APs, not full MOGPAug or an optimized kernel-family comparison',
        code_sha256={str(p):digest(p) for p in files},inputs={str(p):digest(p) for p in [parameter_path,DATA/'trainingData.csv']}))
    (OUT/'code').mkdir()
    for p in files:shutil.copy2(p,OUT/'code'/p.name)
    rows=[]; started=time.perf_counter()
    with (OUT/'scores.jsonl').open('x') as stream:
        for length in LENGTHS:
            for case in cases:
                xyz=case['xyz'];cov=floor_kernel(xyz,xyz,base,length)
                posterior=infer(xyz.new_full((len(xyz),),base['mean']),cov,case['stats'],obs,steps=640)
                score=posterior['loss'].item()/case['stats']['n'].sum().item()
                if not np.isfinite(score):raise RuntimeError('Nonfinite source score')
                row=dict(length=length,building=case['building'],AP=case['AP'],negative_elbo_per_scan=score)
                rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            mean=float(np.mean([r['negative_elbo_per_scan'] for r in rows if r['length']==length]))
            print(json.dumps(dict(length=length,mean_negative_elbo=mean,seconds=time.perf_counter()-started)),flush=True)
    scores={str(l):float(np.mean([r['negative_elbo_per_scan'] for r in rows if r['length']==l])) for l in LENGTHS}
    chosen=min(LENGTHS,key=lambda l:scores[str(l)])
    write(OUT/'selected.json',dict(completed=True,floor_length=chosen,scores=scores,source_cases=len(cases),
        seconds=time.perf_counter()-started,freeze_sha256=digest(OUT/'freeze.json'),target_evaluations=0))


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    main()
