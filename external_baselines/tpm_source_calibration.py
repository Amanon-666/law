"""Source-only TPM calibration, with target floors excluded in source validation.

Uses the existing project estimator unchanged: three variance/mapping sweeps,
with explicit optimizer diagnostics. Neither success flags nor three sweeps
are a claim of a global optimum. No development-building predictions here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from uji_probability_study import load_raw, legacy_cells, load_legacy, DATA, LEGACY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/tpm_source_calibration_20261027'
FLOORS = tuple(f'B1F{i}' for i in range(4))
PAIRS = [(f'B1F{i}',f'B1F{j}') for i in range(4) for j in range(4) if abs(i-j)==1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare():
    OUT.mkdir(exist_ok=False)
    raw=load_raw('trainingData.csv',set(FLOORS))
    code=[Path(__file__),ROOT/'external_baselines/tpm_adapter.py',ROOT/'uji_probability_study.py',
          ROOT/'data.py',LEGACY/'cross_floor.py',LEGACY/'radio_map.py']
    write(OUT/'freeze.json',dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        purpose=__doc__, source_building='UJI_B1',excluded_development_buildings=['UJI_B0','UJI_B2','UAH2016-b20','Pisa2019-b60'],
        source_row_ids={name:raw[name]['row_ids'].tolist() for name in FLOORS},pairs=PAIRS,
        source_validation='For each held source floor, exclude that floor from observation fitting AND both ends of all transfer training pairs. Other floors can form H at validation deployment.',
        final_fit='After source selection, the all-B1 fit is used for four-building development evaluation; no development Q selects transfer or match parameters.',
        adaptation='Both directed adjacent pairs, rather than original upward-only training. Explicit protocol adaptation to histories on either side; no distance function fitted.',
        estimator='Existing fit_observation and fit_transfer(sweeps=3), including original bounds, start points, moments and pooled bins. No optimizer settings changed.',
        optimizer_policy='Save every optimizer termination, objective and projected gradient; any unsuccessful or nonfinite result marks the fit unavailable. Do not silently use its x.',
        reverse_review=['Only selected B1 rows enter sufficient statistics.',
            'Held source floor excluded from BOTH observations and directed-pair fitting.',
            'No target S or Q enters this calibration.',
            'K1 cannot estimate per-AP target rho/beta; transfer is frozen before target deployment.',
            'Strong-signal repeat cells and same-location detected RSS degrees of freedom were checked in source_eligibility.json.',
            'Fixed three alternating sweeps are an original estimator budget, not convergence of joint likelihood.',
            'GP base mean/amplitude/length and field/matcher selection are NOT fitted by this script.'],
        code_sha256={str(p):sha(p) for p in code},input_sha256={str(DATA/'trainingData.csv'):sha(DATA/'trainingData.csv')}))


def run():
    freeze=json.loads((OUT/'freeze.json').read_text())
    for section in ('code_sha256','input_sha256'):
        for path,digest in freeze[section].items():
            if sha(path)!=digest:raise ValueError(f'Changed frozen file: {path}')
    raw=load_raw('trainingData.csv',set(FLOORS))
    for name in FLOORS:
        assert raw[name]['row_ids'].tolist()==freeze['source_row_ids'][name]
    cells={name:legacy_cells(raw[name]) for name in FLOORS}
    radio,cross=load_legacy('radio_map'),load_legacy('cross_floor')
    original_minimize=cross.minimize
    completed=[]
    for held in (*FLOORS,None):
        name=held or 'all_B1'
        path=OUT/f'{name}.json'
        if path.exists():
            raise FileExistsError(f'Calibration already exists: {path}; inspect rather than rerun')
        training=[floor for floor in FLOORS if floor!=held]
        pairs=[(a,b) for a,b in PAIRS if held not in (a,b)]
        traces=[]
        def traced_minimize(fun,x0,*args,**kwargs):
            result=original_minimize(fun,x0,*args,**kwargs)
            gradient=np.asarray(result.jac,dtype=float).copy()
            for j,(low,high) in enumerate(kwargs.get('bounds') or [(None,None)]*len(gradient)):
                if low is not None and result.x[j]<=low+1e-7 and gradient[j]>0:gradient[j]=0
                if high is not None and result.x[j]>=high-1e-7 and gradient[j]<0:gradient[j]=0
            traces.append(dict(success=bool(result.success),status=int(result.status),message=str(result.message),
                               objective=float(result.fun) if np.isfinite(result.fun) else None,iterations=int(result.nit),evaluations=int(result.nfev),
                               projected_gradient_max=float(np.abs(gradient).max()) if np.isfinite(gradient).all() else None,
                               finite=bool(np.isfinite(result.fun) and np.isfinite(result.x).all() and np.isfinite(gradient).all())))
            return result
        cross.minimize=traced_minimize
        obs=radio.fit_observation([cells[floor] for floor in training])
        try:
            fit=cross.fit_transfer([(cells[a],cells[b]) for a,b in pairs],obs,sweeps=3)
        finally:
            cross.minimize=original_minimize
        fit={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fit.items()}
        acceptable=all(row['success'] and row['finite'] for row in traces) and all(np.isfinite(v).all() for v in fit.values())
        write(path,dict(held_source_floor=held,training_floors=training,pairs=pairs,
            observation_statistics=obs,transfer=fit if acceptable else None,optimizer_traces=traces,usable=acceptable,
            freeze_sha256=sha(OUT/'freeze.json'),scope='Source calibration only; no Q performance and no global convergence claim'))
        print(json.dumps(dict(fit=name,usable=acceptable,optimizer_calls=len(traces))),flush=True)
        if not acceptable:
            write(OUT/'execution.json',dict(state='stopped_unusable_fit',fit=name,completed=completed,
                                          reason='Source optimizer failure/nonfinite result; saved and not used.'))
            raise RuntimeError(f'Unusable source fit {name}; inspect recorded optimizer diagnostics')
        completed.append(name)
    write(OUT/'execution.json',dict(state='complete_source_fits',completed=completed,localization_predictions=0,
                                  source_selection_complete=False,HTTP_downloads=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run'])
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else run)()
