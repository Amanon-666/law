"""Regenerate the existing 16-floor H asset under isolated B1 parameters.

Retains original UJI scans and existing public coordinate-FPS rows. No target
S/Q sampling or scoring. A floor's own cache is never an allowed H in its
evaluation episode. Zero-detection fallback is retained and explicitly named.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from data import group_scans
import inference
from multi_history import single_history
from residual_probe.prepare import data_module
from source_base_calibration import OUT as SOURCE
from external_baselines.tpm_source_calibration import ROOT,sha,write

OUT=ROOT/'results/fusion_history_B1_20261028'
PARAMETERS=SOURCE/'parameters.json'
data_source=data_module()


def prepare():
    parameters=json.loads(PARAMETERS.read_text())
    assert parameters['accepted'] and parameters['source_building']=='UJI_B1'
    selections=[]
    inputs={PARAMETERS,SOURCE/'freeze.json'}
    for building in sorted(data_source.CONFIGS):
        data=data_source.building_data(building)
        for floor,raw in sorted(data['floors'].items()):
            if building.startswith('UJI_'):
                rows=np.arange(len(raw['rss']))
                inputs.add(data_source.DATA/'trainingData.csv')
                policy='All existing original historical floor scans, unchanged'
            else:
                path=data_source.PUBLIC_BANK/building/f'F{floor}'/'selection.json'
                rows=np.asarray(json.loads(path.read_text())['row_indices'])
                inputs.add(path)
                inputs.update(p for p in (data_source.PUBLIC_DATA/building).iterdir() if p.is_file())
                policy='Existing at-most-128-position, one-scan coordinate-FPS rows, unchanged'
            scans=group_scans(raw['rss'][rows],raw['xy'][rows])
            selections.append(dict(building=building,floor=floor,rows=rows.tolist(),
                scan_ids=raw['scan_ids'][rows].tolist(),AP_ids=data['ap_ids'].tolist(),
                origin=scans['positions'].mean(0).tolist(),positions=len(scans['positions']),
                scans=len(rows),observed_APs=int((scans['stats']['k']>0).any(0).sum()),policy=policy))
    OUT.mkdir(exist_ok=False)
    write(OUT/'manifest.json',dict(selections=selections))
    paths=[Path(__file__),Path(data_source.__file__),Path(data_source.load_building.__code__.co_filename),ROOT/'uji_probability_study.py',
           ROOT/'data.py',ROOT/'inference.py',ROOT/'multi_history.py',ROOT/'transfer.py',ROOT/'run.py']
    write(OUT/'freeze.json',dict(UTC=datetime.now(timezone.utc).isoformat(),purpose=__doc__,
        parameters_sha256=sha(PARAMETERS),base=parameters['base'],obs=parameters['obs'],
        VI_maxiter=640,max_projected_gradient=1e-3,
        absent_AP_policy='No detection anywhere on H floor -> None fallback, not a claim of no measurements. Count and retain metadata. No numerical failure may use this fallback.',
        input_rights='Each floor is cached only as a possible H for OTHER target floors. Target must be excluded explicitly from every evaluation history list. No cross-building AP identity assumed.',
        reverse_review=['Uses raw H rows, not old posterior arrays.',
            'Only newly accepted B1 base/obs enter fitting and saved posterior coordinates.',
            'Per-floor origin uses retained H positions only, and is saved for exact episode-frame translation.',
            'Public labels remain native metric coordinates; UJI remains original coordinate units.',
            'Failed AP fit stops its building; no partial-AP location scores or relaxed numerical thresholds.',
            'Empty-AP fallback retains the old policy so source-parameter and fusion comparisons remain interpretable.'],
        code_sha256={str(p):sha(p) for p in paths},
        input_sha256={str(p):sha(p) for p in [*inputs,OUT/'manifest.json']}))
    print(json.dumps(dict(stage='prepared',floors=len(selections),
        nonempty_AP_fits=sum(s['observed_APs'] for s in selections),
        uncompressed_covariance_bytes=sum(s['observed_APs']*s['positions']**2*8 for s in selections))),flush=True)


def run(building):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    frozen=json.loads((OUT/'freeze.json').read_text())
    for section in ('code_sha256','input_sha256'):
        for path,digest in frozen[section].items():
            if sha(path)!=digest:raise ValueError(f'Frozen input changed: {path}')
    data=data_source.building_data(building)
    selections=[s for s in json.loads((OUT/'manifest.json').read_text())['selections'] if s['building']==building]
    assert selections
    summaries=[]
    for meta in selections:
        floor=meta['floor'];raw=data['floors'][floor];rows=meta['rows']
        assert raw['scan_ids'][rows].tolist()==meta['scan_ids'] and data['ap_ids'].tolist()==meta['AP_ids']
        scans=group_scans(raw['rss'][rows],raw['xy'][rows]-np.asarray(meta['origin']))
        dest=OUT/building/f'F{floor}';dest.mkdir(parents=True,exist_ok=True)
        statuses=[]
        for ap in range(len(meta['AP_ids'])):
            cache=dest/f'ap{ap}.npz';status=dest/f'ap{ap}.json'
            if cache.exists() or status.exists():
                raise FileExistsError(f'Existing cache at {cache}; inspect prior execution before resuming')
            details={};original=inference.minimize;started=time.perf_counter()
            def monitored(*args,**kwargs):
                result=original(*args,**kwargs)
                gradient=result.jac.copy()
                for j,(lo,hi) in enumerate(kwargs['bounds']):
                    if lo is not None and result.x[j]<=lo+1e-10 and gradient[j]>0:gradient[j]=0
                maximum=float(np.abs(gradient).max())
                details.update(optimizer_success=bool(result.success),iterations=int(result.nit),
                               projected_gradient=maximum,message=str(result.message))
                if not np.isfinite(maximum) or maximum>frozen['max_projected_gradient']:
                    raise RuntimeError(f'Unfinished H inference: {details}')
                return result
            with patch.object(inference,'minimize',monitored):
                h=single_history(dict(scans=scans),ap,frozen['base'],frozen['obs'],steps=frozen['VI_maxiter'])
            arrays=dict(empty=np.array(h is None))
            if h is not None:
                if not torch.isfinite(h.m).all() or not torch.isfinite(h.c).all():raise FloatingPointError('Nonfinite H posterior')
                arrays.update(z=h.z.numpy(),mean=h.m.numpy(),cov=h.c.numpy())
            np.savez_compressed(cache,**arrays)
            record=dict(AP=ap,empty=h is None,empty_reason='zero H detections, legacy fallback' if h is None else None,
                        detections=int(scans['stats']['k'][:,ap].sum()),optimizer=details,seconds=time.perf_counter()-started,
                        cache_sha256=sha(cache),parameters_sha256=frozen['parameters_sha256'])
            write(status,record);statuses.append(record)
            if ap%100==0:print(json.dumps(dict(building=building,floor=floor,AP_done=ap+1,AP_total=len(meta['AP_ids']))),flush=True)
        summary=dict(floor=floor,APs=len(statuses),nonempty=sum(not s['empty'] for s in statuses),
                     max_projected_gradient=max((s['optimizer'].get('projected_gradient',0.) for s in statuses)),
                     fit_seconds=sum(s['seconds'] for s in statuses))
        write(dest/'summary.json',summary);summaries.append(summary)
    write(OUT/f'{building}_finished.json',dict(complete=True,building=building,floors=summaries,freeze_sha256=sha(OUT/'freeze.json')))
    print(json.dumps(dict(building=building,complete=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--building');a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:
        try:run(a.building)
        except Exception as error:
            write(OUT/f'{a.building}_failure.json',dict(error=repr(error),UTC=datetime.now(timezone.utc).isoformat()))
            raise
