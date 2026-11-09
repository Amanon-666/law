"""Full B0F0 H-only fusion preparation; no source training or query scoring.

Uses every AP in the old H/S catalog, but NOT its fitted source weights,
intercept or support posterior. A failed full-anchor representation is reported
as such; no automatic smaller anchor set, noise increase or AP selection.
The third representation retains numerically resolved orthonormal coordinates
of ALL anchors, with a deterministic machine-precision spectral threshold.
It has no extra anchor noise. Previous two failed attempts are preserved.
"""
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from anchored_barycenter import AnchorSubspace, gaussian_barycenter
from building_source_assets import load_field
from fields import Prior, Transfer, rbf
from recomposition_trial import ROOT, digest, now, write

OUT=ROOT/'results/anchored_barycenter_assets_20261107_centered'
CACHE=ROOT/'results/refi_source_selection_learning_20261107/fit_B0F0_support.pt'
MANIFEST=ROOT/'results/refi_source_selection_learning_20261107_checked_selection/histories.json'


def maximum(a,b):
    return float((a-b).abs().max()) if a.numel() else 0.


@torch.no_grad()
def run():
    started=time.monotonic()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    files=('anchored_barycenter_assets.py','anchored_barycenter.py','fields.py','building_source_assets.py')
    write(OUT/'freeze.json',dict(created=now(),code={n:digest(ROOT/n) for n in files},
        inputs={str(p):digest(p) for p in (CACHE,MANIFEST)},
        cpu=2,gpu=0,seconds=240,query_read=False,training_updates=0,
        source_weights='uniform, no use of the old S-fitted weights',intercept=0.,
        mean_tolerance_dB=.01,covariance_tolerance_dB2=.01,
        anchor_rule='full H union, eigenvalues > eps * dimension * spectral radius',
        reference_variance=0.,
        previous_failures=['noiseless full-anchor Cholesky failed',
            '1e-6 noisy-anchor lift changed covariance by >0.01 dB^2',
            'direct subspace projection changed covariance by 0.010674 dB^2'],
        computation='same subspace/model, reference-relative History updates and centered covariance lift'))
    for n in files:(OUT/n).write_bytes((ROOT/n).read_bytes())
    data=torch.load(CACHE,weights_only=False)
    event=data['events'][0];assert event['floor']=='B0F0'
    names=event['histories'];assert len(names)==3
    manifest=json.loads(MANIFEST.read_text())
    for name in names:assert event['history_rows'][name]==manifest['history_rows'][name]
    records={(r['floor'],r['ap']):r for r in manifest['fields']}
    origin=np.asarray(data['origin']);base=data['base']
    coordinates={}
    for name in names:
        rec=next(r for r in manifest['fields'] if r['floor']==name)
        path=ROOT/rec['path'];assert digest(path)==rec['sha256']
        with np.load(path) as f:coordinates[name]=torch.tensor(f['z']+(f['origin']-origin))
    anchors=torch.unique(torch.cat(list(coordinates.values())),dim=0)
    kernel=lambda x,y:rbf(x,y,base['amplitude'],base['length'])
    ka=kernel(anchors,anchors);spectrum=torch.linalg.eigvalsh(ka)
    subspace=AnchorSubspace(anchors,kernel)
    geometry=dict(anchors=len(anchors),source_positions={h:len(z) for h,z in coordinates.items()},
        APs=len(data['cases']),grid=len(data['grid']),minimum_eigenvalue=float(spectrum[0]),
        maximum_eigenvalue=float(spectrum[-1]),
        condition_number=float(spectrum[-1]/spectrum[0]) if spectrum[0]>0 else None,
        reference_variance=0.,rank=len(subspace.values),threshold=subspace.threshold,
        discarded_absolute_trace=subspace.discarded_absolute_trace,
        retained_condition_number=float(subspace.values[-1]/subspace.values[0]))
    write(OUT/'geometry.json',geometry)
    support=torch.as_tensor(data['support_positions']);grid=data['grid']
    weights=torch.full((len(names),),1/len(names))
    reference_field=subspace.lift(torch.zeros(len(subspace.values)),torch.diag(subspace.values),base['mean'])
    reference=reference_field.reference_blocks(support,grid)
    prepared={};checks=[]
    for ap,case in data['cases'].items():
        block=case['blocks'];moments=[];present=[];errors=[]
        for j,name in enumerate(names):
            missing=bool(block['za'][j,0,2])
            h=None if missing else load_field(records[name,ap],origin,base)
            if h is not None:torch.testing.assert_close(h.z,coordinates[name],rtol=0,atol=1e-8)
            ma,ca,delta=subspace.encode_history(h,base['mean'])
            lifted=subspace.lift(ma,ca,base['mean'],delta)
            p=lifted.prior_blocks(support,grid,reference=reference)
            error=dict(source=name,mean=max(maximum(p['sm'],block['sm'][j]),maximum(p['gm'],block['gm'][j])),
                covariance=max(maximum(p['sk'],block['ss'][j,0]),
                    maximum(p['cross'],block['gs'][j,0]),maximum(p['gv'],block['gv'][j,0])))
            errors.append(error);moments.append((ma,ca));present.append(not missing)
        row=dict(AP=ap,sources=errors)
        with (OUT/'checks.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if any(e['mean']>.01 or e['covariance']>.01 for e in errors):
            raise RuntimeError(f'Common-reference lift no longer preserves historical field: {row}')
        means=torch.stack([v[0] for v in moments]);covs=torch.stack([v[1] for v in moments])
        prepared[ap]=dict(anchor_means=means,anchor_covariances=covs,present=present,stats=case['stats'])
        checks.append(row)
        write(OUT/'progress.json',dict(stage='source_preservation',APs_prepared=len(prepared),
            seconds=time.monotonic()-started,last_AP=ap))
    source_path=OUT/'source_moments.pt'
    torch.save(dict(cases=prepared,reference=reference,anchors=anchors,basis=subspace.vectors,
        reference_eigenvalues=subspace.values,base=base,obs=data['obs'],support_positions=support,
        grid=grid,events=data['events'],origin=data['origin']),source_path)
    write(OUT/'source_preservation.json',dict(complete=True,APs=len(prepared),checks=checks,
        source_moments_sha256=digest(source_path),source_moments_bytes=source_path.stat().st_size))
    for count,((ap,item),row) in enumerate(zip(prepared.items(),checks),1):
        means,covs,present=item['anchor_means'],item['anchor_covariances'],item['present']
        block=data['cases'][ap]['blocks']
        if any(present):
            fusion=gaussian_barycenter(means,covs,weights)
            field=subspace.lift(fusion['mean'],fusion['cov'],base['mean'])
            p=field.prior_blocks(support,grid,reference=reference)
            p=dict(p,sk=p['sk']+block['rs'],cross=p['cross']+block['rg'],gv=p['gv']+block['rv'])
        else:
            fusion=None
            p=dict(sm=support.new_full((len(support),),base['mean']),
                gm=grid.new_full((len(grid),),base['mean']),sk=kernel(support,support),
                cross=kernel(grid,support),gv=grid.new_full((len(grid),),base['amplitude']**2))
        torch.linalg.cholesky(p['sk'])
        if not all(bool(torch.isfinite(t).all()) for t in p.values()) or not bool((p['gv']>=0).all()):
            raise RuntimeError(f'Invalid prior AP {ap}')
        item.update(prior=p,fusion=fusion)
        row.update(iterations=None if fusion is None else fusion['iterations'],
            residual=None if fusion is None else fusion['residual'])
        write(OUT/'progress.json',dict(stage='fusion',APs_prepared=count,seconds=time.monotonic()-started,last_AP=ap))
    path=OUT/'prepared.pt'
    torch.save(dict(cases=prepared,anchors=anchors,grid=grid,support_positions=support,
        weights=weights,base=base,obs=data['obs'],origin=data['origin'],events=data['events'],
        reference_variance=0.,basis=subspace.vectors,reference_eigenvalues=subspace.values),path)
    write(OUT/'complete.json',dict(complete=True,seconds=time.monotonic()-started,geometry=geometry,
        checks=checks,prepared_sha256=digest(path),prepared_bytes=path.stat().st_size,
        support_conditioned=False,query_read=False,training_updates=0,performance_evaluated=False))
    print('complete',len(prepared),time.monotonic()-started,flush=True)


if __name__=='__main__':
    try:run()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),type=type(error).__name__,message=str(error)))
        raise
