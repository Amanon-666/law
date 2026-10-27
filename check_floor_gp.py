"""One source-only floor-GP data-flow check; not tuning or localization."""
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from data import group_scans
from floor_gp import FloorGPPrior, fit_floor_history, floor_kernel
from run import ap_stats, tensor
from run_multi_history import digest, write
from source_weight_mixture import field_from_prior
from transfer import rbf
from uji_probability_study import load_raw, DATA


def main():
    root = Path(__file__).resolve().parent
    out = root / 'results/floor_gp_source_audit_20261026'; out.mkdir(exist_ok=False)
    parameter_path = root / 'results/uji_history_bank_v1_20261026/parameters.json'
    manifest_path = root / 'results/uji_conditional_operator_v1_20261026/manifest.json'
    params = json.loads(parameter_path.read_text()); base, obs = params['base'], params['obs']
    names = ['B0F1', 'B0F2', 'B0F3']
    raw = load_raw('trainingData.csv', set(names) | {'B0F0'})
    origin = np.concatenate([raw[n]['positions'] for n in names]).mean(0)
    records = [dict(floor=int(n[-1]), scans=group_scans(raw[n]['rss'],raw[n]['xy']-origin)) for n in names]
    ap = int(np.argmax(sum((raw[n]['rss'] != 100).sum(0) for n in names)))
    floor_length = 1.  # Solely a numerical audit value; not selected for deployment.
    files = [Path(__file__), root / 'floor_gp.py', root / 'inference.py', root / 'reference_pmc.py', root / 'source_weight_mixture.py']
    write(out / 'freeze.json', dict(created_unix=time.time(), scope=__doc__, AP=ap, histories=names,
        target='B0F0', floor_length=floor_length, query_labels_used=False,
        code_sha256={str(p): digest(p) for p in files}, inputs={str(p): digest(p) for p in [parameter_path, manifest_path, DATA / 'trainingData.csv']}))
    (out / 'code').mkdir()
    for p in files:
        shutil.copy2(p, out / 'code' / p.name)
    started = time.perf_counter()
    prior, posterior = fit_floor_history(records, ap, base, obs, floor_length, 0)
    # Query probes are taken from H coordinates; no target Q is read.
    points = tensor(np.concatenate([r['scans']['positions'] for r in records])[::10])
    mean, feature = prior.features(points); cov = prior.covariance(points,points,feature,feature)
    torch.testing.assert_close(prior.diagonal(points,feature), cov.diagonal(), atol=1e-8, rtol=1e-9)
    eigenvalue = float(torch.linalg.eigvalsh((cov+cov.T)/2).min())
    assert eigenvalue > -1e-7
    order = torch.arange(len(prior.z)-1, -1, -1)
    reordered = FloorGPPrior(prior.z[order], dict(mean=posterior['mean'][order], cov=posterior['cov'][order][:,order]), base, floor_length, 0)
    rm, rf = reordered.features(points)
    torch.testing.assert_close(mean, rm, atol=1e-6, rtol=1e-8)
    torch.testing.assert_close(cov, reordered.covariance(points,points,rf,rf), atol=1e-6, rtol=1e-8)
    same_xy = tensor([[0.,0.,0.],[0.,0.,1.]])
    k = floor_kernel(same_xy,same_xy,base,1.)
    np.testing.assert_allclose(float(k[0,1]/k[0,0]), np.exp(-.5), atol=1e-12)
    torch.testing.assert_close(floor_kernel(prior.z,prior.z,base,1e12), rbf(prior.z[:,:2],prior.z[:,:2],base['amplitude'],base['length']), atol=1e-8, rtol=1e-10)
    narrow = FloorGPPrior(prior.z,posterior,base,.01,0)
    nm,nf = narrow.features(points)
    torch.testing.assert_close(nm, nm.new_full(nm.shape,base['mean']), atol=0, rtol=0)
    torch.testing.assert_close(narrow.diagonal(points,nf), nm.new_full(nm.shape,base['amplitude']**2), atol=0, rtol=0)
    meta = next(e for e in json.loads(manifest_path.read_text())['events'] if e['id']=='B0F0_s0_J3')
    f = raw['B0F0']; lookup = {int(r):i for i,r in enumerate(f['row_ids'])}
    sr = [lookup[r] for r in meta['support_rows']]
    support = group_scans(f['rss'][sr], f['xy'][sr]-origin)
    result = field_from_prior(prior,obs,tensor(support['positions']),ap_stats(support,ap),points,261030,defensive_fraction=.1)
    assert result['effective_samples'] >= 1000
    assert torch.isfinite(result['mean']).all() and (result['variance'] >= 0).all()
    output = dict(completed=True, AP=ap, historical_positions=len(prior.z), historical_scans=sum(len(r['scans']['rss']) for r in records),
        floor_length=floor_length, length_is_tuned=False, source_positions=len(support['positions']),
        min_projected_eigenvalue=eigenvalue, history_permutation=True, floor_dimension_preserved=True,
        independent_floor_limit=True, pooled_XY_kernel_limit=True, effective_samples=result['effective_samples'],
        seconds=time.perf_counter()-started, scope=__doc__)
    write(out / 'audit.json',output); print(json.dumps(output))


if __name__=='__main__':
    if socket.gethostname()!='ray-web':
        raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    main()
