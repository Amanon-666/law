"""Post-hoc official-query candidate discrimination; frozen fields, no training."""
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from inference import observation_log_prob
from run import tensor
from run_multi_history import digest,write
from uji_probability_study import load_raw


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).resolve().parent
    out=root/'results/uji_operator_paired_v2_20261026';old=root/'results/uji_history_bank_v1_20261026'
    destination=out/'official_readout_diagnosis.json'
    if destination.exists():raise FileExistsError(destination)
    assert (out/'summary.json').exists()
    obs=json.loads((old/'parameters.json').read_text())['obs']
    manifest=json.loads((old/'manifest.json').read_text())['events']
    official=load_raw('validationData.csv',{'B1F1','B1F2'})
    rows=[]
    for event in manifest:
        d=out/event['id'];f=official[event['floor']]
        lookup={int(r):i for i,r in enumerate(f['row_ids'])};ix=[lookup[r] for r in event['official_rows']]
        with np.load(d/'predictions.npz') as data:
            origin=data['origin'];stored={m:data[m][-len(ix):] for m in ['adapted_bank','operator_bank']}
        xy=f['xy'][ix]-origin;rss=tensor(f['rss'][ix]);bounds=np.array(event['grid_bounds'])-origin
        axes=[np.linspace(bounds[0,k],bounds[1,k],int(np.ceil((bounds[1,k]-bounds[0,k])/2))+1) for k in range(2)]
        grid=tensor(np.stack(np.meshgrid(*axes),-1).reshape(-1,2))
        grid_error,true_index=cKDTree(grid.numpy()).query(xy)
        _,groups,count=np.unique(xy,axis=0,return_inverse=True,return_counts=True)
        weighting=tensor(1/(len(count)*count[groups.ravel()]))
        for method in stored:
            values={k:[] for k in ['position_nll','true_grid_rank','mass_within_10m','entropy','mean_error','map_error','true_scan_nll']}
            means=[];variances=[]
            for a in event['ap_columns']:
                with np.load(d/f'ap{a}.npz') as data:
                    means.append(tensor(data[method+'_mean']));variances.append(tensor(data[method+'_variance']))
            with torch.no_grad():
                for start in range(0,len(rss),32):
                    r=rss[start:start+32];truth=tensor(xy[start:start+32]);ii=torch.tensor(true_index[start:start+32]);batch=torch.arange(len(r))
                    score=r.new_zeros((len(r),len(grid)))
                    for a,m,v in zip(event['ap_columns'],means,variances):
                        score+=observation_log_prob(r[:,a,None],m[None],v[None],obs)
                    logp=score.log_softmax(1);prob=logp.exp();estimate=prob@grid
                    np.testing.assert_allclose(estimate.numpy(),stored[method][start:start+32],rtol=0,atol=1e-9)
                    distance=(grid[None]-truth[:,None]).norm(dim=-1)
                    values['position_nll'].append(-logp[batch,ii])
                    values['true_grid_rank'].append((score>score[batch,ii,None]).sum(1).double()+1)
                    values['mass_within_10m'].append((prob*(distance<=10)).sum(1))
                    values['entropy'].append(-(prob*logp).sum(1))
                    values['mean_error'].append((estimate-truth).norm(dim=-1))
                    values['map_error'].append((grid[score.argmax(1)]-truth).norm(dim=-1))
                    values['true_scan_nll'].append(-score[batch,ii])
            rows.append(dict(event=event['id'],floor=event['floor'],method=method,
                mean_true_to_grid_distance=float(weighting@tensor(grid_error)),
                **{k:float(weighting@torch.cat(v)) for k,v in values.items()}))
        print(json.dumps(dict(event=event['id'],rows=rows[-2:])),flush=True)
    write(destination,dict(scope=__doc__,created_unix=time.time(),script_sha256=digest(Path(__file__)),
        limitations='Nearest true grid point is a diagnostic discretization; all queries are previously viewed development data. MAP is not a selected replacement readout.',rows=rows))
