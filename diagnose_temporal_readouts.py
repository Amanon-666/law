"""Post-hoc readout diagnosis on server; distinguish latent f from RSS|detected."""
import json
from pathlib import Path
import socket

import numpy as np
from scipy.special import ndtr,log_ndtr
import torch

from full_ap_study import event_data
from inference import observation_log_prob
from run import tensor


def main():
    if socket.gethostname()!='ray-web':raise RuntimeError('server only')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    root=Path(__file__).resolve().parent;ab=root/'results/full_ap_v1';rw=root/'results/random_walk_v1'
    setting=json.loads((ab/'freeze.json').read_text());obs=setting['obs'];manifest=json.loads((ab/'manifest.json').read_text())
    rows=[]
    for meta in manifest['events']:
        _,_,query=event_data(meta);g=query['group']
        accum={name:dict(sq=[],abs=[],brier=[],nll=[]) for name in ['single','pooled','multi','random_walk']}
        for ap in meta['ap_columns']:
            old=np.load(ab/meta['id']/f'ap{ap}.npz');new=np.load(rw/meta['id']/f'ap{ap}.npz')
            rss=query['rss'][:,ap];det=rss!=100
            for name,stats in accum.items():
                a=new if name=='random_walk' else old;prefix='' if name=='random_walk' else name+'_'
                mean=a[prefix+'query_mean'][g];variance=a[prefix+'query_variance'][g]
                scale=np.sqrt(variance+obs['detect_scale']**2);z=(mean-obs['beta'])/scale
                conditional_mean=mean+variance/scale*np.exp(-.5*z*z-.5*np.log(2*np.pi)-log_ndtr(z))
                delta=conditional_mean[det]-rss[det]
                stats['sq'].extend((delta**2).tolist());stats['abs'].extend(np.abs(delta).tolist())
                stats['brier'].extend(((obs['alpha']*ndtr(z)-det)**2).tolist())
                stats['nll'].extend((-observation_log_prob(tensor(rss),tensor(mean),tensor(variance),obs)).tolist())
        for name,a in accum.items():rows.append(dict(event=meta['id'],method=name,
            detected_RSS_conditional_mean_RMSE=float(np.sqrt(np.mean(a['sq']))),
            detected_RSS_conditional_mean_MAE=float(np.mean(a['abs'])),
            detection_Brier=float(np.mean(a['brier'])),Gaussian_joint_NLL=float(np.mean(a['nll']))))
    keys=['detected_RSS_conditional_mean_RMSE','detected_RSS_conditional_mean_MAE','detection_Brier','Gaussian_joint_NLL']
    means={name:{key:float(np.mean([r[key] for r in rows if r['method']==name])) for key in keys}
           for name in ['single','pooled','multi','random_walk']}
    result=dict(scope='post-hoc diagnostic, no model choice or training',results=rows,equal_event_means=means,
        interpretation='RSS conditioned on detection has mean mu+v/sqrt(v+s^2)*phi(z)/Phi(z); latent mu alone is not this conditional mean',
        limitation='Gaussian-moment readout metrics; not exact mixture metrics or proof of physical cause')
    (rw/'readout_diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(means))


if __name__=='__main__':main()
