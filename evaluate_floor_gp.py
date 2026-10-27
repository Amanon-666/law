"""Floor-aware GP control on the existing H/S/Q localization events."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from data import group_scans
from floor_gp import FloorGPPrior, fit_floor_history
from full_ap_study import metrics
from history_bank import BankPrior
from inference import observation_log_prob
from run import ap_stats, tensor
from run_multi_history import digest, write
from source_weight_mixture import field_from_prior
from uji_probability_study import load_raw, DATA

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'results/uji_history_bank_v1_20261026'
PAIRED=ROOT/'results/uji_operator_paired_v2_20261026'
FIT=ROOT/'results/floor_gp_source_fit_20261026'
OUT=ROOT/'results/uji_floor_gp_v1_20261026'
SOURCES=['B1F0','B1F3']


def prepare():
    selected=json.loads((FIT/'selected.json').read_text());assert selected['completed']
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),*[ROOT/n for n in ['floor_gp.py','inference.py','source_weight_mixture.py','reference_pmc.py','data.py','transfer.py','full_ap_study.py','uji_probability_study.py']]]
    inputs=[FIT/'selected.json',FIT/'freeze.json',OLD/'parameters.json',OLD/'manifest.json',OLD/'histories/completed.json',
            DATA/'trainingData.csv',DATA/'validationData.csv']
    inputs+=sorted(PAIRED.glob('B*/predictions.npz'))+sorted(PAIRED.glob('B*/completed.json'))
    write(OUT/'freeze.json',dict(created_unix=time.time(),purpose=__doc__,floor_length=selected['floor_length'],
        hypothesis='A conventional GP that retains floor information is a necessary control for learned history-profile use; no superiority is assumed',
        method='Independent-AP RBF over XY and floor index, joint raw-H posterior, same original detection/RSS likelihood and defensive M-PMC S update',
        source_fit='B0/B2 source-only finite-grid variational fit of floor length; same frozen horizontal/base/observation parameters as LAW',
        target='Exact six prior B1 events, H=B1F0/F3, K5R3, same APs and H/S-derived search bounds',
        missing_history='All-H-undetected APs use the same local-only GP fallback as the LAW comparisons',
        claim='Component adaptation, not full MOGPAug reproduction; independent APs and no augmentation/RNN',
        code_sha256={str(p):digest(p) for p in files},inputs={str(p):digest(p) for p in inputs}))
    (OUT/'code').mkdir();(OUT/'histories').mkdir()
    for p in files:shutil.copy2(p,OUT/'code'/p.name)


def run():
    freeze=json.loads((OUT/'freeze.json').read_text())
    for section in ['code_sha256','inputs']:
        for path,expected in freeze[section].items():assert digest(Path(path))==expected,path
    param=json.loads((OLD/'parameters.json').read_text());base,obs=param['base'],param['obs']
    length=freeze['floor_length'];events=json.loads((OLD/'manifest.json').read_text())['events']
    raw=load_raw('trainingData.csv',set(SOURCES)|{'B1F1','B1F2'})
    official=load_raw('validationData.csv',{'B1F1','B1F2'})
    origin=np.array(json.loads((OLD/'histories/completed.json').read_text())['origin'])
    records=[dict(floor=int(n[-1]),scans=group_scans(raw[n]['rss'],raw[n]['xy']-origin)) for n in SOURCES]
    aps=sorted({a for e in events for a in e['ap_columns']})
    for index,ap in enumerate(aps):
        path=OUT/'histories'/f'ap{ap}.npz'
        if path.exists():continue
        if not any(r['scans']['stats']['k'][:,ap].sum()>0 for r in records):
            np.savez_compressed(path,empty=True)
        else:
            prior,posterior=fit_floor_history(records,ap,base,obs,length,0)
            np.savez_compressed(path,empty=False,z=prior.z.numpy(),mean=posterior['mean'].numpy(),cov=posterior['cov'].numpy())
        if index%20==0:print(json.dumps(dict(stage='joint_history',completed=index+1,total=len(aps))),flush=True)
    rows=[]
    for event in events:
        dest=OUT/event['id'];dest.mkdir(exist_ok=True)
        if (dest/'completed.json').exists():
            rows.extend(json.loads((dest/'completed.json').read_text())['results']);continue
        f=raw[event['floor']];v=official[event['floor']]
        def indices(data,ids):
            lookup={int(r):i for i,r in enumerate(data['row_ids'])};return [lookup[r] for r in ids]
        sr=indices(f,event['support_rows']);qi=indices(f,event['internal_rows']);qo=indices(v,event['official_rows'])
        support=group_scans(f['rss'][sr],f['xy'][sr]-origin);sx=tensor(support['positions'])
        qrss=np.concatenate([f['rss'][qi],v['rss'][qo]]);qxy=np.concatenate([f['xy'][qi],v['xy'][qo]])-origin
        assert len(sx)==5 and len(sr)==15
        assert not (qxy[:,None]==support['positions'][None]).all(-1).any()
        bounds=np.array(event['grid_bounds'])-origin
        axes=[np.linspace(bounds[0,d],bounds[1,d],int(np.ceil((bounds[1,d]-bounds[0,d])/2))+1) for d in range(2)]
        grid=tensor(np.stack(np.meshgrid(*axes),-1).reshape(-1,2));fields=[];ess=[]
        for index,ap in enumerate(event['ap_columns']):
            path=dest/f'ap{ap}.npz'
            if not path.exists():
                with np.load(OUT/'histories'/f'ap{ap}.npz') as data:
                    prior=BankPrior([],[],base) if bool(data['empty']) else FloorGPPrior(tensor(data['z']),dict(mean=tensor(data['mean']),cov=tensor(data['cov'])),base,length,int(event['floor'][-1]))
                result=field_from_prior(prior,obs,sx,ap_stats(support,ap),grid,261026+10000*event['seed']+ap,defensive_fraction=.1)
                if result['effective_samples']<1000:raise RuntimeError(f'Low ESS: {event["id"]}, AP {ap}')
                np.savez_compressed(path,mean=result['mean'].numpy(),variance=result['variance'].numpy(),ess=result['effective_samples'])
            with np.load(path) as data:
                fields.append((ap,tensor(data['mean']),tensor(data['variance'])));ess.append(float(data['ess']))
            if index%40==0:print(json.dumps(dict(stage='field',event=event['id'],completed=index+1,total=len(event['ap_columns']))),flush=True)
        predictions=[]
        for start in range(0,len(qrss),64):
            scans=tensor(qrss[start:start+64]);scores=scans.new_zeros((len(scans),len(grid)))
            for ap,mean,var in fields:scores+=observation_log_prob(scans[:,ap,None],mean[None],var[None],obs)
            predictions.append((scores.softmax(1)@grid).numpy())
        pred=np.concatenate(predictions)
        with np.load(PAIRED/event['id']/'predictions.npz') as data:
            np.testing.assert_array_equal(data['query_xy'],qxy);np.testing.assert_array_equal(data['origin'],origin)
            baseline=data['adapted_bank']
        result=[]
        for split,sl in [('internal',slice(0,len(qi))),('official',slice(len(qi),None))]:
            xy=qxy[sl];_,group,count=np.unique(xy,axis=0,return_inverse=True,return_counts=True)
            score_data=dict(xy=xy,scan_weights=1/(len(count)*count[group.ravel()]))
            for name,p in [('floor_gp',pred),('adapted_bank',baseline)]:
                result.append(dict(event=event['id'],floor=event['floor'],method=name,split=split,**metrics(p[sl],score_data)))
        np.savez_compressed(dest/'predictions.npz',floor_gp=pred,query_xy=qxy,origin=origin)
        write(dest/'completed.json',dict(results=result,min_ess=min(ess)));rows.extend(result)
        print(json.dumps(dict(stage='completed',event=event['id'],results=result)),flush=True)
    summary={}
    for split in ['internal','official']:
        mean={m:float(np.mean([r['mde'] for r in rows if r['split']==split and r['method']==m])) for m in ['floor_gp','adapted_bank']}
        summary[split]=mean
    write(OUT/'summary.json',dict(summary=summary,records=rows,scope=freeze['claim']))
    print(json.dumps(dict(stage='summary',summary=summary)),flush=True)


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    {'prepare':prepare,'run':run}[args.stage]()
