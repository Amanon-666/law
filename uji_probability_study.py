"""Server-only UJI source-development adapter for the existing probability LAW."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

import inference
from data import group_scans
from full_ap_study import metrics,wknn
from inference import infer,observation_log_prob,predict_from_posterior
from multi_history import single_history
from reference_pmc import readout
from run import ap_stats,tensor
from run_calibration_learning import fit_samples
from run_multi_history import digest,write
from transfer import Prior,Transfer,rbf


ROOT=Path(__file__).resolve().parent
SERVER=Path('/home/panyushuo/projects/panyushuo')
DATA=SERVER/'UJIIndoorLoc/data/raw/UJIndoorLoc'
LEGACY=SERVER/'FeMLoc-Signal-Calibrated-Map/models'
ALLOWED={'B0F0','B0F1','B0F2','B1F0','B1F1','B1F2'}
METHODS=('local_gp','history_identity','history_affine')
SOLVER_CONTEXT={}


def load_legacy(name):
    spec=importlib.util.spec_from_file_location('legacy_'+name,LEGACY/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def load_raw(filename,allowed):
    storage={}
    with (DATA/filename).open(newline='') as f:
        for rowid,row in enumerate(csv.DictReader(f),1):
            name=f"B{row['BUILDINGID']}F{row['FLOOR']}"
            if name not in allowed:continue
            storage.setdefault(name,[]).append((rowid,[int(row[f'WAP{i:03}']) for i in range(1,521)],
                                                [float(row['LONGITUDE']),float(row['LATITUDE'])]))
    result={}
    for name,rows in storage.items():
        grouped=group_scans(np.array([r[1] for r in rows]),np.array([r[2] for r in rows]))
        grouped['row_ids']=np.array([r[0] for r in rows]);result[name]=grouped
    return result


def legacy_cells(floor):
    return dict(xy=floor['positions'],**floor['stats'])


def prepare(out):
    out.mkdir(exist_ok=False)
    freeze=dict(scope='UJI source development only, not untouched targets',source=['B0F0','B0F1','B0F2'],
       development=['B1F1','B1F2'],reference={'B1F1':'B1F0','B1F2':'B1F1'},k=5,r=3,seeds=[0,1],
       base_updates=32,base_cases=12,base_seed=251025,optimizer='Adam',learning_rate=.02,
       base_initial=dict(mean=-95.,amplitude=20.,length=20.),history_maxiter=640,
       shared_kernel='ordinary spatial drift; Office strength parameters not used',
       source_transfer='existing fit_transfer, three default sweeps, raw scan statistics',
       source_obs='existing fit_observation + transfer detection likelihood',
       coordinate_unit='UJI original coordinate distance',grid_spacing=2.,
       code_sha256={p.name:digest(p) for p in ROOT.glob('*.py')},
       inputs={str(p):digest(p) for p in [DATA/'trainingData.csv',DATA/'validationData.csv',LEGACY/'cross_floor.py',LEGACY/'radio_map.py']},
       protocol_sha256=digest(ROOT/'UJI_SOURCE_PROTOCOL.md'),created_unix=time.time())
    write(out/'freeze.json',freeze);(out/'code').mkdir()
    for p in ROOT.glob('*.py'):shutil.copy2(p,out/'code'/p.name)
    shutil.copy2(ROOT/'UJI_SOURCE_PROTOCOL.md',out/'protocol.md')
    for name in ['cross_floor','radio_map']:shutil.copy2(LEGACY/(name+'.py'),out/'code'/('legacy_'+name+'.py'))
    floors=load_raw('trainingData.csv',set(freeze['source']))
    cells={name:legacy_cells(floors[name]) for name in freeze['source']}
    radio=load_legacy('radio_map');cross=load_legacy('cross_floor')
    obs0=radio.fit_observation(list(cells.values()))
    law=cross.fit_transfer([(cells['B0F0'],cells['B0F1']),(cells['B0F1'],cells['B0F2'])],obs0)
    obs=dict(alpha=law['h'],beta=law['theta'],detect_scale=law['s'],noise_std=law['sigma_total'])
    write(out/'legacy_source_fit.json',{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in law.items()})
    rng=np.random.default_rng(freeze['base_seed']);cases=[];case_meta=[]
    for name in freeze['source']:
        f=floors[name];eligible=np.flatnonzero((f['stats']['k']>0).any(0))
        for ap in rng.choice(eligible,4,replace=False):
            cases.append((tensor(f['positions']-f['positions'].mean(0)),ap_stats(f,int(ap))))
            case_meta.append(dict(floor=name,ap=int(ap),scans=len(f['rss']),positions=len(f['positions'])))
    write(out/'source_cases.json',case_meta)
    shift=torch.nn.Parameter(torch.tensor(0.));logs=torch.nn.Parameter(torch.tensor(np.log([20.,20.])))
    optimizer=torch.optim.Adam([shift,logs],lr=.02)
    with (out/'base_training.jsonl').open('x') as stream:
        for step in range(32):
            SOLVER_CONTEXT.update(phase='source_base',step=step,case=case_meta[step%len(cases)])
            x,stats=cases[step%len(cases)];mean=-95+20*shift;amplitude,length=logs.exp().unbind()
            optimizer.zero_grad()
            posterior=infer(mean.expand(len(x)),rbf(x,x,amplitude,length),stats,obs,steps=640,hyperparameter_fit=True)
            loss=posterior['loss']/stats['n'].sum();loss.backward()
            if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in [shift,logs]):
                raise RuntimeError('nonfinite UJI source GP calibration')
            optimizer.step()
            row=dict(step=step,case=step%len(cases),objective=loss.item(),mean=(-95+20*shift).item(),
                     amplitude=logs[0].exp().item(),length=logs[1].exp().item())
            stream.write(json.dumps(row)+'\n');stream.flush()
            if step%4==0:print(json.dumps(dict(stage='source_base',**row)),flush=True)
    base=dict(mean=(-95+20*shift).item(),amplitude=logs[0].exp().item(),length=logs[1].exp().item(),
              drift=float(np.sqrt(np.median(law['sb2']+law['s_w2']))))
    write(out/'parameters.json',dict(base=base,obs=obs,source_only=True,calibration_updates=32))
    floors.update(load_raw('trainingData.csv',{'B1F0','B1F1','B1F2'}))
    official=load_raw('validationData.csv',{'B1F1','B1F2'});events=[]
    for name in freeze['development']:
        f=floors[name];h=floors[freeze['reference'][name]]
        for seed in freeze['seeds']:
            rng=np.random.default_rng(seed);eligible=np.flatnonzero(f['stats']['n']>=3)
            chosen=np.sort(rng.choice(eligible,5,replace=False))
            sr=np.concatenate([rng.choice(np.flatnonzero(f['group']==p),3,replace=False) for p in chosen])
            qi=np.flatnonzero(~np.isin(f['group'],chosen));q=official[name]
            overlap=(q['xy'][:,None]==f['positions'][chosen][None]).all(-1).any(-1)
            qo=np.flatnonzero(~overlap)
            aps=np.flatnonzero((h['rss']!=100).any(0)|(f['rss'][sr]!=100).any(0))
            known=np.concatenate([h['positions'],f['xy'][sr]])
            events.append(dict(id=f'{name}_s{seed}',floor=name,reference=freeze['reference'][name],seed=seed,
                 support_rows=f['row_ids'][sr].tolist(),internal_query_rows=f['row_ids'][qi].tolist(),
                 official_query_rows=q['row_ids'][qo].tolist(),ap_columns=aps.tolist(),
                 H_scans=len(h['rss']),H_positions=len(h['positions']),S_scans=len(sr),
                 internal_scans=len(qi),official_scans=len(qo),official_overlaps_excluded=int(overlap.sum()),
                 grid_bounds=[known.min(0).tolist(),known.max(0).tolist()]))
    write(out/'manifest.json',dict(events=events,parameters_sha256=digest(out/'parameters.json')))
    print(json.dumps(dict(stage='prepared',base=base,obs=obs,events=[dict(id=e['id'],aps=len(e['ap_columns']),internal=e['internal_scans'],official=e['official_scans']) for e in events])),flush=True)


def run_event(out,freeze,meta):
    destination=out/meta['id'];destination.mkdir(exist_ok=True)
    if (destination/'completed.json').exists():return
    fitted=json.loads((out/'parameters.json').read_text());common,obs=fitted['base'],fitted['obs']
    law=json.loads((out/'legacy_source_fit.json').read_text())
    floors=load_raw('trainingData.csv',{meta['floor'],meta['reference']});official=load_raw('validationData.csv',{meta['floor']})
    f,h,q=floors[meta['floor']],floors[meta['reference']],official[meta['floor']]
    lookup={int(v):i for i,v in enumerate(f['row_ids'])};ql={int(v):i for i,v in enumerate(q['row_ids'])}
    sr=np.array([lookup[r] for r in meta['support_rows']]);qi=np.array([lookup[r] for r in meta['internal_query_rows']]);qo=np.array([ql[r] for r in meta['official_query_rows']])
    origin=h['positions'].mean(0)
    h=group_scans(h['rss'],h['xy']-origin);support=group_scans(f['rss'][sr],f['xy'][sr]-origin)
    query=group_scans(np.concatenate([f['rss'][qi],q['rss'][qo]]),np.concatenate([f['xy'][qi],q['xy'][qo]])-origin)
    query['scan_weights']=np.ones(len(query['rss']))/len(query['rss'])
    bounds=np.array(meta['grid_bounds'])-origin
    axes=[np.linspace(bounds[0,d],bounds[1,d],int(np.ceil((bounds[1,d]-bounds[0,d])/2))+1) for d in range(2)]
    grid=tensor(np.stack(np.meshgrid(*axes),-1).reshape(-1,2));points=torch.cat((grid,tensor(query['positions'])))
    cross=load_legacy('cross_floor');bins=cross.ap_bin(legacy_cells(h))
    for index,ap in enumerate(meta['ap_columns']):
        path=destination/f'ap{ap}.npz';record=destination/f'ap{ap}.json'
        if path.exists() and record.exists():continue
        started=time.perf_counter();base=dict(common,drift=float(np.sqrt(law['sb2'][int(bins[ap])]+law['s_w2'])))
        SOLVER_CONTEXT.update(phase='development_history',event=meta['id'],ap=ap)
        history=single_history(dict(scans=h),ap,base,obs,steps=640)
        saved={};rows=[]
        for name in METHODS:
            model=Transfer('affine' if name=='history_affine' else 'identity').double()
            if name=='history_affine':
                with torch.no_grad():model.a.fill_(law['rho'][int(bins[ap])]);model.b.fill_(law['bbar'][int(bins[ap])]/20)
            case=dict(history=None if name=='local_gp' else history,sx=tensor(support['positions']),qx=tensor(query['positions']),
                      stats=ap_stats(support,ap),rss=tensor(query['rss'][:,ap]),group=query['group'],weights=tensor(query['scan_weights']))
            now=time.perf_counter()
            with torch.no_grad():
                samples,_,logw,info=fit_samples(model,None,case,base,obs,251025+meta['seed'])
                prior=Prior(model,case['history'],base)
                sm,sf=prior.features(case['sx']);sk=prior.covariance(case['sx'],case['sx'],sf,sf)
                gm,gf=prior.features(points);gv=prior.diagonal(points,gf);gc=prior.covariance(points,case['sx'],gf,sf)
                w=logw.exp();pm=w@samples;c=samples-pm;post=dict(mean=pm,cov=(c.T*w)@c)
                pred=predict_from_posterior(gm,gv,gc,sm,sk,post)
                if not torch.isfinite(pred['mean']).all() or not torch.isfinite(pred['variance']).all() or (pred['variance']<0).any():raise RuntimeError('invalid UJI field')
                sl=slice(len(grid),None);g=case['group'];diag=readout(samples,logw,sm,sk,gm[sl][g],gv[sl][g],gc[sl][g],case['rss'],case['weights'],obs)
                for key,value in dict(grid_mean=pred['mean'][:len(grid)],grid_variance=pred['variance'][:len(grid)],scan_nll=tensor(diag['per_observation_nll'])).items():saved[f'{name}_{key}']=value.numpy()
            rows.append(dict(method=name,seconds=time.perf_counter()-now,nll=diag['nll'],ess=info['ess']))
        np.savez_compressed(path,**saved);write(record,dict(ap=ap,band=int(bins[ap]),seconds=time.perf_counter()-started,results=rows))
        if index%10==0:print(json.dumps(dict(stage='ap',event=meta['id'],index=index+1,total=len(meta['ap_columns']))),flush=True)
    scores={name:torch.zeros((len(query['rss']),len(grid))) for name in METHODS};saved={};results=[]
    with torch.no_grad():
        for ap in meta['ap_columns']:
            with np.load(destination/f'ap{ap}.npz') as arrays:
                for name in METHODS:scores[name]+=observation_log_prob(tensor(query['rss'][:,ap])[:,None],tensor(arrays[f'{name}_grid_mean'])[None],tensor(arrays[f'{name}_grid_variance'])[None],obs)
        for name in METHODS:saved[name]=(torch.softmax(scores[name],1)@grid).numpy()
    saved['wknn_raw']=wknn(support,query,meta['ap_columns'],False);saved['wknn_position_mean']=wknn(support,query,meta['ap_columns'],True)
    for split,sl in [('internal',slice(0,len(qi))),('official',slice(len(qi),None))]:
        one=dict(xy=query['xy'][sl],scan_weights=np.ones(len(query['xy'][sl]))/len(query['xy'][sl]))
        for name,pred in saved.items():results.append(dict(method=name,split=split,**metrics(pred[sl],one)))
    np.savez_compressed(destination/'predictions.npz',**{k:v+origin for k,v in saved.items()},query_xy=query['xy']+origin,
              internal_count=len(qi),support_xy=support['positions']+origin,grid=grid.numpy()+origin,ap_columns=meta['ap_columns'],
              query_detected_APs=(query['rss'][:,meta['ap_columns']]!=100).sum(1))
    write(destination/'completed.json',dict(completed=True,event=meta['id'],results=results,parameters_sha256=digest(out/'parameters.json'),predictions_sha256=digest(destination/'predictions.npz')))
    print(json.dumps(dict(stage='completed',event=meta['id'],results=results)),flush=True)


def main():
    if socket.gethostname()!='ray-web':raise RuntimeError('UJI experiments are authorized on ray-web only')
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','event']);p.add_argument('--event')
    p.add_argument('--out',type=Path,default=ROOT/'results/uji_probability_v1');args=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    real=inference.minimize
    def tracked(fun,x0,*a,**kw):
        result=real(fun,x0,*a,**kw)
        row=dict(context=dict(SOLVER_CONTEXT),success=bool(result.success),status=int(result.status),
                 message=str(result.message),nit=int(result.nit),gradient_max=float(np.abs(result.jac).max()))
        with (args.out/'solver_diagnostics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if not result.success:raise RuntimeError(f'UJI historical/base VI unfinished: {row}')
        return result
    inference.minimize=tracked
    if args.stage=='prepare':prepare(args.out);return
    freeze=json.loads((args.out/'freeze.json').read_text());manifest=json.loads((args.out/'manifest.json').read_text())
    for name,sha in freeze['code_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError(f'frozen code changed: {name}')
    if digest(args.out/'parameters.json')!=manifest['parameters_sha256']:raise RuntimeError('parameters changed')
    run_event(args.out,freeze,next(e for e in manifest['events'] if e['id']==args.event))


if __name__=='__main__':main()
