"""Server-only phase C. Reuse frozen A/B inputs and unchanged readout."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

import inference
from full_ap_study import event_data,metrics
from inference import observation_log_prob,predict_from_posterior
from random_walk_history import RandomWalkHistory,check
from run import ap_stats,tensor
from run_calibration_learning import fit_samples,load_mean
from run_multi_history import digest,relation,write
from transfer import Prior


ROOT=Path(__file__).resolve().parent
AB=ROOT/'results/full_ap_v1'


def prepare(out):
    out.mkdir(exist_ok=False)
    old=json.loads((AB/'freeze.json').read_text())
    setting={k:old[k] for k in ['base','obs','calendar']}
    setting.update(mean_checkpoint=str(ROOT/'results/source_transfer_v4_spatial_control/affine.pt'),
                   phase='only Office mechanism intervention: OU to random walk; no learning',
                   source_results_seen_before_design=True,updates=0,history_maxiter=640,
                   code_sha256={p.name:digest(p) for p in ROOT.glob('*.py')},
                   inputs={str(p):digest(p) for p in [AB/'manifest.json',AB/'grid.npy',AB/'initial.pt']},
                   decision_sha256=digest(ROOT/'RANDOM_WALK_DECISION.md'),created_unix=time.time())
    write(out/'freeze.json',setting);(out/'code').mkdir()
    for p in ROOT.glob('*.py'):shutil.copy2(p,out/'code'/p.name)
    shutil.copy2(ROOT/'RANDOM_WALK_DECISION.md',out/'decision.md')
    shutil.copy2(AB/'initial.pt',out/'initial.pt')
    shutil.copy2(AB/'manifest.json',out/'manifest.json');shutil.copy2(AB/'grid.npy',out/'grid.npy')
    write(out/'mechanism_check.json',check())


def evaluate(out,setting,meta):
    destination=out/meta['id'];destination.mkdir(exist_ok=True)
    if (destination/'completed.json').exists():return
    histories,support,query=event_data(meta);base=setting['base'];obs=setting['obs']
    model=relation(setting,out/'initial.pt');mean=load_mean(setting)
    grid=tensor(np.load(out/'grid.npy'));points=torch.cat((grid,tensor(query['positions'])))
    real=inference.minimize;solver=[]
    def tracked(*a,**kw):
        r=real(*a,**kw);solver.append(dict(success=bool(r.success),nit=int(r.nit),gradient=float(np.abs(r.jac).max()),message=str(r.message)))
        if not r.success:raise RuntimeError(str(solver[-1]))
        return r
    inference.minimize=tracked
    for index,ap in enumerate(meta['ap_columns']):
        target=destination/f'ap{ap}.npz';record=destination/f'ap{ap}.json'
        if target.exists() and record.exists():continue
        solver.clear();start=time.perf_counter()
        h=RandomWalkHistory(histories,ap,base,obs,setting['calendar'])
        effective=h.reference_base
        case=dict(history=h,sx=tensor(support['positions']),qx=tensor(query['positions']),stats=ap_stats(support,ap),
                  rss=tensor(query['rss'][:,ap]),group=query['group'],weights=tensor(query['scan_weights']))
        with torch.no_grad():
            samples,_,logw,info=fit_samples(mean,model,case,effective,obs,251025+meta['seed'])
            prior=Prior(mean,h,effective,model)
            sm,sf=prior.features(case['sx']);sk=prior.covariance(case['sx'],case['sx'],sf,sf)
            gm,gf=prior.features(points);gv=prior.diagonal(points,gf);gc=prior.covariance(points,case['sx'],gf,sf)
            w=logw.exp();pm=w@samples;c=samples-pm;post=dict(mean=pm,cov=(c.T*w)@c)
            pred=predict_from_posterior(gm,gv,gc,sm,sk,post)
            if not torch.isfinite(pred['mean']).all() or not torch.isfinite(pred['variance']).all() or (pred['variance']<0).any():raise RuntimeError('invalid random-walk prediction')
            # Structural check of History/Prior amplitude consistency on S.
            if not torch.allclose(prior.diagonal(case['sx'],sf),sk.diag(),atol=1e-7):raise RuntimeError('diagonal inconsistent')
            np.savez_compressed(target,grid_mean=pred['mean'][:len(grid)].numpy(),grid_variance=pred['variance'][:len(grid)].numpy(),
                                query_mean=pred['mean'][len(grid):].numpy(),query_variance=pred['variance'][len(grid):].numpy())
        write(record,dict(ap=ap,seconds=time.perf_counter()-start,history_solver=solver,effective_amplitude=effective['amplitude'],**info))
        if index%10==0:print(json.dumps(dict(stage='AP',event=meta['id'],index=index+1,total=len(meta['ap_columns']))),flush=True)
    inference.minimize=real
    scores=torch.zeros((len(query['rss']),len(grid)))
    with torch.no_grad():
        for ap in meta['ap_columns']:
            a=np.load(destination/f'ap{ap}.npz')
            scores+=observation_log_prob(tensor(query['rss'][:,ap])[:,None],tensor(a['grid_mean'])[None],tensor(a['grid_variance'])[None],obs)
        pred=(torch.softmax(scores,1)@grid).numpy()
    np.savez_compressed(destination/'predictions.npz',predicted_xy=pred,query_xy=query['xy'],scores=scores.numpy(),ap_columns=meta['ap_columns'])
    result=dict(completed=True,event=meta['id'],method='random_walk',**metrics(pred,query),predictions_sha256=digest(destination/'predictions.npz'))
    write(destination/'completed.json',result);print(json.dumps(result),flush=True)


def main():
    if socket.gethostname()!='ray-web':raise RuntimeError('server only')
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','event','summarize']);p.add_argument('--event')
    p.add_argument('--out',type=Path,default=ROOT/'results/random_walk_v1');args=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    if args.stage=='prepare':prepare(args.out);return
    setting=json.loads((args.out/'freeze.json').read_text());manifest=json.loads((args.out/'manifest.json').read_text())
    for name,sha in setting['code_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError(f'code changed {name}')
    if args.stage=='event':evaluate(args.out,setting,next(e for e in manifest['events'] if e['id']==args.event));return
    rows=[json.loads((args.out/e['id']/'completed.json').read_text()) for e in manifest['events']]
    write(args.out/'summary.json',dict(completed=True,results=rows,equal_event_mean={k:float(np.mean([r[k] for r in rows])) for k in ['mde','p90','fraction_error_gt10']},updates=0))
    print(json.dumps(dict(completed=True,mean_mde=float(np.mean([r['mde'] for r in rows])))))


if __name__=='__main__':main()
