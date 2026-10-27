"""Frozen-model localization across single, paired and three-floor histories.

The original two-history results are reused exactly. Source-set ablations share
the old H/S-derived deployment bounds, not hidden target-floor coordinates.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from conditional_operator_learning import fit_episode, prior_for
from data import group_scans
from full_ap_study import metrics, wknn
from history_bank import BankPrior
from inference import observation_log_prob
from operator_prior import OperatorNet
from run import ap_stats, tensor
from run_multi_history import digest, write
from run_operator_prior import load_history
from source_weight_mixture import field_from_prior
from transfer import History
from uji_probability_study import load_raw, DATA

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'results/uji_history_bank_v1_20261026'
SOURCE=ROOT/'results/uji_operator_prior_v1_20261026'
TRAIN=ROOT/'results/uji_conditional_operator_v1_20261026'
EXTRA=ROOT/'results/operator_v11_20261026/histories/UJI_B1'
OUT=ROOT/'results/uji_variable_history_v1_20261026'
METHODS=['adapted_bank','initial','prior','conditional']


def parameters():
    d=json.loads((OLD/'parameters.json').read_text());return d['base'],d['obs']


def origin():
    return np.array(json.loads((OLD/'histories/completed.json').read_text())['origin'])


@lru_cache(maxsize=1)
def raw():
    return load_raw('trainingData.csv',{'B1F0','B1F1','B1F2','B1F3'})


@lru_cache(maxsize=None)
def history(name,ap):
    base,_=parameters()
    if name in ['B1F0','B1F3']:
        path=OLD/'histories'/name/f'ap{ap}.npz'
        if path.exists():return load_history(path,base)
    else:
        directory=EXTRA/f'F{name[-1]}';path=directory/f'ap{ap}.npz'
        if path.exists():
            with np.load(path) as a:
                if bool(a['empty']):return None
                shift=np.array(json.loads((directory/'done.json').read_text())['origin'])-origin()
                return History(tensor(a['z']+shift),dict(mean=tensor(a['mean']),cov=tensor(a['cov'])),base)
    # A missing cache is legitimate only when this source never detected AP.
    assert not (raw()[name]['rss'][:,ap]!=100).any(),(name,ap)
    return None


def support_and_cases(meta):
    f=raw()[meta['floor']];lookup={int(r):i for i,r in enumerate(f['row_ids'])}
    sr=[lookup[r] for r in meta['support_rows']]
    support=group_scans(f['rss'][sr],f['xy'][sr]-origin());sx=tensor(support['positions'])
    assert len(sx)==5 and len(sr)==15 and meta['floor'] not in meta['histories']
    seen=(support['rss']!=100).any(0)
    for name in meta['histories']:seen|=(raw()[name]['rss']!=100).any(0)
    aps=np.flatnonzero(seen).tolist()
    cases={a:dict(histories=[history(n,a) for n in meta['histories']],sx=sx,stats=ap_stats(support,a),
        dz=[int(n[-1])-int(meta['floor'][-1]) for n in meta['histories']]) for a in aps}
    return support,cases


def checkpoint(arm):
    return SOURCE/'operator.pt' if arm=='initial' else TRAIN/arm/'operator.pt'


def prepare():
    OUT.mkdir(exist_ok=False);(OUT/'code').mkdir()
    originals=json.loads((OLD/'manifest.json').read_text())['events'];events=[]
    for e in originals:
        sets=dict(low=['B1F0'],high=['B1F3'],pair=['B1F0','B1F3'],
                  all=[f'B1F{f}' for f in range(4) if f!=int(e['floor'][-1])])
        for bank,names in sets.items():
            events.append(dict(e,id=e['id']+'_'+bank,original=e['id'],bank=bank,histories=names))
    files=[Path(__file__),*[ROOT/n for n in ['conditional_operator_learning.py','multi_source_adaptation.py','operator_prior.py',
        'history_bank.py','inference.py','reference_pmc.py','source_weight_mixture.py','transfer.py','data.py','full_ap_study.py','uji_probability_study.py']]]
    inputs=[OLD/'parameters.json',OLD/'manifest.json',OLD/'histories/completed.json',DATA/'trainingData.csv',DATA/'validationData.csv']
    inputs+=sorted((OLD/'histories').glob('B*/ap*.npz'))+sorted(OLD.glob('B*/adaptation.json'))
    for f in [1,2]:inputs+=sorted((EXTRA/f'F{f}').glob('*.npz'))+[EXTRA/f'F{f}'/'done.json']
    inputs += [checkpoint(a) for a in METHODS[1:]]
    inputs += sorted((TRAIN/'evaluation').glob('*/B*/completed.json'))+[TRAIN/'evaluation/freeze.json']
    write(OUT/'freeze.json',dict(created_unix=time.time(),purpose=__doc__,models=METHODS,
        hypothesis='Historical-content learning should remain useful across source-set changes; improvement due to more history is distinct from learned-module gain',
        geometry='Same legal deployment bounds from candidate-bank F0/F3 coordinates plus S are given to all source-set conditions. Single-source conditions withhold other-source RSS, not this shared region metadata.',
        AP_policy='Union detected in the actually selected histories or S, no Q selection; all methods within a bank share AP information',
        reuse='Pair results copied from the completed matched-policy evaluation, with data/code/checkpoint hashes and input identity checked',
        interpretation='Report each bank and target floor separately. J2-only initial model has an unidentifiable count-feature coefficient outside J2; do not claim variable-J training implies target accuracy.',
        criterion='Learning gain must be assessed against shared EB within each bank; multi-J training against initial within each bank. No after-result tuning; all six old development events per bank.',
        code_sha256={str(p):digest(p) for p in files},inputs={str(p):digest(p) for p in inputs}))
    write(OUT/'manifest.json',dict(events=events))
    for p in files:shutil.copy2(p,OUT/'code'/p.name)
    # Same backend and initial/continued checkpoints as the reused pair study.
    pair_freeze=json.loads((TRAIN/'evaluation/freeze.json').read_text())
    for path,expected in pair_freeze['inputs'].items():
        assert digest(Path(path))==expected,path
    for name in ['history_bank.py','inference.py','reference_pmc.py','source_weight_mixture.py','transfer.py','operator_prior.py']:
        p=ROOT/name;assert digest(p)==pair_freeze['code_sha256'][str(p)],name
    audit=[]
    for e in events:
        support,cases=support_and_cases(e)
        dest=OUT/e['id'];dest.mkdir()
        if e['bank']=='pair':
            assert list(cases)==e['ap_columns']
            weights=json.loads((OLD/e['original']/'adaptation.json').read_text())['weights']
            results=[]
            for method in METHODS:
                arm='initial' if method=='adapted_bank' else method
                source=TRAIN/'evaluation'/arm/e['original']/'completed.json'
                data=json.loads(source.read_text());assert data['weights']==weights
                key='adapted_bank' if method=='adapted_bank' else 'operator_bank'
                results.extend([dict(r,method=method) for r in data['results'] if r['method']==key])
            write(dest/'completed.json',dict(event=e['id'],bank=e['bank'],results=results,weights=weights,reused_pair=True))
        # Audit source-coordinate translation independently of target Q.
        if e['bank']=='all' and e['seed']==0:
            name='B1F2' if e['floor']=='B1F1' else 'B1F1';directory=EXTRA/f'F{name[-1]}'
            ap=next(a for a in cases if history(name,a) is not None)
            original=load_history(directory/f'ap{ap}.npz',parameters()[0]);translated=history(name,ap)
            shift=tensor(np.array(json.loads((directory/'done.json').read_text())['origin'])-origin())
            points=cases[ap]['sx'];oldp=original.projection(points-shift);newp=translated.projection(points)
            torch.testing.assert_close(oldp,newp,atol=1e-8,rtol=1e-8)
            audit.append(dict(history=name,AP=ap,translation_max_error=float((oldp-newp).abs().max())))
    write(OUT/'audit.json',dict(pair_AP_sets_identical=True,translated_histories=audit,query_used=False))


def verify():
    frozen=json.loads((OUT/'freeze.json').read_text())
    for section in ['code_sha256','inputs']:
        for path,expected in frozen[section].items():assert digest(Path(path))==expected,path


def event(meta,nets,base,obs):
    dest=OUT/meta['id']
    if (dest/'completed.json').exists():return
    support,cases=support_and_cases(meta);sx=tensor(support['positions'])
    path=dest/'adaptation.json'
    if not path.exists():
        weights,trials=fit_episode(cases,base,obs)
        write(path,dict(weights=weights,trials=trials,APs=list(cases)))
    fitted=json.loads(path.read_text());assert fitted['APs']==list(cases);weights=fitted['weights']
    bounds=np.array(meta['grid_bounds'])-origin()
    axes=[np.linspace(bounds[0,d],bounds[1,d],int(np.ceil((bounds[1,d]-bounds[0,d])/2))+1) for d in range(2)]
    grid=tensor(np.stack(np.meshgrid(*axes),-1).reshape(-1,2))
    for index,(ap,c) in enumerate(cases.items()):
        path=dest/f'ap{ap}.npz'
        if path.exists():continue
        arrays={}
        for method in METHODS:
            prior=prior_for(None if method=='adapted_bank' else nets[method],c,weights,base)
            result=field_from_prior(prior,obs,sx,c['stats'],grid,261026+10000*meta['seed']+ap,defensive_fraction=.1)
            if result['effective_samples']<1000:raise RuntimeError(f'Low ESS {meta["id"]} {ap} {method}')
            arrays[method+'_mean']=result['mean'].numpy();arrays[method+'_variance']=result['variance'].numpy()
            arrays[method+'_ess']=result['effective_samples']
        np.savez_compressed(path,**arrays)
        if index%40==0:print(json.dumps(dict(stage='fields',event=meta['id'],completed=index+1,total=len(cases))),flush=True)
    # Only now read Q observations; adaptation above is entirely H/S based.
    f=raw()[meta['floor']];v=load_raw('validationData.csv',{meta['floor']})[meta['floor']]
    def indices(data,ids):
        lookup={int(r):i for i,r in enumerate(data['row_ids'])};return [lookup[r] for r in ids]
    qi=indices(f,meta['internal_rows']);qo=indices(v,meta['official_rows'])
    qrss=np.concatenate([f['rss'][qi],v['rss'][qo]]);qxy=np.concatenate([f['xy'][qi],v['xy'][qo]])-origin()
    assert not (qxy[:,None]==support['positions'][None]).all(-1).any()
    field={m:[] for m in METHODS};ess=[]
    for ap in cases:
        with np.load(dest/f'ap{ap}.npz') as d:
            for m in METHODS:
                field[m].append((ap,tensor(d[m+'_mean']),tensor(d[m+'_variance'])))
                ess.append(float(d[m+'_ess']))
    predictions={m:[] for m in METHODS}
    for start in range(0,len(qrss),64):
        scans=tensor(qrss[start:start+64])
        for m in METHODS:
            score=scans.new_zeros((len(scans),len(grid)))
            for ap,mean,var in field[m]:score+=observation_log_prob(scans[:,ap,None],mean[None],var[None],obs)
            predictions[m].append((score.softmax(1)@grid).numpy())
    predictions={m:np.concatenate(v) for m,v in predictions.items()}
    local=dict(rss=support['rss'],xy=support['xy'])
    pooled=dict(rss=np.concatenate([raw()[n]['rss'] for n in meta['histories']]+[support['rss']]),
        xy=np.concatenate([raw()[n]['xy']-origin() for n in meta['histories']]+[support['xy']]))
    predictions['wknn_local']=wknn(local,dict(rss=qrss),list(cases),False)
    predictions['wknn_bank']=wknn(pooled,dict(rss=qrss),list(cases),False)
    results=[]
    for split,sl in [('internal',slice(0,len(qi))),('official',slice(len(qi),None))]:
        xy=qxy[sl];_,group,count=np.unique(xy,axis=0,return_inverse=True,return_counts=True)
        for method,pred in predictions.items():
            results.append(dict(method=method,split=split,**metrics(pred[sl],dict(xy=xy,scan_weights=1/(len(count)*count[group.ravel()])))))
    np.savez_compressed(dest/'predictions.npz',**predictions,query_xy=qxy,origin=origin())
    write(dest/'completed.json',dict(event=meta['id'],bank=meta['bank'],results=results,weights=weights,APs=len(cases),min_ess=min(ess)))
    print(json.dumps(dict(stage='completed',event=meta['id'],mde={r['method']+'_'+r['split']:r['mde'] for r in results})),flush=True)


def run(worker):
    verify();base,obs=parameters();nets={}
    for m in METHODS[1:]:
        net=OperatorNet().double();net.load_state_dict(torch.load(checkpoint(m),weights_only=True));net.eval();nets[m]=net
    events=[e for e in json.loads((OUT/'manifest.json').read_text())['events'] if e['bank']!='pair']
    for meta in events[worker::3]:event(meta,nets,base,obs)


def summarize():
    events=json.loads((OUT/'manifest.json').read_text())['events'];rows=[]
    for meta in events:
        data=json.loads((OUT/meta['id']/'completed.json').read_text())
        rows.extend([dict(event=meta['id'],bank=meta['bank'],floor=meta['floor'],**r) for r in data['results']])
    summary={}
    for bank in ['low','high','pair','all']:
        summary[bank]={}
        for split in ['internal','official']:
            sub=[r for r in rows if r['bank']==bank and r['split']==split]
            means={m:float(np.mean([r['mde'] for r in sub if r['method']==m])) for m in sorted({r['method'] for r in sub})}
            contrasts={}
            for m in METHODS[1:]:
                deltas=[]
                for e in [x for x in events if x['bank']==bank]:
                    pick=lambda method:next(r['mde'] for r in sub if r['event']==e['id'] and r['method']==method)
                    deltas.append(pick(m)-pick('adapted_bank'))
                contrasts[m]=dict(mean_change=float(np.mean(deltas)),improved_events=sum(d<0 for d in deltas))
            summary[bank][split]=dict(mean=means,versus_mature_history=contrasts)
    write(OUT/'summary.json',dict(summary=summary,records=rows,scope='Variable-history development on two previously viewed UJI floors; paired histories reused; no independent-building or arbitrary-registration claim'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','summarize']);parser.add_argument('--worker',type=int,choices=[0,1,2]);args=parser.parse_args()
    if args.stage=='run':run(args.worker)
    else:{'prepare':prepare,'summarize':summarize}[args.stage]()
