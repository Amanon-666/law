"""Matched cross-floor tasks for the existing two/three-parameter LAW kernels.

No new network. B0 histories + disjoint K5 support/query train the correction
covariance; target B1 histories remain inputs at deployment. The supervised
scan score uses the same Gaussian-moment readout as deployed localization.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from calibration_relation import CalibrationRelation
from data import group_scans
from inference import observation_log_prob, predict_from_posterior
from multi_history import GaussianStrengthRelation, single_history
from reference_pmc import setup, sample_reference, log_proposal
from run import ap_stats, tensor
from run_multi_history import digest, write
from sourcewise_calibration_transfer import SourcewiseCalibratedBankPrior
from support_integral import ImportanceSupport
from uji_history_bank_study import OUT as BANK, load_raw
from uji_probability_study import DATA
import run_bank_calibration_transfer as evaluation

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/uji_bank_crossfloor_learning_v1_20261026'
FLOORS = ['B0F0','B0F1','B0F2']
KINDS = ['uniform','strength_rbf']


def log(**row): print(json.dumps(row, allow_nan=False), flush=True)


def relation(kind, base):
    cls = GaussianStrengthRelation if kind=='strength_rbf' else CalibrationRelation
    return cls(kind, base['length']).double()


def prepare():
    OUT.mkdir(exist_ok=False)
    params = json.loads((BANK/'parameters.json').read_text()); base,obs = params['base'],params['obs']
    files = [Path(__file__), *[ROOT/p for p in ['sourcewise_calibration_transfer.py','bank_calibration.py',
        'history_bank.py','inference.py','reference_pmc.py','support_integral.py','transfer.py','multi_history.py',
        'calibration_relation.py','uji_history_bank_study.py','uji_probability_study.py','data.py']]]
    freeze = dict(created_unix=time.time(), hypothesis='Existing correction relations can be learned more usefully when source episodes match multi-history cross-floor deployment; no new representation is introduced',
        base=base, obs=obs, source_floors=FLOORS, histories_per_task=2, target_support_positions=5,
        scans_per_position=3, query_positions=12, source_tasks=18, aps_per_task=4,
        source_seed=261027, updates=80, learning_rate=.02, optimizer='Adam, same number of updates and AP cases for both families',
        objective='Position-balanced conditional scan NLL after S adaptation, using Gaussian posterior moments exactly as in the existing location backend',
        trainable='uniform: local fraction and spatial length; strength_rbf: same plus RSS difference length',
        fixed='Equal source weights, historical states, prior mean and point variance, observation parameters, data and model family',
        source_lineage='B0 observation and base parameters were already fitted and remain frozen; 80 updates are additional relation training',
        historical_input='Two other B0 floors in each source task, full original scans with native XY/AP correspondence',
        query_rights='Training Q labels only define prediction-loss evaluation locations; no Q enters H cache or S posterior',
        ap_rule='Four APs sampled from those detected in either history; S-only APs use the unchanged parameter-free local fallback',
        gradient='Fresh S-only M-PMC proposal each update, fixed-proposal differentiable posterior weights and moments',
        contrast='Task-matched training vs same-family initialization and ordinary shared correction; not a pure Office-vs-UJI causal contrast because readout approximation also matches deployment here',
        target='Exact existing six B1 development events after fixed update 80, no target checkpoint selection',
        success='Lower source loss is insufficient; need improved B1 localization vs initial and mature controls',
        limits='Only three source floors in one building; two or three global scalars cannot represent arbitrary changes or establish innovation',
        reverse_audit=['Two source-floor scan sets -> fixed per-AP H posteriors',
            'Five distinct target-source positions -> fifteen raw S records -> posterior under current relation',
            'Posterior weights -> mean/covariance -> deployed Gaussian field prediction at disjoint Q positions',
            'Q scan score -> relation gradient only; no update to H, observations or mean',
            'Fixed final checkpoint -> identical deployment path with B1 H and S'],
        counterexamples=dict(no_support='All families have identical one-point marginals, so covariance learning alone cannot change single-scan localization',
            one_support='Cannot identify arbitrary distant changes; kernel imposes a shared prior',
            zero_drift='All relation derivatives vanish', missing_history='Common local GP has no relation parameters',
            wrong_prior='Zero-mean covariance correction cannot invent a target spatial pattern without S coverage',
            identical_histories='No gain from source distinctions; reduces to the corresponding mature relation',
            source_order='Weighted source kernel sum is permutation invariant',
            coordinate_translation='All spatial calculations use differences; no target-query-based scale normalization',
            conditional_objective='Scan NLL does not enforce ranking against incorrect candidate positions; localization remains a separate test'),
        code_sha256={str(p):digest(p) for p in files},
        inputs={str(p):digest(p) for p in [BANK/'parameters.json', DATA/'trainingData.csv']})
    write(OUT/'freeze.json',freeze); shutil.copy2(BANK/'parameters.json',OUT/'parameters.json')
    (OUT/'code').mkdir()
    for p in files: shutil.copy2(p,OUT/'code'/p.name)
    floors = load_raw('trainingData.csv',set(FLOORS)); rng = np.random.default_rng(freeze['source_seed'])
    origin = np.concatenate([f['positions'] for f in floors.values()]).mean(0)
    events=[]
    for repetition in range(6):
        for name in FLOORS:
            f=floors[name]; others=[n for n in FLOORS if n!=name]
            chosen=rng.permutation(np.flatnonzero(f['stats']['n']>=3))[:17]
            sr=np.concatenate([rng.choice(np.flatnonzero(f['group']==g),3,replace=False) for g in chosen[:5]])
            qr=np.concatenate([rng.choice(np.flatnonzero(f['group']==g),3,replace=False) for g in chosen[5:]])
            seen=np.concatenate([floors[n]['rss'] for n in others])!=100
            aps=rng.choice(np.flatnonzero(seen.any(0)),4,replace=False)
            events.append(dict(id=len(events),floor=name,histories=others,support_rows=f['row_ids'][sr].tolist(),
                query_rows=f['row_ids'][qr].tolist(),aps=aps.tolist()))
    write(OUT/'manifest.json',dict(events=events,origin=origin.tolist()))
    histories={}
    for name in FLOORS:
        f=floors[name]; centered=group_scans(f['rss'],f['xy']-origin)
        aps=sorted({ap for event in events if name in event['histories'] for ap in event['aps']})
        dest=OUT/'histories'/name; dest.mkdir(parents=True)
        for ap in aps:
            h=single_history(dict(scans=centered),ap,base,obs,steps=640)
            arrays=dict(empty=np.array(h is None),origin=origin)
            if h is not None: arrays.update(z=h.z.numpy(),mean=h.m.numpy(),cov=h.c.numpy())
            np.savez_compressed(dest/f'ap{ap}.npz',**arrays); histories[(name,ap)]=h
        log(stage='histories',floor=name,aps=len(aps))
    batches=[]
    for meta in events:
        f=floors[meta['floor']]; ids={int(v):i for i,v in enumerate(f['row_ids'])}
        sr=np.array([ids[i] for i in meta['support_rows']]); qr=np.array([ids[i] for i in meta['query_rows']])
        s=group_scans(f['rss'][sr],f['xy'][sr]-origin); q=group_scans(f['rss'][qr],f['xy'][qr]-origin)
        assert len(s['positions'])==5 and len(q['positions'])==12
        assert not (s['positions'][:,None]==q['positions'][None]).all(-1).any()
        batches.append([dict(histories=[histories[n,ap] for n in meta['histories']],sx=tensor(s['positions']),
            qx=tensor(q['positions']),stats=ap_stats(s,ap),group=torch.tensor(q['group']),rss=tensor(q['rss'][:,ap]),
            weights=tensor(1/(len(q['positions'])*q['stats']['n'][q['group']])),episode=dict(meta,ap=ap)) for ap in meta['aps']])
    torch.save(batches,OUT/'source_cases.pt')
    for kind in KINDS: torch.save(relation(kind,base).state_dict(),OUT/f'{kind}_initial.pt')
    write(OUT/'prepared.json',dict(completed=True,cases=72,source_cases_sha256=digest(OUT/'source_cases.pt')))
    log(stage='prepared',tasks=len(batches),cases=72)


def prior_for(model, case, base):
    return SourcewiseCalibratedBankPrior(case['histories'],[.5,.5],base,model)


def samples_for(model,case,base,obs,seed):
    with torch.no_grad():
        p=prior_for(model,case,base); m,f=p.features(case['sx']); k=p.covariance(case['sx'],case['sx'],f,f)
        target,proposal,center,root,_=setup(m,k,case['stats'],obs)
        samples,_,info=sample_reference(target,proposal,center,root,seed)
        return samples,log_proposal(samples,center,root,info),info


def loss_for(model,case,base,obs,samples,logq):
    p=prior_for(model,case,base); sm,sf=p.features(case['sx']); sk=p.covariance(case['sx'],case['sx'],sf,sf)
    posterior=ImportanceSupport(sm,sk,case['stats'],obs,samples,logq)
    w=posterior.log_weights.exp(); mean=w@samples; c=samples-mean
    pm=dict(mean=mean,cov=(c.T*w)@c)
    qm,qf=p.features(case['qx']); cross=p.covariance(case['qx'],case['sx'],qf,sf)
    prediction=predict_from_posterior(qm,p.diagonal(case['qx'],qf),cross,sm,sk,pm)
    g=case['group']; lp=observation_log_prob(case['rss'],prediction['mean'][g],prediction['variance'][g],obs)
    return -(case['weights']*lp).sum()


def audit(freeze,batches):
    base,obs=freeze['base'],freeze['obs']; rows=[]; case=batches[0][0]
    for kind in KINDS:
        model=relation(kind,base); samples,logq,info=samples_for(model,case,base,obs,261027)
        value=loss_for(model,case,base,obs,samples,logq)
        gradient=torch.autograd.grad(value,list(model.parameters())); checks=[]
        for (name,p),g in zip(model.named_parameters(),gradient):
            original=p.item(); eps=1e-4
            with torch.no_grad():
                p.fill_(original+eps); plus=loss_for(model,case,base,obs,samples,logq).item()
                p.fill_(original-eps); minus=loss_for(model,case,base,obs,samples,logq).item();p.fill_(original)
            numerical=(plus-minus)/(2*eps)
            np.testing.assert_allclose(g.item(),numerical,rtol=2e-3,atol=2e-5)
            checks.append(dict(parameter=name,autograd=g.item(),finite_difference=numerical))
        rows.append(dict(kind=kind,loss=value.item(),effective_samples=info['effective_samples'],gradients=checks))
    write(OUT/'audit.json',dict(completed=True,scope='One source case, new moment-loss derivative only; not a performance experiment',rows=rows))
    log(stage='audited',rows=rows)


def train(freeze,batches,kind):
    assert (OUT/'audit.json').exists()
    base,obs=freeze['base'],freeze['obs']; model=relation(kind,base)
    optimizer=torch.optim.Adam(model.parameters(),lr=freeze['learning_rate']); started=time.perf_counter()
    with (OUT/f'{kind}_training.jsonl').open('x') as stream:
        for step in range(freeze['updates']):
            optimizer.zero_grad(); values=[]; ess=[]
            for j,case in enumerate(batches[step%len(batches)]):
                samples,logq,info=samples_for(model,case,base,obs,261027+step*4+j)
                value=loss_for(model,case,base,obs,samples,logq); (value/4).backward()
                values.append(value.item()); ess.append(info['effective_samples'])
            norm=torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())).item()
            if not np.isfinite(values).all() or not np.isfinite(norm): raise RuntimeError('Nonfinite source loss/gradient')
            optimizer.step()
            row=dict(step=step,mean_nll=float(np.mean(values)),grad_norm=norm,min_ess=min(ess),
                parameters={n:p.item() for n,p in model.named_parameters()},seconds=time.perf_counter()-started)
            stream.write(json.dumps(row)+'\n');stream.flush()
            if step%10==0 or step==freeze['updates']-1: log(stage='train',kind=kind,**row)
    torch.save(model.state_dict(),OUT/f'{kind}.pt')
    write(OUT/f'{kind}_completed.json',dict(completed=True,updates=freeze['updates'],seconds=time.perf_counter()-started))


def configure_evaluation():
    evaluation.OUT=OUT/'evaluation'; evaluation.SOURCE=OUT
    evaluation.METHODS={name:pair for name,pair in evaluation.METHODS.items() if name.endswith('_trained')}
    evaluation.CalibratedBankPrior=SourcewiseCalibratedBankPrior


def prepare_evaluation():
    for kind in KINDS: assert (OUT/f'{kind}_completed.json').exists()
    configure_evaluation(); evaluation.prepare()
    freeze=json.loads((evaluation.OUT/'freeze.json').read_text())
    freeze.update(source='B0F0/F1/F2 cross-floor tasks, fixed update 80, matched Gaussian-moment readout',
        source_parameter_lineage='Existing B0 base/observation fit plus this 80-update conditional calibration fit',
        new_source_updates=80, source_training_record=json.loads((OUT/'freeze.json').read_text()),
        composition='Equal weighted sum of per-source kernels; same as frozen sourcewise controls',
        hypothesis='Task-matched shared calibration improves target localization over identical initialization and ordinary correction',
        matched_initial_controls={'uniform':str(ROOT/'results/uji_bank_calibration_transfer_v1_20261026'),
            'strength':str(ROOT/'results/uji_bank_sourcewise_transfer_v1_20261026')},
        spatial_unit_rule='Same UJI coordinate units and fixed base, length ratio equals one')
    # Reuse identical initial targets instead of rerunning them: verify the actual
    # model parameters after the earlier prescribed Office->UJI conversion.
    previous=json.loads((ROOT/'results/uji_bank_calibration_transfer_v1_20261026/transferred_parameters.json').read_text())
    for name,kind in [('uniform_initial','uniform'),('strength_initial','strength_rbf')]:
        state=torch.load(OUT/f'{kind}_initial.pt',weights_only=True)
        for key,value in state.items(): np.testing.assert_allclose(value.item(),previous[name][key],rtol=1e-12,atol=1e-12)
    freeze['code_sha256'][str(Path(__file__))]=digest(Path(__file__))
    freeze['code_sha256'][str(ROOT/'sourcewise_calibration_transfer.py')]=digest(ROOT/'sourcewise_calibration_transfer.py')
    write(evaluation.OUT/'freeze.json',freeze)
    shutil.copy2(__file__,evaluation.OUT/'code'/Path(__file__).name)


if __name__=='__main__':
    if socket.gethostname()!='ray-web': raise RuntimeError('Server numerical work only')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','audit','train','prepare-evaluation','event','summarize'])
    p.add_argument('--kind',choices=KINDS);p.add_argument('--event');args=p.parse_args()
    if args.stage=='prepare': prepare()
    else:
        freeze=json.loads((OUT/'freeze.json').read_text())
        for group in ['code_sha256','inputs']:
            for path,expected in freeze[group].items(): assert digest(Path(path))==expected,path
        if args.stage in ['audit','train']:
            assert digest(OUT/'source_cases.pt')==json.loads((OUT/'prepared.json').read_text())['source_cases_sha256']
            batches=torch.load(OUT/'source_cases.pt',weights_only=False)
            if args.stage=='audit': audit(freeze,batches)
            else: train(freeze,batches,args.kind)
        elif args.stage=='prepare-evaluation': prepare_evaluation()
        else:
            configure_evaluation()
            ev=json.loads((evaluation.OUT/'freeze.json').read_text())
            for group in ['code_sha256','history_sha256','inputs','source_weights']:
                for path,expected in ev[group].items(): assert digest(Path(path))==expected,path
            if args.stage=='event': evaluation.event(args.event)
            else: evaluation.summarize()
