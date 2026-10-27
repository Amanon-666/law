"""Frozen 24-source-task raw WKNN and S-only ordinary spatial GP baselines."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from residual_probe import prepare as prepared
# prepare imports existing solvers; its lazy data module must resolve to the snapshot.
sys.path.insert(0,str(prepared.PREPARATION/'code'))

import numpy as np
import torch
from data import group_scans
from full_ap_study import metrics,wknn
from history_bank import BankPrior
from inference import observation_log_prob
from run import ap_stats,tensor
from run_multi_history import digest,write
from source_weight_mixture import field_from_prior

OUT=ROOT/'results/source_external_baselines_v1_20261026'
METHODS=['wknn_S','wknn_HS','gp_S']
KEYS=['n','k','s1','s2']


def log(**r):print(json.dumps(dict(created=prepared.now(),**r),allow_nan=False),flush=True)


def data_module():
    data=prepared.data_module()
    assert Path(data.__file__).resolve().is_relative_to(prepared.PREPARATION/'code')
    assert Path(data.source_task.__code__.co_filename).resolve().is_relative_to(prepared.PREPARATION/'code')
    return data


def task_for(meta):
    task=data_module().task_for(meta)
    assert task['S']['scan_ids'].tolist()==meta['support_scan_ids']
    assert task['Q']['scan_ids'].tolist()==meta['query_scan_ids']
    assert task['ap_indices'].tolist()==meta['AP_indices']
    assert task['ap_ids'].tolist()==meta['AP_ids']
    assert task['metadata']['origin']==meta['origin']
    assert data_module().grid_definition(task)==meta['grid']
    for floor,h in zip(meta['histories'],task['H']):
        saved=meta['history_metadata'][str(floor)]
        assert h['scan_ids'].tolist()==saved['scan_ids']
        assert h['row_indices'].tolist()==saved['rows']
    assert len(task['S']['rss'])==5 and len(task['Q']['rss'])==12
    assert not (task['S']['positions'][:,None]==task['Q']['positions'][None]).all(-1).any()
    return task


def grid_for(meta):
    g=meta['grid']
    axes=[np.linspace(g['lower'][d],g['upper'][d],g['shape_xy'][d]) for d in range(2)]
    points=tensor(np.stack(np.meshgrid(*axes),-1).reshape(-1,2))
    assert len(points)==g['points']
    return points


def seed_for(event):
    return int.from_bytes(hashlib.sha256(('external-baselines-v1|'+event).encode()).digest()[:4],'little')


def freeze():
    data=data_module();manifest=data.verify_freeze(prepared.PREPARATION)
    for directory in [prepared.PREPARATION,prepared.OUT,prepared.OUT.parent/'paired_features']:
        summary=json.loads((directory/'summary.json').read_text())
        assert summary.get('complete',summary.get('completed',False)),directory
    assert len(manifest['events'])==24
    checked=[]
    for meta in manifest['events']:
        task=task_for(meta)
        checked.append(dict(id=meta['id'],building=meta['building'],J=len(meta['histories']),APs=len(meta['AP_indices']),
            H_scans=[len(h['rss']) for h in task['H']],S_scans=5,Q_positions=12,raw_row_alignment=True))
    modules=[prepared.history_bank,prepared.inference,prepared.transfer,prepared.reference_pmc,prepared.source_weight_mixture]
    files=[Path(__file__),ROOT/'full_ap_study.py',ROOT/'run.py',ROOT/'data.py',Path(prepared.__file__),Path(data.__file__),
        Path(data.source_task.__code__.co_filename),*[Path(m.__file__) for m in modules]]
    inputs=[prepared.PREPARATION/n for n in ['manifest.json','freeze.json','summary.json']]+[Path(manifest['parameters']),
        prepared.OUT/'summary.json',prepared.OUT/'freeze.json',prepared.OUT.parent/'paired_features/summary.json',prepared.OUT.parent/'paired_features/freeze.json']
    OUT.mkdir(exist_ok=False);(OUT/'code').mkdir()
    record=dict(created=prepared.now(),purpose=__doc__,source_scope='All 24 previously frozen source-development tasks; 12 Q positions each, not untouched confirmation',
        prior_question='How do established raw nearest-neighbour localization and a source-parameterized S-only GP perform on the exact H/S/Q/AP task set? No superiority/SOTA decision criterion is inferred from these simple controls.',
        methods=METHODS,WKNN='Exact full_ap_study.wknn(mean_by_position=False): k=min(3,reference scans), RMS RSS distance, non-detection=-110, inverse max(distance,1e-8); no query tuning',
        GP='BankPrior([],[],base): ordinary RBF latent RSS GP, fixed B0 source base/obs; existing field_from_prior M-PMC and censored observation_log_prob, no random intercept and no hyperparameter fitting',
        GP_settings=dict(seed='SHA256(external-baselines-v1|event_id) first 4 bytes little endian; same deterministic seed for every AP',
            defensive_fraction=.1,minimum_ESS=1000,all_APs_scored=True,
            cache='Identical S sufficient statistics within an event reuse the identical field; no AP is removed. Grid/coordinates/parameters/seed are fixed within that event.'),
        information='All three arms receive the identical frozen AP catalog. GP/S-WKNN use only S signal labels; GP candidates use common frozen H+S domain metadata. H+S WKNN reads exactly selected raw H rows, including original repeats. No additional source floor or unretained public scans.',
        data_isolation='Only S statistics enter GP update. WKNN predictors receive Q RSS only. Q XY used only for scoring. AP/grid from frozen H+S, no Q selection or fitting.',
        expected_limits='K5 only-S localization has sparse coordinate support; WKNN outputs lie in reference-coordinate convex hull. Fixed B0 hyperparameters are source experience, not locally estimated defaults. Additional J also changes permitted AP/domain, so J curves are whole information policies, not only sample-count effects.',
        aggregation='Per-event original position weights; report building and building/J means. Do not pool UJI original distance with public metre coordinates as universal metres.',
        base=manifest['base'],obs=manifest['obs'],parameter_scope=manifest['parameter_scope'],
        reverse_trace=['Exact frozen raw H/S/Q scan IDs -> existing source_task, identical AP ordering and origin',
            'S or exact H+S raw references + Q RSS -> unmodified raw WKNN -> coordinates',
            'S statistics + ordinary GP -> cached field on frozen grid -> sum all AP query log likelihoods -> original posterior-mean coordinates',
            'Only final predictions meet Q XY in metrics; all 24 event results retained'],
        code_sha256={str(p.resolve()):digest(p) for p in files},inputs_sha256={str(p):digest(p) for p in inputs},alignment=checked,
        resource='ray-web CPU two threads, GPU hidden, existing dependencies, offline; no data movement/download')
    write(OUT/'freeze.json',record);write(OUT/'manifest.json',manifest)
    for p in files:
        snapshot=OUT/'code'/p.resolve().relative_to(ROOT)
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,snapshot)
    log(stage='freeze',events=len(checked),APs=sum(r['APs'] for r in checked))


def verify():
    data_module().verify_freeze(prepared.PREPARATION)
    record=json.loads((OUT/'freeze.json').read_text())
    for section in ['code_sha256','inputs_sha256']:
        for path,sha in record[section].items():assert digest(Path(path))==sha,path
    return json.loads((OUT/'manifest.json').read_text())


def score(pred,task):return metrics(pred,task['Q'])


def run_wknn(meta,task):
    dest=OUT/meta['id'];dest.mkdir(exist_ok=True)
    if (dest/'wknn.json').exists():return
    q=dict(rss=task['Q']['rss']);aps=np.arange(len(meta['AP_indices']))
    hs=group_scans(np.concatenate([h['rss'] for h in task['H']]+[task['S']['rss']]),
        np.concatenate([h['xy'] for h in task['H']]+[task['S']['xy']]))
    predictions=dict(wknn_S=wknn(task['S'],q,aps,False),wknn_HS=wknn(hs,q,aps,False))
    prepared.save_case(dest/'wknn_predictions.npz',dict(**predictions,query_xy=task['Q']['xy']))
    write(dest/'wknn.json',dict(id=meta['id'],results={m:score(p,task) for m,p in predictions.items()}))
    log(stage='wknn',event=meta['id'],results={m:score(p,task)['mde'] for m,p in predictions.items()})


@torch.no_grad()
def run_gp(meta,task,base,obs):
    dest=OUT/meta['id']
    if (dest/'completed.json').exists():return
    fields=dest/'gp_fields';fields.mkdir(exist_ok=True)
    grid=grid_for(meta);sx=tensor(task['S']['positions']);seed=seed_for(meta['id'])
    prior=BankPrior([],[],base);scores=torch.zeros((12,len(grid)));mapping=[]
    for j,ap in enumerate(meta['AP_indices']):
        stats=ap_stats(task['S'],j)
        key=hashlib.sha256(b''.join(stats[k].numpy().tobytes() for k in KEYS)).hexdigest()
        path=fields/(key+'.npz')
        if not path.exists():
            result=field_from_prior(prior,obs,sx,stats,grid,seed,defensive_fraction=.1)
            if result['effective_samples']<1000:raise RuntimeError(f'Insufficient GP ESS: {meta["id"]} AP{ap}: {result["effective_samples"]}')
            prepared.save_case(path,dict(mean=result['mean'].numpy(),variance=result['variance'].numpy(),ess=np.array(result['effective_samples']),seed=np.array(seed)))
        with np.load(path) as values:
            mean,variance=tensor(values['mean']),tensor(values['variance']);ess=float(values['ess'])
        rss=tensor(task['Q']['rss'][:,j,None])
        scores+=observation_log_prob(rss,mean[None],variance[None],obs)
        mapping.append(dict(AP=ap,field_key=key,ESS=ess))
        if j%50==0:log(stage='gp',event=meta['id'],APs_done=j+1,APs_total=len(meta['AP_indices']),distinct_S_statistics=len(list(fields.glob('*.npz'))))
    pred=(scores.softmax(1)@grid).numpy()
    prepared.save_case(dest/'gp_predictions.npz',dict(prediction=pred,query_xy=task['Q']['xy'],AP_indices=meta['AP_indices']))
    old=json.loads((dest/'wknn.json').read_text())
    write(dest/'completed.json',dict(id=meta['id'],building=meta['building'],J=len(meta['histories']),seed=meta['seed'],
        coordinate_unit=meta['coordinate_unit'],position_kind=meta['position_kind'],query_scan_ids=meta['query_scan_ids'],
        results=dict(old['results'],gp_S=score(pred,task)),GP_AP_fields=mapping,minimum_ESS=min(r['ESS'] for r in mapping)))
    log(stage='completed',event=meta['id'],gp_S_mde=score(pred,task)['mde'])


def run():
    manifest=verify();data=data_module()
    # Finish the inexpensive established comparators independently of GP status.
    for meta in manifest['events']:run_wknn(meta,task_for(meta))
    for meta in manifest['events']:run_gp(meta,task_for(meta),manifest['base'],manifest['obs'])
    summarize()


def summarize():
    manifest=json.loads((OUT/'manifest.json').read_text());rows=[]
    for e in manifest['events']:
        result=json.loads((OUT/e['id']/'completed.json').read_text())
        rows.append({k:result[k] for k in ['id','building','J','seed','coordinate_unit','position_kind','results','minimum_ESS']})
    summary=[]
    for building in sorted({r['building'] for r in rows}):
        br=[r for r in rows if r['building']==building]
        for J in [None,*sorted({r['J'] for r in br})]:
            selected=br if J is None else [r for r in br if r['J']==J]
            summary.append(dict(building=building,J=J,events=len(selected),coordinate_unit=selected[0]['coordinate_unit'],
                methods={m:{k:float(np.mean([r['results'][m][k] for r in selected])) for k in ['mde','median','p90','fraction_error_gt10']} for m in METHODS}))
    write(OUT/'summary.json',dict(completed=True,events=len(rows),scope=__doc__,records=rows,summary=summary,
        minimum_ESS=min(r['minimum_ESS'] for r in rows),limits='12 Q positions/event, all source development, UAH/Pisa walking-label positions. Shared AP/domain metadata and fixed B0 parameter experience retained. No TCMS score or SOTA superiority claim.'))
    log(stage='summary',buildings=[r for r in summary if r['J'] is None])


if __name__=='__main__':
    assert socket.gethostname()=='ray-web','All numerical work stays on ray-web'
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','summarize']);args=parser.parse_args()
    dict(freeze=freeze,run=run,summarize=summarize)[args.stage]()
