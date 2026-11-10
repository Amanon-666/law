"""Separate CPU source-fit and explicit target-evaluation entrypoints.

Use an external timeout for each invocation. No resume/retry, download, target
selection, GP fit or Tab model loading/forward is performed. Interrupted artifacts
remain as written; a source complete marker is required for evaluation.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits
from acquisition_tasks import load_acquisition_rows, episode_scans
from source_episode_catalog import make_episode
from observation_translation_model import ObservationTranslationModel
from observation_translation_tasks import ObservationTranslationContext
from building_information_tab_trial import context, digest, now, read_scans, read_truth, write

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
OLD = RESULTS/'tabicl_multitask_full_trial_20261107_checked'
CATALOG = RESULTS/'source_episode_catalog_20261105/catalog.json'
CSV = Path('/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv')
ASSETS = RESULTS/'building_information_assets_20261109'
REFERENCE = RESULTS/'history_training_attribution_20261110'
TRANSLATED = RESULTS/'translated_history_system_20261110'
SEED = 11509159
METHODS = ('observation_translation', 'GP', 'ordinary_HS', 'GP_ordinary_HS_half', 'GP_translated_3NN_half')
CODE = ('observation_translation_trial.py', 'observation_translation_model.py',
        'observation_translation_tasks.py', 'location_decision.py',
        'source_episode_catalog.py', 'acquisition_tasks.py', 'building_information_tab_trial.py', 'uji.py',
        'vertical_translator/translator.py')


def read(path):
    return json.loads(Path(path).read_text())


def budget(start, seconds):
    if time.monotonic()-start >= seconds:
        raise TimeoutError('Explicit segment budget exhausted; retain artifacts, no automatic continuation')


def freeze(out, inputs, contract):
    out.mkdir(exist_ok=False)
    write(out/'freeze.json', dict(created=now(), code={n:digest(ROOT/n) for n in CODE},
        inputs={str(p):digest(p) for p in inputs}, contract=contract,
        resources=dict(CPU_threads=2, GPU=0, downloads=0, external_timeout_required=True),
        runtime=dict(numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__)))


def source_pack(rows, catalog, spec):
    event = make_episode(catalog, **spec)
    protected = set(catalog['protected_rows'])
    assert event['floor'] in ('B0F0', 'B0F1', 'B0F2')
    assert not protected.intersection(event['support_rows'])
    assert all(not protected.intersection(ids) for ids in event['history_rows'].values())
    assert (not protected.intersection(event['query_rows']) if spec['phase']=='fit'
            else set(event['query_rows']) <= protected)
    assert event['role'] == ('train' if spec['phase']=='fit' else 'source_validation')
    pack = episode_scans(rows, event, np.zeros(2))
    assert not set(map(tuple,pack['support']['xy'])).intersection(map(tuple,pack['query']['xy']))
    return event, pack


def source_summary(records):
    return {metric:dict(mean=float(np.mean([np.mean([r[metric] for r in records if r['floor']==f])
                                             for f in ('B0F0','B0F1','B0F2')])),
        by_floor={f:float(np.mean([r[metric] for r in records if r['floor']==f])) for f in ('B0F0','B0F1','B0F2')},
        by_K={str(k):float(np.mean([np.mean([r[metric] for r in records if r['floor']==f and r['K']==k])
                                   for f in ('B0F0','B0F1','B0F2')])) for k in (1,3,5)})
        for metric in ('mde','interval_NLL')}


def train(args):
    start=time.monotonic(); old=read(OLD/'complete.json'); original=read(OLD/'freeze.json')
    assert old['complete'] and old['freeze_sha256']==digest(OLD/'freeze.json')
    catalog=read(CATALOG)
    assert catalog['source_floors']==['B0F0','B0F1','B0F2']
    assert original['inputs'][str(CSV)]==digest(CSV) and original['inputs'][str(CATALOG)]==digest(CATALOG)
    schedule=original['schedule']; validation=original['validation']
    assert len(schedule)==24 and all(len(batch)==3 for batch in schedule) and len(validation)==27
    specs=[s for batch in schedule for s in batch]
    assert all(s['phase']=='fit' for s in specs) and all(s['phase']=='validation' for s in validation)
    freeze(args.output,[CSV,CATALOG,OLD/'freeze.json',OLD/'complete.json'],dict(
        phase='source_train',schedule=specs,validation=validation,seed=SEED,max_iter=200,
        seconds=args.seconds,CPU_threads=2,maximum_classes=17,maximum_trees=3400,
        source_floors=catalog['source_floors'],selection=False,
        information='72 legal fit tables, one fit; persist model before 27 protected source validations; no target access',
        bins='Training-table position/scan/AP weights define at most16 detected intervals plus missing; unchanged in validation'))
    rows=load_acquisition_rows(CSV,set(catalog['source_floors'])); tables=[]
    for spec in specs:
        budget(start,args.seconds);e,p=source_pack(rows,catalog,spec)
        tables.append(ObservationTranslationContext(p['histories'],p['support'],e['floor']).source_scan_table(p['query'],e))
    budget(start,args.seconds); model=ObservationTranslationModel().fit(tables,seed=SEED,max_iter=200)
    model_path=args.output/'model.joblib';model.save(model_path);model_sha=digest(model_path)
    write(args.output/'model_saved.json',dict(created=now(),model_sha256=model_sha,cuts=model.cuts.tolist(),training_tables=72))
    del tables
    records=[]
    for spec in validation:
        budget(start,args.seconds);e,p=source_pack(rows,catalog,spec)
        c=ObservationTranslationContext(p['histories'],p['support'],e['floor'])
        nll=model.source_nll(c.source_scan_table(p['query'],e))
        pred=model.prepare(p['histories'],p['support'],e['floor']).predict(p['query']['rss'])
        path=args.output/f"{e['id']}_unscored.npz"
        np.savez_compressed(path,**pred,query_rows=e['query_rows'])
        _,group,count=np.unique(p['query']['xy'],axis=0,return_inverse=True,return_counts=True)
        weights=1/(len(count)*count[group]);error=np.linalg.norm(pred['projected']-p['query']['xy'],axis=1)
        records.append(dict(event=e['id'],floor=e['floor'],K=e['K'],mde=float(weights@error),interval_NLL=nll,
            prediction_path=str(path),prediction_sha256=digest(path)))
        write(args.output/'progress.json',dict(records=records,model_sha256=model_sha))
    assert digest(model_path)==model_sha
    write(args.output/'complete.json',dict(complete=True,created=now(),phase='source_train',seconds=time.monotonic()-start,
        freeze_sha256=digest(args.output/'freeze.json'),model_path=str(model_path),model_sha256=model_sha,
        cuts=model.cuts.tolist(),training_tables=72,validation_events=27,summary=source_summary(records),records=records,
        target_opened=False,parameters_selected=False,scope='Fixed source validation, no bin/model selection'))


def target_summary(manifest,records):
    samples=[];events={e['id']:e for e in manifest['events']}
    for r in records:
        e=events[r['event']]
        with np.load(r['scored_path']) as a:
            _,g=np.unique(a['truth'],axis=0,return_inverse=True)
            n=int(g.max())+1
            errors={m:np.array([a[m][g==i].mean() for i in range(n)]) for m in METHODS}
        nf=sum(f.startswith(e['building']) for f in manifest['floors'])
        samples.append(dict(event=e,errors=errors,weight=.5/nf/2/n))
    def agg(ss,m):
        v=np.concatenate([s['errors'][m] for s in ss]);w=np.concatenate([np.full(len(s['errors'][m]),s['weight']) for s in ss]);w/=w.sum();order=np.argsort(v)
        return dict(mean=float(w@v),position_mean_p90=float(v[order][np.searchsorted(np.cumsum(w[order]),.9)]))
    return {m:dict(agg(samples,m),**{'by_'+key:{str(c):agg([s for s in samples if s['event'][key]==c],m) for c in choices}
        for key,choices in (('K',(1,3,5)),('floor',manifest['floors']),('building',manifest['buildings']))}) for m in METHODS}


def evaluate(args):
    start=time.monotonic();source=read(args.source/'complete.json')
    assert source['complete'] and source['phase']=='source_train' and source['target_opened'] is False
    assert digest(args.source/'freeze.json')==source['freeze_sha256']
    model_path=Path(source['model_path']);assert digest(model_path)==source['model_sha256']
    # Bind the evaluated implementation to the completed source implementation.
    src_freeze=read(args.source/'freeze.json')
    for n in CODE:assert digest(ROOT/n)==src_freeze['code'][n]
    manifest=read(ASSETS/'manifest.json');asset=read(ASSETS/'complete.json')
    assert asset['manifest_sha256']==digest(ASSETS/'manifest.json') and manifest['csv_sha256']==digest(CSV)
    assert len(manifest['events'])==54 and manifest['buildings']==['B1','B2']
    old=read(REFERENCE/'complete.json');translated=read(TRANSLATED/'complete.json')
    assert old['complete'] and translated['complete']
    assert old['freeze_sha256']==digest(REFERENCE/'freeze.json') and translated['freeze_sha256']==digest(TRANSLATED/'freeze.json')
    assert read(TRANSLATED/'freeze.json')['contract']['events']==manifest['events']
    refs={r['event']:r for r in old['records']};trs={r['event']:r for r in translated['records']}
    assert set(refs)==set(trs)=={e['id'] for e in manifest['events']}
    paths=[CSV,model_path,args.source/'complete.json',args.source/'freeze.json',ASSETS/'manifest.json',
        ASSETS/'complete.json',REFERENCE/'complete.json',REFERENCE/'freeze.json',TRANSLATED/'complete.json',TRANSLATED/'freeze.json']
    for eid in refs:
        for r in (refs[eid],trs[eid]):
            p=Path(r['prediction_path']);assert digest(p)==r['prediction_sha256'];paths.append(p)
    freeze(args.output,paths,dict(phase='target_evaluate',model_sha256=source['model_sha256'],cuts=source['cuts'],
        events=manifest['events'],methods=METHODS,selection=False,seconds=args.seconds,
        old_translated_baseline='GP_translated_3NN_half uses original four-B0-floor source ensemble; not equal source corpus',
        information='H/S prepare, Query RSS predict, persist unscored, then Query truth scoring; no new half combination',
        scope='Exposed development comparison; correlated floors/directions/K, no blind or statistical guarantee'))
    model=ObservationTranslationModel.load(model_path);assert np.array_equal(model.cuts,source['cuts']);records=[]
    for e in manifest['events']:
        budget(start,args.seconds);histories,support,*_=context(CSV,manifest,e,'HS')
        prepared=model.prepare(histories,support,e['floor'])
        qrss=read_scans(CSV,e['query_rows'],False)['rss'];pred=prepared.predict(qrss)
        points={'observation_translation':pred['projected']}
        for r,names in ((refs[e['id']],METHODS[1:4]),(trs[e['id']],METHODS[4:])):
            with np.load(r['prediction_path']) as a:
                assert np.array_equal(a['query_rows'],e['query_rows'])
                points.update({m:a[m].copy() for m in names})
        path=args.output/f"{e['id']}_unscored.npz"
        np.savez_compressed(path,**points,query_rows=e['query_rows'],cand_xy=pred['cand_xy'],decision_gap=pred['decision_gap'])
        truth=read_truth(CSV,e['query_rows']);_,g,c=np.unique(truth,axis=0,return_inverse=True,return_counts=True)
        weights=1/(len(c)*c[g]);errors={m:np.linalg.norm(v-truth,axis=1) for m,v in points.items()}
        scored=args.output/f"{e['id']}_scored.npz";np.savez_compressed(scored,truth=truth,weights=weights,**errors)
        records.append(dict(event=e['id'],floor=e['floor'],K=e['K'],prediction_path=str(path),prediction_sha256=digest(path),
            scored_path=str(scored),scored_sha256=digest(scored),mde={m:float(weights@errors[m]) for m in METHODS}))
        write(args.output/'progress.json',dict(records=records,model_sha256=source['model_sha256']))
    assert digest(model_path)==source['model_sha256']
    write(args.output/'complete.json',dict(complete=True,created=now(),phase='target_evaluate',seconds=time.monotonic()-start,
        freeze_sha256=digest(args.output/'freeze.json'),model_sha256=source['model_sha256'],records=records,
        summary=target_summary(manifest,records),training_updates=0,GP_fits=0,target_selection=False,
        position_unit='Within-XY scan mean; repeated K entries are descriptive, not independent physical locations'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('train','evaluate'),required=True)
    parser.add_argument('--seconds',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    assert socket.gethostname()=='ray-web' and args.seconds>0
    assert args.phase!='evaluate' or args.source is not None
    with threadpool_limits(limits=2):
        (train if args.phase=='train' else evaluate)(args)
