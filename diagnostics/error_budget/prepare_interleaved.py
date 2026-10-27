"""T9 v2 post-hoc coverage revision; two events, no model change or fitting."""
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from run_multi_history import digest,write
from uji_probability_study import DATA,load_raw

OLD=ROOT/'results/uji_history_bank_v1_20261026'
PAIRED=ROOT/'results/uji_operator_paired_v2_20261026'
OUT=ROOT/'results/uji_error_budget_interleaved_v2_20261026/preparation'
IDS=['B1F1_s0','B1F2_s0']


def indices(f,ids):
    lookup={int(r):i for i,r in enumerate(f['row_ids'])}
    return np.array([lookup[r] for r in ids],dtype=int)


def prepare():
    events=[e for e in json.loads((OLD/'manifest.json').read_text())['events'] if e['id'] in IDS]
    assert [e['id'] for e in events]==IDS
    length=json.loads((OLD/'parameters.json').read_text())['base']['length']
    train=load_raw('trainingData.csv',{'B1F1','B1F2'});val=load_raw('validationData.csv',{'B1F1','B1F2'})
    OUT.mkdir(parents=True,exist_ok=False)
    geometry=json.loads((ROOT/'results/uji_error_budget_coverage_design_20261026/summary.json').read_text())
    manifest=[];records=[]
    for meta in events:
        f,o=train[meta['floor']],val[meta['floor']]
        sr=indices(f,meta['support_rows']);ir=indices(f,meta['internal_rows']);vr=indices(o,meta['official_rows'])
        sx=np.unique(f['xy'][sr],axis=0);qxy=np.concatenate([f['xy'][ir],o['xy'][vr]])
        tiles=np.floor((qxy-np.asarray(meta['grid_bounds'])[0])/length).astype(int)
        labels=tiles[:,0]%2+2*(tiles[:,1]%2)
        non_support=~(f['positions'][:,None]==sx[None]).all(-1).any(-1)
        distance=np.full(len(qxy),np.nan);folds=[]
        with np.load(PAIRED/meta['id']/'predictions.npz') as p:
            np.testing.assert_allclose(p['query_xy']+p['origin'],qxy,rtol=0,atol=1e-8)
        for fold in range(4):
            take=labels==fold;held=np.unique(qxy[take],axis=0)
            available=non_support&(cKDTree(held).query(f['positions'])[0]>3.)
            assert available.any()
            rows=np.flatnonzero(available[f['group']])
            assert not np.isin(f['group'][rows],np.flatnonzero(~non_support)).any()
            nearest=cKDTree(held).query(f['xy'][rows])[0]
            assert nearest.min()>3.
            distance[take]=cKDTree(f['positions'][available]).query(qxy[take])[0]
            splits={}
            for split,ix,source,offset in [('internal',ir,f,0),('official',vr,o,len(ir))]:
                keep=take[offset:offset+len(ix)]
                splits[split]=dict(query_rows=source['row_ids'][ix[keep]].tolist(),positions=len(np.unique(source['xy'][ix[keep]],axis=0)),scans=int(keep.sum()))
            folds.append(dict(fold=fold,reference_rows=f['row_ids'][rows].tolist(),reference_positions=int(available.sum()),
                reference_scans=len(rows),support_rows=meta['support_rows'],minimum_query_distance=float(nearest.min()),splits=splits))
        assert np.isfinite(distance).all()
        for split,offset,ids in [('internal',0,meta['internal_rows']),('official',len(ir),meta['official_rows'])]:
            collected=[r for fold in folds for r in fold['splits'][split]['query_rows']]
            assert sorted(collected)==sorted(ids) and len(set(collected))==len(collected)
            xy=qxy[offset:offset+len(ids)];_,ix=np.unique(xy,axis=0,return_index=True);d=distance[offset:offset+len(ids)][ix]
            expected=next(r for r in geometry['records'] if r['event']==meta['id'] and r['split']==split and r['policy']=='interleaved_length_tiles')
            assert abs(d.mean()-expected['mean'])<1e-10
            assert abs(float((d>length).mean())-expected['fraction_farther_than_source_length'])<1e-10
            records.append(dict(event=meta['id'],floor=meta['floor'],split=split,positions=len(d),mean_reference_distance=float(d.mean()),
                median_reference_distance=float(np.median(d)),maximum_reference_distance=float(d.max()),fraction_gt_base_length=float((d>length).mean()),
                baseline_query_order_aligned=True,Q_partition_complete=True,reference_S_and_Q_exclusion_passed=True))
        manifest.append(dict(event=meta['id'],floor=meta['floor'],ap_columns=meta['ap_columns'],grid_bounds=meta['grid_bounds'],folds=folds))
    sources=[Path(__file__),ROOT/'diagnostics/error_budget/coverage_design.py',OLD/'manifest.json',OLD/'parameters.json',DATA/'trainingData.csv',DATA/'validationData.csv',
        ROOT/'results/uji_error_budget_coverage_design_20261026/summary.json',ROOT/'results/uji_error_budget_reference_v1_20261026/freeze.json',ROOT/'results/uji_error_budget_reference_v1_20261026/summary.json']
    write(OUT/'freeze.json',dict(created_unix=time.time(),purpose=__doc__,stage='Metadata only; no GP fitting or predictions',
        revision='Selected after v1 failure using coordinate-only coverage evidence; not independent pre-registered confirmation or model tuning',
        events=IDS,tile_width=length,tile_origin='original grid_bounds lower corner',fold='(floor((x-origin_x)/length) mod2)+2*(floor((y-origin_y)/length) mod2)',
        exclusion='All original S positions/repeats and all current-fold internal+official Query neighbourhoods <=3 raw units',
        unchanged='H/AP/w/base/obs/grid/Q row order and a/b/d models',source_paths={str(p):digest(p) for p in sources}))
    write(OUT/'manifest.json',dict(events=manifest))
    write(OUT/'summary.json',dict(records=records,GP_fits=0,new_predictions=0,all_checks_passed=True))
    print(json.dumps(dict(stage='prepared',records=records),indent=2))


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    prepare()
