"""Apply the frozen unified episode rule to an explicit source-building set.

Only the source-data adapter changes. Each building supplies an isolated view
to the existing task constructor; its local statistics key0 is an internal
index, never a feature. Real building IDs and original CSV row IDs are retained
in provenance. U.task, feature schema, cell sampling, losses and estimators
are reused verbatim. The target-building exclusion is explicit at loading.

This module does not change or retrain the frozen B0 checkpoint currently
under external evaluation. Broader-source fitting is not launched here and
would require its own freeze and source-matched comparison. A source building
must provide cross-acquisition groups for the existing four task modes; absent
groups are an information limitation, not permission to fabricate identities.
The R choice is a cap: all available real rows up to3 are retained, as in the
original sampler. Actual per-position counts remain in the77 input columns.
"""
import csv
import importlib.util
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
from unittest.mock import patch

import claude_hurdle_assets as A
import numpy as np
from candidates import unified_scan_field_20261111 as U


def load_sources(csv_path,buildings,excluded_buildings):
    buildings=tuple(sorted(set(buildings)));excluded_buildings=set(excluded_buildings)
    if not buildings or set(buildings)&excluded_buildings:
        raise ValueError('Source buildings must be explicit and disjoint from excluded targets')
    assert A.digest(csv_path) == U.E.CSV_SHA
    values={k:[] for k in ('ROW_IDS','RSS','XY','FLR','BLD','USER','PHONE','TS')}
    with Path(csv_path).open(newline='') as stream:
        reader=csv.DictReader(stream);waps=[n for n in reader.fieldnames if n.startswith('WAP')]
        for row_id,row in enumerate(reader,1):
            building=int(row['BUILDINGID'])
            if building not in buildings:
                continue
            values['ROW_IDS'].append(row_id);values['BLD'].append(building)
            values['RSS'].append([float(row[n]) for n in waps])
            values['XY'].append([float(row['LONGITUDE']),float(row['LATITUDE'])])
            for key,column in (('FLR','FLOOR'),('USER','USERID'),('PHONE','PHONEID'),('TS','TIMESTAMP')):
                values[key].append(int(row[column]))
    C=ModuleType('unified_source_core')
    for name,value in values.items():
        setattr(C,name,np.asarray(value,dtype=float if name in ('RSS','XY') else np.int64))
    assert set(C.BLD)==set(buildings)
    C.WAPS=waps;C.DET=C.RSS!=100
    # The unchanged fitter needs only FLOORS when supplied our cell_table;
    # its legacy predictor/training constructor are not used by this adapter.
    V=ModuleType('unified_source_floors')
    V.FLOORS={b:sorted(np.unique(C.FLR[C.BLD==b]).tolist()) for b in buildings}
    if any(len(floors)<2 for floors in V.FLOORS.values()):
        raise ValueError('Cross-floor source tasks require at least two source floors')
    names=('core','translator','vt_lib');previous={n:sys.modules[n] for n in names if n in sys.modules}
    try:
        sys.modules.update(core=C,translator=U.T,vt_lib=V)
        spec=importlib.util.spec_from_file_location('unified_source_hurdle',A.PACKAGE/'hurdle.py')
        H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    finally:
        for name in names:sys.modules.pop(name,None)
        sys.modules.update(previous)
    views={}
    for building in buildings:
        rows=np.flatnonzero(C.BLD==building)
        view=SimpleNamespace(**{name:getattr(C,name)[rows] for name in (*values,'DET')})
        statistics=SimpleNamespace(STATS={(0,f):H.STATS[(building,f)] for f in V.FLOORS[building]},det_stats=H.det_stats)
        views[building]=(view,statistics)
    return C,V,H,views


def cell_table(views,records):
    def build(pairs,rng,draws=4,Ks=(1,3,5),max_rows=6000):
        assert draws==4 and Ks==(1,3,5) and max_rows==6000
        parts=[]
        for building,floor in pairs:
            C,H=views[building]
            for K in Ks:
                for draw in range(draws):
                    values,record,_=U.task(C,H,floor,K,draw,rng)
                    selected=rng.choice(len(values[0]),min(len(values[0]),max_rows),replace=False)
                    parts.append(tuple(v[selected] for v in values))
                    records.append(dict(record,building=int(building),eligible_cells=len(values[0]),
                        selected_cells=len(selected),selected_indices_sha256=U.array_sha(selected),
                        cell_sampling='Uniform without replacement, independent of labels'))
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(5))
    return build


def fit(H,views,seed,records):
    """Unchanged estimator and cross-fitting; caller owns freeze and serialization."""
    with patch.object(H,'cell_table',cell_table(views,records)):
        return H.HurdleField(sorted(views),seed=seed,draws=4)


def check_source_views(csv_path,output):
    """No fit: unchanged B0 episodes and isolated real tasks on the other source."""
    old,_,old_H=U.E.load_source(csv_path)
    C,V,H,views=load_sources(csv_path,[0,2],[1])
    for name in ('ROW_IDS','RSS','XY','FLR','USER','PHONE','TS','DET'):
        assert np.array_equal(getattr(old,name),getattr(views[0][0],name))
    a,b=[],[]
    values_a=U.cell_table(old,old_H,a)([(0,0)],np.random.default_rng(1))
    values_b=cell_table(views,b)([(0,0)],np.random.default_rng(1))
    for x,y in zip(values_a,values_b):
        assert np.array_equal(x,y,equal_nan=True)
    assert a==[{k:v for k,v in r.items() if k!='building'} for r in b]
    checks=[]
    for building in (0,2):
        view,statistics=views[building]
        assert (view.BLD==building).all()
        groups=np.column_stack([view.USER,view.PHONE])
        for floor in V.FLOORS[building]:
            values,record,data=U.task(view,statistics,floor,5,3,np.random.default_rng([20261111,building,floor]))
            _,_,context,base,sr,qr,hr=data
            assert not np.intersect1d(base,qr).size
            assert all(f!=floor and (view.FLR[rows]==f).all() for f,rows in hr.items())
            assert (groups[sr]==record['S_group']).all() and (groups[qr]==record['Q_group']).all()
            assert record['S_group']!=record['Q_group']
            assert len(np.unique(view.XY[sr],axis=0))==5 and len(values[0])>0
            checks.append(dict(building=building,floor=floor,cells=len(values[0]),
                feature_sha256=U.array_sha(context['features']),
                actual_S_counts=np.unique(view.XY[sr],axis=0,return_counts=True)[1].tolist()))
    output.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists()
    A.write(output,dict(complete=True,created=A.datetime.now(A.timezone.utc).isoformat(),
        code_sha256=A.digest(__file__),csv_sha256=U.E.CSV_SHA,
        source_buildings=[0,2],excluded_target_buildings=[1],
        loaded_rows_by_building={str(b):int((C.BLD==b).sum()) for b in V.FLOORS},
        B0_identical_episodes=len(a),B0_all_tables_and_provenance_identical=True,
        checked_tasks=checks,fits=0,models_loaded=0,localization_scores=0,
        status='Source adapter checked; broader-source fitting not started'))


if __name__=='__main__':
    import argparse,socket
    from threadpoolctl import threadpool_limits
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert socket.gethostname()=='ray-web'
    with threadpool_limits(limits=2):
        check_source_views(args.csv,args.output)
