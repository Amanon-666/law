"""Explicit H/current-S inputs and a separate Q scoring interface.

Reads the new B1-conditioned cache only. Target-floor caches, future S,
Q RSS and Q coordinates never construct the adaptation inputs or domain.
"""
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
import torch

from data import group_scans
from residual_probe.prepare import data_module
from residual_probe.episode_rows import nested_rows
from run import tensor
from transfer import History

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'results/fusion_history_B1_20261028'
PARAMETERS=ROOT/'results/source_base_B1_20261028/parameters.json'
K_VALUES=(1,3,5,10,20)


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def history_index():
    assert read(CACHE/'summary.json')['complete']
    return {(r['building'],r['floor']):r for r in read(CACHE/'manifest.json')['selections']}


def prepare_manifest(draws):
    """Geometry-only randomized rows; no model scores select cases."""
    events=[];unavailable=[];old=data_module()
    for building in sorted(old.CONFIGS):
        data=old.building_data(building)
        for target,raw in sorted(data['floors'].items()):
            all_h=sorted(f for f in data['floors'] if f!=target)
            far_h=[f for f in all_h if abs(f-target)>=2]
            n=len(np.unique(raw['xy'],axis=0));q_count=min(128,n-max(K_VALUES))
            if q_count<12:raise ValueError(f'Insufficient target positions: {building}/{target}')
            if not far_h:unavailable.append(dict(building=building,target=target,scenario='far',reason='No eligible distant history; not silently replaced by adjacent H'))
            for draw in range(draws):
                rows=nested_rows(raw['xy'],range(len(raw['xy'])),K_VALUES,q_count,
                                 f'fusion-source-isolated-20261028/{building}/{target}/{draw}')
                for scenario,histories in [('all',all_h),('far',far_h)]:
                    if not histories:continue
                    events.append(dict(id=f'{building}_F{target}_{scenario}_d{draw}',building=building,
                        target=target,scenario=scenario,draw=draw,histories=histories,
                        support_rows={str(k):v for k,v in rows['support_rows'].items()},query_rows=rows['query_rows'],
                        support_scan_ids={str(k):raw['scan_ids'][v].tolist() for k,v in rows['support_rows'].items()},
                        query_scan_ids=raw['scan_ids'][rows['query_rows']].tolist(),query_positions=q_count,
                        coordinate_unit='UJI original coordinate distance' if building.startswith('UJI_') else 'metres'))
    return dict(created_UTC=datetime.now(timezone.utc).isoformat(),draws_per_target=draws,K=K_VALUES,
        query_policy='At most 128 fixed positions, reserving 20 distinct S positions; one real scan per position. Whole-position exclusion.',
        scenarios='all: every other floor; far: floor-index gap >=2. Same S/Q in both, H-derived permissions recomputed per scenario.',
        missing_scenarios=unavailable,events=events,
        scope='Known development buildings; independent building verification remains separate.')


def adaptation_inputs(meta,k):
    """No Q access. Returned task has no Q rows, signals, IDs or coordinates."""
    assert meta['target'] not in meta['histories'] and len(set(meta['histories']))==len(meta['histories'])
    data=data_module().building_data(meta['building']);raw=data['floors'][meta['target']]
    sr=meta['support_rows'][str(k)]
    assert len(sr)==k and raw['scan_ids'][sr].tolist()==meta['support_scan_ids'][str(k)]
    hs=[];visible=(raw['rss'][sr]!=100).any(0)
    for floor in meta['histories']:
        info=history_index()[(meta['building'],floor)];h=data['floors'][floor];rows=info['rows']
        assert h['scan_ids'][rows].tolist()==info['scan_ids'] and data['ap_ids'].tolist()==info['AP_ids']
        visible|=(h['rss'][rows]!=100).any(0)
        hs.append(dict(floor=floor,rss=h['rss'][rows],xy=h['xy'][rows]))
    aps=np.flatnonzero(visible)
    histories=[dict(floor=h['floor'],**group_scans(h['rss'][:,aps],h['xy'])) for h in hs]
    support=group_scans(raw['rss'][sr][:,aps],raw['xy'][sr])
    permitted=np.concatenate([h['positions'] for h in histories]+[support['positions']])
    lower=np.floor(permitted.min(0)/2)*2;upper=np.ceil(permitted.max(0)/2)*2
    axes=[np.arange(lower[d],upper[d]+1,2.) for d in range(2)]
    points=np.stack(np.meshgrid(*axes),-1).reshape(-1,2)
    return dict(building=meta['building'],target=meta['target'],histories=histories,support=support,
                AP_indices=aps.tolist(),AP_ids=data['ap_ids'][aps].tolist(),points=tensor(points),
                grid=dict(lower=lower.tolist(),upper=upper.tolist(),spacing=2.,shape_xy=[len(a) for a in axes],
                          flatten_order='x varies fastest',rights='H plus current S only',frame='unscaled original coordinates'))


def query_for_scoring(meta,ap_indices):
    raw=data_module().building_data(meta['building'])['floors'][meta['target']]
    rows=meta['query_rows']
    assert raw['scan_ids'][rows].tolist()==meta['query_scan_ids']
    return dict(rss=raw['rss'][rows][:,ap_indices],xy=raw['xy'][rows],scan_ids=raw['scan_ids'][rows],
                omitted_detected_APs=((raw['rss'][rows]!=100).sum(1)-(raw['rss'][rows][:,ap_indices]!=100).sum(1)))


@lru_cache(maxsize=2048)
def load_history(building,floor,ap,parameter_sha):
    """Uses a fixed original-coordinate frame so H objects can be reused."""
    info=history_index()[(building,floor)];directory=CACHE/building/f'F{floor}'
    status=read(directory/f'ap{ap}.json')
    assert status['parameters_sha256']==parameter_sha
    with np.load(directory/f'ap{ap}.npz',allow_pickle=False) as arrays:
        if bool(arrays['empty']):return None
        return History(tensor(arrays['z'])+tensor(info['origin']),
                       dict(mean=tensor(arrays['mean']),cov=tensor(arrays['cov'])),read(PARAMETERS)['base'])


def histories_for_AP(task,ap,parameter_sha):
    floors=[h['floor'] for h in task['histories']]
    assert task['target'] not in floors
    return [load_history(task['building'],floor,ap,parameter_sha) for floor in floors]


def scoring_strata(task,query_xy,length):
    """Q coordinates enter scoring only; none of these labels choose a model."""
    s=task['support']['positions'];center=s.mean(0);a=s-center;q=query_xy-center
    rank=np.linalg.matrix_rank(a)
    if rank==0:
        inside=np.linalg.norm(q,axis=1)<=1e-8
    elif rank==1:
        direction=np.linalg.svd(a,full_matrices=False)[2][0]
        along=q@direction;projected=a@direction
        inside=(np.linalg.norm(q-along[:,None]*direction,axis=1)<=1e-8)&(along>=projected.min()-1e-8)&(along<=projected.max()+1e-8)
    else:
        hull=ConvexHull(a)
        inside=(q@hull.equations[:,:2].T+hull.equations[:,2]<=1e-8).all(1)
    distances=np.stack([cKDTree(h['positions']).query(query_xy)[0] for h in task['histories']],1)
    covered=distances<=length
    return dict(support_hull_inside=inside.tolist(),source_nearest_distances=distances.tolist(),
        complementary_geometry=(covered.any(1)&~covered.all(1)).tolist(),
        no_geometric_H_coverage=(~covered.any(1)).tolist(),
        geometry_scale=length,meaning='H geometry diagnostic, not a claim that historical precision implies target relevance')
