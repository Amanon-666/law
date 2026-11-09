"""Source-only query rotation with immutable H/S adaptation and candidate space.

Keep twelve positions and three real scans per position per update. Rotate over
all eligible source positions, excluding the entire support-position groups.
This increases unique source supervision; it does not create new environments.
"""
import numpy as np
import torch
from uji import tensor


def query_schedule(events, floors, schedule, seed, positions=12, repeats=3):
    """Freeze all row IDs before optimization, without using RSS or model scores."""
    rng=np.random.default_rng(seed);pools={};cursors={};details={}
    for ident in sorted(set(schedule)):
        event=events[ident]
        assert event['role']=='train'
        raw=floors[event['floor']]
        xy,groups,counts=np.unique(raw['xy'],axis=0,return_inverse=True,return_counts=True)
        support_indices=np.flatnonzero(np.isin(raw['row_ids'],event['support_rows']))
        assert len(support_indices)==len(event['support_rows'])
        support_groups=np.unique(groups[support_indices])
        eligible=np.flatnonzero((counts>=repeats)&~np.isin(np.arange(len(xy)),support_groups))
        assert len(eligible)>=positions
        pools[ident]=(rng.permutation(eligible),groups,xy)
        cursors[ident]=0
        details[ident]=dict(floor=event['floor'],eligible_positions=len(eligible),
                           support_positions=len(support_groups),selected_groups=set(),selected_rows=set())
    batches=[]
    for step,ident in enumerate(schedule,1):
        pool,groups,xy=pools[ident];raw=floors[events[ident]['floor']]
        selected=pool[(cursors[ident]+np.arange(positions))%len(pool)]
        cursors[ident]+=positions
        assert len(np.unique(selected))==positions
        rows=[]
        for group in selected:
            rows.extend(raw['row_ids'][rng.choice(np.flatnonzero(groups==group),repeats,replace=False)].tolist())
        details[ident]['selected_groups'].update(selected.tolist())
        details[ident]['selected_rows'].update(rows)
        batches.append(dict(step=step,event=ident,query_rows=rows,positions=xy[selected].tolist()))
    for item in details.values():
        item['unique_query_positions']=len(item.pop('selected_groups'))
        item['unique_query_rows']=len(item.pop('selected_rows'))
    return dict(seed=seed,positions_per_update=positions,scans_per_position=repeats,
                steps=len(batches),batches=batches,coverage=details)


def set_queries(episode, query):
    """Change only supervised readout data, preserving adaptive numerical state.

    Do not copy case dictionaries: a proposal refreshed during training belongs
    to this fixed H/S task and must survive the next query batch.
    """
    episode['truth']=tensor(query['xy'])
    episode['nearest']=torch.cdist(episode['truth'],episode['grid']).argmin(1)
    episode['query_weights']=tensor(1/(len(query['positions'])*query['stats']['n'][query['group']]))
    for ap,case in episode['cases'].items():case['rss']=tensor(query['rss'][:,ap])
