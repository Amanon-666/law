"""Source-only acquisition-held-out tasks from existing UJI metadata.

Groups are observed (user, phone, UTC date), not certified independent physical
replicates. S/Q use different user-phone pairs and disjoint positions. Metadata
defines partitions only; it is not supplied as a location predictor's feature.
"""
import csv
from collections import Counter,defaultdict
from datetime import datetime,timezone

import numpy as np
from uji import group_scans


def load_acquisition_rows(path, allowed_floors):
    rows = {}
    with open(path,newline='') as f:
        for row_id,r in enumerate(csv.DictReader(f),1):
            name = f"B{r['BUILDINGID']}F{r['FLOOR']}"
            if name not in allowed_floors:
                continue
            stamp = int(r['TIMESTAMP'])
            group = (int(r['USERID']),int(r['PHONEID']),
                     datetime.fromtimestamp(stamp,timezone.utc).date().isoformat())
            rows[row_id] = dict(floor=name,xy=(float(r['LONGITUDE']),float(r['LATITUDE'])),
                rss=[int(r[f'WAP{i:03}']) for i in range(1,521)],
                acquisition=group,timestamp=stamp)
    return rows


def acquisitions(rows, floor):
    groups = defaultdict(lambda:defaultdict(list))
    for row_id,r in rows.items():
        if r['floor'] == floor:
            groups[r['acquisition']][r['xy']].append(row_id)
    for positions in groups.values():
        for ids in positions.values():
            ids.sort(key=lambda i:(rows[i]['timestamp'],i))
    return groups


def make_tasks(rows, source_events, seed=1150535):
    """One fixed nested K1/3/5 family per eligible directed acquisition pair.

    R3 uses three actual records, without deduplication or replacement. A fixed
    12-position Q pool excludes the complete K5 support pool, including at K1.
    H uses exactly the existing source event's rows; it may contain the query
    user-phone on OTHER floors. This is not a globally unseen-device protocol.
    """
    rng = np.random.default_rng(seed)
    parents = {}
    for e in source_events:
        assert e['role'] == 'train' and e['floor'].startswith('B0')
        parents.setdefault(e['floor'],e)
    families = []; rejected = []; catalog = []
    for floor,parent in sorted(parents.items()):
        groups = acquisitions(rows,floor)
        for key,positions in sorted(groups.items()):
            catalog.append(dict(floor=floor,acquisition=key,positions=len(positions),
                R3_positions=sum(len(ids)>=3 for ids in positions.values())))
        for sg,sp in sorted(groups.items()):
            for qg,qp in sorted(groups.items()):
                if sg[:2] == qg[:2]:
                    continue
                available_s = sorted(x for x,ids in sp.items() if len(ids)>=3)
                available_q = sorted(x for x,ids in qp.items() if len(ids)>=3)
                if len(available_s)<5:
                    rejected.append(dict(floor=floor,S=sg,Q=qg,reason='fewer than 5 R3 support positions'))
                    continue
                sxy = [available_s[i] for i in rng.permutation(len(available_s))[:5]]
                available_q = [x for x in available_q if x not in sxy]
                if len(available_q)<12:
                    rejected.append(dict(floor=floor,S=sg,Q=qg,reason='fewer than 12 disjoint R3 query positions'))
                    continue
                qxy = [available_q[i] for i in rng.permutation(len(available_q))[:12]]
                events = []
                for k in (1,3,5):
                    events.append(dict(id=f'acquisition_{len(families)}_K{k}',role='train',floor=floor,
                        parent_event=parent['id'],histories=parent['histories'],history_rows=parent['history_rows'],
                        support_rows=[i for x in sxy[:k] for i in sp[x][:3]],
                        query_rows=[i for x in qxy for i in qp[x][:3]],
                        support_acquisition=list(sg),query_acquisition=list(qg),K=k,R=3))
                families.append(dict(floor=floor,support_acquisition=sg,query_acquisition=qg,events=events))
    return dict(seed=seed,families=families,rejected=rejected,catalog=catalog)


def verify_tasks(rows, manifest):
    for family in manifest['families']:
        events = family['events']
        assert [e['K'] for e in events] == [1,3,5]
        assert events[0]['support_rows'] == events[1]['support_rows'][:3]
        assert events[1]['support_rows'] == events[2]['support_rows'][:9]
        assert all(e['query_rows'] == events[0]['query_rows'] for e in events)
        maximal_s = {rows[i]['xy'] for i in events[-1]['support_rows']}
        for e in events:
            sr,qr = e['support_rows'],e['query_rows']
            assert len(set(sr)) == len(sr) == 3*e['K'] and len(set(qr)) == len(qr) == 36
            assert not maximal_s.intersection(rows[i]['xy'] for i in qr)
            assert all(list(rows[i]['acquisition']) == e['support_acquisition'] for i in sr)
            assert all(list(rows[i]['acquisition']) == e['query_acquisition'] for i in qr)
            assert e['support_acquisition'][:2] != e['query_acquisition'][:2]
            assert sorted(Counter(rows[i]['xy'] for i in sr).values()) == [3]*e['K']
            assert sorted(Counter(rows[i]['xy'] for i in qr).values()) == [3]*12
            assert all(rows[i]['floor'] == e['floor'] for i in sr+qr)


def episode_scans(rows, event, origin):
    """Return predictor-compatible arrays and S-only group provenance.

    The caller performs the existing H/S adaptation afresh. Old mixed-S cached
    weights, posterior draws and proposals cannot be reused for these tasks.
    """
    def collect(ids):
        return group_scans(np.asarray([rows[i]['rss'] for i in ids],dtype=np.int16),
                           np.asarray([rows[i]['xy'] for i in ids])-origin)
    query = dict(collect(event['query_rows']),acquisition_group=tuple(event['query_acquisition']))
    return dict(histories={h:collect(ids) for h,ids in event['history_rows'].items()},
                support=collect(event['support_rows']),query=query,
                support_acquisition_groups=[tuple(event['support_acquisition'])])


def adapt_acquisition(histories, support, acquisition, base, obs, history_reader):
    """Fresh existing H/S fit, with a declared single-group sampling contract."""
    from episodic_tasks import adapt_task
    task = adapt_task(histories,support,base,obs,history_reader)
    task['support_acquisition_groups'] = [acquisition]
    return task
