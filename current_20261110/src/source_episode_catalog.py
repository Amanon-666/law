"""Metadata-only source tasks with validation positions excluded from every H/S/Q.

Keeps the existing plan's source buildings. No RSS parsing, fitted statistics,
model loading, or training. Task multiplicity is not environment independence.
"""
import csv
import hashlib
import json
import math
import random
import socket
from collections import Counter
from datetime import datetime, timezone

from acquisition_tasks import acquisitions

SEED=1150545


def generator(*parts):
    seed=int.from_bytes(hashlib.sha256(json.dumps([SEED,*parts],sort_keys=True).encode()).digest()[:8],'little')
    return random.Random(seed)


def read_metadata(path, floors):
    rows={}
    with open(path,newline='') as f:
        for row_id,r in enumerate(csv.DictReader(f),1):
            floor=f"B{r['BUILDINGID']}F{r['FLOOR']}"
            if floor not in floors:
                continue
            stamp=int(r['TIMESTAMP'])
            rows[row_id]=dict(floor=floor,xy=(float(r['LONGITUDE']),float(r['LATITUDE'])),timestamp=stamp,
                acquisition=(int(r['USERID']),int(r['PHONEID']),
                    datetime.fromtimestamp(stamp,timezone.utc).date().isoformat()))
    return rows


def build_catalog(rows, plan):
    parents={}
    for e in plan['events']:
        if e['role']=='train':
            if e['floor'] in parents:
                assert e['history_rows']==parents[e['floor']]['history_rows']
            parents.setdefault(e['floor'],e)
    floors=sorted(parents)
    assert set(r['floor'] for r in rows.values())==set(floors)
    protected={}
    for floor in floors:
        positions=sorted({r['xy'] for r in rows.values() if r['floor']==floor})
        protected[floor]=set(generator('position_split',floor).sample(positions,math.ceil(len(positions)*.25)))
    protected_rows=sorted(i for i,r in rows.items() if r['xy'] in protected[r['floor']])
    forbidden=set(protected_rows)
    history_rows={}
    groups={}
    for floor,parent in parents.items():
        assert set(parent['histories'])==set(parent['history_rows'])
        assert floor not in parent['history_rows']
        assert set(parent['history_rows'])<=set(floors)
        history_rows[floor]={name:[i for i in ids if i not in forbidden]
                             for name,ids in parent['history_rows'].items()}
        for index,(key,positions) in enumerate(sorted(acquisitions(rows,floor).items())):
            pools={phase:[] for phase in ('train','validation')}
            for xy,ids in sorted(positions.items()):
                if len(ids)<3:
                    continue
                phase='validation' if xy in protected[floor] else 'train'
                pools[phase].append(dict(xy=list(xy),rows=ids[:3]))
            groups[f'{floor}_group{index}']=dict(floor=floor,acquisition=list(key),**pools)
    families=[];rejected=[]
    for sid,sg in groups.items():
        for qid,qg in groups.items():
            if sg['floor']!=qg['floor'] or sg['acquisition'][:2]==qg['acquisition'][:2]:
                continue
            reasons=[]
            if len(sg['train'])<5:reasons.append('support has fewer than 5 nonreserved R3 positions')
            if len(qg['train'])<17:reasons.append('query has fewer than 17 nonreserved R3 positions')
            if len(qg['validation'])<6:reasons.append('query has fewer than 6 reserved R3 positions')
            record=dict(id=f'{sid}_to_{qid}',floor=sg['floor'],support_group=sid,query_group=qid)
            if reasons:rejected.append(dict(record,reasons=reasons))
            else:families.append(record)
    counts=Counter(e['floor'] for e in families)
    assert set(counts)==set(floors),'A source floor has no eligible tasks; do not silently drop the floor'
    for e in families:e['sampling_probability']=1/(len(floors)*counts[e['floor']])
    catalog=dict(seed=SEED,source_floors=floors,
        protected_positions={f:[list(x) for x in sorted(p)] for f,p in protected.items()},
        protected_rows=protected_rows,history_rows=history_rows,groups=groups,
        families=families,rejected=rejected,
        contract=dict(K=[1,3,5],R=3,fit_query_positions=12,validation_query_positions=6,
            validation_fraction=.25,history='Original plan sources, removing all protected rows',
            initialization='Fresh public weights and training-only global statistics; old B0 fits prohibited',
            sampling='Equal source floor, then equal directed acquisition pair; variants are correlated tasks',
            scope='Source position validation, not unseen-floor/building or unseen-device validation'))
    verify_catalog(rows,catalog)
    return catalog


def make_episode(catalog, family_id, draw_index, *, k=5, histories=None, phase='fit'):
    assert k in (1,3,5) and phase in ('fit','validation') and draw_index>=0
    family=next(e for e in catalog['families'] if e['id']==family_id)
    floor=family['floor'];sg=catalog['groups'][family['support_group']];qg=catalog['groups'][family['query_group']]
    support=generator('support',family['support_group'],draw_index).sample(sg['train'],5)
    maximum_support={tuple(p['xy']) for p in support}
    if phase=='fit':
        pool=[p for p in qg['train'] if tuple(p['xy']) not in maximum_support]
        query=generator('fit',family_id,draw_index).sample(pool,12)
    else:
        query=generator('validation',family['query_group']).sample(qg['validation'],6)
    allowed=catalog['history_rows'][floor]
    names=sorted(allowed) if histories is None else sorted(histories)
    assert names and len(names)==len(set(names)) and set(names)<=set(allowed)
    return dict(id=f'{family_id}_draw{draw_index}_K{k}_H{"-".join(names)}_{phase}',
        role='train' if phase=='fit' else 'source_validation',floor=floor,K=k,R=3,
        histories=names,history_rows={h:allowed[h] for h in names},
        support_rows=[i for p in support[:k] for i in p['rows']],
        query_rows=[i for p in query for i in p['rows']],
        support_acquisition=sg['acquisition'],query_acquisition=qg['acquisition'],
        sampling_probability=family['sampling_probability'])


def verify_catalog(rows,catalog):
    protected=set(catalog['protected_rows'])
    floors=set(catalog['source_floors'])
    reserved={f:{tuple(x) for x in v} for f,v in catalog['protected_positions'].items()}
    assert protected=={i for i,r in rows.items() if r['xy'] in reserved[r['floor']]}
    for target,histories in catalog['history_rows'].items():
        for floor,ids in histories.items():
            assert target!=floor and ids and floor in floors
            assert not protected.intersection(ids)
            assert all(rows[i]['floor']==floor for i in ids)
    for group in catalog['groups'].values():
        for phase in ('train','validation'):
            for p in group[phase]:
                assert len(p['rows'])==len(set(p['rows']))==3
                assert all(rows[i]['floor']==group['floor'] and list(rows[i]['xy'])==p['xy']
                           and list(rows[i]['acquisition'])==group['acquisition'] for i in p['rows'])
                assert all((i in protected)==(phase=='validation') for i in p['rows'])
    for family in catalog['families']:
        sg=catalog['groups'][family['support_group']];qg=catalog['groups'][family['query_group']]
        assert sg['floor']==qg['floor']==family['floor']
        assert sg['acquisition'][:2]!=qg['acquisition'][:2]
        assert len(sg['train'])>=5 and len(qg['train'])>=17 and len(qg['validation'])>=6


def verify_draws(rows,catalog):
    """Check deployable identities and counterexample that would leak via H."""
    count=0
    for family in catalog['families']:
        histories=sorted(catalog['history_rows'][family['floor']])
        for draw in (0,1):
            for phase in ('fit','validation'):
                full=make_episode(catalog,family['id'],draw,k=5,phase=phase)
                assert full==make_episode(catalog,family['id'],draw,k=5,phase=phase)
                maximum_s={rows[i]['xy'] for i in full['support_rows']}
                for names in ([h] for h in histories):
                    smaller=make_episode(catalog,family['id'],draw,histories=names,phase=phase)
                    assert smaller['support_rows']==full['support_rows'] and smaller['query_rows']==full['query_rows']
                for k in (1,3,5):
                    e=make_episode(catalog,family['id'],draw,k=k,phase=phase)
                    assert e['support_rows']==full['support_rows'][:3*k] and e['query_rows']==full['query_rows']
                    assert len(set(e['support_rows']))==3*k
                    assert len(set(e['query_rows']))==3*(12 if phase=='fit' else 6)
                    assert not maximum_s.intersection(rows[i]['xy'] for i in e['query_rows'])
                    assert sorted(Counter(rows[i]['xy'] for i in e['support_rows']).values())==[3]*k
                    count+=1
    # A reserved row entering ANY other task's history must be rejected.
    changed=json.loads(json.dumps(catalog))
    target=changed['source_floors'][0]
    history=next(iter(changed['history_rows'][target]))
    leaked=next(i for i in changed['protected_rows'] if rows[i]['floor']==history)
    changed['history_rows'][target][history].append(leaked)
    rejected=False
    try:verify_catalog(rows,changed)
    except AssertionError:rejected=True
    assert rejected,'Cross-task validation leakage was not detected'
    return count


def main():
    from recomposition_trial import ROOT, PLAN, CSV, digest, write, now
    out=ROOT/'results/source_episode_catalog_20261105'
    out.mkdir(exist_ok=False)
    plan=json.loads(PLAN.read_text())
    floors={e['floor'] for e in plan['events'] if e['role']=='train'}
    names=['source_episode_catalog.py','acquisition_tasks.py','uji.py','recomposition_trial.py']
    write(out/'freeze.json',dict(created=now(),seed=SEED,
        code={n:digest(ROOT/n) for n in names},inputs={str(p):digest(p) for p in (PLAN,CSV)},
        task='Metadata task construction only; no RSS conversion, model fitting, prediction or GPU use',
        limits=dict(cpu=1,seconds=120,downloads=0)))
    for n in names:(out/n).write_bytes((ROOT/n).read_bytes())
    rows=read_metadata(CSV,floors)
    catalog=build_catalog(rows,plan)
    count=verify_draws(rows,catalog)
    write(out/'catalog.json',catalog)
    examples=[make_episode(catalog,e['id'],0) for e in catalog['families']]
    write(out/'example_events.json',examples)
    summary=dict(complete=True,created=now(),source_floors=catalog['source_floors'],
        metadata_rows=len(rows),protected_rows=len(catalog['protected_rows']),groups=len(catalog['groups']),
        families_by_floor=dict(Counter(e['floor'] for e in catalog['families'])),
        excluded_pairs=len(catalog['rejected']),verified_draws=count,
        history_rows={f:{h:len(ids) for h,ids in hs.items()} for f,hs in catalog['history_rows'].items()},
        protected_positions={f:len(p) for f,p in catalog['protected_positions'].items()},
        cross_task_leakage_rejected=True,RSS_parsed=False,model_updates=0,localization_scores=0)
    write(out/'complete.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    main()
