"""ReFi source-environment selection: fit B0, select B2, assess B1 later.

Builds row-level tasks from existing catalogs without reading RSS or fitting.
B2 previously used in training/development is NOT a fresh blind evaluation.
Old B0+B2 checkpoints cannot enter this new B0-only fitting protocol.
"""
import json
import math
import socket
from collections import Counter
from pathlib import Path
from paired_acquisition_tasks import make_quartet,verify_quartet
from source_episode_catalog import read_metadata

ROOT=Path(__file__).resolve().parent
CATALOG=ROOT/'results/building_source_tasks_20261106'
OUT=ROOT/'results/source_environment_validation_20261107'
FIT_BUILDING='B0'
SELECTION_BUILDING='B2'
ASSESSMENT_BUILDING='B1'


def make_events(catalog,pairs,role,*,k=3,draw=0):
    """One metadata-selected acquisition pair per floor; AA/AB share S.

    Both matched and changed acquisition scans are retained. This default
    defines 4 fitting and 5 selection H/S contexts, not 18 independent tasks.
    Creating descriptors does not authorize fitting every context at once.
    """
    building={'fit':FIT_BUILDING,'selection':SELECTION_BUILDING}[role]
    eligible=sorted((p for p in pairs if p['floor'].startswith(building+'F')),key=lambda p:p['id'])
    chosen={}
    for p in eligible:chosen.setdefault(p['floor'],p)
    events=[]
    for floor,pair in sorted(chosen.items()):
        quartet=make_quartet(catalog,pair,draw,k=k,phase='fit' if role=='fit' else 'validation')
        # Each floor has equal mass, split equally between matched/cross scans.
        for e in quartet[:2]:
            events.append(dict(e,role=role,selection_weight=1/(2*len(chosen))))
    assert events and math.isclose(sum(e['selection_weight'] for e in events),1.)
    return events


def select_checkpoint(records,events):
    """Pick an already-frozen checkpoint using ONLY source-selection MDE.

    Each record gives its step, fitted_buildings, checkpoint identity, and
    event MDEs (position-balanced geometric-median readout). The initial
    unlearned checkpoint is mandatory and remains a substantive comparator.
    This routine selects an existing model; it never refits on selection RSS.
    """
    assert events and all(e['role']=='selection' and e['floor'].startswith(SELECTION_BUILDING+'F') for e in events)
    weights={e['id']:e['selection_weight'] for e in events}
    assert len(weights)==len(events) and math.isclose(sum(weights.values()),1.)
    assert records and len({r['step'] for r in records})==len(records)
    assert any(r['step']==0 for r in records)
    scored=[]
    for r in records:
        assert r['step']>=0 and r['fitted_buildings']==[FIT_BUILDING]
        assert set(r['event_mde'])==set(weights)
        assert all(math.isfinite(v) and v>=0 for v in r['event_mde'].values())
        score=sum(weights[e]*r['event_mde'][e] for e in weights)
        scored.append(dict(step=r['step'],checkpoint=r['checkpoint'],source_selection_mde=score))
    return min(scored,key=lambda r:(r['source_selection_mde'],r['step']))


def run():
    from building_source_tasks import sha,write
    from datetime import datetime,timezone
    import time
    started=time.monotonic();OUT.mkdir(exist_ok=False)
    catalog=json.loads((CATALOG/'catalog.json').read_text())
    pairs=json.loads((CATALOG/'paired_catalog.json').read_text())['pairs']
    csv=ROOT.parent/'UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv'
    parent=json.loads((CATALOG/'freeze.json').read_text());assert sha(csv)==parent['CSV_sha256']
    code=['source_environment_validation.py','paired_acquisition_tasks.py','source_episode_catalog.py']
    frozen=dict(created=datetime.now(timezone.utc).isoformat(),code={n:sha(ROOT/n) for n in code},
        inputs={str(p):sha(p) for p in (CATALOG/'catalog.json',CATALOG/'paired_catalog.json',csv)},
        roles=dict(shared_fit=FIT_BUILDING,source_selection=SELECTION_BUILDING,later_development=ASSESSMENT_BUILDING),
        design='K3 R3 draw0, first eligible pair by ID per floor, AA/AB equal, floors equal',
        selection='Position-balanced geometric-median MDE; identity step0 included; tie selects earlier step',
        scope='Executable task and model-selection interface; no GP preparation, training, actual model selection or target scoring',
        constraints=['Global model/base/observation/normalization fits restricted to B0',
            'B2 H/S may fit only task-specific state and historical fields, never shared parameters',
            'B2 Q only scores frozen checkpoints; B1 entirely absent until checkpoint choice is fixed',
            'B2/B1 have been used previously; this is a new development protocol, not an unseen test',
            'Old B0+B2 trained weights are prohibited, not relabeled as B0-only',
            'Only one fitting building and one selection building; no claim of sufficient environment coverage'],
        resources=dict(cpu=1,gpu=0,seconds=60,downloads=0,training_updates=0))
    write(OUT/'freeze.json',frozen)
    for n in code:(OUT/n).write_bytes((ROOT/n).read_bytes())
    rows=read_metadata(csv,set(catalog['source_floors']))
    plans={role:make_events(catalog,pairs,role) for role in ('fit','selection')}
    protected=set(catalog['protected_rows']);used={}
    for role,events in plans.items():
        allowed_building=FIT_BUILDING if role=='fit' else SELECTION_BUILDING
        used[role]=set()
        for e in events:
            full=make_quartet(catalog,next(p for p in pairs if p['id']==e['pair_id']),0,k=3,
                              phase='fit' if role=='fit' else 'validation')
            verify_quartet(rows,catalog,full,'fit' if role=='fit' else 'validation')
            ids=set(e['support_rows']+e['query_rows']).union(*(set(v) for v in e['history_rows'].values()))
            assert all(rows[i]['floor'].startswith(allowed_building+'F') for i in ids)
            assert not protected.intersection(e['support_rows'])
            assert all(not protected.intersection(v) for v in e['history_rows'].values())
            if role=='fit':assert not protected.intersection(e['query_rows'])
            else:assert set(e['query_rows'])<=protected
            used[role].update(ids)
        assert not any(rows[i]['floor'].startswith(ASSESSMENT_BUILDING+'F') for i in used[role])
    assert not used['fit'].intersection(used['selection'])
    # Deliberately imbalanced scans/duplicate identities do not define weights.
    fake=[dict(id='floor_a_AA',role='selection',floor='B2F0',selection_weight=.25),
          dict(id='floor_a_AB',role='selection',floor='B2F0',selection_weight=.25),
          dict(id='floor_b_AA',role='selection',floor='B2F1',selection_weight=.25),
          dict(id='floor_b_AB',role='selection',floor='B2F1',selection_weight=.25)]
    rec=lambda step,vals:dict(step=step,checkpoint=f'fixture_{step}',fitted_buildings=['B0'],event_mde=dict(zip([e['id'] for e in fake],vals)))
    records=[rec(0,[10,10,10,10]),rec(4,[8,8,11,11]),rec(8,[7,7,15,15])]
    assert select_checkpoint(records,fake)['step']==4
    assert select_checkpoint([records[0],rec(4,[10,10,10,10])],fake)['step']==0
    bad={**records[1],'fitted_buildings':['B0','B2']}
    try:select_checkpoint([records[0],bad],fake)
    except AssertionError:pass
    else:raise AssertionError('Selection-contaminated fitted checkpoint accepted')
    write(OUT/'tasks.json',plans)
    summary=dict(complete=True,seconds=time.monotonic()-started,
        events={r:len(es) for r,es in plans.items()},
        floors={r:sorted({e['floor'] for e in es}) for r,es in plans.items()},
        H_S_contexts={r:len(es)//2 for r,es in plans.items()},
        query_positions={r:{e['floor']:len(e['query_rows'])//3 for e in es} for r,es in plans.items()},
        source_data_roles_disjoint=True,RSS_parsed=False,model_updates=0,GP_refits=0,
        selection_interface_checks_passed=True,real_checkpoints_selected=0,
        artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    write(OUT/'complete.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    run()
