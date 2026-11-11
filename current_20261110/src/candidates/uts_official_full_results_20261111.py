"""Add complete retained-GP/system outcomes to the fixed official UTS scoring.

No inference, tuning, partial-floor ranking or prediction alteration. Require
all48 GP/pool tasks and all original A/simple-reference results, then apply
the same position-balanced and nested-K metrics to the two added methods.
"""
import argparse
import csv
import socket

import numpy as np
from candidates import uts_official_evaluation_20261111 as E
from candidates import uts_official_gp_20261111 as G


def freeze():
    E.check()
    G.check()
    path = E.OUTPUT/'full_system_scoring_freeze.json'
    assert not path.exists()
    E.write(path,dict(created=E.now(),code_sha256=E.digest(__file__),
        evaluator_freeze_sha256=E.digest(E.OUTPUT/'freeze.json'),
        GP_freeze_sha256=E.digest(G.OUTPUT/'freeze.json'),
        added_methods=['retained_GP','original_A02_GP'],
        metrics='Identical to the already frozen A/WKNN evaluator; all16 floors and48 tasks required',
        query_XY_opened=False,adoption='Research judgment only, no automatic replacement'))


def score():
    E.check()
    _,tasks = G.check()
    frozen = E.read(E.OUTPUT/'full_system_scoring_freeze.json')
    assert frozen['code_sha256']==E.digest(__file__)
    assert frozen['evaluator_freeze_sha256']==E.digest(E.OUTPUT/'freeze.json')
    assert frozen['GP_freeze_sha256']==E.digest(G.OUTPUT/'freeze.json')
    complete = E.read(G.OUTPUT/'predictions_complete.json')
    assert complete['complete'] and complete['freeze_sha256']==frozen['GP_freeze_sha256']
    assert len(complete['records'])==len(tasks['events'])==48
    for item in complete['records']:
        assert E.digest(item['path'])==item['sha256']
    base = E.read(E.OUTPUT/'scores.json')
    assert base['complete'] and base['freeze_sha256']==frozen['evaluator_freeze_sha256']
    assert [r['event'] for r in base['records']]==[e['event'] for e in tasks['events']]
    with (E.DATA/'UTS_test.csv').open(newline='') as stream:
        truth = np.array([[float(r['Pos_x']),float(r['Pos_y'])] for r in csv.DictReader(stream)])
    for event,record,binding in zip(tasks['events'],base['records'],complete['records']):
        saved = E.read(binding['path'])
        assert saved['event']==event and saved['freeze_sha256']==frozen['GP_freeze_sha256']
        for item in saved['files']:
            assert E.digest(item['path'])==item['sha256']
        with np.load(G.OUTPUT/event['event']/'predictions.npz') as values:
            assert values['rows'].tolist()==record['rows']==event['Q_rows_zero_based']
            predictions = dict(retained_GP=values['GP_projected'],original_A02_GP=values['probability_half'])
            qxy = truth[record['rows']]
            positions,groups = np.unique(qxy,axis=0,return_inverse=True)
            assert positions.tolist()==record['positions']
            for method,pred in predictions.items():
                record['scores'][method]=E.position_score(pred,dict(xy=qxy,groups=groups.ravel()))
                record['predictions'][method]=pred.tolist()
    for method in frozen['added_methods']:
        base['summary'][method],base['paired'][method]={},{}
        for k in (1,3,5):
            selected=[r for r in base['records'] if r['K']==k]
            base['summary'][method][str(k)]=dict(
                positions=E.metrics([e for r in selected for e in r['scores'][method]['position_errors']]),
                scans=E.metrics([e for r in selected for e in r['scores'][method]['scan_errors']]),
                equal_floor_mean=float(np.mean([r['scores'][method]['mean'] for r in selected])),
                by_floor={str(r['floor']):E.metrics(r['scores'][method]['position_errors']) for r in selected})
        for a,b in ((1,3),(3,5),(1,5)):
            differences,floors=[],{}
            for floor in range(-3,13):
                ra=next(r for r in base['records'] if r['floor']==floor and r['K']==a)
                rb=next(r for r in base['records'] if r['floor']==floor and r['K']==b)
                assert ra['positions']==rb['positions'] and ra['rows']==rb['rows']
                delta=np.asarray(rb['scores'][method]['position_errors'])-ra['scores'][method]['position_errors']
                differences.extend(delta.tolist()); floors[str(floor)]=float(delta.mean())
            delta=np.asarray(differences)
            base['paired'][method][f'{a}_to_{b}']=dict(n=len(delta),mean_change=float(delta.mean()),
                improved=int((delta < -1e-8).sum()),worsened=int((delta > 1e-8).sum()),
                improved_over5m=int((delta < -5).sum()),worsened_over5m=int((delta > 5).sum()),
                floor_mean_changes=floors)
    base.update(created=E.now(),full_system_complete=True,
        original_scoring_sha256=E.digest(E.OUTPUT/'scores.json'),
        full_system_scoring_freeze_sha256=E.digest(E.OUTPUT/'full_system_scoring_freeze.json'),
        GP_complete_sha256=E.digest(G.OUTPUT/'predictions_complete.json'),
        remaining='One fixed support draw; cross-draw stability and other environments remain separate questions')
    assert not (E.OUTPUT/'full_scores.json').exists()
    E.write(E.OUTPUT/'full_scores.json',base)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','score'))
    args=parser.parse_args()
    assert socket.gethostname()=='ray-web'
    globals()[args.phase]()
