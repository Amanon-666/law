"""Post-score attribution: compare spatial updating with omitting new-AP factors.

All original K-conditioned A fields, empirical S atoms and AP-count tempering
remain. S-only APs still enter A's learned context features, but their Query
likelihood factors are omitted. This is an information-path ablation, not a
new trained method or an independent confirmation of the candidate.
"""
from candidates import uts_new_ap_spatial_20261111 as C
from frozen_hurdle_translator import PreparedFrozenHurdle
import csv
import socket
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits


def run():
    E,G=C.E,C.G
    C.check()
    result_path=C.OUTPUT/'omission_reference_after_scoring.json'
    assert not result_path.exists()
    E.write(C.OUTPUT/'omission_reference_freeze.json',dict(created=E.now(),code_sha256=E.digest(__file__),
        candidate_scores_sha256=E.digest(C.OUTPUT/'scores.json'),
        timing='Post-score attribution; same48 tasks, unchanged candidate geometry and full-AP temperature',
        question='Does spatial conditioning help beyond omitting new-AP Query likelihood factors?'))
    train,query=E.table(E.DATA/'UTS_training.csv',history=True),E.table(E.DATA/'UTS_test.csv',history=False)
    source=E.read(E.A_OUTPUT/'freeze.json');records=[]
    for event in source['events']:
        directory=E.A_OUTPUT/event['event'];complete=E.read(directory/'complete.json')
        for item in complete['files']:assert E.digest(item['path'])==item['sha256']
        with np.load(directory/'field.npz') as f:field={k:f[k].copy() for k in f.files}
        hseen=(train['rss'][train['floor']!=event['floor']]!=100).any(0)
        keep=hseen[field['aps']]
        if keep.all():
            with np.load(directory/'query.npz') as q:mass=q['log_position_mass'].copy()
        else:
            reader=PreparedFrozenHurdle(field['cand_xy'],field['aps'][keep],
                *[field[k][:,keep] for k in ('p','m','s2')],field['bounds'],int(field['ap_count']))
            mass=reader.predict(query['rss'][event['Q_rows_zero_based']])['log_position_mass']
            mass=mass*keep.sum()/len(keep)
            mass=mass-logsumexp(mass,axis=1,keepdims=True)
        prediction=np.clip(np.exp(mass)@field['cand_xy'],field['bounds'][0],field['bounds'][1])
        with np.load(G.OUTPUT/event['event']/'predictions.npz') as gp:
            assert gp['rows'].tolist()==event['Q_rows_zero_based']
            pooled=C.fuse_hurdle_gp(field['cand_xy'],mass,gp['GP_grid'],gp['GP_scores'],gp['GP_raw'],field['bounds'])
        records.append(dict(event=event['event'],floor=event['floor'],K=event['K'],rows=event['Q_rows_zero_based'],
            omitted_APs=field['aps'][~keep].tolist(),predictions=dict(omit_new_AP=prediction.tolist(),
                omit_new_AP_GP_pool=pooled['predictions']['probability_half'].tolist())))
    E.write(C.OUTPUT/'omission_reference_predictions.json',dict(complete=True,created=E.now(),records=records,
        freeze_sha256=E.digest(C.OUTPUT/'omission_reference_freeze.json'),query_XY_opened=False))
    with (E.DATA/'UTS_test.csv').open(newline='') as stream:
        truth=np.array([[float(r['Pos_x']),float(r['Pos_y'])] for r in csv.DictReader(stream)])
    methods=('omit_new_AP','omit_new_AP_GP_pool')
    for r in records:
        xy=truth[r['rows']];_,groups=np.unique(xy,axis=0,return_inverse=True)
        r['scores']={m:E.position_score(r['predictions'][m],dict(xy=xy,groups=groups.ravel())) for m in methods}
    summary={m:{str(k):dict(positions=E.metrics([e for r in records if r['K']==k for e in r['scores'][m]['position_errors']]),
        by_floor={str(r['floor']):E.metrics(r['scores'][m]['position_errors']) for r in records if r['K']==k}) for k in (1,3,5)} for m in methods}
    E.write(result_path,dict(complete=True,created=E.now(),summary=summary,records=records,
        freeze_sha256=E.digest(C.OUTPUT/'omission_reference_freeze.json'),
        predictions_sha256=E.digest(C.OUTPUT/'omission_reference_predictions.json')))


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    with threadpool_limits(limits=2):run()
