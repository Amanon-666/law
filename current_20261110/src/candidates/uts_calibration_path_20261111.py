"""Post-score K3->K5 computational-path decomposition on every UTS floor.

Question: which changed input path contributes to the observed calibration
regression? Reuse sealed original fields, without models, training or tuning.
Sequence: original K3 -> K5 fields on K3 APs/candidate entries -> add K5 APs
on those same entries -> original K5 with its two extra empirical S entries.
Thus the first transition changes only the learned historical field: the old
empirical S entries must be exactly unchanged. The second includes the original
reader's AP-count tempering; it is not an isolated AP-likelihood intervention.
The third adds empirical S atoms. Attribution is conditional on this stated
order, not unique causal attribution or new independent performance evidence.

All16 floors and official Query rows are retained. Two diagnostic readouts per
floor are sealed before scoring; original endpoints are reused verbatim.
"""
from candidates import uts_official_evaluation_20261111 as E
from frozen_hurdle_translator import PreparedFrozenHurdle
import argparse
import csv
from pathlib import Path
import socket

import numpy as np
from threadpoolctl import threadpool_limits

OUTPUT=E.ROOT/'results/uts_calibration_path_20261111'
STAGES=('K3','updated_common_field','added_APs','K5')


def freeze():
    E.check()
    complete=E.read(E.A_OUTPUT/'A_complete.json')
    assert complete['complete'] and complete['freeze_sha256']==E.A_FREEZE
    OUTPUT.mkdir(exist_ok=False)
    E.write(OUTPUT/'freeze.json',dict(created=E.now(),code_sha256=E.digest(__file__),
        A_complete_sha256=E.digest(E.A_OUTPUT/'A_complete.json'),
        A_scores_sha256=E.digest(E.OUTPUT/'scores.json'),
        evaluator_sha256=E.digest(E.ROOT/'candidates/uts_official_evaluation_20261111.py'),
        reader_sha256=E.digest(E.ROOT/'frozen_hurdle_translator.py'),stages=list(STAGES),
        timing='Post-score mechanism diagnostic; K3->K5 chosen because observed tail regression',
        scope='All16 floors, same official Queries; no outcome-selected floor exclusion',
        attribution='Ordered computational decomposition; AP expansion includes original tempering',
        training=False,adoption=False,source_field_regeneration=False,CPU_threads=2,GPU=0))


def run():
    E.check()
    frozen=E.read(OUTPUT/'freeze.json')
    assert frozen['code_sha256']==E.digest(__file__)
    for key,path in [('A_complete_sha256',E.A_OUTPUT/'A_complete.json'),
                     ('A_scores_sha256',E.OUTPUT/'scores.json'),
                     ('evaluator_sha256',E.ROOT/'candidates/uts_official_evaluation_20261111.py'),
                     ('reader_sha256',E.ROOT/'frozen_hurdle_translator.py')]:
        assert frozen[key]==E.digest(path)
    query=E.table(E.DATA/'UTS_test.csv',history=False)
    records=[]
    for floor in range(-3,13):
        fields,events,original={},{},{}
        for k in (3,5):
            directory=E.A_OUTPUT/f'F{floor}_K{k}'
            complete=E.read(directory/'complete.json')
            assert complete['freeze_sha256']==E.A_FREEZE
            for item in complete['files']:
                assert E.digest(item['path'])==item['sha256']
            events[k]=complete['event']
            with np.load(directory/'field.npz') as saved:
                fields[k]={name:saved[name].copy() for name in saved.files}
            with np.load(directory/'query.npz') as saved:
                assert saved['rows'].tolist()==events[k]['Q_rows_zero_based']
                original[k]=saved['projected'].tolist()
        a,b=fields[3],fields[5]
        assert events[3]['Q_rows_zero_based']==events[5]['Q_rows_zero_based']
        assert events[3]['S_rows_zero_based']==events[5]['S_rows_zero_based'][:9]
        nh=len(a['cand_xy'])-3
        assert nh==len(b['cand_xy'])-5 and np.array_equal(a['cand_xy'][:nh],b['cand_xy'][:nh])
        match=(a['cand_xy'][nh:,None]==b['cand_xy'][None,nh:]).all(-1)
        assert (match.sum(1)==1).all()
        entries=np.r_[np.arange(nh),nh+match.argmax(1)]
        assert np.array_equal(a['cand_xy'],b['cand_xy'][entries])
        columns=np.searchsorted(b['aps'],a['aps'])
        assert np.array_equal(b['aps'][columns],a['aps'])
        for name in ('p','m','s2'):
            assert np.array_equal(a[name][nh:],b[name][entries[nh:]][:,columns])
        rss=query['rss'][events[3]['Q_rows_zero_based']]
        predictions=dict(K3=original[3],K5=original[5])
        for name,cols in [('updated_common_field',columns),('added_APs',np.arange(len(b['aps'])))]:
            if name=='added_APs' and np.array_equal(a['aps'],b['aps']):
                predictions[name]=predictions['updated_common_field']
                continue
            field=PreparedFrozenHurdle(a['cand_xy'],b['aps'][cols],
                *[b[n][entries][:,cols] for n in ('p','m','s2')],a['bounds'],int(b['ap_count']))
            predictions[name]=field.predict(rss)['projected'].tolist()
        records.append(dict(floor=floor,rows=events[3]['Q_rows_zero_based'],
            history_candidates=nh,APs_K3=len(a['aps']),APs_K5=len(b['aps']),predictions=predictions))
        print('Completed calibration path F'+str(floor),flush=True)
    E.write(OUTPUT/'predictions.json',dict(created=E.now(),complete=True,
        freeze_sha256=E.digest(OUTPUT/'freeze.json'),records=records,query_XY_opened=False))
    # The readout intervention is now sealed for every floor; score afterwards.
    with (E.DATA/'UTS_test.csv').open(newline='') as stream:
        truth=np.array([[float(r['Pos_x']),float(r['Pos_y'])] for r in csv.DictReader(stream)])
    for record in records:
        xy=truth[record['rows']]
        _,groups=np.unique(xy,axis=0,return_inverse=True)
        record['scores']={name:E.position_score(pred,dict(xy=xy,groups=groups.ravel()))
                          for name,pred in record['predictions'].items()}
    summary={name:E.metrics([e for r in records for e in r['scores'][name]['position_errors']]) for name in STAGES}
    transitions=[]
    for a,b in zip(STAGES[:-1],STAGES[1:]):
        floors={str(r['floor']):r['scores'][b]['mean']-r['scores'][a]['mean'] for r in records}
        delta=np.concatenate([np.asarray(r['scores'][b]['position_errors'])-r['scores'][a]['position_errors'] for r in records])
        transitions.append(dict(before=a,after=b,mean_change=float(delta.mean()),
            improved=int((delta < -1e-8).sum()),worsened=int((delta > 1e-8).sum()),
            improved_over5m=int((delta < -5).sum()),worsened_over5m=int((delta > 5).sum()),
            floor_mean_changes=floors))
    E.write(OUTPUT/'scores.json',dict(created=E.now(),complete=True,
        freeze_sha256=E.digest(OUTPUT/'freeze.json'),predictions_sha256=E.digest(OUTPUT/'predictions.json'),
        summary=summary,transitions=transitions,records=records,
        interpretation='Post-score ordered mechanism diagnostic; no automatic new method or adoption'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','run'))
    args=parser.parse_args()
    assert socket.gethostname()=='ray-web'
    with threadpool_limits(limits=2):
        globals()[args.phase]()
