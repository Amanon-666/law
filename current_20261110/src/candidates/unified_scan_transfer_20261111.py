"""Unchanged unified field and reader on the existing complete IPIN protocol.

Only the protocol adapter differs from static UJI. Model, features, unique
H/S candidate union, readout and GP pool are identical. Existing GP states
are reused. Native GP must be unchanged; its candidate-grid endpoint may
change because duplicated coordinates are intentionally removed.
"""
import claude_hurdle_assets as A
import argparse
from pathlib import Path
import socket
import time
from unittest.mock import patch

import numpy as np
import torch

import ipin2025_confirmation as O
import ipin2025_detection_ipw as S
from candidates import unified_scan_field_20261111 as U

MODELS = A.ROOT/'results/unified_scan_b0_ens5_20261111'


def freeze(args):
    for name,sha in S.OLD_SHA.items():
        assert A.digest(S.BASELINE/name) == sha
    source = O.read(MODELS/'complete.json')
    frozen = O.read(MODELS/'freeze.json')
    assert source['complete'] and source['recipe'] == frozen['recipe'] == U.RECIPE
    assert source['freeze_sha256'] == args.model_freeze_sha256 == A.digest(MODELS/'freeze.json')
    assert frozen['code']['candidates/unified_scan_field_20261111.py'] == A.digest(U.__file__)
    old = O.read(S.BASELINE/'freeze.json')
    contexts = O.read(S.BASELINE/'contexts_complete.json')
    assert contexts['complete'] and len(contexts['contexts']) == 15
    inputs = [S.BASELINE/n for n in S.OLD_SHA]
    inputs += [MODELS/'complete.json',MODELS/'freeze.json']
    inputs += [Path(r['path']) for r in old['inputs']]
    inputs += [Path(r['path']) for r in contexts['contexts']]
    inputs += [MODELS/r['path'] for r in source['models']]
    code = (*O.CODE,'cached_retained_readout.py','ipin2025_detection_ipw.py',
            'candidates/finite_observation_20261111/features.py',
            str(Path(U.__file__).relative_to(A.ROOT)),str(Path(__file__).relative_to(A.ROOT)))
    args.output.mkdir(parents=True,exist_ok=False)
    A.write(args.output/'freeze.json',dict(created=S.now(),
        **{k:old[k] for k in ('eligible_floors','K','R','methods','logs')},
        inputs=[O.binding(p) for p in dict.fromkeys(inputs)],code={n:A.digest(A.ROOT/n) for n in code},
        model_generation=dict(path=str(MODELS),freeze_sha256=args.model_freeze_sha256),
        contract=dict(recipe=U.RECIPE,question='Does one unified next-scan field improve transfer without dataset-specific adaptation?',
            fixed='Identical model and raw reader to UJI; original H/S/Q, APs, bounds, GP state and fixed half pool',
            changed='Unique candidate coordinates and learned distributions at every H/S position; no empirical-S append',
            native_GP='Unchanged; candidate GP endpoint recomputed on unique coordinates',
            status='Exposed external evaluation; all30 streams, no model/reader/weight selection or adoption',
            training=0,GP_fits=0,resources=dict(CPU_threads=2,GPU=0,segment_seconds=args.seconds))))


def prepare(args):
    if not args.resume:
        freeze(args)
    O.verify_freeze(args.output)
    translator = U.UnifiedScanField(MODELS)
    manifest = O.read(O.ASSETS/'manifest.json')
    records = []; start = time.monotonic()
    for record in O.read(S.BASELINE/'contexts_complete.json')['contexts']:
        directory = args.output/record['event']; path = directory/'complete.json'
        if path.exists():
            complete = O.read(path)
            assert complete['freeze_sha256'] == A.digest(args.output/'freeze.json')
            assert A.digest(complete['hurdle']['path']) == complete['hurdle']['sha256']
        else:
            if time.monotonic()-start >= args.seconds:
                return 2
            parent = S.checked(record)
            histories,support,ap_ids = O.context(manifest,record['floor'],record['K'])
            assert parent['ap_ids'] == ap_ids and parent['sources'] == list(histories)
            field = translator.prepare(histories,support,f"IPIN2025_B2025F{record['floor']}")
            assert A.digest(parent['hurdle']['path']) == parent['hurdle']['sha256']
            with np.load(parent['hurdle']['path']) as old:
                assert np.array_equal(np.unique(old['cand_xy'],axis=0),field.cand_xy)
                for n in ('aps','bounds','ap_count'):
                    assert np.array_equal(old[n],getattr(field,n))
            directory.mkdir(exist_ok=False)
            fp = directory/'hurdle.npz'
            np.savez_compressed(fp,**{n:getattr(field,n) for n in ('cand_xy','aps','p','m','s2','bounds','ap_count')})
            complete = dict(parent,created=S.now(),hurdle=O.binding(fp),parent_context=record,
                freeze_sha256=A.digest(args.output/'freeze.json'),source_model_freeze_sha256=args.model_freeze_sha256,
                empirical_support_appended=False,candidates_unique=True)
            A.write(path,complete)
        records.append(dict(event=record['event'],floor=record['floor'],K=record['K'],**O.binding(path)))
        print(f'Prepared {len(records)}/15 unified contexts',flush=True)
    A.write(args.output/'contexts_complete.json',dict(complete=True,created=S.now(),contexts=records,
        freeze_sha256=A.digest(args.output/'freeze.json'),official_query_opened=False,GT_opened=False))
    return 0


def verify_predictions(output):
    baseline = {(r['log_id'],r['floor'],r['K']):r for r in O.read(S.BASELINE/'predictions.json')['predictions']}
    saved = O.read(output/'predictions_progress.json')
    checks = []
    for r in saved:
        old = baseline[r['log_id'],r['floor'],r['K']]
        assert A.digest(r['path']) == r['sha256'] and A.digest(old['path']) == old['sha256']
        for n in ('receipt_predictions','ticks','duplicate_bssid_rows','conflicting_rss_rows'):
            assert r[n] == old[n]
        with np.load(r['path']) as a,np.load(old['path']) as b:
            assert np.array_equal(a['times'],b['times']) and np.array_equal(a['snapshot_times'],b['snapshot_times'])
            difference = float(np.max(abs(a['GP']-b['GP'])))
            assert difference <= 1e-7
        checks.append(dict(log_id=r['log_id'],floor=r['floor'],K=r['K'],native_GP_difference=difference))
    assert len({(r['log_id'],r['floor'],r['K']) for r in checks}) == len(checks)
    A.write(output/'native_GP_parity.json',dict(complete=len(checks)==30,checks=checks,
        candidate_GP='Different candidate multiplicity is intentional; not asserted identical'))
    return len(checks)==30


def predict(args):
    with patch.object(O,'score_retained_fields',S.score_cached_fields):
        status = O.predict_streams(args)
    verify_predictions(args.output)
    return status


def score(args):
    O.verify_freeze(args.output)
    assert verify_predictions(args.output)
    from ipin2025_score import score as score_saved
    result = score_saved(args.output/'predictions.json')
    previous = O.read(S.BASELINE/'scores.json')
    for key in ('schema','gt_sha256','eligible_floors','methods','gt_rows','metric','coverage','reference'):
        assert previous[key] == result[key]
    assert len(previous['points']) == len(result['points'])
    for a,b in zip(previous['points'],result['points']):
        assert {k:v for k,v in a.items() if k!='errors'} == {k:v for k,v in b.items() if k!='errors'}
    A.write(args.output/'scores.json',result)
    A.write(args.output/'comparison.json',dict(complete=True,recipe=U.RECIPE,
        baseline=O.binding(S.BASELINE/'scores.json'),current=O.binding(args.output/'scores.json'),
        by_K=S.compare_group(previous['by_K'],result['by_K']),adopted=False,
        status='Same complete unified recipe as UJI; no IPIN-specific tuning'))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-freeze-sha256',required=True)
    parser.add_argument('--seconds',type=float,default=540.)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(); args.output,args.device = args.output.resolve(),'cpu'
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 540
    assert A.digest(MODELS/'freeze.json') == args.model_freeze_sha256
    torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
    raise SystemExit({'prepare':prepare,'predict':predict,'score':score}[args.phase](args))
