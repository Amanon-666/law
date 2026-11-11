"""Evaluate the already-trained, unchanged acquisition recipe on all IPIN streams.

Reuse the existing context replacement, stream replay and scorer. The ORIGINAL
raw reader is used, exactly as on UJI; no count conditioning, selected weight,
new source fit, GP fit or target adaptation rule. This is exposed external
evaluation of one frozen candidate, not a new IPIN-specific training recipe.
"""
import claude_hurdle_assets as A  # original CPU2/offline environment
import argparse
from pathlib import Path
import socket
from unittest.mock import patch

import torch

import ipin2025_confirmation as O
import ipin2025_detection_ipw as S
from candidates.finite_observation_20261111 import ipin2025_trial as F
from frozen_hurdle_translator import FrozenHurdleTranslator

MODELS = A.ROOT/'results/claude_hurdle_b0_acquisition_ens5_20261111'


def freeze(args):
    assert args.output != S.BASELINE and S.BASELINE not in args.output.parents
    for name, sha in S.OLD_SHA.items():
        assert A.digest(S.BASELINE/name) == sha
    source = O.read(MODELS/'complete.json')
    assert source['complete'] and source['recipe'] == 'coherent_acquisition_55_v1'
    assert source['freeze_sha256'] == args.model_freeze_sha256 == A.digest(MODELS/'freeze.json')
    old = O.read(S.BASELINE/'freeze.json')
    contexts = O.read(S.BASELINE/'contexts_complete.json')
    assert contexts['complete'] and len(contexts['contexts']) == 15
    inputs = [S.BASELINE/n for n in S.OLD_SHA]
    inputs += [MODELS/'complete.json',MODELS/'freeze.json']
    inputs += [Path(r['path']) for r in old['inputs']]
    inputs += [Path(r['path']) for r in contexts['contexts']]
    inputs += [MODELS/r['path'] for r in source['models']]
    for r in contexts['contexts']:
        S.checked(r)
    code = (*O.CODE,'cached_retained_readout.py','ipin2025_detection_ipw.py',
            str(Path(F.__file__).relative_to(A.ROOT)),str(Path(__file__).relative_to(A.ROOT)))
    args.output.mkdir(parents=True,exist_ok=False)
    A.write(args.output/'freeze.json',dict(created=S.now(),
        **{k:old[k] for k in ('eligible_floors','K','R','methods','logs')},
        inputs=[O.binding(p) for p in dict.fromkeys(inputs)],
        code={n:A.digest(A.ROOT/n) for n in code},
        model_generation=dict(path=str(MODELS),freeze_sha256=args.model_freeze_sha256),
        contract=dict(recipe=source['recipe'],question='Does coherent source acquisition training improve cross-acquisition transfer?',
            changed='Only the already-frozen five source models',
            fixed='Same raw FrozenHurdle reader as UJI; old H/S/Q, candidates, APs, empirical S, GP and fixed half pool',
            status='Exposed external evaluation; all30 streams, no dataset-specific selection or adoption',
            training=0,GP_fits=0,model_selection=False,resources=dict(CPU_threads=2,GPU=0,segment_seconds=args.seconds))))


def prepare(args):
    # Reuse its generic field-replacement path only; do not invoke its count reader.
    with patch.multiple(F,MODELS=MODELS,FiniteObservationTranslator=FrozenHurdleTranslator,freeze=freeze):
        return F.prepare(args)


def score(args):
    O.verify_freeze(args.output)
    parity = S.verify_streams(args.output)
    assert parity['complete']
    from ipin2025_score import score as score_saved
    result = score_saved(args.output/'predictions.json')
    previous = O.read(S.BASELINE/'scores.json')
    for key in ('schema','gt_sha256','eligible_floors','methods','gt_rows','metric','coverage','reference'):
        assert previous[key] == result[key]
    assert len(previous['points']) == len(result['points'])
    for a,b in zip(previous['points'],result['points']):
        assert {k:v for k,v in a.items() if k != 'errors'} == {k:v for k,v in b.items() if k != 'errors'}
    A.write(args.output/'scores.json',result)
    A.write(args.output/'comparison.json',dict(complete=True,recipe='coherent_acquisition_55_v1',
        baseline=O.binding(S.BASELINE/'scores.json'),current=O.binding(args.output/'scores.json'),
        by_K=S.compare_group(previous['by_K'],result['by_K']),adopted=False,
        status='Frozen source candidate, unchanged cross-dataset inference; no IPIN tuning'))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-freeze-sha256',required=True)
    parser.add_argument('--seconds',type=float,default=540.)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    args.output,args.device = args.output.resolve(),'cpu'
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 540
    assert A.digest(MODELS/'freeze.json') == args.model_freeze_sha256
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    raise SystemExit({'prepare':prepare,'predict':S.predict,'score':score}[args.phase](args))
