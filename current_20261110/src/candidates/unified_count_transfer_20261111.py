"""Complete the unified field with the existing count-conditioned reader.

Forward review: H/S produce the already frozen77 field. Query detection count
M is observed at inference; subtract log P_field(M) from the raw score before
the unchanged temperature. This reuses the existing conditional-Bernoulli
working score, not a new learned structure or target-fitted calibration.

Reverse review: coordinates <- fixed GP pool and count-conditioned field
mass <- frozen p/m/s2 and the current query mask/RSS <- legal H/S and source
models. No target coordinates enter prediction. All-query mask pre-caching
only amortizes the DP: a receipt's score depends on its own count, not future
counts. Complete cached fields and all30 streams are reused, not re-prepared.

Counterconditions: conditioning removes count's useful location information
as well as its acquisition sensitivity. A common log-odds nuisance cancels
only in the detection factor with conditional intensity held fixed; AP-wise
changes and RSS shifts are not removed. M=0 or M=A leaves no detection-pattern
information. The existing exact enumeration check covers those DP endpoints;
no duplicate mathematical test or training is needed here.

Decision: one complete exposed IPIN comparison, frozen before inference;
judge all K and floors against the same raw field and existing finite-count
recipe. No target parameter selection, per-dataset switch or automatic adoption.
If useful, the same fixed reader must also retain UJI gains before selection.
"""
import argparse
from pathlib import Path
import socket
from unittest.mock import patch

from candidates.count_conditioned_20261111 import ipin2025_trial as C

O, S = C.original, C.shared
BASE = O.ROOT/'results/ipin2025_unified_scan_20261111'
RECIPE = 'unified_scan_field_77_v1_count'


def freeze(args):
    old = O.verify_freeze(BASE)
    contexts = C.read(BASE/'contexts_complete.json')
    predictions = C.read(BASE/'predictions.json')
    assert contexts['complete'] and len(contexts['contexts']) == 15
    assert predictions['complete'] and len(predictions['predictions']) == 30
    assert predictions['contexts_complete_sha256'] == C.digest(BASE/'contexts_complete.json')
    paths = [BASE/n for n in ('freeze.json','contexts_complete.json','predictions.json','scores.json')]
    paths += [Path(r['path']) for r in old['inputs']]
    paths += [Path(r['path']) for r in contexts['contexts']]
    paths += [Path(r['path']) for r in predictions['predictions']]
    code = dict(old['code'])
    for p in (Path(__file__),Path(C.__file__),Path(C.__file__).with_name('reader.py')):
        code[str(p.relative_to(O.ROOT))] = C.digest(p)
    args.output.mkdir(parents=True,exist_ok=False)
    C.write(args.output/'freeze.json',dict(created=C.now(),recipe=RECIPE,code=code,
        **{k:old[k] for k in ('eligible_floors','K','R','methods','logs','model_generation')},
        inputs=[C.binding(p) for p in dict.fromkeys(paths)],
        contract=dict(question='Does conditioning the observation count make the same unified learned field transfer reliably?',
            changed='Only raw reader -> already implemented exact count-conditioned reader',
            fixed='All model assets, fields, H/S, APs, unique candidates, robust terms, temperature, offsets, GP and half pool',
            review=__doc__,status='Exposed post-hoc mechanism comparison; not an independent test',
            resources=dict(CPU_threads=2,GPU=0,fits=0,field_preparations=0,downloads=0,segment_seconds=args.seconds))))
    C.write(args.output/'contexts_complete.json',dict(complete=True,created=C.now(),
        contexts=contexts['contexts'],freeze_sha256=C.digest(args.output/'freeze.json'),
        original_contexts=C.binding(BASE/'contexts_complete.json'),official_query_opened=False,GT_opened=False))
    return 0


def run(args):
    if args.phase == 'freeze':
        return freeze(args)
    # Reuse receipt-wise raw-field equivalence and GP/clock checks against the
    # actual unified baseline. Patches are process-local; source files stay fixed.
    hashes = {n:C.digest(BASE/n) for n in S.OLD_SHA}
    with patch.object(C,'BASELINE',BASE), patch.object(S,'BASELINE',BASE), patch.object(S,'OLD_SHA',hashes):
        status = {'predict':C.predict,'score':C.score}[args.phase](args)
    if args.phase == 'score':
        path = args.output/'comparison.json'; result = C.read(path)
        result.update(recipe=RECIPE,status='Complete unified77 plus existing count reader; no automatic adoption')
        reference = O.ROOT/'results/ipin2025_finite_observation_20261111/scores.json'
        result['finite_count_reference'] = dict(scores=C.binding(reference),
            by_K=S.compare_group(C.read(reference)['by_K'],C.read(args.output/'scores.json')['by_K']))
        C.write(path,result)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','predict','score'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=540.)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args();args.output,args.device = args.output.resolve(),'cpu'
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 540
    C.torch.set_num_threads(2);C.torch.set_default_dtype(C.torch.float64)
    raise SystemExit(run(args))
