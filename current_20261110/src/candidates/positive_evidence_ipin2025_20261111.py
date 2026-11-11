"""Isolate negative visibility evidence in the adopted learned field.

Keep every original field, RSS factor, offset, temperature and GP operation.
L_raw = log mean_o prod_detected[t_a(o)] + sum_missing log miss_a,
where miss_a=(1-EPS)*(1-p_a)+EPS*U_MISS. Remove ONLY the missing term
before the unchanged BETA/A. Positive factors, including p and the robust
background term, stay exactly as in the adopted reader. No parameter fit.

Park et al., INFOCOM 2011, found cross-device AP visibility can hurt Bayesian
localization: https://people.csail.mit.edu/jgpark/docs/PCTL11.pdf . Their
RSS-only model motivates isolating negative evidence; this control retains
positive detection factors and is not a reproduction of their estimator.

Forward: fixed learned H/S field -> original positive score on observed
APs -> unchanged position readout and fixed GP pool. Reverse: Q determines
only its observed set/RSS, never a coordinate, floor mask or selected score.
An empty observed set leaves uniform H/S candidate mass. Nonempty scans
retain learned detection/RSS evidence; missing-AP counts cannot favor an
otherwise silent candidate. Empirical S, H multiplicities and bounds remain.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from candidates.count_conditioned_20261111 import ipin2025_trial as runner
from frozen_hurdle_translator import PreparedFrozenHurdle, BETA, EPS, U_MISS

original, shared = runner.original, runner.shared
ROOT, BASELINE = original.ROOT, runner.BASELINE
read, binding, digest, write, now = runner.read, runner.binding, runner.digest, runner.write, runner.now
OUTPUT = ROOT/'results/ipin2025_positive_evidence_20261111'


class PositiveEvidenceHurdle(PreparedFrozenHurdle):
    def __init__(self, prepared):
        super().__init__(*(getattr(prepared, name) for name in
            ('cand_xy', 'aps', 'p', 'm', 's2', 'bounds', 'ap_count')))
        self.log_miss = np.log((1-EPS)*(1-self.p)+EPS*U_MISS)

    def cache_counts(self, counts):
        # The shared sealed-stream runner supplies counts for its old reader.
        # Removing negative evidence needs no count DP or Query-wide fit.
        pass

    def predict(self, query_rss):
        result = super().predict(query_rss)
        detected = (np.asarray(query_rss)[:, self.aps] != 100).astype(float)
        missing_score = (1-detected)@self.log_miss.T
        scores = result['log_position_mass']-(BETA/len(self.aps))*missing_score
        log_mass = scores-logsumexp(scores, axis=1, keepdims=True)
        return dict(projected=np.clip(np.exp(log_mass)@self.cand_xy, self.bounds[0], self.bounds[1]),
            log_position_mass=log_mass, cand_xy=self.cand_xy.copy(), aps=self.aps.copy(),
            original_projected=result['projected'])


def freeze(args):
    assert args.output != BASELINE and BASELINE not in args.output.parents
    for name, sha in shared.OLD_SHA.items():
        assert digest(BASELINE/name) == sha
    old = read(BASELINE/'freeze.json')
    contexts = read(BASELINE/'contexts_complete.json')
    predictions = read(BASELINE/'predictions.json')
    assert contexts['complete'] and len(contexts['contexts']) == 15
    assert predictions['complete'] and len(predictions['predictions']) == 30
    assert contexts['freeze_sha256'] == shared.OLD_SHA['freeze.json']
    assert predictions['contexts_complete_sha256'] == shared.OLD_SHA['contexts_complete.json']
    inputs = [BASELINE/name for name in shared.OLD_SHA]
    for record in old['inputs']:
        assert digest(record['path']) == record['sha256']
        inputs.append(Path(record['path']))
    for record in contexts['contexts']:
        assert shared.checked(record)['complete']
        inputs.append(Path(record['path']))
    for record in predictions['predictions']:
        assert digest(record['path']) == record['sha256']
        inputs.append(Path(record['path']))
    inputs += [ROOT/'results'/name/'scores.json' for name in
        ('ipin2025_count_conditioned_20261111', 'ipin2025_finite_observation_20261111')]
    assert digest(ROOT/'cached_retained_readout.py') == shared.FAST_SHA
    code = (*original.CODE, 'cached_retained_readout.py', 'ipin2025_detection_ipw.py',
        'candidates/count_conditioned_20261111/ipin2025_trial.py',
        'candidates/count_conditioned_20261111/reader.py',
        'candidates/frozen_v1_ipin2025_20261111.py', str(Path(__file__).relative_to(ROOT)))
    args.output.mkdir(exist_ok=False)
    write(args.output/'freeze.json', dict(created=now(), **{k: old[k] for k in
        ('eligible_floors', 'K', 'R', 'methods', 'logs')}, inputs=[binding(p) for p in dict.fromkeys(inputs)],
        code={name: digest(ROOT/name) for name in code},
        contract=dict(status='Exposed IPIN, prospective fixed observation-model comparison',
            question='Does negative visibility evidence damage the useful learned positive signal match across acquisition conditions?',
            change='Remove only sum_missing log((1-EPS)*(1-p)+EPS*U_MISS) before the unchanged BETA/A',
            fixed='Original adopted five-model cached field, all H/S atoms/APs, p/m/s2, empirical S, RSS factors, EPS/OFFSETS, GP and fixed half pool',
            provenance='Park et al. INFOCOM2011 motivates isolating negative evidence; positive detection factors remain, unlike their RSS-only model',
            no_fit='No model load, field inference, source training, new parameters, GP adaptation or outcome-based selection',
            source_A_tolerance_m=1e-8, GP_tolerance_m=1e-7,
            scoring='All original30 streams seal and raw-A/GP/clock parity checks pass before scoring; no K pooling',
            decision='Evaluate full method and single path against original A/count/finite and GP, including all floors and tails; no automatic adoption',
            limits='Discarding absence evidence can lose spatial information; this comparison isolates that tradeoff while positive detection information remains',
            resources=dict(CPU_threads=2, GPU=0, fits=0, field_preparations=0, segment_seconds=args.seconds))))
    write(args.output/'contexts_complete.json', dict(complete=True, created=now(), contexts=contexts['contexts'],
        freeze_sha256=digest(args.output/'freeze.json'), original_contexts=binding(BASELINE/'contexts_complete.json'),
        official_query_opened=False, GT_opened=False))


def predict(args):
    # Reuse exact raw-A, receipt-clock and GP endpoint comparisons in the
    # existing full-stream runner; only its score transformer is replaced.
    with patch.object(runner, 'CountConditionedHurdle', PositiveEvidenceHurdle):
        return runner.predict(args)


def score(args):
    original.verify_freeze(args.output)
    parity = shared.verify_streams(args.output)
    assert parity['complete'] and parity['predictions_sha256'] == digest(args.output/'predictions.json')
    checks = read(args.output/'source_A_checks.json')
    assert checks['freeze_sha256'] == digest(args.output/'freeze.json')
    assert len(checks['streams']) == 30 and all(r['passed'] for r in checks['streams'].values())
    from ipin2025_score import score as original_score
    from candidates.frozen_v1_ipin2025_20261111 import compatible
    result = original_score(args.output/'predictions.json')
    write(args.output/'scores.json', result)
    references = {}
    for name, directory in dict(original_A=BASELINE,
            count=ROOT/'results/ipin2025_count_conditioned_20261111',
            finite_observation=ROOT/'results/ipin2025_finite_observation_20261111').items():
        baseline = read(directory/'scores.json')
        compatible(baseline, result)
        references[name] = dict(source=binding(directory/'scores.json'),
            by_K=shared.compare_group(baseline['by_K'], result['by_K']))
    write(args.output/'comparison.json', dict(complete=True, created=now(),
        status='Original A field without its missing-signal penalty; no new training or automatic adoption',
        freeze_sha256=digest(args.output/'freeze.json'), scores=binding(args.output/'scores.json'),
        GP_parity=binding(args.output/'cached_GP_parity.json'), source_A_check=binding(args.output/'source_A_checks.json'),
        references=references, delta_convention='Current minus reference; negative improves error'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'predict', 'score'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seconds', type=float, default=780.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output, args.device = args.output.resolve(), 'cpu'
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 780
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    with threadpool_limits(limits=2):
        result = {'freeze': freeze, 'predict': predict, 'score': score}[args.phase](args)
    raise SystemExit(0 if result is None else result)
