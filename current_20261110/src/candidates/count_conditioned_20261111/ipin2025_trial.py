"""Original IPIN cached fields, count-conditioned A, unchanged GP and fixed pool.

No model load/prepare/fit. All 15 original contexts and all 30 original streams.
freeze -> predict (natural stream resume) -> separate score after all30 seal.
The unconditioned A computed inside the reader and both GP endpoints must
reproduce original saved coordinates. This is exposed-data post-hoc evaluation.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
import torch

import ipin2025_confirmation as original
import ipin2025_detection_ipw as shared
from building_information_tab_trial import digest, now, write
from .reader import CountConditionedHurdle

ROOT = original.ROOT
BASELINE = shared.BASELINE
read, binding = original.read, original.binding


def freeze(args):
    assert args.output != BASELINE and BASELINE not in args.output.parents
    assert args.output != original.ASSETS and original.ASSETS not in args.output.parents
    for name, sha in shared.OLD_SHA.items():
        assert digest(BASELINE/name) == sha
    old = read(BASELINE/'freeze.json')
    contexts = read(BASELINE/'contexts_complete.json')
    predictions = read(BASELINE/'predictions.json')
    assert contexts['complete'] and contexts['freeze_sha256'] == shared.OLD_SHA['freeze.json']
    assert predictions['complete'] and predictions['contexts_complete_sha256'] == shared.OLD_SHA['contexts_complete.json']
    expected = {(f, k) for f in range(2, 7) for k in (1, 3, 5)}
    assert len(contexts['contexts']) == 15 and {(r['floor'], r['K']) for r in contexts['contexts']} == expected
    assert len(predictions['predictions']) == 30 and {
        (r['log_id'], r['floor'], r['K']) for r in predictions['predictions']} == {
        (log, f, k) for log in ('03', '04') for f, k in expected}
    inputs = [BASELINE/name for name in shared.OLD_SHA]
    for record in old['inputs']:
        assert digest(record['path']) == record['sha256']
        inputs.append(Path(record['path']))
    for record in contexts['contexts']:
        complete = shared.checked(record)
        assert complete['complete'] and complete['freeze_sha256'] == shared.OLD_SHA['freeze.json']
        inputs.append(Path(record['path']))
    for record in predictions['predictions']:
        assert digest(record['path']) == record['sha256']
        inputs.append(Path(record['path']))
    execution = shared.checked(predictions['readout_execution'])
    assert digest(ROOT/'cached_retained_readout.py') == execution['binding']['source']['sha256'] == shared.FAST_SHA
    assert read(BASELINE/'cached_readout_check.json')['passed']
    inputs += [BASELINE/'cached_readout_check.json', Path(predictions['readout_execution']['path'])]
    code = (*original.CODE, 'cached_retained_readout.py', 'ipin2025_detection_ipw.py',
            str(Path(__file__).relative_to(ROOT)), str(Path(__file__).with_name('reader.py').relative_to(ROOT)))
    args.output.mkdir(exist_ok=False)
    write(args.output/'freeze.json', dict(created=now(), **{k: old[k] for k in ('eligible_floors', 'K', 'R', 'methods', 'logs')},
        inputs=[binding(p) for p in dict.fromkeys(inputs)], code={n: digest(ROOT/n) for n in code},
        contract=dict(status='Exposed IPIN failure, post-hoc single-mechanism comparison',
            change='Original L_raw minus log Poisson-binomial P(M), then original BETA/A',
            fixed='Original B0 cached p/m/s2, candidates/APs/bounds, EPS/OFFSETS, GP fields and half pool; no Gamma/IPW',
            count_axis='Original prepared AP axis; all used receipts from both logs; no GT filtering',
            source_A_tolerance_m=1e-8, GP_tolerance_m=1e-7,
            fields='Original complete files transitively bind every field/state/grid; load_prepared verifies them',
            scoring='All30 seal and source-A/GP/clock checks precede original scorer; no K pooling or adoption rule',
            resources=dict(CPU_threads=2, GPU=0, fits=0, field_preparations=0, segment_seconds=args.seconds))))
    write(args.output/'contexts_complete.json', dict(complete=True, created=now(),
        contexts=contexts['contexts'], freeze_sha256=digest(args.output/'freeze.json'),
        original_contexts=binding(BASELINE/'contexts_complete.json'),
        official_query_opened=False, GT_opened=False))


def predict(args):
    original.verify_freeze(args.output)
    baseline = {(r['log_id'], r['floor'], r['K']): r
                for r in read(BASELINE/'predictions.json')['predictions']}
    progress = args.output/'predictions_progress.json'
    saved = read(progress) if args.resume and progress.exists() else []
    done = {(r['log_id'], r['floor'], r['K']) for r in saved}
    check_path = args.output/'source_A_checks.json'
    checks = read(check_path) if args.resume and check_path.exists() else dict(
        freeze_sha256=digest(args.output/'freeze.json'), streams={})
    assert checks['freeze_sha256'] == digest(args.output/'freeze.json')
    for key in done:
        assert '/'.join(map(str, key)) in checks['streams']
    streams = {}
    read_stream, load = original.read_query_stream, original.load_prepared

    def capture_stream(path):
        value = read_stream(path)
        streams[str(path)] = value
        return value

    def load_conditioned(record, device):
        h, gp, ap_ids = load(record, device)
        if all((log, record['floor'], record['K']) in done for log in ('03', '04')):
            return h, gp, ap_ids
        pending, counts = [], []
        for log in read(args.output/'freeze.json')['logs']:
            stream = streams[log['query_path']]
            used = stream['used_snapshot_indices']
            rss = original.rss_matrix([stream['snapshots'][i] for i in used], ap_ids)
            counts.append((rss[:, h.aps] != 100).sum(1))
            key = log['id'], record['floor'], record['K']
            if key in done:
                continue
            base = baseline[key]
            assert digest(base['path']) == base['sha256']
            expand = np.searchsorted(used, stream['snapshot_indices'])
            first = np.searchsorted(expand, np.arange(len(used)))
            assert np.array_equal(expand[first], np.arange(len(used)))
            with np.load(base['path'], allow_pickle=False) as old:
                pending.append(dict(key=key, expected=old['hurdle'][first], binding=base))

        class VerifiedReader(CountConditionedHurdle):
            def __init__(self, field):
                super().__init__(field)
                self.stream_index, self.offset, self.max_difference = 0, 0, 0.

            def predict(self, query):
                result = super().predict(query)
                item = pending[self.stream_index]
                end = self.offset+len(query)
                delta = float(np.max(np.abs(result['original_projected']-item['expected'][self.offset:end])))
                assert delta <= 1e-8, (item['key'], delta)
                self.max_difference = max(self.max_difference, delta)
                self.offset = end
                if end == len(item['expected']):
                    checks['streams']['/'.join(map(str, item['key']))] = dict(
                        passed=True, receipts=end, max_difference_m=self.max_difference,
                        baseline=item['binding'])
                    write(check_path, checks)
                    self.stream_index += 1
                    self.offset, self.max_difference = 0, 0.
                return result

        conditioned = VerifiedReader(h)
        conditioned.cache_counts(np.concatenate(counts))
        return conditioned, gp, ap_ids

    with patch.object(original, 'read_query_stream', capture_stream), patch.object(original, 'load_prepared', load_conditioned):
        return shared.predict(args)  # Original runner + verified fast GP hook + GP/clock checks.


def score(args):
    original.verify_freeze(args.output)
    parity = shared.verify_streams(args.output)
    assert parity['complete'] and parity['predictions_sha256'] == digest(args.output/'predictions.json')
    checks = read(args.output/'source_A_checks.json')
    assert checks['freeze_sha256'] == digest(args.output/'freeze.json')
    sealed = read(args.output/'predictions.json')
    expected = {'/'.join(map(str, (r['log_id'], r['floor'], r['K']))) for r in sealed['predictions']}
    assert set(checks['streams']) == expected and all(r['passed'] for r in checks['streams'].values())
    from ipin2025_score import score as original_score
    result = original_score(args.output/'predictions.json')
    write(args.output/'scores.json', result)
    baseline = read(BASELINE/'scores.json')
    for key in ('schema', 'gt_sha256', 'eligible_floors', 'methods', 'gt_rows', 'metric', 'coverage', 'reference'):
        assert baseline[key] == result[key]
    assert len(baseline['points']) == len(result['points'])
    for a, b in zip(baseline['points'], result['points']):
        assert {k: v for k, v in a.items() if k != 'errors'} == {k: v for k, v in b.items() if k != 'errors'}
    write(args.output/'comparison.json', dict(complete=True, created=now(),
        status='Exposed-data post-hoc count-conditioned original A; no Gamma/IPW or automatic adoption',
        baseline=binding(BASELINE/'scores.json'), current=binding(args.output/'scores.json'),
        source_A_check=binding(args.output/'source_A_checks.json'),
        GP_parity=binding(args.output/'cached_GP_parity.json'),
        delta_convention='Current minus original; per-K overall/log/floor/log-floor mean and p90; no K pooling',
        by_K=shared.compare_group(baseline['by_K'], result['by_K'])))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'predict', 'score'))
    parser.add_argument('--output', type=Path, default=ROOT/'results/ipin2025_count_conditioned_20261111')
    parser.add_argument('--seconds', type=float, default=780.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output, args.device = args.output.resolve(), 'cpu'
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 780
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    result = {'freeze': freeze, 'predict': predict, 'score': score}[args.phase](args)
    raise SystemExit(0 if result is None else result)
