"""H-only empirical S prior correction on all 30 exposed IPIN streams.

freeze -> predict (natural stream resume) -> separate score after all30 seal.
Original cached fields, raw reader, fast GP and fixed half pooling; no training,
field inference, count conditioning, IPW or Gamma. H/S alone set the prior.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from pathlib import Path
import socket
from unittest.mock import patch

import torch

import ipin2025_confirmation as original
import ipin2025_detection_ipw as shared
from building_information_tab_trial import digest, now, write
from .prepared import replace_support_prior

ROOT, BASELINE = original.ROOT, shared.BASELINE
read, binding = original.read, original.binding


def freeze(args):
    for protected in (BASELINE, original.ASSETS, original.SOURCE):
        assert args.output != protected and protected not in args.output.parents
    for name, sha in shared.OLD_SHA.items():
        assert digest(BASELINE/name) == sha
    old = read(BASELINE/'freeze.json')
    contexts = read(BASELINE/'contexts_complete.json')
    predictions = read(BASELINE/'predictions.json')
    assert contexts['complete'] and contexts['freeze_sha256'] == shared.OLD_SHA['freeze.json']
    assert predictions['complete'] and predictions['freeze_sha256'] == shared.OLD_SHA['freeze.json']
    assert predictions['contexts_complete_sha256'] == shared.OLD_SHA['contexts_complete.json']
    expected = {(f, k) for f in range(2, 7) for k in (1, 3, 5)}
    assert len(contexts['contexts']) == 15 and {(r['floor'], r['K']) for r in contexts['contexts']} == expected
    assert len(predictions['predictions']) == 30 and {
        (r['log_id'], r['floor'], r['K']) for r in predictions['predictions']} == {
        (log, f, k) for log in ('03', '04') for f, k in expected}
    for key, value in dict(eligible_floors=[2, 3, 4, 5, 6], K=[1, 3, 5], methods=original.METHODS).items():
        assert old[key] == predictions[key] == value
    assert old['logs'] == predictions['logs'] and old['R'] == 1
    inputs = [BASELINE/name for name in shared.OLD_SHA]
    for record in old['inputs']:
        assert digest(record['path']) == record['sha256']
        inputs.append(Path(record['path']))
    manifest = read(original.ASSETS/'manifest.json')
    for record in [*manifest['histories'].values(), *manifest['support'].values()]:
        assert digest(record['path']) == record['sha256']
        inputs.append(Path(record['path']))
    for record in contexts['contexts']:
        complete = shared.checked(record)
        assert complete['complete'] and complete['freeze_sha256'] == shared.OLD_SHA['freeze.json']
        assert (complete['floor'], complete['K']) == (record['floor'], record['K'])
        inputs.append(Path(record['path']))
    for record in predictions['predictions']:
        assert digest(record['path']) == record['sha256']
        inputs.append(Path(record['path']))
    execution = shared.checked(predictions['readout_execution'])
    assert digest(ROOT/'cached_retained_readout.py') == execution['binding']['source']['sha256'] == shared.FAST_SHA
    assert read(BASELINE/'cached_readout_check.json')['passed']
    inputs += [BASELINE/'cached_readout_check.json', Path(predictions['readout_execution']['path'])]
    code = (*original.CODE, 'cached_retained_readout.py', 'ipin2025_detection_ipw.py',
            str(Path(__file__).relative_to(ROOT)), str(Path(__file__).with_name('prepared.py').relative_to(ROOT)))
    args.output.mkdir(exist_ok=False)
    write(args.output/'freeze.json', dict(created=now(), **{k: old[k] for k in ('eligible_floors', 'K', 'R', 'methods', 'logs')},
        inputs=[binding(p) for p in dict.fromkeys(inputs)], code={n: digest(ROOT/n) for n in code},
        contract=dict(status='Exposed IPIN data, post-hoc single-mechanism comparison',
            change='Only empirical S p: p0=(dH+.5)/(nH+1); pS=(ndS+p0)/(nS+1)',
            prior_concentration=1, clipping=[1e-4, 1-1e-4],
            history='Actual original aligned deployment H rows pooled equally; no deduplication or floor reweighting',
            interpretation='Global historical detection frequency, not true target spatial detection probability',
            fixed='Learned H p, all m/s2/candidates/APs/bounds, original raw reader, cached GP and fixed half pool',
            verification='Reconstruct original empirical S fields before replacement; no repeated original A inference',
            fields='Original complete files bind every field/state/grid; original load_prepared verifies them',
            timing=old['contract']['timing'], coverage=old['contract']['coverage'], GP_tolerance_m=1e-7,
            scoring='All30 seal and transformation/GP/clock checks precede original scorer; no K pooling or adoption rule',
            resources=dict(CPU_threads=2, GPU=0, fits=0, field_preparations=0, segment_seconds=args.seconds))))
    write(args.output/'contexts_complete.json', dict(complete=True, created=now(),
        contexts=contexts['contexts'], freeze_sha256=digest(args.output/'freeze.json'),
        original_contexts=binding(BASELINE/'contexts_complete.json'),
        official_query_opened=False, GT_opened=False))


def predict(args):
    original.verify_freeze(args.output)
    manifest = read(original.ASSETS/'manifest.json')
    check_path = args.output/'support_prior_checks.json'
    checks = read(check_path) if args.resume and check_path.exists() else dict(
        freeze_sha256=digest(args.output/'freeze.json'), contexts={})
    assert checks['freeze_sha256'] == digest(args.output/'freeze.json')
    load = original.load_prepared

    def load_transformed(record, device):
        h, gp, ap_ids = load(record, device)
        complete = read(record['path'])
        histories, support, current_ids = original.context(manifest, record['floor'], record['K'])
        assert current_ids == ap_ids and list(histories) == complete['sources']
        transformed, details = replace_support_prior(h, histories, support)
        current = dict(original_context=record, original_field=complete['hurdle'], **details)
        if record['event'] in checks['contexts']:
            assert checks['contexts'][record['event']] == current
        checks['contexts'][record['event']] = current
        write(check_path, checks)
        return transformed, gp, ap_ids

    with patch.object(original, 'load_prepared', load_transformed):
        return shared.predict(args)


def score(args):
    original.verify_freeze(args.output)
    parity = shared.verify_streams(args.output)
    assert parity['complete'] and parity['predictions_sha256'] == digest(args.output/'predictions.json')
    checks = read(args.output/'support_prior_checks.json')
    assert checks['freeze_sha256'] == digest(args.output/'freeze.json')
    expected = {r['event'] for r in read(args.output/'contexts_complete.json')['contexts']}
    assert set(checks['contexts']) == expected
    assert all(r[key] for r in checks['contexts'].values() for key in (
        'original_S_verified', 'learned_H_p_unchanged', 'same_m_s2_candidates_AP_bounds'))
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
        status='Exposed-data post-hoc H-only empirical Support prior correction; no automatic adoption',
        baseline=binding(BASELINE/'scores.json'), current=binding(args.output/'scores.json'),
        support_prior_check=binding(args.output/'support_prior_checks.json'),
        GP_parity=binding(args.output/'cached_GP_parity.json'),
        delta_convention='Current minus original; per-K overall/log/floor/log-floor mean and p90; no K pooling',
        by_K=shared.compare_group(baseline['by_K'], result['by_K'])))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'predict', 'score'))
    parser.add_argument('--output', type=Path, default=ROOT/'results/ipin2025_support_history_prior_20261111')
    parser.add_argument('--seconds', type=float, default=780.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output, args.device = args.output.resolve(), 'cpu'
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 780
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    result = {'freeze': freeze, 'predict': predict, 'score': score}[args.phase](args)
    raise SystemExit(0 if result is None else result)
