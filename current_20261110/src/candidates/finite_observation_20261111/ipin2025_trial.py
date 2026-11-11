"""Finite-observation five-model fields with the existing exact count reader.

prepare: one new field per original IPIN H/S context, retaining cached GP.
predict: all original 30 streams, fixed count conditioning and fixed half pool.
score: compare against the completed original-field count candidate first.
No target fitting, new GP adaptation, reader selection or automatic adoption.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
from pathlib import Path
import socket
import time
from unittest.mock import patch

import numpy as np
import torch

import ipin2025_confirmation as original
import ipin2025_detection_ipw as shared
from building_information_tab_trial import digest, now, write
from candidates.count_conditioned_20261111.reader import CountConditionedHurdle
from .deployment import FiniteObservationTranslator

ROOT, BASELINE = original.ROOT, shared.BASELINE
COUNT_BASE = ROOT/'results/ipin2025_count_conditioned_20261111'
MODELS = ROOT/'results/claude_hurdle_b0_finite_obs_ens5_20261111'
read, binding = original.read, original.binding


def verify(args):
    frozen = original.verify_freeze(args.output)
    assert frozen['model_generation']['freeze_sha256'] == args.model_freeze_sha256
    return frozen


def freeze(args):
    for protected in (BASELINE, COUNT_BASE, MODELS, original.ASSETS, original.SOURCE):
        assert args.output != protected and protected not in args.output.parents
    for name, sha in shared.OLD_SHA.items():
        assert digest(BASELINE/name) == sha
    old = read(BASELINE/'freeze.json')
    contexts = read(BASELINE/'contexts_complete.json')
    count = read(COUNT_BASE/'predictions.json')
    assert contexts['complete'] and contexts['freeze_sha256'] == shared.OLD_SHA['freeze.json']
    assert count['complete'] and len(count['predictions']) == 30
    assert digest(COUNT_BASE/'scores.json') == '1b75d463d24c64335a09abab3e9fcd4ff598be43762ec6c4f83777d118078019'
    expected = {(f, k) for f in range(2, 7) for k in (1, 3, 5)}
    assert len(contexts['contexts']) == 15 and {(r['floor'], r['K']) for r in contexts['contexts']} == expected
    source = read(MODELS/'complete.json')
    assert source['complete'] and source['freeze_sha256'] == digest(MODELS/'freeze.json')
    assert source['freeze_sha256'] == args.model_freeze_sha256
    assert sorted(r['seed'] for r in source['models']) == [1, 2, 3, 4, 5]
    assert source['source_buildings'] == [0]
    FiniteObservationTranslator(MODELS)
    inputs = [BASELINE/name for name in shared.OLD_SHA]
    inputs += [COUNT_BASE/name for name in ('freeze.json', 'predictions.json', 'scores.json')]
    inputs += [MODELS/name for name in ('complete.json', 'freeze.json')]
    for record in old['inputs']:
        assert digest(record['path']) == record['sha256']
        inputs.append(Path(record['path']))
    manifest = read(original.ASSETS/'manifest.json')
    for record in [*manifest['histories'].values(), *manifest['support'].values()]:
        assert digest(record['path']) == record['sha256']
        inputs.append(Path(record['path']))
    for record in contexts['contexts']:
        parent = shared.checked(record)
        assert parent['complete'] and parent['freeze_sha256'] == shared.OLD_SHA['freeze.json']
        inputs.append(Path(record['path']))
    for record in source['models']:
        path = MODELS/record['path']
        assert digest(path) == record['sha256']
        inputs.append(path)
    assert digest(ROOT/'cached_retained_readout.py') == shared.FAST_SHA
    args.output.mkdir(exist_ok=False)
    candidate_code = sorted(Path(__file__).parent.glob('*.py'))
    code = [*original.CODE, 'cached_retained_readout.py', 'ipin2025_detection_ipw.py',
            'candidates/count_conditioned_20261111/reader.py',
            *[str(p.relative_to(ROOT)) for p in candidate_code]]
    write(args.output/'freeze.json', dict(created=now(),
        **{k: old[k] for k in ('eligible_floors', 'K', 'R', 'methods', 'logs')},
        inputs=[binding(p) for p in dict.fromkeys(inputs)], code={n: digest(ROOT/n) for n in code},
        model_generation=dict(path=str(MODELS), freeze_sha256=source['freeze_sha256']),
        contract=dict(status='Exposed IPIN data; complete finite-observation learning recipe comparison',
            changed='Learned H fields from the new source recipe; original H/S rows, candidates and empirical S preserved',
            primary='New fields plus exact count conditioning versus original fields plus the same count conditioning',
            secondary='Original raw-reader scores retained separately, not the primary comparator',
            fixed='Exact count reader, EPS/OFFSETS/BETA, GP fields and fixed half pool; no S-prior/IPW/Gamma',
            timing=old['contract']['timing'], coverage=old['contract']['coverage'],
            scoring='All30 streams sealed and original GP/clock parity checked before the original scorer',
            source_training='Already completed and frozen B0-only five-model recipe; no target training',
            attribution='Task organization and observation representation jointly changed; not a single-feature ablation',
            resources=dict(CPU_threads=2, GPU=0, target_fits=0, GP_fits=0, segment_seconds=args.seconds))))


def prepare(args):
    if not args.resume:
        freeze(args)
    verify(args)
    assert not (args.output/'contexts_complete.json').exists()
    translator = FiniteObservationTranslator(MODELS)
    manifest = read(original.ASSETS/'manifest.json')
    records, completed = [], 0
    start = time.monotonic()
    for record in read(BASELINE/'contexts_complete.json')['contexts']:
        directory = args.output/record['event']
        path = directory/'complete.json'
        if path.exists():
            current = read(path)
            assert current['freeze_sha256'] == digest(args.output/'freeze.json')
            assert digest(current['hurdle']['path']) == current['hurdle']['sha256']
        else:
            if time.monotonic()-start >= args.seconds:
                return 2
            parent = shared.checked(record)
            histories, support, ap_ids = original.context(manifest, record['floor'], record['K'])
            assert parent['ap_ids'] == ap_ids and parent['sources'] == list(histories)
            h = translator.prepare(histories, support, f"IPIN2025_B2025F{record['floor']}")
            assert digest(parent['hurdle']['path']) == parent['hurdle']['sha256']
            ns = len(np.unique(support['xy'], axis=0))
            with np.load(parent['hurdle']['path'], allow_pickle=False) as old:
                for name in ('cand_xy', 'aps', 'bounds', 'ap_count'):
                    assert np.array_equal(getattr(h, name), old[name]), (record['event'], name)
                for name in ('p', 'm', 's2'):
                    assert np.array_equal(getattr(h, name)[-ns:], old[name][-ns:]), (record['event'], 'empirical S', name)
            directory.mkdir(exist_ok=True)
            field_path = directory/'hurdle.npz'
            np.savez_compressed(field_path, **{name: getattr(h, name) for name in
                ('cand_xy', 'aps', 'p', 'm', 's2', 'bounds', 'ap_count')})
            current = dict(parent, created=now(), hurdle=binding(field_path),
                freeze_sha256=digest(args.output/'freeze.json'), parent_context=record,
                inherited=dict(candidates_AP_bounds=True, empirical_S_p_m_s2=True, GP_fields=True),
                source_model_freeze_sha256=args.model_freeze_sha256)
            write(path, current)
            completed += 1
        records.append(dict(event=record['event'], floor=record['floor'], K=record['K'], **binding(path)))
        print(f'Prepared {len(records)}/15 finite-observation contexts', flush=True)
    write(args.output/'contexts_complete.json', dict(complete=True, created=now(), contexts=records,
        freeze_sha256=digest(args.output/'freeze.json'), official_query_opened=False, GT_opened=False,
        new_contexts_this_segment=completed))
    return 0


def predict(args):
    verify(args)
    streams = {}
    read_stream, load = original.read_query_stream, original.load_prepared

    def capture(path):
        stream = read_stream(path)
        streams[str(path)] = stream
        return stream

    def load_counted(record, device):
        h, gp, ap_ids = load(record, device)
        counted = CountConditionedHurdle(h)
        counts = []
        for stream in streams.values():
            rss = original.rss_matrix([stream['snapshots'][i] for i in stream['used_snapshot_indices']], ap_ids)
            counts.append((rss[:, h.aps] != 100).sum(1))
        counted.cache_counts(np.concatenate(counts))
        return counted, gp, ap_ids

    with patch.object(original, 'read_query_stream', capture), patch.object(original, 'load_prepared', load_counted):
        return shared.predict(args)


def score(args):
    verify(args)
    parity = shared.verify_streams(args.output)
    assert parity['complete'] and parity['predictions_sha256'] == digest(args.output/'predictions.json')
    from ipin2025_score import score as original_score
    result = original_score(args.output/'predictions.json')
    write(args.output/'scores.json', result)
    references = dict(count_conditioned=COUNT_BASE, original_raw=BASELINE)
    comparisons = {}
    for label, path in references.items():
        previous = read(path/'scores.json')
        for key in ('schema', 'gt_sha256', 'eligible_floors', 'methods', 'gt_rows', 'metric', 'coverage', 'reference'):
            assert previous[key] == result[key]
        assert len(previous['points']) == len(result['points'])
        for a, b in zip(previous['points'], result['points']):
            assert {k: v for k, v in a.items() if k != 'errors'} == {k: v for k, v in b.items() if k != 'errors'}
        comparisons[label] = dict(scores=binding(path/'scores.json'),
                                  by_K=shared.compare_group(previous['by_K'], result['by_K']))
    write(args.output/'comparison.json', dict(complete=True, created=now(),
        current=binding(args.output/'scores.json'), primary_reference='count_conditioned',
        model_freeze_sha256=args.model_freeze_sha256, GP_parity=binding(args.output/'cached_GP_parity.json'),
        status='Exposed-data complete recipe comparison; no automatic adoption or reader selection',
        comparisons=comparisons))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'predict', 'score'))
    parser.add_argument('--model-freeze-sha256', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'results/ipin2025_finite_observation_20261111')
    parser.add_argument('--seconds', type=float, default=780.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output, args.device = args.output.resolve(), 'cpu'
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 780
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    raise SystemExit({'prepare': prepare, 'predict': predict, 'score': score}[args.phase](args))
