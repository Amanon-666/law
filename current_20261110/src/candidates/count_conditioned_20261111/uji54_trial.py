"""Exact count conditioning on original B0-five-model strict54, no retraining.

Thin wrapper of hurdle_gp_pool_trial.run: one original field preparation per
context, the existing reader, all five readouts, original save-before-score
ordering and hierarchical metrics. Original A parity reuses the unconditioned
readout already computed inside CountConditionedHurdle; GP scores are cached.
The frozen reader and all original code/assets/results remain untouched.
"""
import hurdle_gp_pool_trial as original  # Original CPU2/offline setup first.

import argparse
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from .reader import CountConditionedHurdle, BETA

ROOT = original.ROOT
MODELS = ROOT/'results/claude_hurdle_b0_ens5_20261110'
BASELINE = ROOT/'results/hurdle_gp_pool_20261110/complete.json'


def run(args):
    args.models, args.baseline = MODELS, BASELINE
    assert original.digest(MODELS/'freeze.json') == original.BASE_FREEZE_SHA
    assert original.digest(BASELINE) == original.BASELINE_SHA
    assert original.digest(original.ASSETS/'manifest.json') == original.MANIFEST_SHA
    assert args.output != MODELS and MODELS not in args.output.parents
    assert args.output != BASELINE.parent and BASELINE.parent not in args.output.parents
    manifest = original.read(original.ASSETS/'manifest.json')
    events = manifest['events']
    baseline = {r['event']: r for r in original.read(BASELINE)['records']}
    parity, prepared_events, fields = {}, [], []
    real_write, real_fuse = original.write, original.fuse_hurdle_gp

    class CheckedReader(CountConditionedHurdle):
        def __init__(self, field, event):
            super().__init__(field)
            self.event = event

        def predict(self, query):
            event = self.event
            counts = (np.asarray(query)[:, self.aps] != 100).sum(1)
            self.cache_counts(counts)  # All original Query rows; one DP per context.
            result = super().predict(query)
            # Add back the count term and renormalize: no second old-A forward.
            restored = result['log_position_mass'] + (BETA/len(self.aps))*self._log_counts[:, counts].T
            restored -= logsumexp(restored, axis=1, keepdims=True)
            record = baseline[event['id']]
            path = ROOT/record['prediction_path']
            assert original.digest(path) == record['prediction_sha256']
            with np.load(path, allow_pickle=False) as old:
                assert np.array_equal(old['query_rows'], event['query_rows'])
                assert np.array_equal(old['aps'], self.aps)
                assert np.array_equal(old['cand_xy'], self.cand_xy)
                coordinate_delta = float(np.max(np.abs(result['original_projected']-old['hurdle'])))
                log_mass_delta = float(np.max(np.abs(restored-old['log_hurdle'])))
            assert coordinate_delta <= 1e-8, (event['id'], 'original A coordinates', coordinate_delta)
            assert log_mass_delta <= 1e-8, (event['id'], 'original A log mass', log_mass_delta)
            assert event['id'] not in parity
            parity[event['id']] = dict(original_A_coordinate_max_difference_m=coordinate_delta,
                original_A_log_mass_max_difference=log_mass_delta,
                query_rows=len(query), candidates=len(self.cand_xy), APs=len(self.aps),
                min_detected=int(counts.min()), max_detected=int(counts.max()),
                original_prediction_sha256=record['prediction_sha256'])
            return result

    class CountTranslator(original.FrozenHurdleTranslator):
        def prepare(self, histories, support, target_floor):
            event = events[len(prepared_events)]
            assert event['floor'] == target_floor
            field = super().prepare(histories, support, target_floor)
            path = args.output/f"{event['id']}_field.npz"
            np.savez_compressed(path, **{key: getattr(field, key) for key in
                ('cand_xy', 'aps', 'p', 'm', 's2', 'bounds', 'ap_count')})
            fields.append(dict(event=event['id'], path=str(path),
                               sha256=original.digest(path), query_used=False))
            prepared_events.append(event['id'])
            return CheckedReader(field, event)

    def checked_fuse(*values):
        result = real_fuse(*values)
        eid = prepared_events[-1]
        with np.load(ROOT/baseline[eid]['prediction_path'], allow_pickle=False) as old:
            for method in ('GP', 'GP_candidates'):
                delta = float(np.max(np.abs(result['predictions'][method]-old[method])))
                assert delta <= 1e-8, (eid, method, delta)
                parity[eid][method+'_coordinate_max_difference_m'] = delta
        return result

    def write_current(path, value):
        if path == args.output/'freeze.json':
            # Replace the old pool-selection question BEFORE the freeze is written.
            value['contract'].update(
                question='Does exact count conditioning retain the adopted original A gains on strict54?',
                change='Only L_raw - log P_qeff(sum D=M), followed by unchanged BETA/A',
                condition='Conditional intensity score fixed; common detection log-odds shift cancels given M; AP-specific/RSS changes are outside this guarantee',
                controls='Count-conditioned original A, unchanged native GP, candidate GP, coordinate_half, fixed probability_half',
                pool='Fixed0.5*count-conditioned A log mass+0.5*original interpolated GP log density',
                adoption='No automatic adoption rule or target-tuned threshold; report paired baseline deltas for research judgment',
                scope='Already exposed UJI strict54 retention comparison; original hierarchical metrics, not IPIN scoring',
                execution='54 original field preparations, original five trained models; no source training, GP fit or baseline rerun',
                field_cache='Save each prepared H/S field for reuse; no Query information in field preparation',
                source_parity='Old candidate/AP/query binding; original A coordinates and reconstructed log mass; both GP coordinates',
                source_parity_tolerances=dict(coordinate_m=1e-8, log_mass=1e-8))
        elif path == args.output/'progress.json':
            value['source_parity'] = dict(parity)
            value['fields'] = list(fields)
        elif path == args.output/'complete.json':
            assert prepared_events == [e['id'] for e in events] and len(parity) == 54
            assert all('GP_candidates_coordinate_max_difference_m' in r for r in parity.values())
            value['adoption'] = dict(status='requires_main_thread_judgment', automatic_rule=False)
            value['source_parity'] = dict(passed=True, events=parity)
            value['field_preparations'] = len(prepared_events)
            value['fields'] = fields
            value['comparison_pairs'] = dict(single_path='original A -> count-conditioned original A',
                                             fixed_pool='original half pool -> count-conditioned A with same GP half pool')
            value['purpose'] = 'Strict54 gains-retention check of the same count-conditioning mechanism; no new parameter selection'
        real_write(path, value)

    code = (*original.CODE, str(Path(__file__).relative_to(ROOT)),
            str(Path(__file__).with_name('reader.py').relative_to(ROOT)))
    with patch.object(original, 'FrozenHurdleTranslator', CountTranslator), \
         patch.object(original, 'fuse_hurdle_gp', checked_fuse), \
         patch.object(original, 'write', write_current), patch.object(original, 'CODE', code):
        original.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/uji54_count_conditioned_20261111')
    parser.add_argument('--seconds', type=float, default=840.)
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 840
    with threadpool_limits(limits=2):
        run(args)
