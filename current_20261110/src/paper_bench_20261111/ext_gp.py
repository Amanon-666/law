"""Original completed paper ReFi + retained GP, one GP evaluation per task.

Only the cached original five-model ReFi and existing full-history GP fields
are used. GP-only predictions are saved alongside the fixed half-log-pool;
export_gp scores those same predictions, without a second GP computation.
"""
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np

from law_bench import run
from law_bench.method_retained_gp import RetainedGPMethod
from hurdle_gp_pool import fuse_hurdle_gp


class ReFiGP:
    name = 'refi_GP'

    def __init__(self, source_run, gp_options):
        self.source_run = Path(source_run)
        if gp_options.get('fit_missing_history', False):
            raise ValueError('Paper extension reuses existing full-H fields only')
        self.gp = RetainedGPMethod(**gp_options)

    def fit_source(self, sources):
        root = self.source_run
        freeze = json.loads((root/'freeze.json').read_text())
        fit = json.loads((root/'source_models/refi.json').read_text())
        model_path = root/'source_models/refi.joblib'
        identity = run._data_identity({s['building']: s['floors'] for s in sources},
                                      {s['building']: s['ap_ids'] for s in sources})
        if (any(s['dataset'] != 'uji' for s in sources)
                or identity != freeze['source_data'] or fit['status'] != 'completed'
                or run._sha(model_path) != fit['sha256']):
            raise ValueError('Completed paper ReFi model/source identity mismatch')
        self.refi = joblib.load(model_path)
        if (len(self.refi.translator.models) != 5
                or self.refi.provenance['seeds'] != [1, 2, 3, 4, 5]
                or sorted(self.refi.sources) != sorted(identity)):
            raise ValueError('Expected original five-seed paper ReFi')
        source_root = Path(__file__).resolve().parents[1]
        for name, sha in self.refi.provenance['code'].items():
            if run._sha(source_root/name) != sha:
                raise ValueError(f'Original ReFi code changed: {name}')
        self.gp.fit_source(sources)
        self.provenance = dict(source_run=str(root), model_sha256=fit['sha256'],
            source_data=identity, refi=self.refi.provenance,
            extension_code_sha256=run._sha(__file__), training_performed=False)
        return self

    def predict(self, task):
        self.diagnostics = dict(provenance=self.provenance, completed=False)
        if task.dataset == 'uji' and task.building in self.refi.sources:
            raise ValueError('Target building is a source')
        tick = time.monotonic()
        prepared = self.refi.translator.prepare(
            {f'{task.building}F{f}': d for f, d in task.H.items()}, task.S,
            f'{task.building}F{task.floor}')
        field = prepared.predict(task.Q_rss)
        refi_seconds = time.monotonic()-tick
        original = json.loads((self.source_run/'events/refi'/(task.task_id+'.json')).read_text())
        if not np.allclose(field['projected'], original['prediction'], rtol=0, atol=1e-9):
            raise ValueError('ReFi predictions differ from the original completed paper task')
        tick = time.monotonic()
        try:
            gp = self.gp.predict_scores(task)
        finally:
            self.diagnostics['GP'] = self.gp.diagnostics
        gp_seconds = time.monotonic()-tick
        if not (np.array_equal(field['aps'], gp['aps'])
                and np.array_equal(prepared.bounds, gp['bounds'])):
            raise ValueError('ReFi/GP AP axis or H/S coordinate bounds differ')
        fused = fuse_hurdle_gp(prepared.cand_xy, field['log_position_mass'],
            gp['grid'], gp['scores'], gp['raw'], prepared.bounds)
        self.diagnostics.update(completed=True, pool='probability_half',
            ReFi_projected=field['projected'].tolist(), GP_projected=gp['projected'].tolist(),
            refi_seconds=refi_seconds, gp_seconds=gp_seconds,
            original_event_sha256=run._sha(self.source_run/'events/refi'/(task.task_id+'.json')))
        return fused['predictions']['probability_half']


def export_gp(output):
    """Use the same frozen tasks and GP predictions to emit standard table rows."""
    from law_bench.data import load
    from law_bench.protocol import make_tasks
    from law_bench.score import score, aggregate

    output = Path(output)
    freeze = json.loads((output/'freeze.json').read_text())
    c = freeze['configuration']
    if c['protocol'] != 'static-resample-v2':
        raise ValueError('Paper export expects the static paper protocol')
    buildings, aps = load(c['dataset'], root=c['root'])
    target = buildings[c['building']]
    if run._data_identity({c['building']: target}, {c['building']: aps[c['building']]}) != freeze['target_data']:
        raise ValueError('Target data changed since predictions')
    records = []
    for floor in c['floors'] or sorted(target):
        for repeat in c['repeats']:
            tasks = make_tasks(c['dataset'], c['building'], target, aps[c['building']], floor,
                budgets=tuple(c['budgets']), repeat=repeat, seed=c['seed'], R=c['R'],
                query_positions=c['query_positions'])
            for task, truth in tasks:
                source = output/'events/refi_GP'/(task.task_id+'.json')
                combined = json.loads(source.read_text())
                record = {k: combined[k] for k in ('task_id', 'floor', 'K', 'repeat', 'meta', 'status')}
                if combined['status'] == 'completed':
                    if combined['query_row_indices'] != truth['row_indices'].tolist():
                        raise ValueError('Query row binding changed')
                    d = combined['diagnostics']
                    record.update(prediction=d['GP_projected'], scores=score(d['GP_projected'], truth),
                        query_row_indices=combined['query_row_indices'], query_groups=combined['query_groups'],
                        seconds=d['gp_seconds'], diagnostics=d['GP'],
                        derived_from=dict(path=str(source), sha256=run._sha(source)))
                else:
                    record.update(stage=combined.get('stage'), error=combined.get('error'))
                run._write(output/'events/retained_gp'/(task.task_id+'.json'), record)
                records.append(record)
    unavailable = json.loads((output/'task_inventory.json').read_text())['unavailable']
    all_records = records+unavailable
    summary = dict(method='retained_gp', by_K=aggregate(all_records),
        by_floor={str(f): aggregate([r for r in all_records if r['floor'] == f])
                  for f in sorted({r['floor'] for r in all_records})},
        task_records=[f"events/retained_gp/{r['task_id']}.json" for r in records],
        unavailable=unavailable, computation='Same GP predictions as refi_GP; no recomputation')
    run._write(output/'retained_gp.json', summary)
    summaries = json.loads((output/'summary.json').read_text())
    summaries['retained_gp'] = summary
    run._write(output/'summary.json', summaries)
    print(json.dumps(summary['by_K']), flush=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['export-gp']:
        export_gp(sys.argv[2])
    else:
        from paper_bench_20261111 import ext_gp
        sys.modules['law_bench.paper_ext_gp'] = ext_gp
        run.METHODS['refi_GP'] = ('paper_ext_gp', 'ReFiGP', {})
        run.main()
