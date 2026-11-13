"""Original runner for a new paired five-budget UTS curve; fixed assets only."""
import hashlib
import json
from pathlib import Path
import sys

BASE = Path('/data/users/yanrongen/projects/panyushuo/PaperGP-20261113')
WORK = BASE/'uts_kcurve'
REBUILD = BASE/'refi_rebuild'
MODEL = REBUILD/'results/TIE1/source_models/refi.joblib'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    method = sys.argv[1]
    assert method in ('refi', 'wknn_HS', 'retained_gp')
    source = REBUILD/'src' if method == 'refi' else BASE/'src'
    sys.path.insert(0, str(source))
    from law_bench import run
    manifest = json.loads((WORK/'preparation.json').read_text())
    for path, expected in manifest['inputs_sha256'].items():
        assert sha(path) == expected, path
    # Every method waits for the completed fixed-model dependency, not an active fit.
    tie1 = json.loads((REBUILD/'results/TIE1/refi.json').read_text())
    assert sum(v['completed_tasks'] for v in tie1['by_K'].values()) == 18
    assert all(v['complete'] and v['failed_tasks'] == 0 for v in tie1['by_K'].values())
    model_meta = json.loads(MODEL.with_suffix('.json').read_text())
    assert model_meta['status'] == 'completed' and sha(MODEL) == model_meta['sha256']
    bound_model = dict(path=str(MODEL), sha256=model_meta['sha256'],
        source_metadata_sha256=sha(MODEL.with_suffix('.json')),
        completed_TIE1_summary_sha256=sha(REBUILD/'results/TIE1/refi.json'))
    if method == 'refi':
        import joblib
        from paper_bench_20261111 import refi

        def fixed_source(self, sources):
            loaded = joblib.load(MODEL)
            assert loaded.sources == ['B0', 'B1', 'B2']
            assert loaded.provenance['seeds'] == [1, 2, 3, 4, 5]
            assert loaded.provenance['draws'] == 4 and len(loaded.translator.models) == 5
            identity = run._data_identity({s['building']: s['floors'] for s in sources},
                                         {s['building']: s['ap_ids'] for s in sources})
            assert identity == loaded.provenance['data']
            self.__dict__.update(loaded.__dict__)
            return self
        # This replaces the only training entry before the original runner constructs ReFi.
        refi.ReFi.fit_source = fixed_source
        sys.modules['law_bench.paper_refi'] = refi
        run.METHODS['refi'] = ('paper_refi', 'ReFi', {})
    original_tasks = run.make_tasks

    def paired_tasks(*args, **kwargs):
        tasks = original_tasks(*args, **kwargs)
        first = tasks[0][1]
        assert [task.K for task, _ in tasks] == [1, 3, 5, 10, 20]
        for _, truth in tasks:
            assert (truth['row_indices'] == first['row_indices']).all()
            assert (truth['groups'] == first['groups']).all()
        return tasks
    run.make_tasks = paired_tasks
    original_write = run._write

    def write(path, value):
        original_write(path, value)
        if path.name == 'freeze.json':
            original_write(path.parent/'curve_provenance.json', dict(
                preparation_sha256=sha(WORK/'preparation.json'), fixed_refi=bound_model,
                new_source_training=False, fit_missing_GP_history=False,
                old_K135_predictions_reused=False,
                query_policy='Original make_tasks with all five budgets; identical Query across K within floor/repeat',
                dependency_policy='Existing completed TIE1 model and formal full-H GP cache only'))
    run._write = write
    output = WORK/'results'/method
    sys.argv = [str(Path(run.__file__)), '--dataset', 'uts', '--building', 'New IT',
        '--root', str(BASE/'datasets/utsindoorloc/UTS_training.csv'),
        '--source-dataset', 'uji', '--source-root', manifest['uji_csv'],
        '--source-buildings', 'B0', 'B1', 'B2', '--protocol', 'static-resample-v2',
        '--seed', '20261111', '--budgets', '1', '3', '5', '10', '20',
        '--repeats', '0', '1', '2', '--R', '3', '--query-positions', '60',
        '--methods', method, '--method-options', str(WORK/'options.json'), '--output', str(output)]
    run.main()
    report = json.loads((output/(method+'.json')).read_text())
    assert sum(v['completed_tasks'] for v in report['by_K'].values()) == 135
    assert all(v['complete'] and v['failed_tasks'] == 0 for v in report['by_K'].values())


if __name__ == '__main__':
    main()
