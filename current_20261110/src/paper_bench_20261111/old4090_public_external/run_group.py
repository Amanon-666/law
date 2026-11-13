"""Original public Tab GPU consumer on one fixed external paper group; no training."""
import hashlib
import json
from pathlib import Path
import sys

WORK = Path('/data/users/yanrongen/projects/panyushuo/PaperGP-20261113/public_tab')
ASSETS = Path('/data/users/yanrongen/projects/panyushuo/TabAssets-B1-20261111')
CHECKPOINT_SHA = '0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    group = sys.argv[1]
    manifest = json.loads((WORK/'preparation.json').read_text())
    selected = manifest['groups'][group]
    for path, expected in manifest['inputs_sha256'].items():
        assert sha(path) == expected, path
    assert sha(manifest['checkpoint']) == CHECKPOINT_SHA
    from law_bench import run
    from law_bench.data import load
    from law_bench.protocol import make_tasks
    source = json.loads(Path(selected['bindings']).read_text())
    ref = source[group] if selected['dataset'] == 'tampere' else source['datasets'][selected['dataset']]
    records = {r['task_id']: r for r in ref['tasks']}
    buildings, ap_ids = load(selected['dataset'], root=selected['root'])
    floors = buildings[selected['building']]
    checked = []
    for floor in sorted(floors):
        for task, truth in make_tasks(selected['dataset'], selected['building'], floors,
                ap_ids[selected['building']], floor, budgets=(1, 3, 5), repeat=0,
                seed=20261111, R=3, query_positions=60):
            original = records[task.task_id]
            assert task.meta == original['meta']
            assert truth['row_indices'].tolist() == original['query_row_indices']
            assert truth['groups'].tolist() == original['query_groups']
            checked.append(task.task_id)
    assert len(checked) == selected['tasks'] == len(records)
    output = WORK/'results'/selected['output']
    original_write = run._write

    def write(path, value):
        original_write(path, value)
        if path.name == 'freeze.json':
            original_write(path.parent/'public_tab_provenance.json', dict(
                preparation_sha256=sha(WORK/'preparation.json'), checkpoint_sha256=CHECKPOINT_SHA,
                query_bindings_verified=checked, device='cuda', n_estimators=8,
                context='All supplied H scans and current S; original preprocessing and projected readout',
                source_updates=0, new_weights_downloaded=False,
                method='Unchanged PaperTab-20261112 public consumer; data/root/output only changed',
                inputs_sha256=manifest['inputs_sha256']))
    run._write = write
    sys.argv = [str(Path(run.__file__)), '--dataset', selected['dataset'],
        '--building', selected['building'], '--root', selected['root'],
        '--source-dataset', 'uji', '--source-root', manifest['uji_csv'],
        '--source-buildings', 'B0', 'B1', 'B2', '--protocol', 'static-resample-v2',
        '--seed', '20261111', '--budgets', '1', '3', '5', '--repeats', '0',
        '--R', '3', '--query-positions', '60', '--methods', 'tabicl',
        '--method-options', str(WORK/'options.json'), '--output', str(output)]
    run.main()
    summary = json.loads((output/'tabicl.json').read_text())
    assert sum(v['completed_tasks'] for v in summary['by_K'].values()) == selected['tasks']
    assert all(v['complete'] and v['failed_tasks'] == 0 for v in summary['by_K'].values())


if __name__ == '__main__':
    main()
