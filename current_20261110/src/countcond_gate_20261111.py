"""Run loaded-model ablation and stop before countcond if same differs from ours_A."""
import json
import os
from pathlib import Path

from law_bench import run_claude

R = run_claude.R
write = R._write
original = Path(os.environ['COUNTCOND_ORIGINAL'])
fields = ('task_id', 'floor', 'K', 'repeat', 'meta', 'status', 'scores',
          'prediction', 'query_row_indices', 'query_groups')


def checked_write(path, value):
    if path.name == 'ours_loaded_same.json' and path.parent.name != 'source_models':
        output = path.parent
        prior = json.loads((original / 'ours_A.json').read_text())
        if len(value['task_records']) != len(prior['task_records']):
            raise RuntimeError('same task count differs from ours_A')
        for relative in value['task_records']:
            task = json.loads((output / relative).read_text())
            reference = json.loads((original / relative.replace('ours_loaded_same', 'ours_A')).read_text())
            for field in fields:
                if task[field] != reference[field]:
                    raise RuntimeError(f'same differs: {relative} {field}')
        print(f'SAME_GATE_PASS {len(value["task_records"])} {output}', flush=True)
    write(path, value)


R._write = checked_write
R.main()
