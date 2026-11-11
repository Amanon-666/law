"""Serial paper runs; preserve failures and commit lightweight evidence after each building."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
RESULTS = ROOT/'results'
OPTIONS = ROOT/'src/paper_bench_20261111/options.json'


def main():
    runs = [('uji', b, [0, 1, 2]) for b in ('B0', 'B1', 'B2')]
    runs += [('tampere', b, [0]) for b in ('TIE1', 'SAH1')]
    runs += [('sod', 'CETC331', [0]), ('uts', 'New IT', [0])]
    state = []
    for dataset, building, repeats in runs:
        output = RESULTS/f'paper_{dataset}_{building}_20261111'.replace(' ', '_')
        cmd = [sys.executable, '-u', '-m', 'paper_bench_20261111.run',
               '--dataset', dataset, '--building', building, '--repeats', *map(str, repeats),
               '--methods', 'wknn_S', 'wknn_HS', 'gp_S', 'gp_histmean', 'refi',
               '--method-options', str(OPTIONS), '--output', str(output)]
        if dataset != 'uji':
            cmd += ['--source-dataset', 'uji', '--source-buildings', 'B0', 'B1', 'B2']
        tick = time.monotonic()
        with (RESULTS/(output.name+'.log')).open('w') as stream:
            child = subprocess.Popen(cmd, cwd=ROOT/'src', stdout=stream, stderr=subprocess.STDOUT)
            entry = dict(dataset=dataset, building=building, pid=child.pid, command=cmd, status='running')
            state.append(entry)
            (RESULTS/'paper_batch_20261111.json').write_text(json.dumps(state, indent=2))
            print(json.dumps(entry), flush=True)
            rc = child.wait()
        entry.update(status='finished' if rc == 0 else 'runner_failed', exit_code=rc,
                     seconds=time.monotonic()-tick)
        (RESULTS/'paper_batch_20261111.json').write_text(json.dumps(state, indent=2))
        paths = [str(p.relative_to(REPO)) for p in output.rglob('*.json')]
        paths += [str((RESULTS/'paper_batch_20261111.json').relative_to(REPO)),
                  str((RESULTS/(output.name+'.log')).relative_to(REPO))]
        subprocess.run(['git', 'add', '-f', '--', *paths], cwd=REPO, check=True)
        subprocess.run(['git', 'commit', '-m', f'Paper benchmark {dataset} {building}: preserve results and failures'],
                       cwd=REPO, check=True)
        entry['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        print(json.dumps(entry), flush=True)


if __name__ == '__main__':
    main()
