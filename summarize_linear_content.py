"""Summarize the frozen matched-information comparison; no model selection."""
import json
from pathlib import Path
import socket

import numpy as np

from run_multi_history import digest, write

ROOT = Path(__file__).resolve().parent
LINEAR = ROOT / 'results/uji_linear_content_paired_v1_20261026'
NONLINEAR = ROOT / 'results/uji_operator_paired_v2_20261026'


def main():
    events = json.loads((ROOT / 'results/uji_history_bank_v1_20261026/manifest.json').read_text())['events']
    freeze = json.loads((LINEAR / 'freeze.json').read_text())
    for path, expected in freeze['reused_baseline_sha256'].items():
        assert digest(Path(path)) == expected, path
    records, evidence, ess = [], {}, []
    for event in events:
        name = event['id']
        paths = [LINEAR / name / 'completed.json', NONLINEAR / name / 'completed.json']
        linear, nonlinear = [json.loads(p.read_text()) for p in paths]
        evidence.update({str(p): digest(p) for p in paths})
        for split in ['internal', 'official']:
            def result(data, method):
                return next(r for r in data['results'] if r['split'] == split and r['method'] == method)
            assert result(linear, 'adapted_bank') == result(nonlinear, 'adapted_bank')
            records.append(dict(event=name, floor=event['floor'], split=split,
                baseline=result(linear, 'adapted_bank')['mde'],
                linear=result(linear, 'operator_bank')['mde'],
                nonlinear=result(nonlinear, 'operator_bank')['mde']))
        for ap in event['ap_columns']:
            d = json.loads((LINEAR / name / f'ap{ap}.json').read_text())
            ess.append(d['operator_bank']['ess'])
    summary = {}
    for split in ['internal', 'official']:
        rows = [r for r in records if r['split'] == split]
        means = {m: float(np.mean([r[m] for r in rows])) for m in ['baseline', 'linear', 'nonlinear']}
        comparisons = {}
        for candidate, reference in [('linear', 'baseline'), ('nonlinear', 'linear'), ('nonlinear', 'baseline')]:
            delta = [r[candidate] - r[reference] for r in rows]
            comparisons[f'{candidate}_minus_{reference}'] = dict(mean=float(np.mean(delta)),
                improved_events=sum(d < 0 for d in delta),
                by_floor={f: float(np.mean([r[candidate] - r[reference] for r in rows if r['floor'] == f]))
                    for f in sorted({r['floor'] for r in rows})})
        summary[split] = dict(mean_mde=means, comparisons=comparisons)
    output = dict(summary=summary, records=records, min_linear_ess=min(ess), completed_sha256=evidence,
        script_sha256=digest(Path(__file__)),
        scope='Six previously viewed UJI B1 K5R3 events. Both learned controls use the same B0/B2 tasks and features; linear received extra source-only optimization. Fixed two-source tasks. No independent confirmation or nonlinear-mechanism necessity claim.')
    with (LINEAR / 'summary.json').open('x') as f:
        json.dump(output, f, indent=2)
    print(json.dumps(dict(summary=summary, min_linear_ess=min(ess))))


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical work stays on ray-web')
    main()
