"""Run untouched K-curve events while preserving failed events for diagnosis.

Only scheduling changes: each event still calls the frozen run_event, and a
failed AP stops its whole event. No failed event is retried by this entry.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import traceback

import torch
import run_k_curve as original


def run(building):
    original.verify()
    out = original.OUT / 'continuation'
    out.mkdir(exist_ok=True)
    record_path = out / f'{building}.json'
    if record_path.exists():
        raise FileExistsError(record_path)
    initial_failure = original.OUT / f'{building}_failure.json'
    failed_event = original.read(initial_failure)['event']
    metas = [e for e in original.manifest()['events'] if e['building'] == building]
    complete = [e['id'] for e in metas if (original.OUT/e['id']/'completed.json').exists()]
    record = dict(started=datetime.now(timezone.utc).isoformat(), building=building,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  original_freeze_sha256=original.digest(original.OUT/'freeze.json'),
                  old_failure=str(initial_failure), old_failure_sha256=original.digest(initial_failure),
                  previously_completed=complete, not_retried=[failed_event], completed=[], failures={},
                  policy='Original model/solver/budget; stop failed event, continue other predetermined events. Failures remain unresolved, never counted as complete.')
    original.write(record_path, record)
    m = original.manifest()
    for event in metas:
        if event['id'] in complete or event['id'] == failed_event:
            continue
        try:
            original.run_event(event, m['base'], m['obs'])
            record['completed'].append(event['id'])
        except Exception as error:
            record['failures'][event['id']] = dict(error=repr(error),
                time=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc())
            original.log(stage='event_failed_not_omitted', event=event['id'], error=repr(error))
        original.write(record_path, record)
    record['finished'] = datetime.now(timezone.utc).isoformat()
    record['all_events_complete'] = all((original.OUT/e['id']/'completed.json').exists() for e in metas)
    original.write(record_path, record)


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('building')
    run(parser.parse_args().building)
