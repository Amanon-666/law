"""Run one building of the immutable joint-v2 experiment; no shared summary writes."""
import argparse
from pathlib import Path
import socket

import torch

from residual_probe import run_change_mode_joint_v2 as run
from run_multi_history import digest


def main(building):
    frozen=run.loc.read(run.OUT/'freeze.json');run.loc.verify()
    for path,sha in frozen['code_sha256'].items():assert digest(Path(path))==sha,path
    assert digest(run.CHECK)==frozen['checks_sha256']
    assert digest(run.loc.PREPARED/'manifest.json')==frozen['prepared_manifest_sha256']
    for item in frozen['source_models'].values():assert digest(Path(item['path']))==item['sha256']
    events=[e for e in run.loc.manifest()['events'] if e['building']==building]
    if not events:raise ValueError(building)
    for event in events:
        if all((run.OUT/event['id']/arm/'result.json').exists() for arm in run.ARMS):continue
        run.event_score(event,frozen)
    run.loc.write(run.OUT/f'completed_{building}.json',dict(completed=run.loc.now(),building=building,
                                                         events=[e['id'] for e in events]))


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('building');args=parser.parse_args()
    main(args.building)
