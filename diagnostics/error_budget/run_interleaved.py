"""Thin v2 entry: freeze revised folds, reuse unchanged v1 inference/readout."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import run_reference as reference

ROOT=reference.ROOT
reference.OUT=ROOT/'results/uji_error_budget_interleaved_v2_20261026'
reference.PREP=reference.OUT/'preparation'
EXTRA=[Path(__file__),Path(__file__).with_name('prepare_interleaved.py')]


def check():
    # Reuse already-passed numerical checks; only the folding has changed.
    old=ROOT/'results/uji_error_budget_reference_v1_20261026'
    original=json.loads((old/'freeze.json').read_text())
    for path,sha in original['code_sha256'].items():assert reference.digest(Path(path))==sha,path
    previous=json.loads((old/'check.json').read_text())
    assert previous['same_mean_field_exact'] and previous['missing_reference_fallback']
    summary=json.loads((reference.PREP/'summary.json').read_text());assert summary['all_checks_passed']
    assert {r['event'] for r in summary['records']}=={'B1F1_s0','B1F2_s0'}
    reference.write(reference.OUT/'check.json',dict(created_unix=time.time(),
        script_sha256=reference.digest(Path(reference.__file__)),reused_numerical_check=str(old/'check.json'),
        reused_numerical_check_sha256=reference.digest(old/'check.json'),new_fold_checks=summary,
        entry_sha256=reference.digest(Path(__file__)),prepare_sha256=reference.digest(EXTRA[1])))
    reference.log(stage='v2_check',passed=True)


def freeze():
    reference.freeze()
    path=reference.OUT/'freeze.json';record=json.loads(path.read_text())
    record.update(version='v2 post-hoc interleaved coverage revision, two events',
        revision='v1 macro-block failure preserved; width chosen from source base.length by coordinate-only evidence. No model tuning and not independent confirmation.',
        hypothesis='With improved extra-label coverage, does b improve true-position scan prediction AND localization versus a? d is an empirical reference for this fixed GP family, never a best-achievable bound.',
        stopping='A source-GP/inference failure stops visibly. No tile-width or model revision from v2 scores. If reference signal still does not improve, stop using this GP as an error-upper-limit construction.',
        events=['B1F1_s0','B1F2_s0'])
    for p in EXTRA:
        record['code_sha256'][str(p.resolve())]=reference.digest(p)
        shutil.copy2(p,reference.OUT/'code'/p.name)
    record['inputs'][str(reference.OUT/'check.json')]=reference.digest(reference.OUT/'check.json')
    reference.write(path,record)
    reference.log(stage='v2_freeze_complete',entry_and_preparation_hashed=True)


def run():
    record=json.loads((reference.OUT/'freeze.json').read_text())
    for p in EXTRA:assert record['code_sha256'][str(p.resolve())]==reference.digest(p)
    assert record['events']==['B1F1_s0','B1F2_s0']
    reference.run()


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    reference.torch.set_default_dtype(reference.torch.float64);reference.torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['check','freeze','run','summarize']);args=parser.parse_args()
    dict(check=check,freeze=freeze,run=run,summarize=reference.summarize)[args.stage]()
