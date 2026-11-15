"""Serially finish the frozen source selection, three seeds and B1 evaluation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def remaining(deadline):
    seconds = math.ceil(deadline-time.time())
    if seconds <= 0:
        raise TimeoutError('Frozen eight-hour formal-stage wall budget exhausted')
    return seconds


def wait_external(work,name,pid,deadline):
    path = work/(name+'.exit.json')
    while not path.exists():
        remaining(deadline)
        try:
            os.kill(pid,0)
        except ProcessLookupError:
            time.sleep(1)
            if not path.exists():
                raise RuntimeError(f'{name} process stopped without its exit record')
        time.sleep(15)
    result = read(path)
    assert result['exit_code'] == 0, result
    return result


def stage(work,name,script,arguments,output,deadline):
    exit_path = work/(name+'.exit.json')
    complete_path = output/'complete.json'
    if complete_path.exists():
        complete = read(complete_path)
        assert complete['complete']
        print(json.dumps(dict(stage='reuse_completed_stage',name=name)),flush=True)
        return complete
    command = [sys.executable,'-u',str(Path(__file__).parent/script),*arguments]
    write(work/'driver_status.json',dict(stage=name,status='running',command=command,
        deadline_epoch=deadline,source_selection_complete=(work/'selected_epochs.json').exists()))
    started = time.time()
    timeout = remaining(deadline)
    with (work/(name+'.log')).open('a') as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    write(exit_path,dict(command=command,exit_code=result.returncode,start_epoch=started,
        end_epoch=time.time(),elapsed_seconds=time.time()-started))
    assert result.returncode == 0, f'{name} exited with {result.returncode}'
    complete = read(complete_path)
    assert complete['complete']
    return complete


def model_record(path):
    assert read(path)['source_only'] is True
    return dict(path=str(path),sha256=digest(path))


def main(args):
    work, source = args.work, Path(__file__).parent
    deadline = args.start_epoch+8*3600
    freeze_path = source/'formal_freeze_20261115.json'
    freeze = read(freeze_path)
    for name,expected in freeze['source_code_sha256'].items():
        assert digest(source/name) == expected
    code_names = ['formal_driver.py','formal_source.py','formal_direct.py','evaluate_final.py',
        'field_predictor.py','native_tab.py','source_cells.py','source_probe.py','direct_coordinate.py',
        'cache_original_gp.py']
    plan = dict(formal_freeze_sha256=digest(freeze_path),
        code_sha256={name:digest(source/name) for name in code_names},
        source_development_start_epoch=args.start_epoch,
        source_development_start_evidence='Actual launcher3563650 ps lstart, read by independent execution monitor',
        deadline_epoch=deadline,maximum_formal_single_GPU_wall_seconds=8*3600,
        accounting='Conservatively count all wall time since the first formal source stage, including CPU preparation, OOF, source validation and target evaluation.',
        source_selection='B0 train, B2 development only, choose epoch0/1/2/3 independently for B and C before B1 evaluation.',
        final='Restart seeds1/2/3 from public weights on B0+B2. Report all seeds. D uses frozen public weights and its own OOF labels.',
        target_evaluation='Only after all source selections and final model assets are committed. Original B1 36 tasks.',
        stages=['existing field_development','direct_development','field_final_seed1..3',
            'frozen_field_seed1..3','direct_final_seed1..3','target_B1'],
        resources=dict(gpu=1,GPU_index=0,CPU_threads=2,shared_GP_cache_CPU_threads=2),
        relation_encoding_ablation='Separate later CPU-only stage, outside this driver and the A/B/C/D comparison.')
    plan_path = work/'driver_plan.json'
    if plan_path.exists():
        assert read(plan_path) == plan
    else:
        write(plan_path,plan)
    write(work/'driver_status.json',dict(stage='field_development',status='waiting_for_existing_stage',
        existing_launcher=args.field_development_pid,deadline_epoch=deadline))
    field_exit = wait_external(work,'field_development',args.field_development_pid,deadline)
    assert field_exit['start_epoch'] >= args.start_epoch
    field = read(work/'field_development/complete.json')
    assert field['complete'] and field['source_only'] and not field['target_evaluation']
    direct_output = work/'direct_development'
    direct = stage(work,'direct_development','formal_direct.py',
        ['--phase','development','--seed','1','--epochs','3','--output',str(direct_output)],direct_output,deadline)
    selection_path = work/'selected_epochs.json'
    selection = dict(field_epochs=field['selected_source_epoch'],direct_epochs=direct['selected_source_epoch'],
        source_training_buildings=['B0'],source_development_building='B2',target_evaluation_used=False,
        rule='Minimum source B2 pooled position MDE among epochs0/1/2/3, earliest exact tie.',
        field_complete_sha256=digest(work/'field_development/complete.json'),
        field_curve_sha256=digest(work/'field_development/learning_curve.json'),
        direct_complete_sha256=digest(direct_output/'complete.json'),
        driver_plan_sha256=digest(plan_path))
    if selection_path.exists():
        assert read(selection_path) == selection
    else:
        write(selection_path,selection)
    print(json.dumps(dict(stage='source_selection_frozen',**selection)),flush=True)
    manifest = dict(source_selection_sha256=digest(selection_path),B={},C={},D={})
    for seed in (1,2,3):
        output = work/f'field_final_seed{seed}'
        stage(work,f'field_final_seed{seed}','formal_source.py',
            ['--phase','final','--seed',str(seed),'--epochs',str(selection['field_epochs']),'--output',str(output)],output,deadline)
        manifest['B'][str(seed)] = model_record(output/'model_assets.json')
    for seed in (1,2,3):
        if selection['field_epochs'] == 0:
            manifest['D'][str(seed)] = dict(manifest['B'][str(seed)],
                identical_to=f'B seed{seed}',reason='Source selection chose public epoch0; reuse identical frozen assets without another computation.')
        else:
            output = work/f'frozen_field_seed{seed}'
            stage(work,f'frozen_field_seed{seed}','formal_source.py',
                ['--phase','final','--seed',str(seed),'--epochs','0','--output',str(output)],output,deadline)
            manifest['D'][str(seed)] = model_record(output/'model_assets.json')
        output = work/f'direct_final_seed{seed}'
        stage(work,f'direct_final_seed{seed}','formal_direct.py',
            ['--phase','final','--seed',str(seed),'--epochs',str(selection['direct_epochs']),'--output',str(output)],output,deadline)
        manifest['C'][str(seed)] = model_record(output/'model_assets.json')
    manifest_path = work/'model_manifest.json'
    if manifest_path.exists():
        assert read(manifest_path) == manifest
    else:
        write(manifest_path,manifest)
    cache_exit = read(work/'gp_cache.exit.json')
    assert cache_exit['exit_code'] == 0 and read(work/'gp_surfaces/complete.json')['complete']
    result = stage(work,'target_evaluation','evaluate_final.py',['--work',str(work)],work/'target_B1',deadline)
    write(work/'complete.json',dict(complete=True,source_selection=selection,
        model_manifest_sha256=digest(manifest_path),target_complete=result,
        start_epoch=args.start_epoch,end_epoch=time.time(),
        conservative_single_GPU_stage_wall_seconds=time.time()-args.start_epoch,
        driver_plan_sha256=digest(plan_path)))
    write(work/'driver_status.json',dict(stage='complete',status='completed',
        complete_sha256=digest(work/'complete.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--field-development-pid',type=int,required=True)
    parser.add_argument('--start-epoch',type=int,required=True)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        write(arguments.work/'driver_status.json',dict(status='failed',error=repr(error),
            time_epoch=time.time(),source_selection_complete=(arguments.work/'selected_epochs.json').exists()))
        raise
