"""Matched source training and paired evaluation, with no B1 tuning."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

import operator_prior
import run_operator_prior as original
import evaluate_operator_paired as evaluation
from linear_content_control import LinearContentNet
from run_multi_history import digest,write

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'results/uji_operator_prior_v1_20261026'
OUT=ROOT/'results/uji_linear_content_v1_20261026'
EVAL=ROOT/'results/uji_linear_content_paired_v1_20261026'
PAIRED=ROOT/'results/uji_operator_paired_v2_20261026'
STEPS=1500


def train():
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'linear_content_control.py',ROOT/'operator_prior.py',ROOT/'run_operator_prior.py',
           ROOT/'inference.py',ROOT/'transfer.py',ROOT/'history_bank.py']
    inputs=[SOURCE/'training.json',SOURCE/'operator.pt',original.OLD/'parameters.json']
    inputs+=sorted((SOURCE/'train_histories').glob('B*/*.npz'))
    write(OUT/'freeze.json',dict(created_unix=time.time(),hypothesis='At equal B0/B2 historical training information, determine whether nonlinear content representation adds localization benefit beyond an affine rule over the same historical summaries',
        train_floors=original.TRAIN_FLOORS,target_data='No B1 target scans or outcomes read by training',
        tasks='Reuse exact 42 source target/two-history tasks and position/AP objective weights',
        model='Mean pool six original history features; affine 6-to-2 head; same shift and softplus scale outputs',
        steps=STEPS,learning_rate=1e-3,weight_update_every=50,weight_grid=original.W_GRID.tolist(),
        initialization='Exactly same g=0,s=10*softplus(-4) outputs as original OperatorNet',
        optimization='Full-batch Adam; accumulate task gradients before each update to bound memory; no checkpoint selection on B1',
        interpretation='Low capacity/content representation contrast; not a faithful reproduction of a named paper, and not yet general-J training',
        code_sha256={str(p):digest(p) for p in files},inputs={str(p):digest(p) for p in inputs}))
    (OUT/'code').mkdir()
    for p in files:shutil.copy2(p,OUT/'code'/p.name)
    shutil.copy2(ROOT/'LINEAR_CONTENT_PROTOCOL.md',OUT/'PROTOCOL.md')
    base,obs=original.params()
    tasks=original.build_tasks(SOURCE,base)
    assert len(tasks)==42
    model=LinearContentNet().double()
    first=tasks[0];neural=original.OperatorNet().double()
    for weight in [0.,.5,1.]:
        torch.testing.assert_close(original.task_loss(model,first,base,obs,weight),
                                   original.task_loss(neural,first,base,obs,weight),rtol=0,atol=0)
    # Validate the new trainable path against the actual non-Gaussian source loss.
    value=original.task_loss(model,first,base,obs)
    grad,=torch.autograd.grad(value,model.head.weight)
    checks=[]
    for row,col in [(0,0),(0,4),(1,1)]:
        epsilon=1e-5
        with torch.no_grad():
            model.head.weight[row,col]=epsilon;plus=original.task_loss(model,first,base,obs).item()
            model.head.weight[row,col]=-epsilon;minus=original.task_loss(model,first,base,obs).item()
            model.head.weight[row,col]=0.
        fd=(plus-minus)/(2*epsilon)
        np.testing.assert_allclose(grad[row,col].item(),fd,rtol=2e-3,atol=2e-5)
        checks.append(dict(index=[row,col],gradient=grad[row,col].item(),finite_difference=fd))
    write(OUT/'audit.json',dict(initial_loss_exact=True,gradients=checks,parameter_count=sum(p.numel() for p in model.parameters()),tasks=len(tasks)))
    # The reserved GPU is checked before launch; no package or data download.
    device=torch.device('cuda:1')
    model.to(device)
    operator_prior.GH_NODES=operator_prior.GH_NODES.to(device)
    operator_prior.GH_WEIGHTS=operator_prior.GH_WEIGHTS.to(device)
    for task in tasks:
        for key in ['tokens','means','vars']:task[key]=task[key].to(device)
        task['stats']={k:v.to(device) for k,v in task['stats'].items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    curve=[];started=time.perf_counter()
    with (OUT/'training.jsonl').open('x') as stream:
        for step in range(STEPS+1):
            if step%50==0:
                with torch.no_grad():
                    for task in tasks:
                        losses=[original.task_loss(model,task,base,obs,w).item() for w in original.W_GRID]
                        task['w']=float(original.W_GRID[int(np.argmin(losses))])
                    row=dict(step=step,loss=float(np.mean([original.task_loss(model,t,base,obs).item() for t in tasks])),
                             weights=[t['w'] for t in tasks],seconds=time.perf_counter()-started)
                    curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    print(json.dumps({k:v for k,v in row.items() if k!='weights'}),flush=True)
            if step==STEPS:break
            optimizer.zero_grad()
            for task in tasks:
                loss=original.task_loss(model,task,base,obs)/len(tasks)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite source loss')
                loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite source gradient')
            optimizer.step()
    torch.save({k:v.cpu() for k,v in model.state_dict().items()},OUT/'operator.pt')
    write(OUT/'training.json',dict(completed=True,steps=STEPS,curve=curve,checkpoint_sha256=digest(OUT/'operator.pt'),
        last_update_gradient_norm=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))),
        parameters={k:v.detach().cpu().tolist() for k,v in model.state_dict().items()},
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        tasks=[dict(target=t['target'],sources=t['sources'],aps=int(t['tokens'].shape[1]),w=t['w']) for t in tasks]))


def configure_evaluation():
    evaluation.TRAIN=ROOT/'results/uji_linear_content_polished_v1_20261026'
    evaluation.OUT=EVAL;evaluation.OperatorNet=LinearContentNet


def prepare_evaluation():
    configure_evaluation();evaluation.prepare()
    freeze=json.loads((EVAL/'freeze.json').read_text())
    freeze.update(candidate='Linear content control, not the nonlinear operator',
        nonlinear_comparator=str(ROOT/'results/uji_operator_paired_v2_20261026'),
        attribution='Same original 42 source tasks and training objective; 1500 Adam updates plus declared source-only convergence polish, candidate receives the same raw historical tokens')
    for p in [Path(__file__),ROOT/'linear_content_control.py',ROOT/'polish_linear_content.py']:
        freeze['code_sha256'][str(p)]=digest(p);shutil.copy2(p,EVAL/'code'/p.name)
    baseline_files=[PAIRED/'freeze.json',PAIRED/'summary.json']
    baseline_files+=sorted(PAIRED.glob('B*/*.npz'))+sorted(PAIRED.glob('B*/*.json'))
    freeze['reused_baseline_sha256']={str(p):digest(p) for p in baseline_files}
    freeze['baseline_reuse']='Exact completed paired-v2 baseline fields and predictions; event query coordinates/origin must be identical'
    write(EVAL/'freeze.json',freeze)


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(261026)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','prepare-evaluation','event']);parser.add_argument('--event');args=parser.parse_args()
    if args.stage=='train':train()
    elif args.stage=='prepare-evaluation':prepare_evaluation()
    else:
        configure_evaluation()
        freeze=json.loads((EVAL/'freeze.json').read_text())
        for path,expected in freeze['reused_baseline_sha256'].items():
            if args.event in Path(path).parts:assert digest(Path(path))==expected,path
        evaluation.event(args.event,baseline_directory=PAIRED)
