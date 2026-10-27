"""Source-only convergence check/polish for the 14-parameter control.

Added before any B1 evaluation of the linear model: its Adam curve still fell
at step 1500. Keep that checkpoint; do not handicap a mature simple baseline
merely by giving it the same number of updates as the nonlinear model.
"""
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
from scipy.optimize import minimize
import torch

import operator_prior
import run_operator_prior as original
from linear_content_control import LinearContentNet
from run_multi_history import digest,write


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).resolve().parent;parent=root/'results/uji_linear_content_v1_20261026'
    source=root/'results/uji_operator_prior_v1_20261026'
    out=root/'results/uji_linear_content_polished_v1_20261026';out.mkdir(exist_ok=False)
    base,obs=original.params();start=json.loads((parent/'training.json').read_text())
    write(out/'freeze.json',dict(created_unix=time.time(),scope=__doc__,
        reason='Adam last gradient norm 0.01237 and continuing source loss decrease; no linear-model B1 localization inspected',
        inputs={str(p):digest(p) for p in [parent/'operator.pt',parent/'training.json',parent/'freeze.json',Path(__file__)]},
        optimizer='At most five alternating rounds: L-BFGS-B maxiter100 on 14 coefficients, then original 17-point source weight grid',
        selection='Lowest final source objective only; B1 and confirmation data are never read',
        extra_budget='Reported separately; original 1500-update Adam checkpoint preserved'))
    shutil.copy2(__file__,out/Path(__file__).name)
    shutil.copy2(root/'LINEAR_CONTENT_PROTOCOL.md',out/'PROTOCOL.md')
    with (out/'PROTOCOL.md').open('a') as f:f.write('\n源端收敛补充：1500步仍有可见梯度/下降，未评价B1前追加最多五轮L-BFGS与原配比网格交替；保留原checkpoint，额外预算单列，不据目标结果选择。\n')
    tasks=original.build_tasks(source,base)
    for task,old in zip(tasks,start['tasks']):
        assert task['target']==old['target'] and task['sources']==old['sources']
        task['w']=old['w']
    device=torch.device('cuda:1');model=LinearContentNet().double().to(device)
    model.load_state_dict(torch.load(parent/'operator.pt',weights_only=True))
    operator_prior.GH_NODES=operator_prior.GH_NODES.to(device);operator_prior.GH_WEIGHTS=operator_prior.GH_WEIGHTS.to(device)
    for task in tasks:
        for k in ['tokens','means','vars']:task[k]=task[k].to(device)
        task['stats']={k:v.to(device) for k,v in task['stats'].items()}
    parameters=list(model.parameters());calls=0
    def set_parameters(x):
        offset=0
        with torch.no_grad():
            for p in parameters:
                p.copy_(torch.as_tensor(x[offset:offset+p.numel()],device=device).reshape_as(p));offset+=p.numel()
    def objective(x):
        global calls
        calls+=1;set_parameters(x);model.zero_grad();total=0.
        for task in tasks:
            loss=original.task_loss(model,task,base,obs)/len(tasks)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite source objective')
            total+=loss.item();loss.backward()
        grad=torch.cat([p.grad.reshape(-1) for p in parameters]).cpu().numpy()
        if not np.isfinite(grad).all():raise RuntimeError('Nonfinite source gradient')
        return total,grad
    x=torch.cat([p.detach().reshape(-1) for p in parameters]).cpu().numpy();rounds=[]
    for iteration in range(5):
        result=minimize(objective,x,jac=True,method='L-BFGS-B',options=dict(maxiter=100,ftol=1e-12,gtol=1e-7,maxls=30))
        x=result.x.copy();set_parameters(x);changed=0
        with torch.no_grad():
            for task in tasks:
                values=[original.task_loss(model,task,base,obs,w).item() for w in original.W_GRID]
                w=float(original.W_GRID[int(np.argmin(values))]);changed+=int(w!=task['w']);task['w']=w
        value,gradient=objective(x)
        row=dict(round=iteration,loss=value,gradient_max=float(abs(gradient).max()),changed_source_weights=changed,
            optimizer_success=bool(result.success),message=str(result.message),iterations=int(result.nit),evaluations=calls)
        rounds.append(row);print(json.dumps(row),flush=True)
        if changed==0 and abs(gradient).max()<1e-5:break
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'operator.pt')
    write(out/'training.json',dict(completed=True,parent_checkpoint=digest(parent/'operator.pt'),checkpoint_sha256=digest(out/'operator.pt'),
        source_loss_before=start['curve'][-1]['loss'],rounds=rounds,evaluations=calls,
        converged=bool(rounds[-1]['changed_source_weights']==0 and rounds[-1]['gradient_max']<1e-5),
        parameters={k:v.detach().cpu().tolist() for k,v in model.state_dict().items()},
        tasks=[dict(target=t['target'],sources=t['sources'],w=t['w']) for t in tasks]))
