"""All data-dependent fits in this file use training-file B0 and B2 ONLY.
Reconstruction, not the bytes of the 2026-10-26 checkpoint. No B1 validation opened.
Usage: python -m research.uji_core.round2.paper_replay.limit_study.source data out
"""
import argparse, hashlib, json, sys, time, zipfile, io
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from research.uji_core.protocol import HASHES
import uji_probability_study as uji
ROOT=Path(__file__).resolve().parents[5]; sys.path.insert(0,str(ROOT/'operator_prior_v1'))
import run_operator_prior as R

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(data,out):
    assert sha(data/'trainingData.csv')==HASHES['trainingData.csv']; out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4);torch.set_default_dtype(torch.float64);torch.manual_seed(261026)
    pp=json.loads((ROOT/'research/uji_core/field_parameters.json').read_text())['0'];base,obs=pp['base'],pp['obs']
    R.params=lambda:(base,obs);uji.DATA=data
    original=R.load_raw
    def source_only(filename,names):
        assert filename=='trainingData.csv' and all(n.startswith(('B0F','B2F')) for n in names)
        return original(filename,names)
    R.load_raw=source_only
    R.prepare(out); tasks=R.build_tasks(out,base); assert len(tasks)==42
    # Same scalar correction for every historical target profile. Fit the same source likelihood,
    # same 17-point task mixture update, variance fixed to the GP. Never fit on B1.
    class Constant(torch.nn.Module):
        def __init__(self,x):super().__init__();self.x=x
        def forward(self,z):
            a=z.new_full((z.shape[1],),self.x);return a,torch.zeros_like(a)
    with torch.no_grad():
        c=0.; last=None
        for i in range(8):
            for t in tasks:t['w']=float(R.W_GRID[np.argmin([R.task_loss(Constant(c),t,base,obs,w).item() for w in R.W_GRID])]);
            def risk(x):return np.mean([R.task_loss(Constant(float(x)),t,base,obs).item() for t in tasks])
            fit=minimize_scalar(risk,bounds=(-20.,20.),method='bounded',options={'xatol':.002});assert fit.success;c=float(fit.x)
            weights=tuple(t['w'] for t in tasks)
            if weights==last:break
            last=weights
        (out/'simple.json').write_text(json.dumps({'constant_db':c,'tasks':len(tasks),'loss':float(risk(c)),'seed':261026})+'\n')
    # Full-batch V1. Accumulate each task gradient separately to avoid storing 42 autodiff graphs.
    # Same gradients in real arithmetic; floating-point execution may differ from original.
    torch.manual_seed(261026);net=R.OperatorNet().double(); opt=torch.optim.Adam(net.parameters(),lr=R.LR); curve=[]
    for step in range(R.STEPS+1):
        if step%R.W_EVERY==0:
            with torch.no_grad():
                for t in tasks:t['w']=float(R.W_GRID[np.argmin([R.task_loss(net,t,base,obs,w).item() for w in R.W_GRID])]);
                val=float(np.mean([R.task_loss(net,t,base,obs).item() for t in tasks]));assert np.isfinite(val);curve.append([step,val]);print('source',step,val,flush=True)
        if step==R.STEPS:break
        opt.zero_grad()
        for t in tasks:(R.task_loss(net,t,base,obs)/len(tasks)).backward()
        opt.step()
    torch.save(net.state_dict(),out/'operator.pt');(out/'freeze.json').write_text(json.dumps({'source_buildings':[0,2],'tasks':42,'training':curve,'source_only':True,'operator_sha256':sha(out/'operator.pt'),'simple_sha256':sha(out/'simple.json'),'source_csv':HASHES['trainingData.csv'],'seed':261026,'steps':R.STEPS,'scope':'V1 numerical reconstruction, not recovered historical weights'},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.data,a.out)
