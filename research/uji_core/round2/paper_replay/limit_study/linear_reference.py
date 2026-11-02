"""A secondary, simpler, same-information challenger, not a substitute identity for V1. Compute mean shift from six existing pooled history tokens and an intercept; no neural and no variance head. Predeclared, no B1 tuning and no architecture search. Both scalar and linear use identical B0/B2 42 tasks, equal task weighting, matching stat loss, same draws, fixed identity-only source mixtures and 1500 steps. Fit and target jobs separate."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from .source import ROOT,R,sha,uji,HASHES
from . import paired
from unittest.mock import patch
class Affine(torch.nn.Module):
    def __init__(self):super().__init__();self.beta=torch.nn.Parameter(torch.zeros(7))
    def forward(self,tokens):
        z=tokens.mean(0);g=20*(self.beta[0]+z@self.beta[1:]);return g,torch.zeros_like(g)
def train(data,out):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);out.mkdir(exist_ok=True,parents=True);pp=json.loads((ROOT/'research/uji_core/field_parameters.json').read_text())['0'];base,obs=pp['base'],pp['obs'];assert sha(data/'trainingData.csv')==HASHES['trainingData.csv'];orig=R.load_raw
    def sourceonly(name,keys):
        assert name=='trainingData.csv' and all(k.startswith(('B0F','B2F')) for k in keys);return orig(name,keys)
    R.params=lambda:(base,obs);R.load_raw=sourceonly;uji.DATA=data;R.prepare(out);tasks=R.build_tasks(out,base);assert len(tasks)==42
    starts=[];arrays=[];p0=0
    with torch.no_grad():
        for t in tasks:
            # Source task's H-only identity mean, fitted without either correction, same rule in both arms.
            w=float(R.W_GRID[np.argmin([R.task_loss(R.ZERO,t,base,obs,v).item() for v in R.W_GRID])]);mu=(w*t['means'][0]+(1-w)*t['means'][1]).reshape(-1);vv=(base['drift']**2+w*w*t['vars'][0]+(1-w)**2*t['vars'][1]).reshape(-1);zz=t['tokens'].mean(0).reshape(-1,6);a=torch.cat((zz,mu[:,None],vv[:,None],*[t['stats'][k].reshape(-1,1) for k in ('n','k','s1','s2')]),dim=1);starts.append((p0,len(a)));p0+=len(a);arrays.append(a)
    pack=torch.cat(arrays);del tasks,arrays
    pr=Affine().double();c=torch.nn.Parameter(torch.zeros(()));opt=torch.optim.Adam([pr.beta,c],lr=.001);rng=np.random.default_rng(261026);curve=[]
    for step in range(1500):
        idx=np.concatenate([o+rng.integers(m,size=64) for o,m in starts]);z=pack[idx];features=z[:,:6];base_mu,v,n,k,s1,s2=z[:,6:].T;g=20*(pr.beta[0]+features@pr.beta[1:]);losses=[-R.marginal_loglik(base_mu+gg,v,n,k,s1,s2,obs).mean() for gg in (g,20*c)];loss=sum(losses);assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if step%250==0:curve.append([step,*[float(a.detach()) for a in losses]])
    torch.save(pr.state_dict(),out/'operator.pt');(out/'simple.json').write_text(json.dumps({'constant_db':20*float(c.detach())})+'\n');(out/'freeze.json').write_text(json.dumps(dict(operator_sha256=sha(out/'operator.pt'),simple_sha256=sha(out/'simple.json'),mean_only_affine_beta=pr.beta.detach().tolist(),curve=curve,tasks=42,source=[0,2],seed=261026,steps=1500,sampling='64 AP-position pairs per task per step, identical for scalar & affine',scope='Seven coefficients on original pooled tokens, not V1 checkpoint. Coefficients not selected on B1.'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase');p.add_argument('data',type=Path);p.add_argument('src',type=Path);p.add_argument('out',type=Path,nargs='?');p.add_argument('f',type=int,nargs='?');p.add_argument('seed',type=int,nargs='?');a=p.parse_args()
    if a.phase=='train':train(a.data,a.src)
    elif a.phase=='score':
        with patch.object(R,'OperatorNet',Affine):paired.run(a.data,a.src,a.out,a.f,a.seed)
    else: raise ValueError(a.phase)
