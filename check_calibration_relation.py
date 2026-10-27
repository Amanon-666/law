"""Check the information controls and their new length-scale derivatives."""
import hashlib
import json

import numpy as np
import torch

from calibration_relation import CalibrationRelation
from data import draw_episode, load_period
from reference_pmc import ROOT, sample_reference, setup, log_proposal
from run import make_case
from support_integral import importance_loss
from transfer import Prior, Transfer


class KnownField:
    """Same center values but opposite local shapes, with no H uncertainty."""
    def sample_mean(self, x):
        return (-75 + 4 * x[:, 0].square())[:, None]

    def shape(self, x):
        center = self.sample_mean(x)
        offsets = x.new_tensor([[.5,0],[-.5,0],[0,.5],[0,-.5]])
        neighbors = self.sample_mean((x[:,None]+offsets).reshape(-1,2)).reshape(len(x),4,1).transpose(1,2)
        return center, neighbors-center[...,None]


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    setting=json.loads((ROOT/'results/change_kernel_v1/freeze.json').read_text())
    base,obs=setting['base'],setting['obs']
    kinds=('uniform','strength_rbf','pointwise','linear','shape')
    fields=KnownField();x=torch.tensor([[-1.,0.],[1.,0.],[0.,0.],[2.,1.]])
    report=dict(role='Information and derivative checks only; no training',
                code_sha256=hashlib.sha256((ROOT/'calibration_relation.py').read_bytes()).hexdigest(), structure={},gradients={})
    for kind in kinds:
        torch.manual_seed(231024)
        model=CalibrationRelation(kind,base['length']).double()
        k=model.kernel(x,x,fields,base)
        torch.testing.assert_close(k,k.T,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(k.diag(),torch.full((len(x),),base['drift']**2),atol=1e-12,rtol=1e-12)
        assert torch.linalg.eigvalsh(k).min()>-1e-9
        # Remove the common spatial part to expose the historical information.
        from transfer import rbf
        relation=2*k/base['drift']**2-rbf(x,x,1.,model.log_length.exp())
        if kind in ('uniform','strength_rbf','pointwise'):
            torch.testing.assert_close(relation[0,1],torch.tensor(1.),atol=1e-12,rtol=1e-12)
        else:
            assert relation[0,1]<1-1e-6
        report['structure'][kind]=dict(same_center_relation=relation[0,1].item(),min_eigenvalue=torch.linalg.eigvalsh(k).min().item())
    path=ROOT.parent/'Law-Continuous-Change-20261023/data/OfficeP2.npz'
    ep=draw_episode(load_period(path,1),load_period(path,2),.25,5,12,np.random.default_rng(231023))
    counts=ep['S']['stats']['k']
    j=int(np.flatnonzero(ep['historical_positive'] & (counts>0).any(0) & (counts==0).any(0))[0])
    case=make_case(ep,j,base,obs,160)
    mean=Transfer('affine').double();mean.load_state_dict(torch.load(setting['mean_checkpoint'],weights_only=True));mean.requires_grad_(False)
    for kind in kinds:
        torch.manual_seed(231024)
        model=CalibrationRelation(kind,base['length']).double()
        with torch.no_grad():
            prior=Prior(mean,case['history'],base,model)
            sm,sf=prior.features(case['sx']);sk=prior.covariance(case['sx'],case['sx'],sf,sf)
            target,proposal,center,root,_=setup(sm,sk,case['stats'],obs)
            samples,_,info=sample_reference(target,proposal,center,root,231024)
            logq=log_proposal(samples,center,root,info)
        named=list(model.named_parameters())
        loss,_=importance_loss(Prior(mean,case['history'],base,model),case,obs,samples,logq)
        gradients=torch.autograd.grad(loss,[p for _,p in named])
        checks=[]
        # New hyperparameters, including the independently learned local scale.
        for (name,p),g in zip(named,gradients):
            if name.startswith('encoder.'):
                continue
            original=p.detach().clone(); values=[]
            with torch.no_grad():
                for sign in (-1,1):
                    p.copy_(original+sign*1e-4)
                    value,_=importance_loss(Prior(mean,case['history'],base,model),case,obs,samples,logq)
                    values.append(value.item())
                p.copy_(original)
            fd=(values[1]-values[0])/2e-4
            torch.testing.assert_close(g,torch.tensor(fd),rtol=2e-4,atol=2e-7)
            checks.append(dict(parameter=name,gradient=g.item(),finite_difference=fd))
        report['gradients'][kind]=checks
        print(json.dumps(dict(kind=kind,checks=checks)),flush=True)
    output=ROOT/'results/calibration_relation_checks_v1.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
