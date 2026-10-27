"""Post-first-run composition check: average source kernels, not source inputs.

Convex kernel addition is a mature construction. This checks whether applying
a pretrained single-history rule to an averaged history discarded information;
it is not presented as a new algorithmic contribution.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import torch

from bank_calibration import CalibratedBankPrior
from history_bank import BankPrior
from multi_history import GaussianStrengthRelation
from transfer import History, rbf
from run_multi_history import digest, write
import run_bank_calibration_transfer as common


PARENT=common.OUT
OUT=common.ROOT/'results/uji_bank_sourcewise_transfer_v1_20261026'


class SourcewiseCalibratedBankPrior(CalibratedBankPrior):
    def covariance(self, x, y, fx, fy):
        if not self.sources or self.relation.kind=='uniform':
            return super().covariance(x,y,fx,fy)
        base,relation=self.base,self.relation
        original=BankPrior.covariance(self,x,y,fx,fy)
        historical=original-rbf(x,y,base['drift'],base['length'])
        ell2=relation.log_strength_length.exp().square()
        combined=x.new_zeros((len(x),len(y)))
        for weight,source,xf,yf in zip(self.weights,self.sources,fx,fy):
            mx,_=source.features(x);my,_=source.features(y)
            vx=source.diagonal(x,xf);vy=source.diagonal(y,yf)
            cxy=source.covariance(x,y,xf,yf)  # Each source Prior already has drift=0.
            v=vx[:,None]+vy[None]-2*cxy
            if bool((v < -1e-7).any()):raise RuntimeError('Invalid source Gaussian difference variance')
            v=v.clamp_min(0)
            combined+=weight*torch.sqrt(ell2/(ell2+v))*torch.exp(-.5*(mx[:,None]-my[None]).square()/(ell2+v))
        local=relation.logit_local.sigmoid()
        spatial=rbf(x,y,1.,relation.log_length.exp())
        return historical+base['drift']**2*(local*spatial+(1-local)*combined)


@torch.no_grad()
def check():
    base=dict(mean=-95.,amplitude=16.,length=5.,drift=7.)
    z=torch.tensor([[0.,0.],[10.,0.]])
    covariance=torch.zeros((2,2))
    h0=History(z,dict(mean=torch.tensor([-50.,-90.]),cov=covariance),base)
    h1=History(z,dict(mean=torch.tensor([-90.,-50.]),cov=covariance),base)
    same=History(z,dict(mean=torch.tensor([-70.,-70.]),cov=covariance),base)
    left=BankPrior([h0,h1],[.5,.5],base);right=BankPrior([same,same],[.5,.5],base)
    lm,lf=left.features(z);rm,rf=right.features(z)
    torch.testing.assert_close(lm,rm)
    torch.testing.assert_close(left.covariance(z,z,lf,lf),right.covariance(z,z,rf,rf))
    relation=GaussianStrengthRelation('strength_rbf',5.).double()
    relation.logit_local.fill_(-100.)
    mean_field=CalibratedBankPrior([h0,h1],[.5,.5],base,relation)
    per_source=SourcewiseCalibratedBankPrior([h0,h1],[.5,.5],base,relation)
    m,f=per_source.features(z);_,mf=mean_field.features(z)
    torch.testing.assert_close(m,lm)
    torch.testing.assert_close(per_source.diagonal(z,f),left.diagonal(z,lf))
    latent=BankPrior([h0,h1],[.5,.5],dict(base,drift=0.));_,hf=latent.features(z)
    hc=latent.covariance(z,z,hf,hf)
    old=(mean_field.covariance(z,z,mf,mf)-hc)[0,1]/base['drift']**2
    new=(per_source.covariance(z,z,f,f)-hc)[0,1]/base['drift']**2
    if not old>.99 or not new<.14:raise RuntimeError('Source-cancellation counterexample did not hold')
    full=per_source.covariance(z,z,f,f)
    torch.testing.assert_close(full.diag(),per_source.diagonal(z,f))
    if torch.linalg.eigvalsh(full).min()<=0:raise RuntimeError('Nonpositive sourcewise covariance')
    single=SourcewiseCalibratedBankPrior([h0],[1.],base,relation)
    reference=CalibratedBankPrior([h0],[1.],base,relation)
    _,sf=single.features(z);_,qf=reference.features(z)
    torch.testing.assert_close(single.covariance(z,z,sf,sf),reference.covariance(z,z,qf,qf))
    return dict(aggregate_mean_covariance_identical=True,matched_pointwise_variance=True,
        mean_field_strength_correlation=float(old),sourcewise_strength_correlation=float(new),
        single_history_reduction=True,positive_covariance=True,
        interpretation='Retains source distinctions; does not prove improved localization or explain the whole prior result')


def configure():
    common.OUT=OUT
    common.METHODS={k:v for k,v in common.METHODS.items() if k.startswith('strength_')}
    common.CalibratedBankPrior=SourcewiseCalibratedBankPrior


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Server only')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['check','prepare','event','summarize']);p.add_argument('--event')
    args=p.parse_args();configure()
    if args.stage=='check':common.log(stage='sourcewise_checks',**check())
    elif args.stage=='prepare':
        common.prepare()
        freeze=json.loads((OUT/'freeze.json').read_text())
        freeze.update(composition='Weighted sum of per-source expected-strength kernels; never strength kernel of averaged field',
            post_hoc_hypothesis=True,hypothesis_created_unix=time.time(),
            hypothesis='Preserving the pretrained per-source representation improves transfer over mean-field composition; trained-vs-initial remains the separate learning test',
            parent_comparison=str(PARENT),new_trainable_parameters=0,
            source_missing='Unobserved individual source retains its original GP prior; all-H-undetected policy unchanged',
            uniform_control='Identical across compositions; reuse parent uniform_initial/uniform_trained',
            added_rule='Convex kernel addition, not a novelty claim',
            contrast_boundary='This retains differing source patterns but does not create negative one-anchor correction correlations')
        freeze['code_sha256'][str(Path(__file__))]=digest(Path(__file__))
        freeze['inputs'][str(PARENT/'freeze.json')]=digest(PARENT/'freeze.json')
        write(OUT/'freeze.json',freeze);shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
        write(OUT/'composition_checks.json',check())
    else:
        freeze=json.loads((OUT/'freeze.json').read_text())
        if 'composition' not in freeze:raise RuntimeError('Composition freeze incomplete')
        for collection in ['code_sha256','history_sha256','inputs','source_weights']:
            for path,sha in freeze[collection].items():
                if digest(Path(path))!=sha:raise RuntimeError(f'Frozen input changed: {path}')
        if args.stage=='event':common.event(args.event)
        else:common.summarize()
