"""One mature non-mean-reverting temporal alternative; no fitted parameters."""
import torch

from inference import infer
from multi_history import project_posterior
from run import ap_stats,tensor
from transfer import History,rbf


def random_walk_covariance(x,t,y,u,base,unit_days):
    variance=base['amplitude']**2+base['drift']**2*torch.minimum(t[:,None],u[None])/unit_days
    return variance*rbf(x,y,1.,base['length'])


class RandomWalkHistory(History):
    def __init__(self,records,ap,base,obs,calendar,steps=640):
        positions=[tensor(r['scans']['positions']) for r in records]
        x=torch.cat(positions)
        times=torch.cat([p.new_full((len(p),),calendar['days'][r['period']-1]) for p,r in zip(positions,records)])
        counts=[ap_stats(r['scans'],ap) for r in records]
        stats={key:torch.cat([c[key] for c in counts]) for key in counts[0]}
        cov=random_walk_covariance(x,times,x,times,base,calendar['unit_days'])
        posterior=infer(x.new_full((len(x),),base['mean']),cov,stats,obs,steps=steps)
        z=torch.unique(x,dim=0);last=times.max().item();target=z.new_full((len(z),),last)
        cross=random_walk_covariance(z,target,x,times,base,calendar['unit_days'])
        target_cov=random_walk_covariance(z,target,z,target,base,calendar['unit_days'])
        projected=project_posterior(cov,cross,posterior,target_cov,base['mean'])
        effective=dict(base,amplitude=(base['amplitude']**2+base['drift']**2*last/calendar['unit_days'])**.5)
        super().__init__(z,projected,effective)
        self.reference_base=effective


def check():
    torch.set_default_dtype(torch.float64)
    base=dict(mean=-100.,amplitude=20.,drift=4.,length=2.)
    x=torch.zeros((1,2));t=torch.tensor([7.]);u=torch.tensor([28.])
    ctt=random_walk_covariance(x,t,x,t,base,7.)
    crt=random_walk_covariance(x,u,x,t,base,7.)
    crr=random_walk_covariance(x,u,x,u,base,7.)
    predicted=base['mean']+(crt/ctt).item()*(-50-base['mean'])
    uncertainty=(crr-crt.square()/ctt).item()
    assert predicted==-50. and abs(uncertainty-48.)<1e-10
    points=torch.tensor([[0.,0.],[1.,0.],[0.,0.]])
    dates=torch.tensor([0.,7.,14.]);k=random_walk_covariance(points,dates,points,dates,base,7.)
    torch.linalg.cholesky(k)
    return dict(noiseless_old_level=-50.,later_conditional_mean=predicted,increment_variance=uncertainty,
                minimum_joint_eigenvalue=torch.linalg.eigvalsh(k).min().item())


if __name__=='__main__':
    import json
    print(json.dumps(check()))
