"""Bounded algebra/gradient checks for the new manuscript implementation."""
import json
import torch
from fields import History,Prior,Transfer,InterceptBankPrior,rbf
from recomposition import Parameters,Source,prepare,prior,field,ap_scores
from posterior import SupportIntegral
from observations import _chol

def checks():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(113)
    base=dict(mean=-90.,amplitude=7.,length=3.,drift=2.)
    obs=dict(alpha=.85,beta=-85.,detect_scale=4.,noise_std=5.)
    a=torch.tensor([[0.,0.],[2.,0.],[0.,3.]])
    h1=History(a,dict(mean=torch.tensor([-70.,-76.,-88.]),cov=torch.eye(3)*8),base)
    h2=History(a+1,dict(mean=torch.tensor([-90.,-80.,-72.]),cov=torch.eye(3)*12),base)
    sx=torch.tensor([[.5,.2],[2.,2.]])
    grid=torch.tensor([[0.,1.],[1.,1.],[2.,1.],[3.,1.]])
    w=torch.tensor([.3,.7])
    b=prepare([h1,h2],[a,a+1],[1,-1],sx,grid,base,3.)
    model=Parameters(torch.zeros(9),torch.ones(9))
    ref=InterceptBankPrior([h1,h2],w,base,3.)
    sm,sf=ref.features(sx);gm,gf=ref.features(grid);out=prior(model,b,w)
    expected=dict(sm=sm,gm=gm,sk=ref.covariance(sx,sx,sf,sf),
                  cross=ref.covariance(grid,sx,gf,sf),gv=ref.diagonal(grid,gf))
    identity=max(float((out[k]-v).abs().max().detach()) for k,v in expected.items())
    assert identity<1e-9
    with torch.no_grad(): model.theta.copy_(torch.linspace(-.2,.2,12))
    out=prior(model,b,w);phi,eta,lg,ls=model.values()
    x=torch.cat((sx,grid));anchors=torch.unique(torch.cat((a,a+1)),dim=0)
    z=torch.cat((x,anchors));k=rbf(x,anchors,1.,base["length"])
    transform=torch.cat((torch.diag(1-eta+eta/(1+k.sum(1))),eta*k/(1+k.sum(1))[:,None]),1)
    alpha=model.weights(torch.cat((b["zs"],b["zg"]),1),b["za"],w)
    mean=x.new_zeros(len(x));cov=rbf(x,x,base["drift"],base["length"])+9.
    for j,h in enumerate((h1,h2)):
        p=Prior(Transfer("identity"),h,dict(base,drift=0.));m,f=p.features(z)
        mean+=alpha[j]*(transform@m)
        c=p.covariance(z,z,f,f);d=alpha[j,:,None]*transform
        cov+=ls.mul(2).exp()*d@c@d.T
    mean=-75+lg.exp()*(mean+75)
    expected=dict(sm=mean[:2],gm=mean[2:],sk=cov[:2,:2],cross=cov[2:,:2],gv=cov.diag()[2:])
    dense=max(float((out[k]-v).abs().max().detach()) for k,v in expected.items());assert dense<1e-8
    # Source permutation and coordinate translation must preserve the field.
    perm=prepare([h2,h1],[a+1,a],[-1,1],sx,grid,base,3.)
    op=prior(model,perm,w.flip(0))
    permutation=max(float((op[k]-out[k]).abs().max().detach()) for k in out);assert permutation<1e-8
    shift=torch.tensor([500.,-700.])
    hs=[History(h.z+shift,dict(mean=h.m,cov=h.c),base) for h in (h1,h2)]
    translated=prior(model,prepare(hs,[a+shift,a+1+shift],[1,-1],sx+shift,grid+shift,base,3.),w)
    translation=max(float((translated[k]-out[k]).abs().max().detach()) for k in out);assert translation<1e-8
    stats=dict(n=torch.ones(2),k=torch.ones(2),s1=torch.tensor([-76.,-81.]),s2=torch.tensor([76.**2,81.**2]))
    pi=SupportIntegral(out["sm"].detach(),out["sk"].detach(),stats,obs,power=13,seed=113)
    proposal=(pi.samples,torch.distributions.MultivariateNormal(pi.rss["mean"],scale_tril=_chol(pi.rss["cov"])).log_prob(pi.samples))
    case=dict(blocks=b,stats=stats,rss=torch.tensor([-75.,100.]),proposal=proposal)
    coef=torch.arange(8,dtype=torch.float64).reshape(2,4)/10
    objective=lambda:(ap_scores(model,case,w,obs)[0]*coef).sum()
    val=objective();grad,=torch.autograd.grad(val,model.theta);finite=[]
    for j in range(12):
        original=model.theta[j].item();step=1e-5
        with torch.no_grad():model.theta[j]=original+step
        plus=float(objective().detach())
        with torch.no_grad():model.theta[j]=original-step
        minus=float(objective().detach())
        with torch.no_grad():model.theta[j]=original
        finite.append((plus-minus)/(2*step))
    error=float((grad-torch.tensor(finite)).abs().max());assert error<2e-5
    empty=prepare([h1,h2],[a,a+1],[1,-1],sx[:0],grid,base,3.)
    ef,_=field(model,dict(blocks=empty),w,obs);assert torch.isfinite(ef["mean"]).all()
    no=prepare([],[],[],sx,grid,base,3.)
    nofield=prior(model,no,w[:0]);assert not nofield["gm"].requires_grad
    # One missing source is a real base GP; all missing collapses once to base.
    for hs in ([h1,None],[None,None]):
        b0=prepare(hs,[a,a+1],[1,-1],sx[:1],grid,base,3.)
        q=prior(model,b0,w);assert torch.linalg.eigvalsh(q["sk"]).min()>0
    return dict(identity_error=identity,dense_transform_error=dense,permutation_error=permutation,
        translation_error=translation,posterior_gradient_error=error,gradient=grad.tolist(),
        empty_support=True,single_support=True,missing_history=True,no_history=True)
if __name__=="__main__":print(json.dumps(checks()))

