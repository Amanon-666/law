"""A=(1-e)I+eP, P an H-only convex average at history anchors. No dense grid x grid matrix.
All operations are on the historical posterior, not only its inducing-point interpolant.
"""
import torch
from transfer import rbf, Prior, Transfer, History
from history_bank import BankPrior

class Source:
    def __init__(self,h,base,eta,propagate,point=0.):
        self.h=h;self.b=base;self.e=float(eta) if h is not None else 0.;self.t=self.e if propagate else 0.;self.point=point
        self.p=Prior(Transfer('identity'),h,dict(base,drift=0.))
        if self.e:
            z=h.z;self.mz,self.fz=self.p.features(z);self.kzz=self.p.covariance(z,z,self.fz,self.fz)
    def features(self,x):
        m,f=self.p.features(x);w=v=None
        if self.e:
            w=torch.softmax(-torch.cdist(x,self.h.z).square()/(2*(self.b['length']*.5)**2),1)
            m=(1-self.e)*m+self.e*(w@self.mz)
            if self.t: v=self.p.covariance(x,self.h.z,f,self.fz)
        return m+self.point*(m-self.b['mean']), (f,w,v)
    def cov(self,x,y,s,q):
        fx,wx,vx=s;fy,wy,vy=q;k=self.p.covariance(x,y,fx,fy);t=self.t
        if not t:return k
        return (1-t)**2*k + t*(1-t)*(vx@wy.T+wx@vy.T) + t*t*((wx@self.kzz)@wy.T)
    def diag(self,x,s):
        fx,w,v=s;k=self.p.diagonal(x,fx);t=self.t
        if not t:return k
        return (1-t)**2*k+2*t*(1-t)*(w*v).sum(1)+t*t*((w@self.kzz)*w).sum(1)

class Law:
    def __init__(self,hs,w,b,eta=0.,propagate=False,point=0.,hb=None):
        self.base=b;self.w=w;self.h=[Source(h,hb or b,eta,propagate,point) for h in hs]
        self.empty=not any(h is not None for h in hs)
    def features(self,x):
        if self.empty:return x.new_full((len(x),),self.base['mean']),[]
        xs=[p.features(x) for p in self.h];return sum(w*m for w,(m,_) in zip(self.w,xs)),[a for _,a in xs]
    def covariance(self,x,y,a,b):
        if self.empty:return rbf(x,y,self.base['amplitude'],self.base['length'])
        return rbf(x,y,self.base['drift'],self.base['length'])+sum(w*w*p.cov(x,y,s,q) for w,p,s,q in zip(self.w,self.h,a,b))
    def diagonal(self,x,a):
        if self.empty:return x.new_full((len(x),),self.base['amplitude']**2)
        return self.base['drift']**2+sum(w*w*p.diag(x,s) for w,p,s in zip(self.w,self.h,a))

def selftest():
    torch.set_default_dtype(torch.float64);torch.manual_seed(271);z=torch.randn(9,2)*7;x=torch.randn(11,2)*7;b=dict(mean=-95.,length=11.,amplitude=14.,drift=6.);k=rbf(z,z,14.,11.);h=History(z,dict(mean=-80+torch.randn(9)*4,cov=.35*k),b);p=Prior(Transfer('identity'),h,dict(b,drift=0.));u=torch.cat((x,z));m,f=p.features(u);c=p.covariance(u,u,f,f);n=len(x);w=(-torch.cdist(x,z).square()/(2*5.5**2)).softmax(1);worst=0.
    for e in (0.,-.5,.25,.5):
        a=torch.cat(((1-e)*torch.eye(n),e*w),1);s=Source(h,b,e,True);mm,ff=s.features(x);kk=s.cov(x,x,ff,ff);ref=a@c@a.T
        for l,r in [(mm,a@m),(kk,ref),(s.diag(x,ff),ref.diagonal()),(a.sum(1),torch.ones(n))]:worst=max(worst,float((l-r).abs().max()));torch.testing.assert_close(l,r,rtol=1e-9,atol=1e-8)
        assert torch.linalg.eigvalsh(kk).min()>-1e-8
        mo=Source(h,b,e,False);mv,mf=mo.features(x);torch.testing.assert_close(mv,mm);torch.testing.assert_close(mo.cov(x,x,mf,mf),c[:n,:n]);
    for hh in ([h,h],[None,h],[None,None]):
        bp=BankPrior(hh,[.4,.6],b) if any(i is not None for i in hh) else BankPrior([],[],b);op=Law(hh,[.4,.6],b);m1,s1=bp.features(x);m2,s2=op.features(x);torch.testing.assert_close(m1,m2);torch.testing.assert_close(bp.covariance(x,x,s1,s1),op.covariance(x,x,s2,s2))
    return worst
if __name__=='__main__': print('dense_identity_psd_max_error',selftest())
