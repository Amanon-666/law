"""GP -> expected observed Wi-Fi scan (for a future like-for-like WKNN). No parameter or localization fit.
python -m research.eta_field_v2.r1_observable_reference /path/to/a/variance-field-archive.zip
"""
import io, json, sys, zipfile, hashlib
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad_vec

def map_moment(mu,var,alpha,beta,tau,missing=-110.):
    if tau<=0 or not 0<alpha<=1 or not np.isfinite(mu).all() or not np.isfinite(var).all() or np.any(var<0):raise ValueError('invalid detector/latent posterior')
    den=np.sqrt(tau*tau+var);z=(mu-beta)/den;p=alpha*ndtr(z)
    return p,missing+(mu-missing)*p+alpha*var/den*np.exp(-.5*z*z)/np.sqrt(2*np.pi)

def check(path):
    with zipfile.ZipFile(path) as zz:
        n=next(n for n in zz.namelist() if n.endswith('_field.npz'));raw=zz.read(n);a=np.load(io.BytesIO(raw),allow_pickle=False);mu=a['mu'].ravel();v=a['variance'].ravel();ix=np.random.default_rng(261102).choice(len(v),256,replace=False);mu=mu[ix];v=v[ix]
    alpha=.8672161172161172;beta=-85.39309326431558;tau=4.466755164467143;p,proto=map_moment(mu,v,alpha,beta,tau)
    def integrand(s):
        f=mu+np.sqrt(v)*s;d=alpha*ndtr((f-beta)/tau);return np.concatenate((d,-110+(f+110)*d))*np.exp(-.5*s*s)/np.sqrt(2*np.pi)
    ref,_=quad_vec(integrand,-14.,14.,epsabs=1e-11,epsrel=1e-12);p0,z0=np.split(ref,2);ae=np.abs(p-p0).max();be=np.abs(proto-z0).max();assert ae<1e-10 and be<1e-8
    return dict(field_sha256=hashlib.sha256(raw).hexdigest(),n_cells=len(v),probability_error=float(ae),observable_mean_error_db=float(be),scope='Identity check, not a localization or source-tuning result.')
if __name__=='__main__':print(json.dumps(check(sys.argv[1]),indent=2))
