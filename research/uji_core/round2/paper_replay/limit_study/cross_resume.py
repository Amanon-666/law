"""Retry the same numerical objective only on optimizer failure. No query outcome is used to retry, accept or select. Never turn a failure flag into success. Accept only a second successful SciPy solve no worse than the previous endpoint (round-off tolerance); dense_gp also enforces the unmodified projected-gradient threshold. Keep diagnostics."""
import sys,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import inference
from . import density_cross

def main(argv):
    data,src,old,out=map(Path,argv[:4]);f,seed=map(int,argv[4:]); records=[]; real=inference.minimize
    def solver(fun,x0,*args,**kwargs):
        result=real(fun,x0,*args,**kwargs)
        if result.success:return result
        initial=float(result.fun); opt=dict(kwargs.get('options',{})); vals=[]
        for maxls in (120,400):
            cfg=dict(kwargs,options=dict(opt,maxls=maxls,maxiter=max(opt.get('maxiter',800),1200)))
            result=real(fun,np.asarray(result.x).copy(),*args,**cfg)
            vals.append({'maxls':maxls,'objective':float(result.fun),'success':bool(result.success),'message':str(result.message)})
            if result.success:break
        records.append(dict(initial_objective=initial,trials=vals)); print('NUMERICAL_RESTART',records[-1],flush=True)
        if not result.success or not np.isfinite(result.fun) or result.fun > initial+1e-9*max(abs(initial),1.):raise RuntimeError('Original optimizer gate not passed on continuation')
        return result
    with patch.object(inference,'minimize',solver): density_cross.run(data,src,old,out,f,seed)
    (out/f'B1F{f}_{seed}.solver-receipt').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__': main(sys.argv[1:])
