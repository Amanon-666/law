"""Secondary all-budget and multiplicity sensitivity. No model fits, same original labels.
Python + NumPy: python family_check.py <unpacked primary ten-draw archive> family.json
Within each fixed regime use the same spatial and support draws for both learners and all K.
The prior first-two-draw interim had been inspected; this is NOT a new preregistered primary test.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def analyze(root:Path,nboot:int=16000,pilot:bool=False):
    nd=2 if pilot else 10;ks=(1,3,5,10,20);arms=('seven','nonlinear');rg=np.random.default_rng(121103)
    out={'draws':nd,'bootstraps':nboot,'method':'Secondary max absolute standardized centered bootstrap contrast (97.5% within each of two regimes; Bonferroni across regimes). Fixed four floors; common 8-U spatial block and original support-order resamples. Marginal intervals of the within-regime macro average across five pre-existing budgets are also included. No selection, no new-building inference.'}
    for regime in ('official','phone'):
        mat=np.zeros((nboot,10));ctr=np.zeros(10)
        for f in range(4):
            pos0=None;history=[]
            for d in range(nd):
                fld=root/["rows","draw1","remaining"][min(d,2)]/(('gridpilot' if d==0 else f'draw{d}')+f'-{regime}-F{f}');q=[]
                for k in ks:
                    with np.load(fld/f'K{k}.npz',allow_pickle=False) as z:
                        true=z['truth'];pos,g,c=np.unique(true,axis=0,return_inverse=True,return_counts=True);g=g.ravel()
                        if pos0 is not None:np.testing.assert_array_equal(pos0,pos)
                        pos0=pos;basic=np.linalg.norm(z['gp']-true,axis=1)
                        for a in arms:q.append(np.bincount(g,np.linalg.norm(z[a]-true,axis=1)-basic)/c)
                history.append(np.stack(q,axis=1))
            es=np.stack(history,axis=1);ctr+=es.mean((0,1))/4;_,g=np.unique(np.floor(pos0/8).astype(int),axis=0,return_inverse=True);g=g.ravel();n=g.max()+1
            for i in range(0,nboot,512):
                m=min(512,nboot-i);c=rg.multinomial(n,[1/n]*n,size=m)[:,g];c=c/c.sum(1)[:,None];d=rg.multinomial(nd,[1/nd]*nd,size=m)/nd;tmp=(c@es.reshape(len(pos0),-1)).reshape(m,nd,-1);mat[i:i+m]+=(tmp*d[:,:,None]).sum(1)/4
        se=mat.std(0,ddof=1);assert np.all(se>0);h=float(np.quantile(np.max(np.abs((mat-ctr)/se),axis=1),.975));lo=ctr-h*se;hi=ctr+h*se;record={};whole={}
        for idx,(a,k) in enumerate((a,k) for k in ks for a in arms):record[f'{a}-gp/K{k}']={'delta_U':float(ctr[idx]),'simultaneous_approx_interval_U':[float(lo[idx]),float(hi[idx])],'individual_95_U':np.quantile(mat[:,idx],[.025,.975]).tolist()}
        for i,a in enumerate(arms):whole[a]={'delta_average_over_K_U':float(ctr[i::2].mean()),'conditional_95_U':np.quantile(mat[:,i::2].mean(1),[.025,.975]).tolist()}
        out[regime]={'factor':h,'comparisons':record,'equal_K_summary':whole,'negative_band_count':int(np.sum(hi<0)),'positive_band_count':int(np.sum(lo>0))}
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);p.add_argument('--pilot',action='store_true');a=p.parse_args();a.out.write_text(json.dumps(analyze(a.root,pilot=a.pilot),indent=2,allow_nan=False)+'\n')
