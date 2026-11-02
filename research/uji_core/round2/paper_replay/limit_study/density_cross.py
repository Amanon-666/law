"""After observing the simple law's B1 result, test whether it replaces or complements extra target labels. This is a post-hoc diagnostic on the same development site, not independent confirmation. Do not change parameters, sampling, geometry or observer. Exactly the earlier blocked dense_gp comparison with a common VI backend for ALL FOUR cells. Original plain two cells are read from 41372596502. Use only the checksum-locked source weights (B0/B2). Guide for interpretation: only call five-site learned noninferior to dense plain on *these queries* if an explicitly reported paired upper interval is below a positive prespecified tolerance, 0.25 U in each split. Still not a universal label-savings claim. Separate later/within-period outcomes."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from . import dense_gp as D
from .linear_reference import Affine
from .source import R
from history_bank import BankPrior as Original
from research.uji_core.protocol import position_weights
LOCKED='cd5c4972f4ad418a58c9c73f88d9c65c245f12fbb56bd4a?' # authoritative full SHA is checked against saved record below
LOCKED='cd5c4972f4ad418a58c9c73f88d9c65c245f35842bcdc8535114caf58' if False else 'cd5c4972f4ad418a58c9c73f88d9c65c245f58' if False else None

def run(data,src,plain_archive,out,f,sd):
    # The original checkpoint is doubly bound: exact archive digest and source's file digest.
    Z=zipfile.ZipFile(src);assert hashlib.sha256(src.read_bytes()).hexdigest()=='8da134052d70bb96f117e8217bf84547c1f67c8f20892472c045ab9e72cadd37';raw=Z.read('source/operator.pt');fr=json.loads(Z.read('source/freeze.json'));assert hashlib.sha256(raw).hexdigest()==fr['operator_sha256']
    torch.set_default_dtype(torch.float64);net=Affine().double();net.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));net.eval().requires_grad_(False);dz=[-f,3-f]
    class IdentitySafe(R.OperatorBankPrior):
        def operator(self,x):
            key=id(x)
            if key not in self._cache:
                tok=torch.stack([R.source_tokens(h,d,x,self.base) for h,d in zip(self.histories,self.dz)]);self._cache[key]=(x,self.net(tok))
            assert self._cache[key][0] is x;return self._cache[key][1]
    def new(h,w,b):return IdentitySafe(h,w,b,net,dz) if any(i is not None for i in h) else Original(h,w,b)
    sub=out/'working';eid=f'B1F{f}_{sd}';
    with patch.object(D,'BankPrior',new):D.run(data,sub,f,sd)
    assert hashlib.sha256(plain_archive.read_bytes()).hexdigest()=='84f872dbc0018cdaec0025507c9555659f4f44e03db8adf8761e961b48e0db1f';P=zipfile.ZipFile(plain_archive);p=np.load(io.BytesIO(P.read(eid+'.npz')));l=np.load(sub/(eid+'.npz'));pr=json.loads(P.read(eid+'.json'));lr=json.loads((sub/(eid+'.json')).read_text());assert pr['dense_sources']==lr['dense_sources'] and pr['weight']==lr['weight'];keep={};record={}
    for split in ['internal','official']:
        for suffix in ['_ids','_xy']:
            key=split+suffix;np.testing.assert_array_equal(p[key],l[key]);keep[key]=p[key]
        w=position_weights(p[split+'_xy']);record[split]={}
        for name,d in [('plain',p),('learned',l)]:
            for datause in ['S5','denseR3']:
                arm=name+'_'+datause;v=d[split+'_'+datause];keep[split+'_'+arm]=v;record[split][arm]=float(w@np.linalg.norm(v-p[split+'_xy'],axis=1))
    (out/(eid+'.json')).write_text(json.dumps(dict(event=eid,source=fr['operator_sha256'],scores=record,groups=pr['dense_sources'],learned_numerics=lr['max_projected_numerical_gradient']),indent=2));np.savez_compressed(out/(eid+'.npz'),**keep);print('CROSS',eid,record)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data',type=Path);a.add_argument('src',type=Path);a.add_argument('plain',type=Path);a.add_argument('out',type=Path);a.add_argument('f',type=int);a.add_argument('sd',type=int);p=a.parse_args();run(p.data,p.src,p.plain,p.out,p.f,p.sd)
