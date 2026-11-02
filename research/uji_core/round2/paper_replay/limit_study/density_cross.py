"""Post-hoc data-efficiency diagnostic, not independent confirmation.
Run the unchanged seven coefficients at 5 and 42–70 calibration sites. Both budgets use R=3 and one common VI solver. Match every query row and added training location to the previous dense GP experiment. A 0.25-U noninferiority tolerance for sparse learned vs dense plain is fixed before scoring the new cells; require the paired upper uncertainty bound below this tolerance in BOTH regimes, conditional on these two floors. Extra labels are never a K5 prediction.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from history_bank import BankPrior
from research.uji_core.protocol import position_weights
from . import dense_gp
from .linear_reference import Affine
from .source import R

CHECKPOINT = 'cd5c4972f4ad418a58c9c73f88d9c65c245f24ecf5c35842bcdc8535114caf58'
AFFINE_ZIP = '8da134052d70bb96f117e8217bf84547c1f67c8f20892472c045ab9e72cadd37'
DENSE_ZIP = '84f872dbc0018cdaec0025507c9555659f4f44e03db8adf8761e961b48e0db1f'

def verified_archive(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Unexpected archive: {path}')
    return zipfile.ZipFile(path)

class CachedPrior(R.OperatorBankPrior):
    def operator(self, x):
        # Retain the tensor itself: allocator reuse across folds must not produce a stale field.
        key = id(x)
        if key not in self._cache:
            tokens = torch.stack([R.source_tokens(h, dz, x, self.base)
                                  for h, dz in zip(self.histories, self.dz)])
            self._cache[key] = (x, self.net(tokens))
        assert self._cache[key][0] is x
        return self._cache[key][1]

def run(data, source, plain_archive, out, floor, seed):
    torch.set_default_dtype(torch.float64)
    with verified_archive(source, AFFINE_ZIP) as archive:
        state = archive.read('source/operator.pt')
        manifest = json.loads(archive.read('source/freeze.json'))
    assert hashlib.sha256(state).hexdigest() == CHECKPOINT == manifest['operator_sha256']
    net = Affine().double()
    net.load_state_dict(torch.load(io.BytesIO(state), weights_only=True))
    net.eval().requires_grad_(False)
    gaps = [-floor, 3-floor]
    def prior(histories, weights, base):
        if any(h is not None for h in histories):
            return CachedPrior(histories, weights, base, net, gaps)
        return BankPrior([], [], base)
    work = out/'working'
    event = f'B1F{floor}_{seed}'
    with patch.object(dense_gp, 'BankPrior', prior):
        dense_gp.run(data, work, floor, seed)
    with verified_archive(plain_archive, DENSE_ZIP) as archive:
        old = np.load(io.BytesIO(archive.read(event+'.npz')))
        old_meta = json.loads(archive.read(event+'.json'))
    learned = np.load(work/(event+'.npz'))
    new_meta = json.loads((work/(event+'.json')).read_text())
    assert old_meta['dense_sources'] == new_meta['dense_sources']
    assert old_meta['weight'] == new_meta['weight']
    keep, scores = {}, {}
    for split in ['internal', 'official']:
        for suffix in ['_ids', '_xy']:
            key = split + suffix
            np.testing.assert_array_equal(old[key], learned[key]); keep[key] = old[key]
        truth = old[split+'_xy']; sample_weight = position_weights(truth); scores[split] = {}
        for model, results in [('plain',old), ('learned', learned)]:
            for budget in ['S5','denseR3']:
                arm = model + '_' + budget; pred = results[split+'_'+budget]
                keep[split+'_'+arm] = pred
                scores[split][arm] = float(sample_weight @ np.linalg.norm(pred-truth, axis=1))
    report = dict(event=event, source=CHECKPOINT, scores=scores, groups=old_meta['dense_sources'],
                  learned_max_gradient=new_meta['max_projected_numerical_gradient'])
    (out/(event+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/(event+'.npz'), **keep)
    print(event, scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['data','source','plain','out']: parser.add_argument(arg,type=Path)
    parser.add_argument('floor',type=int,choices=(1,2)); parser.add_argument('seed',type=int,choices=(0,1,2))
    args = parser.parse_args(); run(args.data,args.source,args.plain,args.out,args.floor,args.seed)
