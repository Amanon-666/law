"""Server-only replay with the OLD LAW history-availability rule in every arm.

Use a fresh --out copied from frozen configuration, without old per-AP caches.
For an AP never detected in the selected H, all families use the same local GP.
The saved post-hoc score substitution is exactly this branch with the same seed,
parameters and S; no new fitting or learned gate is involved.
"""
from pathlib import Path
import json
import socket
import sys

import full_ap_study as study


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('server only')
    if '--out' not in sys.argv:
        raise RuntimeError('Provide a separate --out directory for the common-policy run')
    out=Path(sys.argv[sys.argv.index('--out')+1])
    if out.resolve()==(Path(__file__).resolve().parent/'results/full_ap_v1').resolve():
        raise RuntimeError('Do not mix corrected policy into original outputs')
    if not (out/'availability_policy.json').exists():
        raise RuntimeError('Prepared common-policy provenance is required before replay')
    original_mean=study.load_mean
    original_history=study.MultiHistory
    def mean_at_server(setting):
        adapted=dict(setting,mean_checkpoint=str(Path(__file__).resolve().parent/'results/source_transfer_v4_spatial_control/affine.pt'))
        return original_mean(adapted)
    def available_history(records,ap,base,obs,calendar,steps=640):
        if not any((r['scans']['stats']['k'][:,ap]>0).any() for r in records):
            return None
        return original_history(records,ap,base,obs,calendar,steps=steps)
    study.load_mean=mean_at_server
    study.MultiHistory=available_history
    study.main()
