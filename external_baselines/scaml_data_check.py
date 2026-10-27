"""Read-only interface audit on frozen development rows; no GP fitting/scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from residual_probe import prepare
from public_corpus import k_curve_episodes as episodes
from external_baselines.scaml_adapter import encode_rss


def main(output):
    manifest = episodes.OUT/'manifest.json'
    rows=[]
    for meta in episodes.load_manifest()['events']:
        task=episodes.build_task(meta)
        assert task['S']['scan_ids'].tolist()==meta['support_scan_ids']
        assert task['Q']['scan_ids'].tolist()==meta['query_scan_ids']
        assert task['ap_indices'].tolist()==meta['AP_indices']
        hxy=np.concatenate([h['xy'] for h in task['H']])
        # Subtracting large absolute map coordinates leaves floating-point roundoff.
        np.testing.assert_allclose(hxy.mean(0),0,atol=1e-6)
        assert len(task['S']['xy'])==meta['K']
        inputs=[encode_rss(h['rss']).numpy() for h in task['H']]+[encode_rss(task['S']['rss']).numpy()]
        assert all(np.isfinite(x).all() for x in inputs)
        assert all(x.shape[1]==len(meta['AP_indices']) for x in inputs)
        rows.append(dict(id=meta['id'],building=meta['building'],K=meta['K'],
            H_scans=[len(h['rss']) for h in task['H']],APs=len(meta['AP_indices']),
            H_S_encoded_min=min(float(x.min()) for x in inputs),
            H_S_encoded_max=max(float(x.max()) for x in inputs),
            H_center_max_abs=float(np.abs(hxy.mean(0)).max())))
    result=dict(scope='Data interface only; no fitting, no Query performance scores.',
        fit_inputs='H RSS/xy and current S RSS/xy only; prediction accepts Query RSS only.',
        normalization='RSS fixed (replace 100 with -110, then +110)/110; coordinates use retained-H origin from old manifest. Source/target outcome transforms see H/S labels only.',
        geometry='No extra map geometry or Q-derived domain; native direct coordinate output, not forced through LAW grid/readout.',
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),events=rows,
        all_H_S_in_unit_interval=all(r['H_S_encoded_min']>=0 and r['H_S_encoded_max']<=1 for r in rows))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(events=len(rows),all_H_S_in_unit_interval=result['all_H_S_in_unit_interval'],first=rows[0])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
