"""s=0 diagnostic on UJI B1 (PROTOCOL_s0.md). ray-web only."""
import json, socket, sys
from pathlib import Path
import numpy as np, torch
import run_operator_lodo as L, run_operator_v11 as V
from full_ap_study import metrics
from history_bank import BankPrior
from inference import observation_log_prob
from operator_prior import OperatorBankPrior, OperatorNet
from run import ap_stats, tensor
from run_multi_history import write
from source_weight_mixture import field_from_prior
OUT = Path(__file__).resolve().parent / 'results/operator_s0_20261026'
ARMS = ['v1dec', 'v1dec_s0']

class NoS(torch.nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, tokens):
        g, s = self.net(tokens); return g, torch.zeros_like(s)

def event(eid):
    dest = OUT / eid; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'completed.json').exists(): return
    case = L.uji_case(eid); base, obs = V.params(); sup = case['support']; sx = tensor(sup['positions']); aps = case['aps']; hist = case['hist']
    net = OperatorNet().double(); net.load_state_dict(torch.load(V.V1 / 'operator.pt', weights_only=True)); net.eval()
    nets = dict(v1dec=net, v1dec_s0=NoS(net))
    stats = {k: torch.stack([ap_stats(sup, a)[k] for a in aps]) for k in ('n', 'k', 's1', 's2')}
    w, _ = V.fit_w(hist, aps, sx, stats, base, obs)
    fields = {}
    for a in aps:
        hs = hist[a]; fields[a] = {}
        for m in ARMS:
            prior = OperatorBankPrior(hs, w, base, nets[m], case['dz']) if any(h is not None for h in hs) else BankPrior([], [], base)
            with torch.no_grad():
                r = field_from_prior(prior, obs, sx, ap_stats(sup, a), case['grid'], 261026 + 10000 * case['seed'] + a, defensive_fraction=.1)
            fields[a][m] = (r['mean'], r['variance'], r['effective_samples'])
    grid = case['grid']; qrss = case['qrss']; res = []; low = {m: sum(1 for a in aps if fields[a][m][2] is not None and fields[a][m][2] < 1000) for m in ARMS}
    for m in ARMS:
        preds, mass = [], []
        with torch.no_grad():
            for st in range(0, len(qrss), 64):
                scans = tensor(qrss[st:st + 64].astype(float)); sc = scans.new_zeros((len(scans), len(grid)))
                for a in aps:
                    mu, var, _ = fields[a][m]; sc += observation_log_prob(scans[:, a, None], mu[None], var[None], obs)
                p = torch.softmax(sc, 1); preds.append((p @ grid).numpy())
                d = torch.cdist(tensor(case['qxy'][st:st + 64]), grid); mass.append((p * (d <= 10)).sum(1).numpy())
        pred = np.concatenate(preds); mass = np.concatenate(mass)
        for sp in ('internal', 'official'):
            sel = case['split'] == sp; xy = case['qxy'][sel]
            _, grp, cnt = np.unique(np.round(xy, 3), axis=0, return_inverse=True, return_counts=True); grp = grp.ravel()
            res.append(dict(method=m, split=sp, mass10=float(mass[sel].mean()), **metrics(pred[sel], dict(xy=xy, scan_weights=1 / (len(cnt) * cnt[grp])))))
    write(dest / 'completed.json', dict(event=eid, w=w, low_ess=low, results=res))
    print(json.dumps(dict(event=eid, results=[(r['method'], r['split'], round(r['mde'], 3), round(r['mass10'], 3)) for r in res])), flush=True)

if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2); event(sys.argv[1])
