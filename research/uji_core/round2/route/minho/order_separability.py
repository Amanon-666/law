"""Single-scan nearest-neighbour ordering-risk diagnostic, Minho OPEN visits only.

Run: python order_separability.py UM_DSI_DB_v1.0.0_lite.zip output.json
Four predefined early/later pairs only; two reserved late pairs never decoded.
Only one earlier scan per site is used as template and risk-predictor input.
Later scans are evaluation labels, never model/predictor features.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import check_order as ref

N_SEEDS = 10
N_BOOT = 400
BOOT_SEED = 20261102
SCORES = ('contrast_per_m', 'rss_contrast', 'ap_jaccard',
          'ap_scarcity', 'geometry', 'uniform')


def auc(score, bad):
    """Expected AUC over future wrong/correct scans, equal-weight positions."""
    s = np.asarray(score, dtype=float)
    p = np.asarray(bad, dtype=float)
    weights = p[:, None] * (1. - p[None, :])
    den = weights.sum()
    if den < 1e-12:
        return 0.5
    win = (s[:, None] > s[None, :]).astype(float)
    win += 0.5 * (s[:, None] == s[None, :])
    return float((weights * win).sum() / den)


def one_pair(z, early_day, late_day, receipts):
    e, emap, _ = ref.visit(z, early_day, receipts)
    l, lmap, _ = ref.visit(z, late_day, receipts)
    pts = sorted(set(emap) & set(lmap))
    xy = np.asarray(pts, dtype=np.int64)
    sq = np.sum((xy[:, None] - xy[None, :])**2, axis=2)
    np.fill_diagonal(sq, np.iinfo(np.int64).max)
    foil = sq.argmin(1)
    dist = np.sqrt(sq[np.arange(len(pts)), foil]) / 1e6
    all_seeds = []
    for seed in range(N_SEEDS):
        indices = ref.sample_rows(early_day, seed, pts, emap)
        # Earlier repeated RSS is never an input; only the single R1 scan is.
        axis = sorted({a for i in indices for a in e[i]})
        template = ref.matrix([e[i] for i in indices], axis)
        queries = ref.matrix(l, axis)
        gap = np.sqrt(np.mean((template - template[foil])**2, axis=1))
        obs = template > ref.MISS
        and_count = np.count_nonzero(obs & obs[foil], axis=1)
        or_count = np.count_nonzero(obs | obs[foil], axis=1)
        risk = {
            'contrast_per_m': -(gap / np.maximum(dist, 1e-9)),
            'rss_contrast': -gap,
            'ap_jaccard': and_count / np.maximum(or_count, 1),
            'ap_scarcity': -np.count_nonzero(obs, axis=1).astype(float),
            'geometry': -dist,
            'uniform': np.zeros(len(pts))
        }
        bad = []
        ties = []
        for i, p in enumerate(pts):
            sign, _ = ref.distances(queries[lmap[p]], template[i], template[foil[i]])
            bad.append(float(np.mean(sign < 0) + 0.5 * np.mean(sign == 0)))
            ties.append(float(np.mean(sign == 0)))
        all_seeds.append(dict(risk={k:v.tolist() for k,v in risk.items()},
                              bad=bad, ties=ties))
    return dict(early=early_day, late=late_day, n_sites=len(pts),
                n_query_rows=sum(len(lmap[p]) for p in pts),
                foil_median_m=float(np.median(dist)), seeds=all_seeds)


def score_pair(pair, site_indices=None):
    seeds = pair['seeds']
    n = pair['n_sites']
    idx = np.arange(n) if site_indices is None else site_indices
    return {s:float(np.mean([auc(np.asarray(z['risk'][s])[idx],
                                  np.asarray(z['bad'])[idx]) for z in seeds]))
            for s in SCORES}


def evaluate(pairs):
    per_pair = []
    for p in pairs:
        scores = score_pair(p)
        per_pair.append(dict(early=p['early'], late=p['late'], sites=p['n_sites'],
                             query_rows=p['n_query_rows'],
                             median_foil_m=p['foil_median_m'],
                             error_rate=float(np.mean([np.mean(z['bad'])
                                                        for z in p['seeds']])),
                             risk_auc=scores))
    macro = {s:float(np.mean([row['risk_auc'][s] for row in per_pair]))
             for s in SCORES}
    rng = np.random.default_rng(BOOT_SEED)
    diffs = {s:[] for s in SCORES if s != 'contrast_per_m'}
    for _ in range(N_BOOT):
        boot = [score_pair(p, rng.integers(p['n_sites'], size=p['n_sites']))
                for p in pairs]
        avg = {s:float(np.mean([z[s] for z in boot])) for s in SCORES}
        for s in diffs:
            diffs[s].append(avg['contrast_per_m'] - avg[s])
    ci = {s:[float(np.quantile(v,0.025)),float(np.quantile(v,0.975))]
          for s,v in diffs.items()}
    return per_pair, macro, ci


def run(archive, target):
    with archive.open('rb') as f:
        digest = hashlib.file_digest(f, 'md5').hexdigest()
    if digest != '4c15884cb7edcd2c602ec5d567ddd8a8':
        raise ValueError('Not the published Minho lite v1.0.0 file')
    receipts = {}
    with ZipFile(archive) as z:
        pairs = [one_pair(z, early, late, receipts)
                 for early,late in ref.VISITS]
    expected = {ref.PREFIX+day+'/'+name for pair in ref.VISITS
                for day in pair for name in ref.SENSORS}
    if set(receipts) != expected:
        raise ValueError('Unexpected archive access (possibly reserved data)')
    per_pair, macro, ci = evaluate(pairs)
    summary = dict(
        method='One-scan contrast-per-metre ranking; no fitted threshold or weights',
        source='UM DSI lite DOI 10.5281/zenodo.6646008',
        md5=digest,
        holdout='2019-12-11/2020-01-15 and 2020-02-19/2021-04-23 excluded',
        evaluation='Earlier common-location R1 prototypes and fixed nearest foil for all methods; '
                   'later queries only as labels; survey/site equal weight; R1-only earlier AP axis; '
                   'missing=-110 dBm. AUC=0.5 chance. Survey-site block bootstrap 400 replicates; '
                   'conditional on four known dates and spatially dependent candidate sites.',
        scores={'contrast_per_m':'negative RSS RMS separation / spatial metres',
                'rss_contrast':'negative RSS RMS separation',
                'ap_jaccard':'detected AP Jaccard similarity',
                'ap_scarcity':'negative true-anchor AP detection count',
                'geometry':'negative nearest-foil distance',
                'uniform':'constant risk'},
        visits=per_pair, macro_auc=macro,
        macro_contrast_per_m_minus_controls_ci95=ci,
        macro_contrast_per_m_minus_controls={s:macro['contrast_per_m']-macro[s]
                                             for s in ci},
        reserved_payloads_read=0, read_manifest=receipts)
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    print('MINHO_RESULT_JSON='+json.dumps({k:summary[k] for k in
        ('visits','macro_auc','macro_contrast_per_m_minus_controls',
         'macro_contrast_per_m_minus_controls_ci95')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    run(args.archive,args.output)
