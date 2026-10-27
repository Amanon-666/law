"""Draw fixed-Q K curves and paired learning increments, including failures.

Thin lines show the three actual split draws, not confidence intervals.
Unfinished runs are not drawn. Terminal numerical failures stay as gaps;
an aggregate point requires all three predetermined split draws.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS = ('WKNN_S', 'WKNN_HS', 'GP_S', 'H_base', 'H_source', 'H_support')
LABELS = ('WKNN: S', 'WKNN: H + S', 'GP: S', 'History base', 'Source learned', 'S-only feature fit')
COLORS = ('#777777', '#D55E00', '#CC79A7', '#0072B2', '#009E73', '#E69F00')
MARKERS = ('x', 's', 'v', 'o', '^', 'D')
EXPORT_VECTORS = False


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    manifest = read(root / 'manifest.json')
    terminal_path = root/'numerical_continuation/terminal_failures.json'
    terminal = {r['event']:r for r in read(terminal_path)} if terminal_path.exists() else {}
    events, missing, pending = [], [], []
    for meta in manifest['events']:
        p = root / meta['id'] / 'completed.json'
        if not p.exists():
            missing.append(meta['id'])
            if meta['id'] not in terminal:
                pending.append(meta['id'])
                continue
            available = {method:read(root/meta['id']/f'{method}.json') for method in METHODS
                         if (root/meta['id']/f'{method}.json').exists()}
            event = dict(id=meta['id'],building=meta['building'],K=meta['K'],seed=meta['seed'],
                         unit=meta['coordinate_unit'],query_scan_ids=meta['query_scan_ids'],
                         outside_grid=meta['query_outside_grid_positions'],results=available,
                         status='numerical_failure',failure=terminal[meta['id']])
        else:
            event = read(p)
            assert set(event['results']) == set(METHODS)
        assert event['query_scan_ids'] == meta['query_scan_ids']
        events.append(event)
    groups = defaultdict(list)
    for event in events:
        groups[event['building'], event['K']].append(event)
    rows = []
    for (building, k), selected in sorted(groups.items()):
        selected.sort(key=lambda e:e['seed'])
        row = dict(building=building, K=k, unit=selected[0]['unit'], splits=len(selected), methods={})
        for method in METHODS:
            valid = [e for e in selected if method in e['results']]
            errors = np.concatenate([e['results'][method]['errors'] for e in valid]) if valid else np.array([])
            full = len(valid) == len(manifest['seeds'])
            row['methods'][method] = dict(available_splits=len(valid),
                mde=float(errors.mean()) if full else None,
                median=float(np.median(errors)) if full else None,
                p90=float(np.quantile(errors, .9)) if full else None,
                split_mde=[e['results'].get(method,{}).get('mde') for e in selected])
        row['learning_gain'] = {
            reference: [e['results'][reference]['mde']-e['results']['H_source']['mde']
                        if reference in e['results'] and 'H_source' in e['results'] else None for e in selected]
            for reference in ('H_base', 'H_support')}
        row['query_outside_grid_fraction'] = sum(e['outside_grid'] for e in selected)/(12*len(selected))
        rows.append(row)
    result = dict(complete=not missing, finished=not pending,
                  completed_events=len(manifest['events'])-len(missing),
                  resolved_events=len(events), expected_events=len(manifest['events']),
                  missing=missing,pending=pending,numerical_failures=list(terminal.values()),groups=rows,events=events,
                  manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
                  interpretation='Three split draws per building; Q fixed across K within a draw; S nested; source model trained at K5. Positive learning gain means source learned has smaller error. No independent-building confidence interval or significance claim. A mean point requires all 3 draws; numerical failures remain gaps, never available-case averages.',
                  source_scope='Development only, fixed B0 base/obs; UJI distances and public metres are separate.',
                  GP_S='S-only signal labels with shared H-derived AP/domain metadata; ordinary GP lacks the random intercept of H_base.',
                  coverage='Domain is H union current S; K curves include coverage expansion. Learning contrasts are paired at the same K and domain.')
    result['original_numerical_failures'] = {p.name:read(p) for p in root.glob('*_failure.json')}
    continuation = root/'numerical_continuation/freeze.json'
    if continuation.exists():
        result['numerical_continuation'] = dict(freeze=read(continuation),
            field_decisions=[json.loads(line) for p in sorted((root/'numerical_continuation').glob('*_fields.jsonl'))
                             for line in p.read_text().splitlines() if line.strip()])
    (root/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result, manifest


def axes_style(ax, building, unit, ylabel):
    ax.set_title(building.replace('-b', ' / building '), fontsize=11, loc='left')
    ax.set_xlabel('Labeled support locations, K')
    ax.set_ylabel(ylabel + (' (original units)' if building.startswith('UJI') else ' (m)'))
    ax.set_xticks([1,3,5,10,20])
    ax.grid(axis='y', alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)


def save(fig, root, name):
    fig.savefig(root/f'{name}.png', dpi=220, facecolor='white')
    if EXPORT_VECTORS:
        fig.savefig(root/f'{name}.pdf', facecolor='white')
        fig.savefig(root/f'{name}.svg', facecolor='white')
    plt.close(fig)


def draw(root, result, manifest):
    if not result['finished']:
        raise RuntimeError(f"Still pending: {len(result['pending'])} events; summary saved, no final figure emitted")
    buildings = ['UJI_B0', 'UJI_B2', 'UAH2016-b20', 'Pisa2019-b60']
    ks, seeds = manifest['K'], manifest['seeds']
    lookup = {(e['building'],e['seed'],e['K']):e for e in result['events']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(2,2,figsize=(11,7.7))
    for ax, building in zip(axes.flat, buildings):
        for method, label, color, marker in zip(METHODS,LABELS,COLORS,MARKERS):
            values = np.array([[lookup[building,s,k]['results'].get(method,{}).get('mde',np.nan)
                                for k in ks] for s in seeds])
            for line in values:
                ax.plot(ks,line,color=color,linewidth=.65,alpha=.24)
            means = [v.mean() if np.isfinite(v).all() else np.nan for v in values.T]
            ax.plot(ks,means,color=color,marker=marker,markersize=4.5,
                    linewidth=1.8,label=label)
        axes_style(ax,building,lookup[building,seeds[0],ks[0]]['unit'],'Mean location error')
        ax.set_ylim(bottom=0)
        for k in ks:
            if any(lookup[building,s,k].get('status')=='numerical_failure' for s in seeds):
                ax.axvline(k,color='#aaaaaa',linestyle=':',linewidth=1)
                ax.text(k,.98,'numerical gap',transform=ax.get_xaxis_transform(),
                        ha='right',va='top',rotation=90,color='#555555',fontsize=8)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False)
    fig.suptitle('Label budget curves with fixed queries and nested support',fontsize=14,y=.98)
    fig.text(.5,.012,'Bold: all 3 draws required; thin: available individual draws. Numerical failures stay as gaps.\nDevelopment data; no confidence intervals. GP shares H-derived metadata.',
             ha='center',va='bottom',fontsize=8)
    fig.subplots_adjust(left=.09,right=.98,top=.91,bottom=.19,hspace=.37,wspace=.28)
    save(fig,root,'k_curve_mde')

    fig, axes = plt.subplots(2,2,figsize=(11,7.4))
    for ax, building in zip(axes.flat,buildings):
        for reference,color,marker,label in [('H_base','#0072B2','o','History base minus source learned'),
                                              ('H_support','#E69F00','s','S-only fit minus source learned')]:
            values = np.array([[lookup[building,s,k]['results'].get(reference,{}).get('mde',np.nan)-
                                lookup[building,s,k]['results'].get('H_source',{}).get('mde',np.nan)
                                for k in ks] for s in seeds])
            for line in values:
                ax.plot(ks,line,color=color,linewidth=.75,alpha=.28)
            means = [v.mean() if np.isfinite(v).all() else np.nan for v in values.T]
            ax.plot(ks,means,color=color,marker=marker,linewidth=1.9,label=label)
        ax.axhline(0,color='#555555',linestyle='--',linewidth=.8)
        ax.axvline(5,color='#aaaaaa',linestyle=':',linewidth=.7)
        axes_style(ax,building,lookup[building,seeds[0],ks[0]]['unit'],'Source-learning gain')
        if any(lookup[building,s,k].get('status')=='numerical_failure' for s in seeds for k in ks):
            gaps = [str(k) for k in ks if any(lookup[building,s,k].get('status')=='numerical_failure' for s in seeds)]
            ax.text(.98,.98,'Numerical gap at K = '+', '.join(gaps),transform=ax.transAxes,
                    ha='right',va='top',color='#555555',fontsize=8)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.048),ncol=2,frameon=False)
    fig.suptitle('Does source learning improve use of the same history?',fontsize=14,y=.98)
    fig.text(.5,.012,'Positive = lower error after source learning. Same H, S, APs and domain; no averaging of incomplete pairs.\nThin: individual draws; dotted K=5: original training budget.',
             ha='center',va='bottom',fontsize=8)
    fig.subplots_adjust(left=.10,right=.98,top=.91,bottom=.17,hspace=.37,wspace=.30)
    save(fig,root,'k_curve_learning_gain')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--summary-only',action='store_true')
    parser.add_argument('--export-vectors',action='store_true')
    args=parser.parse_args()
    EXPORT_VECTORS = args.export_vectors
    result,manifest=summarize(args.root)
    if not args.summary_only:
        draw(args.root,result,manifest)
    print(json.dumps(dict(complete=result['complete'],finished=result['finished'],
                         events=result['completed_events'],pending=len(result['pending']))))
