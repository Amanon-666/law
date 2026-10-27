"""Fixed-panel K curves from the completed, frozen development study.

Missing methods/K values remain gaps. No winner selection, smoothing,
monotonic fitting, or complete-case substitution is performed.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'results/fusion_study_20261028'
K=[1,3,5,10,20]
BUILDINGS=['UJI_B0','UJI_B2','UAH2016-b20','Pisa2019-b60']
GROUPS={
    'fusion': ['global_identity','global_affine','local_identity','local_affine','WKNN_HS','GP_S'],
    'references': ['global_identity','legacy_global_identity','TPM_pool','SCMT_pool','WKNN_HS','WKNN_S'],
}
LABELS={'global_identity':'Global / identity','global_affine':'Global / affine',
    'local_identity':'Local / identity','local_affine':'Local / affine','WKNN_HS':'WKNN(H+S)',
    'GP_S':'GP(S)','legacy_global_identity':'Legacy adaptation','TPM_pool':'TPM (multi-H adaptation)',
    'SCMT_pool':'SCM-T (multi-H adaptation)','WKNN_S':'WKNN(S)'}
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#333333','#8C6D00']
MARKERS=['o','s','^','D','x','v']


def unit(building):
    return 'UJI coordinate units' if building.startswith('UJI') else 'm'


def save(fig,name):
    for extension in ('png','pdf','svg'):
        fig.savefig(ROOT/f'{name}.{extension}',dpi=220)
    plt.close(fig)


def curves(summary):
    rows={(r['building'],r['scenario'],r['method'],r['K']):r for r in summary['secondary']
          if r['floor'] is None and r['stratum']=='all_queries'}
    for group,methods in GROUPS.items():
        fig,axes=plt.subplots(2,4,figsize=(15,8),layout='constrained')
        for bi,building in enumerate(BUILDINGS):
            for si,scenario in enumerate(('all','far')):
                ax=axes[si,bi];missing=0
                for mi,method in enumerate(methods):
                    cells=[rows[(building,scenario,method,k)] for k in K]
                    mean=np.array([r['mean'] if r['mean'] is not None else np.nan for r in cells])
                    low=np.array([r['pointwise95'][0] if r['pointwise95'] is not None else np.nan for r in cells])
                    high=np.array([r['pointwise95'][1] if r['pointwise95'] is not None else np.nan for r in cells])
                    missing+=int(np.isnan(mean).sum())
                    ax.plot(K,mean,color=COLORS[mi],marker=MARKERS[mi],ls='-' if mi<4 else '--',
                            lw=1.2,ms=4,label=LABELS[method])
                    ax.fill_between(K,low,high,color=COLORS[mi],alpha=.10)
                ax.set(title=f'{building}: {scenario} H',xlabel='Labeled support positions K',
                       ylabel=f'Mean distance error ({unit(building)})',xticks=K)
                ax.grid(alpha=.2);ax.set_ylim(bottom=0)
                if missing:ax.text(.02,.02,f'Missing method/K values: {missing}',transform=ax.transAxes,fontsize=8)
            # Common vertical scale for the same building, not across unlike units.
            upper=max(axes[0,bi].get_ylim()[1],axes[1,bi].get_ylim()[1])
            for ax in axes[:,bi]:ax.set_ylim(0,upper)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=9)
        fig.supxlabel('128 paired draws; equal-floor means. Shading: exploratory pointwise 95% bootstrap CI. '
                      'Missing outcomes are gaps; no complete-case substitution.',fontsize=9)
        save(fig,f'k_curves_{group}')


def contrasts(summary):
    rows={(r['building'],r['method'],r.get('reference'),r['K']):r for r in summary['primary_contrasts']
          if r['type']=='reference'}
    methods=GROUPS['fusion'][:4]
    fig,axes=plt.subplots(2,4,figsize=(15,8),layout='constrained')
    for bi,building in enumerate(BUILDINGS):
        for ri,reference in enumerate(('WKNN_HS','GP_S')):
            ax=axes[ri,bi];missing=0;ax.axhline(0,color='#555555',lw=.8)
            for mi,method in enumerate(methods):
                cells=[rows[(building,method,reference,k)]['interval'] for k in K]
                mean=np.array([r['mean'] if r else np.nan for r in cells])
                low=np.array([r['simultaneous95'][0] if r else np.nan for r in cells])
                high=np.array([r['simultaneous95'][1] if r else np.nan for r in cells])
                missing+=int(np.isnan(mean).sum())
                ax.plot(K,mean,color=COLORS[mi],marker=MARKERS[mi],lw=1.2,ms=4,label=LABELS[method])
                ax.vlines(K,low,high,color=COLORS[mi],alpha=.55,lw=.8)
            ax.set(title=f'{building} vs {LABELS[reference]}',xlabel='Labeled support positions K',
                   ylabel=f'Error difference ({unit(building)})',xticks=K)
            ax.grid(alpha=.2)
            if missing:ax.text(.02,.02,f'Missing/undefined contrasts: {missing}',transform=ax.transAxes,fontsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=4,fontsize=9)
    fig.supxlabel('Negative favors candidate. 95% simultaneous intervals over the prespecified 224 contrasts; '
                  'fixed development corpora, not unseen-building uncertainty.',fontsize=9)
    save(fig,'paired_reference_differences')


if __name__=='__main__':
    summary=json.loads((ROOT/'summary.json').read_text())
    assert summary['complete']
    with plt.rc_context({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'}):
        curves(summary);contrasts(summary)
