"""Plot only the completed frozen ScaML comparison; retain missing references."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plot_k_curve as style


def draw(root, vectors=False, original=None):
    result=json.loads((root/'summary.json').read_text())
    if not result['finished']:raise RuntimeError('Predetermined run is not finished')
    if original is not None:
        prior=json.loads((original/'summary.json').read_text())
        if not prior['finished']:raise RuntimeError('Original run is not finished')
        prior_lookup={(e['building'],e['seed'],e['K']):e for e in prior['events']}
    style.EXPORT_VECTORS=vectors
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    buildings=['UJI_B0','UJI_B2','UAH2016-b20','Pisa2019-b60']
    ks=sorted({e['K'] for e in result['events']})
    seeds=sorted({e['seed'] for e in result['events']})
    lookup={(e['building'],e['seed'],e['K']):e for e in result['events']}
    definitions=[('ScaML','#009E73','^','ScaML-GP: RSS to coordinates'),
                 ('H_base','#0072B2','o','LAW history base'),
                 ('WKNN_HS','#D55E00','s','WKNN: H + S'),
                 ('GP_S','#777777','D','GP on S (H metadata)')]
    if original is not None:
        definitions[0]=('ScaML','#009E73','^','ScaML: training-objective selection')
        definitions.append(('ScaML_original','#CC79A7','x','ScaML: official selection'))
    fig,axes=plt.subplots(2,2,figsize=(11,7.7))
    for ax,b in zip(axes.flat,buildings):
        for method,color,marker,label in definitions:
            source=prior_lookup if method=='ScaML_original' else lookup
            key='ScaML' if method=='ScaML_original' else method
            values=np.array([[source[b,s,k]['results'].get(key,{}).get('mde',np.nan) for k in ks] for s in seeds])
            for row in values:ax.plot(ks,row,color=color,linewidth=.65,alpha=.23)
            avg=[v.mean() if np.isfinite(v).all() else np.nan for v in values.T]
            ax.plot(ks,avg,color=color,marker=marker,linewidth=1.9,label=label)
        style.axes_style(ax,b,lookup[b,seeds[0],ks[0]]['unit'],'Mean location error')
        ax.set_ylim(bottom=0)
        missing=[str(k) for k in ks if any(('ScaML' if m=='ScaML_original' else m) not in (prior_lookup if m=='ScaML_original' else lookup)[b,s,k]['results'] for m,_,_,_ in definitions for s in seeds)]
        if missing:ax.text(.98,.98,'Some methods missing at K = '+', '.join(missing),transform=ax.transAxes,ha='right',va='top',fontsize=8,color='#555555')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False)
    fig.suptitle('Same-information development comparison: fixed-query K curves',fontsize=14,y=.98)
    fig.text(.5,.012,'Bold: all 3 draws required; thin: individual draws. Missing results remain gaps.\nNative model outputs; shared allowed H/S/AP information. Separate coordinate units; no independent-site claim.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.10,right=.98,top=.91,bottom=.23,hspace=.37,wspace=.28)
    style.save(fig,root,'scaml_k_curve_mde')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--export-vectors',action='store_true');p.add_argument('--original',type=Path)
    a=p.parse_args();draw(a.root,a.export_vectors,a.original)
