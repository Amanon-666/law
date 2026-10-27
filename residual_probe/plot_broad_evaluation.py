"""Plot completed paired K curves; never fill a numerical gap or pool units."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import plot_k_curve as style


def draw(root, export_vectors=False):
    result=json.loads((root/'summary.json').read_text())
    if not result['finished']:
        raise RuntimeError('The predetermined evaluation has not finished')
    style.EXPORT_VECTORS=export_vectors
    events=result['events']
    buildings=['UJI_B0','UJI_B2','UAH2016-b20','Pisa2019-b60']
    ks=sorted({e['K'] for e in events});seeds=sorted({e['seed'] for e in events})
    lookup={(e['building'],e['seed'],e['K']):e for e in events}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    def values(building,method):
        return np.array([[lookup[building,s,k]['results'].get(method,{}).get('mde',np.nan)
                          for k in ks] for s in seeds])
    def lines(ax,y,color,marker,label):
        for row in y:ax.plot(ks,row,color=color,linewidth=.65,alpha=.23)
        means=[v.mean() if np.isfinite(v).all() else np.nan for v in y.T]
        ax.plot(ks,means,color=color,marker=marker,linewidth=1.9,label=label)
    definitions=[('H_base','#0072B2','o','History base'),
                 ('WKNN_HS','#D55E00','s','WKNN: H + S'),
                 ('H_support','#E69F00','D','Same-feature S fit'),
                 ('new_source','#009E73','^','Broad-task source learner')]
    fig,axes=plt.subplots(2,2,figsize=(11,7.7))
    for ax,b in zip(axes.flat,buildings):
        for m,c,marker,label in definitions:lines(ax,values(b,m),c,marker,label)
        style.axes_style(ax,b,lookup[b,seeds[0],ks[0]]['unit'],'Mean location error')
        ax.set_ylim(bottom=0)
        gaps=[str(k) for k in ks if any('new_source' not in lookup[b,s,k]['results'] for s in seeds)]
        if gaps:ax.text(.98,.98,'Numerical gap at K = '+', '.join(gaps),transform=ax.transAxes,ha='right',va='top',fontsize=8,color='#555555')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
    fig.suptitle('Training-task coverage intervention: fixed-query K curves',fontsize=14,y=.98)
    fig.text(.5,.012,'Bold: all 3 draws required; thin: individual draws. Numerical failures remain gaps.\nSame H, S, APs and domain within K. Development buildings; separate distance units.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.09,right=.98,top=.91,bottom=.19,hspace=.37,wspace=.28)
    style.save(fig,root,'broad_k_curve_mde')
    fig,axes=plt.subplots(2,2,figsize=(11,7.7))
    refs=[('H_base','#0072B2','o','Base minus new learner'),
          ('H_support','#E69F00','s','S fit minus new learner'),
          ('H_source','#777777','^','Old source learner minus new learner')]
    for ax,b in zip(axes.flat,buildings):
        for ref,c,marker,label in refs:lines(ax,values(b,ref)-values(b,'new_source'),c,marker,label)
        ax.axhline(0,color='#555555',linestyle='--',linewidth=.8)
        style.axes_style(ax,b,lookup[b,seeds[0],ks[0]]['unit'],'Paired gain')
        gaps=[str(k) for k in ks if any('new_source' not in lookup[b,s,k]['results'] for s in seeds)]
        if gaps:ax.text(.98,.98,'Numerical gap at K = '+', '.join(gaps),transform=ax.transAxes,ha='right',va='top',fontsize=8,color='#555555')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
    fig.suptitle('Does broader source training improve use of the same history?',fontsize=14,y=.98)
    fig.text(.5,.012,'Positive = lower error for the new learner. All 3 paired draws required for a bold point.\nNo confidence intervals or independent-site claims; no interpolation across missing results.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.10,right=.98,top=.91,bottom=.19,hspace=.37,wspace=.30)
    style.save(fig,root,'broad_k_curve_gain')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--export-vectors',action='store_true');a=p.parse_args()
    draw(a.root,a.export_vectors)
