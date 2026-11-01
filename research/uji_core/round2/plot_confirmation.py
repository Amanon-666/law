"""Plot the pre-frozen model on the 40 fresh support draws, with paired deltas."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def draw(data, target):
    df=pd.read_csv(data)
    with plt.rc_context({'font.size':9,'svg.hashsalt':'uji-refi-round2','svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,3,figsize=(10,3.25),sharey=True,layout='constrained')
        for b,ax in enumerate(axes):
            ax.axhline(0,color='#516070',lw=.9)
            for regime,label,color in (('official','Later queries','#00756C'),('phone','Held-out phone','#B15E2F')):
                v=df[(df.building==b)&(df['mode']==regime)].sort_values('K')
                ax.fill_between(v.K,v.gap_lo,v.gap_hi,color=color,alpha=.12,linewidth=0)
                ax.plot(v.K,v.gap,'o-',color=color,ms=4,label=label)
            ax.set(xticks=[1,3,5,10,20],xlabel='Calibration locations (K)',title=f'Building {b}')
            ax.grid(axis='y',alpha=.16)
        axes[0].set_ylabel('Model error minus WKNN error\n(UJI coordinate units)')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncols=2,frameon=False)
        fig.savefig(target,metadata={'Date':None})
        plt.close(fig)
        target.write_text('\n'.join(s.rstrip() for s in target.read_text().splitlines())+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();draw(a.data,a.out)
