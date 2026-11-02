"""Score fixed model arms only; never use target Q to set a weight, coefficient or candidate grid."""
import argparse, json, hashlib
from pathlib import Path
import torch,numpy as np
from data import group_scans
from run import tensor,ap_stats
from history_bank import fit_two_source_weight,BankPrior
from multi_history import single_history
from source_weight_mixture import field_from_prior
from inference import observation_log_prob
from transfer import Prior,Transfer,rbf
from research.uji_core.protocol import position_weights
from ..prior_bridge import load
from ..replay import hashids
from .source import ROOT,R,sha

class Head(torch.nn.Module):
    def __init__(self,net=None,c=0.):super().__init__();self.net=net;self.c=c
    def forward(self,z):
        g=self.net(z)[0] if self.net is not None else z.new_full((z.shape[1],),self.c)
        return g,torch.zeros_like(g)

def run(data,src,out,f,seed):
    out.mkdir(parents=True,exist_ok=True);torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    fr=json.loads((src/'freeze.json').read_text()); assert fr['operator_sha256']==sha(src/'operator.pt') and fr['simple_sha256']==sha(src/'simple.json')
    net=R.OperatorNet().double();net.load_state_dict(torch.load(src/'operator.pt',weights_only=True));net.eval();c=json.loads((src/'simple.json').read_text())['constant_db'];mean=Head(net);scalar=Head(c=c)
    pkg=ROOT/'research/uji_core/round2/paper_replay';events={e['key']:e for e in json.loads((pkg/'results/replay/replay.json').read_text())['manifest']};eid=f'B1F{f}_{seed}';manifest=events[eid+'_internal']
    par=json.loads((ROOT/'research/uji_core/field_parameters.json').read_text())['0'];base,obs=par['base'],par['obs'];tr=load(data,'trainingData.csv');t=tr[f];hs=[tr[0],tr[3]];orig=np.vstack([h['positions'] for h in hs]).mean(0);sidx=[dict(zip(t['ids'],range(len(t['ids']))))[i] for i in manifest['support_1based']]
    s=group_scans(t['rss'][sidx],t['xy'][sidx]-orig);sx=tensor(s['positions']);hdata=[group_scans(h['rss'],h['xy']-orig) for h in hs];aa=np.flatnonzero(np.logical_or.reduce([(z['rss']!=100).any(0) for z in [*hs,s]]));assert len(aa)==manifest['aps']
    pts=np.vstack([h['positions'] for h in hdata]+[s['positions']]);ax=[np.linspace(v.min(),v.max(),int(np.ceil(np.ptp(v)/2))+1) for v in pts.T];grid=tensor(np.stack(np.meshgrid(*ax),-1).reshape(-1,2));dz=[-f,3-f]
    banks=[[single_history({'scans':h},int(a),base,obs,steps=640) for h in hdata] for a in aa];stats=[ap_stats(s,a) for a in aa];allst={k:torch.stack([d[k] for d in stats]) for k in stats[0]}
    def getw(use_net):
        M=[];C=[];E=[]
        with torch.no_grad():
            for bank in banks:
                active=any(h is not None for h in bank);op=R.OperatorBankPrior(bank,[.5,.5],base,net if use_net else None,dz)
                g,sd=op.operator(sx) if (use_net and active) else (sx.new_zeros(5),sx.new_zeros(5));m=[];cc=[]
                for h in bank:
                    p=Prior(Transfer('identity'),h,dict(base,drift=0.));a,ft=p.features(sx);m.append(a+g);cc.append(p.covariance(sx,sx,ft,ft))
                M.append(torch.stack(m));cs=torch.stack(cc);C.append(cs if active else torch.zeros_like(cs));E.append(rbf(sx,sx,base['drift'] if active else base['amplitude'],base['length'])+sd[:,None]*sd[None]*rbf(sx,sx,1.,base['length']))
        return fit_two_source_weight(torch.stack(M),torch.stack(C),torch.stack(E),allst,obs)[0]
    w=getw(False);wc=getw(True); arms={'plain':(None,w),'scalar':(scalar,w),'coupled_full':(net,wc),'decoupled_full':(net,w),'mean_only':(mean,w)}
    v=load(data,'validationData.csv')[f];ii=np.flatnonzero(~np.isin(t['group'],t['group'][sidx]));oi=np.flatnonzero(~(v['xy'][:,None]==t['positions'][np.unique(t['group'][sidx])][None]).all(-1).any(-1));sets=[('internal',t,ii),('official',v,oi)]
    for name,z,i in sets: assert hashids(z['ids'][i])==events[eid+'_'+name]['query_ids_sha256']
    Q=np.vstack([z['rss'][i][:,aa] for name,z,i in sets]);store={};metrics={};ess={}
    for name, (n,weight) in arms.items():
        l=np.zeros((len(Q),len(grid)));b=np.zeros(len(grid));esss=[]
        with torch.no_grad():
            for j,(hsj,sc) in enumerate(zip(banks,stats)):
                p=R.OperatorBankPrior(hsj,weight,base,n,dz) if any(h is not None for h in hsj) else BankPrior([],[],base)
                field=field_from_prior(p,obs,sx,sc,grid,261026+10000*seed+int(aa[j]),defensive_fraction=.1);esss.append(float(field['effective_samples']));mu,var=field['mean'],field['variance'];assert torch.isfinite(mu).all() and torch.isfinite(var).all() and (var>=0).all()
                missing=observation_log_prob(tensor(100.),mu,var,obs).numpy();b+=missing;seen=np.flatnonzero(Q[:,j]!=100)
                if len(seen):
                    vals,inv=np.unique(Q[seen,j],return_inverse=True);tab=observation_log_prob(tensor(vals)[:,None],mu,var,obs).numpy()-missing;l[seen]+=tab[inv.ravel()]
        pred=(tensor(l+b).softmax(1)@grid).numpy()+orig;ess[name]=min(esss);assert ess[name]>=1000 and np.isfinite(pred).all()
        off=0;metrics[name]={}
        for sp,z,ix in sets:
            stop=off+len(ix);x=z['xy'][ix];pr=pred[off:stop];ww=position_weights(x);er=np.linalg.norm(pr-x,axis=1);ord_=np.argsort(er);metrics[name][sp]={'mde':float(ww@er),'p90':float(er[ord_][min(len(er)-1,np.searchsorted(ww[ord_].cumsum(),.9))])};store[sp+'_ids']=z['ids'][ix];store[sp+'_xy']=x;store[sp+'_'+name]=pr;off=stop
    record=dict(event=eid,source=fr['operator_sha256'],w=w,coupled_w=wc,metrics=metrics,min_ess=ess);(out/(eid+'.json')).write_text(json.dumps(record,indent=2)+'\n');np.savez_compressed(out/(eid+'.npz'),**store);print('PAIRED',json.dumps(record))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('source',type=Path);p.add_argument('out',type=Path);p.add_argument('f',type=int);p.add_argument('seed',type=int);a=p.parse_args();run(a.data,a.source,a.out,a.f,a.seed)
