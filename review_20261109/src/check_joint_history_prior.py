"""Bounded integration of correlated historical evidence into original ReFi."""
import hashlib
import json
import socket
import sys
import time
import torch
from fields import History,rbf
from recomposition import Parameters,prepare,prior,condition_field
from posterior import proposal
from recomposition_predict import locate
from joint_history_prior import JointBank,prepare_joint,joint_prior
from neighbor_gpu_check import source_task
from recomposition_trial import ROOT,digest,write,now

OUT=ROOT/'results/joint_history_integration_20261105'
CODE=('joint_history_prior.py','module_evidence.py','check_joint_history_prior.py',
      'recomposition.py','fields.py','posterior.py','observations.py','recomposition_predict.py')


def difference(a,b):
    return max(float((a[k]-b[k]).detach().abs().max()) if a[k].numel() else 0. for k in a)


def dense(model,block,weights):
    geometry=block['geometry'];bank=block['bank'];base=block['base']
    points=torch.cat((block['support']['x'],block['grid']))
    data=geometry.points(points);allx=torch.cat((points,geometry.anchors));projection=bank.projection(allx)
    _,eta,lg,ls=model.values()
    alpha=model.weights(data['z'],block['za'],weights)
    transform=torch.cat([alpha[j,:,None]*torch.cat((torch.diag(1+eta*data['b']),eta*data['w']),1)
                         for j in range(len(bank.histories))],1)
    means=(base['mean']+projection@bank.mean).reshape(-1)
    kernel=rbf(allx,allx,base['amplitude'],base['length'])
    covariance=torch.cat([torch.cat([bank.R[j,k]*kernel+projection[j]@bank.change@projection[k].T
                                   for k in range(len(bank.histories))],1)
                          for j in range(len(bank.histories))],0)
    mean=-75+lg.exp()*(transform@means+75)
    cov=ls.mul(2).exp()*transform@covariance@transform.T
    cov=cov+rbf(points,points,base['drift'],base['length'])+block['intercept']**2
    n=len(block['support']['x'])
    return dict(sm=mean[:n],gm=mean[n:],sk=cov[:n,:n],cross=cov[n:,:n],gv=cov.diag()[n:]),cov


def synthetic():
    base=dict(mean=-90.,amplitude=7.,length=3.,drift=2.)
    a=torch.tensor([[0.,0.],[2.,0.],[0.,3.]])
    histories=[History(a,dict(mean=torch.tensor([-70.,-76.,-88.]),cov=torch.eye(3)*8),base),
               History(a+1,dict(mean=torch.tensor([-90.,-80.,-72.]),cov=torch.eye(3)*12),base)]
    positions=[a,a+1];delta=[1,-1];weights=torch.tensor([.3,.7])
    sx=torch.tensor([[.5,.2],[2.,2.]])
    grid=torch.tensor([[0.,1.],[1.,1.],[2.,1.],[3.,1.]])
    model=Parameters(torch.zeros(9),torch.ones(9))
    with torch.no_grad():model.theta.copy_(torch.linspace(-.2,.2,12))
    R=torch.tensor([[1.,.5],[.5,1.]])
    independent=prepare_joint(histories,positions,delta,sx,grid,base,3.,torch.eye(2),chunk=2)
    legacy=prior(model,prepare(histories,positions,delta,sx,grid,base,3.),weights)
    nested=difference(joint_prior(model,independent,weights),legacy)
    assert nested < 1e-8
    block=prepare_joint(histories,positions,delta,sx,grid,base,3.,R,chunk=2)
    output=joint_prior(model,block,weights);reference,cov=dense(model,block,weights)
    dense_gap=difference(output,reference);assert dense_gap < 1e-8
    assert float(torch.linalg.eigvalsh(cov).min().detach()) > 0
    objective=lambda x:sum(v.square().mean() for v in x.values())
    grads=torch.autograd.grad(objective(output),model.theta)[0]
    refgrads=torch.autograd.grad(objective(reference),model.theta)[0]
    gradient_gap=float((grads-refgrads).abs().max());assert gradient_gap < 1e-7
    perm=prepare_joint(histories[::-1],positions[::-1],delta[::-1],sx,grid,base,3.,R.flip((0,1)))
    permutation=difference(output,joint_prior(model,perm,weights.flip(0)));assert permutation < 1e-8
    for hs,ps,ds,r,w,n in ((histories,positions,delta,R,weights,0),
                         (histories,positions,delta,R,weights,1),
                         ([histories[0],None],positions,delta,torch.eye(2),weights,1),
                         ([None,None],positions,delta,R,weights,1),
                         ([],[],[],torch.empty(0,0),weights[:0],1)):
        b=prepare_joint(hs,ps,ds,sx[:n],grid,base,3.,r)
        p=joint_prior(model,b,w)
        assert all(bool(torch.isfinite(v).all()) for v in p.values())
        if not any(h is not None for h in hs) or (len(hs)==2 and hs[1] is None):
            assert difference(p,prior(model,prepare(hs,ps,ds,sx[:n],grid,base,3.),w)) < 1e-8
    shift=torch.tensor([500.,-700.])
    moved=[History(h.z+shift,dict(mean=h.m,cov=h.c),base) for h in histories]
    translated=prepare_joint(moved,[p+shift for p in positions],delta,sx+shift,grid+shift,base,3.,R)
    translation=difference(output,joint_prior(model,translated,weights));assert translation < 1e-8
    return dict(complete=True,nonzero_legacy_gap=nested,dense_covariance_gap=dense_gap,
                all_twelve_gradient_gap=gradient_gap,source_permutation_gap=permutation,
                translation_gap=translation,empty_single_missing_H_checked=True,
                code={n:digest(ROOT/n) for n in CODE})


def refresh(p,stats,obs,event,ap):
    k=len(p['sm'])
    if not k:return dict(stats=stats,proposal=None)
    seed=int.from_bytes(hashlib.sha256(f'episodic-refi/{event}/{ap}'.encode()).digest()[:4],'little')
    prepared=dict(sm=p['sm'],sk=p['sk'],phi_s=p['sm'].new_zeros((k,0)),**stats)
    numerical,_=proposal(prepared,obs,p['sm'].new_empty(0),seed)
    return dict(stats=stats,proposal=(numerical['samples'],numerical['log_proposal']))


def run():
    started=time.monotonic()
    check=json.loads((OUT/'synthetic.json').read_text())
    assert all(digest(ROOT/n)==h for n,h in check['code'].items())
    write(OUT/'freeze.json',dict(created=now(),code=check['code'],
        protocol='Existing B0F1 newS K5, all APs. Inspect all source joint precisions first; any invalid correlated AP stops that route.',
        rho=.5,threads=2,wall_seconds=600,training_updates=0,
        reference=str(ROOT/'results/neighbor_learning_uji_20261105/source_gate.json'),
        reference_sha256=digest(ROOT/'results/neighbor_learning_uji_20261105/source_gate.json'),
        performance='Only independent-source baseline replay is scored; correlated output is a structural probe, not a selected model.'))
    cpu,task,event,delta,query=source_task()
    model=Parameters(cpu.center,cpu.scale)
    weights=torch.tensor(task['state']['weights']);J=len(weights)
    matrices={'independent':torch.eye(J),'correlated':.5*torch.ones(J,J)+.5*torch.eye(J)}
    banks={name:{} for name in matrices};failures=[]
    with torch.no_grad():
        for ap,c in task['cases'].items():
            for name,R in matrices.items():
                if not any(h is not None for h in c['histories']):banks[name][ap]=None;continue
                try:banks[name][ap]=JointBank(c['histories'],R,task['base'])
                except torch.linalg.LinAlgError as error:
                    failures.append(dict(AP=ap,mode=name,error=str(error)))
    write(OUT/'joint_admissibility.json',dict(APs=len(task['cases']),failures=failures,
        seconds=time.monotonic()-started,complete=True))
    if failures:
        write(OUT/'complete.json',dict(complete=True,created=now(),seconds=time.monotonic()-started,
            status='joint_family_not_admissible_for_all_APs',failed_APs=len({r['AP'] for r in failures}),
            field_integration_completed=False,training_updates=0,localization_evaluations=0))
        print((OUT/'complete.json').read_text(),flush=True);return
    results={}
    with torch.no_grad():
        for name,R in matrices.items():
            means=[];variances=[];esses=[]
            for i,(ap,c) in enumerate(task['cases'].items(),1):
                block=prepare_joint(c['histories'],task['positions'],delta,c['sx'],task['grid'],task['base'],
                    task['state']['sigma']['intercept_std'],R,bank=banks[name][ap],chunk=128)
                p=joint_prior(model,block,weights)
                case=refresh(p,c['stats'],task['obs'],event['id'],ap)
                field,ess=condition_field(p,case,task['obs'])
                means.append(field['mean']);variances.append(field['variance'])
                if ess is not None:esses.append(ess)
                if i%25==0:
                    progress=dict(mode=name,APs=i,seconds=time.monotonic()-started)
                    write(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
            cached=dict(APs=list(task['cases']),grid=task['grid'],mean=torch.stack(means),
                        variance=torch.stack(variances),obs=task['obs'])
            located=locate(cached,query['rss'][:,cached['APs']])
            assert bool(torch.isfinite(located['scores']).all() and torch.isfinite(located['predictions']).all())
            results[name]=dict(minimum_ESS=min(esses),predictions=located['predictions'].tolist(),
                               all_finite=True,seconds=time.monotonic()-started)
            if name=='independent':
                errors=(located['predictions']-torch.tensor(query['xy'])).norm(dim=1)
                old=json.loads((ROOT/'results/neighbor_learning_uji_20261105/source_gate.json').read_text())['metrics']['identity']
                gap=float((errors-torch.tensor(old['errors'])).abs().max())
                results[name]['baseline_error_gap']=gap;assert gap < 1e-6
            write(OUT/(name+'.json'),results[name])
    write(OUT/'complete.json',dict(complete=True,created=now(),seconds=time.monotonic()-started,
        status='field_integration_completed',APs=len(task['cases']),grid_points=len(task['grid']),
        baseline_error_gap=results['independent']['baseline_error_gap'],
        minimum_ESS={n:r['minimum_ESS'] for n,r in results.items()},training_updates=0,
        new_candidate_accuracy_evaluations=0,scope='Fixed-rho field integration only, no performance or learned-relationship claim.'))
    print((OUT/'complete.json').read_text(),flush=True)


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    try:
        if sys.argv[1:]==['--check']:
            OUT.mkdir(exist_ok=False);result=synthetic();write(OUT/'synthetic.json',result)
            print(json.dumps(result),flush=True)
        elif not sys.argv[1:]:run()
        else:raise ValueError(sys.argv[1:])
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),error=repr(error)))
        raise
