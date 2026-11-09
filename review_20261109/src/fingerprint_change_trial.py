"""One bounded source-only learning comparison of full-fingerprint geometry."""
import gc
import json
import socket
import time
from collections import Counter
import numpy as np
import torch
from fields import rbf
from uji import load_uji,group_scans,ap_stats
from fingerprint_change import FingerprintChange,expected_rbf,prepare_fingerprint,evaluate_fingerprint
from check_joint_history_prior import refresh
from neighbor_learning_trial import load_original,PREP
from neighbor_gpu_check import source_task
from source_context_replay import FIXED,ROTATED
from recomposition_trial import ROOT,PLAN,CSV,digest,write,now,subset

OUT = ROOT/'results/fingerprint_change_uji_20261105'
CODE = ('fingerprint_change.py','fingerprint_change_trial.py','fields.py','posterior.py',
        'observations.py','recomposition.py','neighbor_learning_trial.py','neighbor_gpu_check.py',
        'conditional_budget_trial.py','episodic_tasks.py','uji.py','check_joint_history_prior.py')


def checks():
    torch.manual_seed(11508)
    x = torch.tensor([[0.,0.],[1.,0.],[2.,0.],[3.,0.]])
    mean = torch.tensor([[-82.,-80.,-78.,-76.],[-70.,-80.,-85.,-79.]])
    covariance = torch.stack((rbf(x,x,2.,2.),rbf(x,x,3.,1.)))
    difference = mean[:,:,None]-mean[:,None,:]
    diagonal = covariance.diagonal(dim1=-2,dim2=-1)
    variance = diagonal[:,:,None]+diagonal[:,None,:]-2*covariance
    length = torch.tensor(8.)
    kernel = expected_rbf(difference,variance,length)
    assert float(torch.linalg.eigvalsh(kernel).min()) > 0
    assert torch.equal(kernel.diag(),torch.ones(4))
    assert torch.allclose(kernel,expected_rbf(difference.flip(0),variance.flip(0),length),atol=1e-14)
    assert torch.allclose(expected_rbf(difference,torch.zeros_like(variance),length),
        (-difference.square().sum(0)/(2*len(mean)*length.square())).exp(),atol=1e-14)
    nodes, weights = np.polynomial.hermite.hermgauss(30)
    nodes = torch.tensor(nodes)*2**.5;weights = torch.tensor(weights)/np.pi**.5
    quadrature = torch.ones_like(kernel)
    for m,v in zip(difference,variance):
        quadrature *= ((-(m[...,None]+v.sqrt()[...,None]*nodes).square()/(2*len(mean)*length.square())).exp()*weights).sum(-1)
    quadrature_gap = float((quadrature-kernel).abs().max());assert quadrature_gap < 1e-12
    flat = torch.zeros_like(difference)
    altered = flat.clone();altered[1] = difference[1]
    other_AP_effect = float((expected_rbf(flat,variance*0,length)-expected_rbf(altered,variance*0,length)).abs().max())
    assert other_AP_effect > .05
    # Full likelihood -> posterior -> shared-kernel gradient, fixed integration.
    sx,grid = x[:2],x[2:]
    spatial_ss = rbf(sx,sx,1.,2.); spatial_gs = rbf(grid,sx,1.,2.)
    geometry = dict(history_APs=[0,1],amplitude=8.,spatial_ss=spatial_ss,spatial_gs=spatial_gs,
        mean_ss=difference[:,:2,:2],mean_gs=difference[:,2:,:2],
        variance_ss=variance[:,:2,:2],variance_gs=variance[:,2:,:2])
    obs = dict(alpha=.85,beta=-85.,detect_scale=4.,noise_std=5.)
    support = group_scans(np.array([[-79,-74],[-81,-76],[-75,-82],[-77,-84]]),
                          np.array([[0.,0.],[0.,0.],[1.,0.],[1.,0.]]))
    cases = {}
    for ap in range(2):
        p = dict(sm=mean[ap,:2],gm=mean[ap,2:],sk=9*spatial_ss+torch.eye(2)*2,
                 cross=9*spatial_gs,gv=torch.full((2,),11.))
        c = refresh(p,ap_stats(support,ap),obs,'fingerprint-check',ap)
        cases[ap] = dict(c,prior=p,amplitude=2.)
    episode = dict(cases=cases,geometry=geometry,grid=grid,support=sx,obs=obs)
    query = group_scans(np.array([[-77,-84],[-74,-78]]),grid.numpy())
    model = FingerprintChange()
    evaluate_fingerprint(model,episode,query,backward=True)
    gradient = model.theta.grad.clone(); numerical = []
    initial = model.theta.detach().clone();epsilon=1e-5
    for j in range(3):
        with torch.no_grad():model.theta.copy_(initial);model.theta[j]+=epsilon
        plus = evaluate_fingerprint(model,episode,query)['nll']
        with torch.no_grad():model.theta.copy_(initial);model.theta[j]-=epsilon
        minus = evaluate_fingerprint(model,episode,query)['nll']
        numerical.append((plus-minus)/(2*epsilon))
    gap = float((gradient-torch.tensor(numerical)).abs().max());assert gap < 1e-7
    return dict(quadrature_gap=quadrature_gap,minimum_kernel_eigenvalue=float(torch.linalg.eigvalsh(kernel).min()),
                other_AP_changes_geometry=other_AP_effect,complete_likelihood_gradient_gap=gap,
                AP_permutation_and_unit_diagonal=True,gradient=gradient.tolist())


def progress(stage,**kwargs):
    row=dict(created=now(),stage=stage,**kwargs)
    write(OUT/'progress.json',row);print(json.dumps(row),flush=True)


def report_parameters(model):
    w,r=model.values()
    return dict(theta=model.theta.detach().cpu().tolist(),weights=w.detach().cpu().tolist(),length_ratio=float(r.detach()))


def run():
    OUT.mkdir(exist_ok=False);started=time.monotonic()
    plan=json.loads(PLAN.read_text());events={e['id']:e for e in plan['events']}
    manifest=json.loads((PREP/'complete.json').read_text());rows={r['id']:r for r in manifest['episodes']}
    batches=json.loads((ROTATED/'queries.json').read_text())['batches'][:24]
    assert Counter(b['event'] for b in batches)=={f'train_{i}':4 for i in range(6)}
    reference=json.loads((ROTATED/'freeze.json').read_text())
    assert digest(CSV)==reference['csv_sha256'] and digest(PLAN)==reference['plan_sha256']
    write(OUT/'freeze.json',dict(created=now(),code={n:digest(ROOT/n) for n in CODE},
        inputs={str(p):digest(p) for p in (PLAN,CSV,PREP/'complete.json',ROTATED/'queries.json',
                    FIXED/'source_initial.json',ROOT/'results/neighbor_learning_uji_20261105/source_gate.json')},
        learning_object='Shared target-change covariance using all historical APs, not per-AP shape',
        initializer='weights .8/.1/.1; fingerprint length=base.amplitude; identity control=1/0/0',
        steps=24,schedule=batches,optimizer='Adam lr=.03; .001 quadratic displacement from initialization',
        original_twelve_parameters='Frozen identity',checkpoint='Final24 only',
        source_gate='Known B0F1 newS K5: final beats identity AND untrained by >=3%',
        development_evaluations=0,wall_seconds=900,threads=2,downloads=0,
        numerical='Existing original-density proposals; ESS<1000 stops; no refresh',
        caveats=['Known source labels and all-B0 nuisance fit','Three source floors, no new independent environments',
                 'No full Bayesian integration over S-fitted weights/intercept','Short budget, no convergence claim']))
    for n in CODE:(OUT/n).write_bytes((ROOT/n).read_bytes())
    write(OUT/'checks.json',checks())
    assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats()
    model=FingerprintChange().cuda();initial={k:v.clone() for k,v in model.state_dict().items()}
    floors=load_uji(CSV);origin=np.array(plan['origin']);tasks={};used={};source_initial={}
    old_initial=json.loads((FIXED/'source_initial.json').read_text())
    for ident in (f'train_{i}' for i in range(6)):
        task,proposals,delta,query=load_original(events[ident],rows[ident],plan,floors,used)
        episode=prepare_fingerprint(task,proposals,'cuda')
        baseline=evaluate_fingerprint(model,episode,query,identity=True)
        old=old_initial[ident]
        gap=max(abs(baseline['mde']-old['mde']),max(abs(a-b) for a,b in zip(baseline['errors'],old['errors'])))
        assert gap < 1e-6
        source_initial[ident]=dict(**baseline,old_error_gap=gap)
        tasks[ident]=episode
        del task,proposals;gc.collect()
        progress('source_prepared',event=ident,seconds=time.monotonic()-started)
    write(OUT/'source_initial.json',source_initial);write(OUT/'history_inputs.json',used)
    # Prepare the held S configuration once, without passing Q into adaptation.
    cpu,gate_task,event,delta,gate_query=source_task()
    from neighbor_stream import prepare_cpu_proposals
    proposals=prepare_cpu_proposals(cpu,gate_task,delta,event['id'])
    gate=prepare_fingerprint(gate_task,proposals,'cuda')
    identity=evaluate_fingerprint(model,gate,gate_query,identity=True)
    old=json.loads((ROOT/'results/neighbor_learning_uji_20261105/source_gate.json').read_text())['metrics']['identity']
    gap=max(abs(x-y) for x,y in zip(identity['errors'],old['errors']));assert gap<1e-6
    untrained=evaluate_fingerprint(model,gate,gate_query)
    write(OUT/'gate_before_training.json',dict(identity=identity,untrained=untrained,old_error_gap=gap))
    del gate_task,proposals,cpu;gc.collect()
    optimizer=torch.optim.Adam(model.parameters(),lr=.03);records=[]
    for b in batches:
        query=subset(floors[events[b['event']]['floor']],b['query_rows'],origin)
        optimizer.zero_grad()
        metric=evaluate_fingerprint(model,tasks[b['event']],query,backward=True)
        penalty=model.penalty();penalty.backward()
        assert model.theta.grad is not None and bool(torch.isfinite(model.theta.grad).all())
        row=dict(step=b['step'],event=b['event'],nll=metric['nll'],mde=metric['mde'],
                 minimum_ESS=metric['minimum_ESS'],gradient=model.theta.grad.detach().cpu().tolist(),
                 **report_parameters(model))
        optimizer.step();records.append(row)
        with (OUT/'train.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=b['step']),OUT/'latest.pt')
        if b['step']%4==0:progress('learning',step=b['step'],seconds=time.monotonic()-started)
    trained=evaluate_fingerprint(model,gate,gate_query)
    gain_identity=1-trained['mde']/identity['mde'];gain_untrained=1-trained['mde']/untrained['mde']
    write(OUT/'source_gate.json',dict(identity=identity,untrained=untrained,trained=trained,
        gain_identity=gain_identity,gain_untrained=gain_untrained,
        passed=gain_identity>=.03 and gain_untrained>=.03,parameters=report_parameters(model)))
    write(OUT/'complete.json',dict(complete=True,created=now(),seconds=time.monotonic()-started,
        steps=len(records),mde=dict(identity=identity['mde'],untrained=untrained['mde'],trained=trained['mde']),
        gain_identity=gain_identity,gain_untrained=gain_untrained,
        passed=gain_identity>=.03 and gain_untrained>=.03,parameters=report_parameters(model),
        minimum_training_ESS=min(r['minimum_ESS'] for r in records),B1_B2_evaluations=0,
        GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        scope='One source-only bounded learning trial; no cross-environment performance claim'))
    print((OUT/'complete.json').read_text(),flush=True)


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    try:run()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(created=now(),error=repr(error)))
        raise
