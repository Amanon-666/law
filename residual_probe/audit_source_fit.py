"""Post-hoc paired source objective diagnostic; never fits or selects parameters."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import torch
from residual_probe import broad_source as source
from residual_probe.learning import load_case, observation_loss

@torch.no_grad()
def audit(heldout):
    source.verify()
    out=source.OUT/'source_fit_audit'
    out.mkdir(exist_ok=True)
    model_path=source.OUT/'models'/heldout/'model.json'
    model=source.read(model_path)
    evaluation=source.read(source.OUT/'evaluation'/'freeze.json')
    assert source.digest(model_path)==evaluation['input_sha256'][str(model_path)]
    assert model['freeze_sha256']==source.digest(source.OUT/'freeze.json')
    assert model['feature_indices']==source.INDICES
    prepared={}
    for building in model['training_buildings']:
        record=source.read(source.OUT/'prepared'/f'{building}.json')
        assert record['complete']
        prepared.update({r['path']:r['sha256'] for r in record['cases']})
    steps=source.read(source.OUT/'schedule.json')['schedules'][heldout]
    events={e['id']:e for e in source.row_plan()['events']}
    counts=Counter((s['event'],ap) for s in steps for ap in s['APs'])
    assert len(steps)==512 and sum(counts.values())==2048
    assert all(events[e]['building']!=heldout for e,ap in counts)
    theta=torch.tensor(model['theta'],dtype=torch.float64)
    obs=source.read(source.data_module().PARAMETERS)['obs']
    protocol=dict(created_UTC=datetime.now(timezone.utc).isoformat(),heldout=heldout,
        script_sha256=source.digest(__file__),model_sha256=source.digest(model_path),
        schedule_sha256=source.digest(source.OUT/'schedule.json'),
        question='Does final theta reduce observation NLL on the identical realized source-training draws relative to zero?',
        scope='Post-hoc training fit diagnostic, not convergence proof or generalization evidence.',
        aggregation='2048 scheduled AP occurrences equally weighted, including repeated draws; report source-building/K strata.',
        numerical='Existing frozen proposals; one existing bounded refresh on ESS<1000 per arm/case; stop on any unresolved numerical failure.',
        reverse_audit='Zero and fixed final theta use identical realized scheduled source AP occurrences; repeated draws retain their frequency. The original observation_loss includes the S posterior and source query supervision. Held-out building excluded; cache bytes checked against prepared records and final model against evaluation freeze. One fixed H/S-only numerical refresh per arm if needed, never fit/update/select parameters. Ridge .001 matches the training objective.',
        no_training=True,no_checkpoint_selection=True,independent_validation_opened=False)
    with (out/f'{heldout}_protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    totals=[0.,0.]; strata={};minimum=float('inf');refresh_count=0
    with (out/f'{heldout}_cases.jsonl').open('x') as stream:
        for i,((event,ap),weight) in enumerate(sorted(counts.items())):
            path=source.OUT/'cache'/event/f'ap{ap}.npz'
            cache_sha=source.digest(path)
            assert cache_sha==prepared[str(path)],path
            case=load_case(path); values=[];ess_values=[];refreshes=[]
            for arm,t in [('zero',torch.zeros_like(theta)),('final',theta)]:
                current=dict(case)
                try: loss,ess=observation_loss(t,current,obs,source.INDICES)
                except RuntimeError as error:
                    if 'proposal has insufficient ESS:' not in str(error):raise
                    numeric,info=source.proposal(current,obs,t,source.seed_for(f'source-fit-audit/{heldout}/{event}/{ap}/{arm}'))
                    current.update(numeric)
                    loss,ess=observation_loss(t,current,obs,source.INDICES)
                    refreshes.append(dict(arm=arm,proposal=info));refresh_count+=1
                assert torch.isfinite(loss)
                values.append(float(loss));ess_values.append(float(ess))
            minimum=min(minimum,*ess_values)
            meta=events[event];key=f"{meta['building']}|K{meta['K']}"
            group=strata.setdefault(key,dict(occurrences=0,zero_sum=0.,final_sum=0.))
            group['occurrences']+=weight
            for j,arm in enumerate(['zero','final']):
                totals[j]+=weight*values[j];group[arm+'_sum']+=weight*values[j]
            stream.write(json.dumps(dict(event=event,AP=ap,weight=weight,cache_sha256=cache_sha,
                zero_nll=values[0],final_nll=values[1],ESS=ess_values,refreshes=refreshes))+'\n')
            if i%100==0:stream.flush();print(json.dumps(dict(heldout=heldout,cases=i,total=len(counts))),flush=True)
    for group in strata.values():
        for arm in ['zero','final']:group[arm+'_nll']=group.pop(arm+'_sum')/group['occurrences']
        group['delta']=group['final_nll']-group['zero_nll']
    result=dict(**protocol,complete=True,unique_cases=len(counts),occurrences=2048,
        zero_nll=totals[0]/2048,final_nll=totals[1]/2048,
        ridge_final=.001*float(theta.square().sum()),min_ESS=minimum,refreshes=refresh_count,strata=strata)
    result['delta']=result['final_nll']-result['zero_nll']
    result['penalized_delta']=result['delta']+result['ridge_final']
    source.write(out/f'{heldout}_summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='strata'}),flush=True)

if __name__=='__main__':
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('heldout');audit(p.parse_args().heldout)
