"""One legal source backward, zero optimizer updates; not a performance run."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
import gc
import json
from pathlib import Path
import socket
import sys
import time
ROOT = Path('/home/panyushuo/projects/panyushuo/ReFi-UJI-Core-20261031')
OVERLAY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OVERLAY))
import numpy as np
import torch
from building_information_tab_trial import digest, now, read_scans, write
from building_information_gp_trial import move_task
from context_influence_consumer import InfluenceHistoryRegressor
from context_influence_system import InfluencePositionSystem
from position_fusion_trial import prepare, CHECKPOINT, PARAMETERS, CATALOG, CSV
from tabicl_history import state_hash
ASSETS = ROOT/'results/advised_source_assets_20261109'
OUT = ROOT/'results/context_influence_backward_20261110'
CID = '8a250dea041113046284025fb9ed22e3cfc57d570fde8e4972f4581697d9c14a'
EVENT = 'B0F0_group0_to_B0F0_group2_draw5_K1_HB0F1-B0F2_fit'
SEED = 11510176
CODE = ('context_influence_objective.py', 'context_influence_consumer.py',
        'context_influence_system.py', 'context_influence_adapter.py',
        'context_influence_features.py', 'position_fusion_gp.py',
        'context_influence_backward_check.py')
EXPECTED = {'manifest.json':'0b9c58c3b661832ca00ba6efcc8c484dd4bdf651bf03a361f067453a7e6157a2',
    'histories.json':'a4e308cf07a13459bc44235ebedd64922ac495c0b4d747a817b16133fa2c6c7d',
    f'states/{CID}.json':'1811761b2a52ed9eeae1fd20bc9099ea2f76773d44070c63085c2912c2e66cf5'}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert socket.gethostname() == 'ray-web' and torch.cuda.is_available()
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    for name, expected in EXPECTED.items():
        assert digest(ASSETS/name) == expected
    entry = read(ASSETS/'manifest.json')['schedule'][0][0]
    event = entry['event']
    assert event['id'] == EVENT and entry['context_id'] == CID
    assert event['role'] == 'train' and entry['spec']['phase'] == 'fit'
    assert event['K'] == 1 and len(event['support_rows']) == 3 and len(event['query_rows']) == 36
    protected = set(read(CATALOG)['protected_rows'])
    all_ids = event['support_rows']+event['query_rows']+[i for ids in event['history_rows'].values() for i in ids]
    assert not protected.intersection(all_ids)
    index = read(ASSETS/'histories.json')
    assert event['history_rows'] == {f:index['history_rows'][f] for f in event['histories']}
    assert index['parameters_sha256'] == digest(PARAMETERS)
    state_record = read(ASSETS/f'states/{CID}.json')
    assert state_record['context_id'] == CID
    source = read(ROOT/'results/tabicl_multitask_full_trial_20261107_checked/complete.json')
    assert digest(CHECKPOINT) == source['checkpoint_sha256']['24']
    OUT.mkdir(exist_ok=False)
    freeze = dict(created=now(), event=event, canonical_context_id=CID, seed=SEED,
        code={n:digest(OVERLAY/n) for n in CODE},
        inputs={str(p):digest(p) for p in (ASSETS/'manifest.json', ASSETS/'histories.json',
            ASSETS/f'states/{CID}.json', CATALOG, PARAMETERS, CHECKPOINT, CSV)},
        scope='One legal source zero-update backward wiring contract; no performance improvement evidence',
        resources=dict(gpu=1,cpu_threads=2,hard_seconds=600,downloads=0),
        objective='Existing projected fixed-half system source MDE; one encoder gradient accumulation',
        AMP=dict(dtype='float16',GradScaler_initial_scale=1024.),
        forbidden=['optimizer construction/step','protected validation labels','target data','new GP fit','retry'])
    write(OUT/'freeze.json',freeze)
    for name in CODE:
        (OUT/name).write_bytes((OVERLAY/name).read_bytes())
    histories,support,bounds,aps,task = prepare(event,'source',None,index,read(PARAMETERS))
    assert len(aps)==170 and sum(len(h['rss']) for h in histories.values())==2085
    assert np.array_equal(aps,state_record['aps'])
    assert np.allclose(task['origin'],state_record['origin'],rtol=0,atol=1e-8)
    move_task(task,torch.device('cuda:0'))
    consumer = InfluenceHistoryRegressor(CHECKPOINT,device='cuda:0')
    recompute = consumer.enable_recompute()
    model = consumer.loaded.model_
    assert model.dropout == 0. and model.quantile_dist.crossing_method == 'sort'
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert torch.count_nonzero(consumer.relation_encoder.output.weight)==0
    before = state_hash(model)
    encoder_before = state_hash(consumer.relation_encoder)
    system = InfluencePositionSystem(consumer)
    tick=time.monotonic()
    system.fit(histories,support,event['floor'],aps,bounds,1150546,
        task=task,state=state_record['state'],event_id=CID,k=1)
    preparation_seconds=time.monotonic()-tick
    del task
    gc.collect()
    torch.cuda.empty_cache()
    write(OUT/'progress.json',dict(stage='H_S_prepared',seconds=preparation_seconds,
        relation_shape=list(consumer.relation_features.shape)))
    print(json.dumps(dict(stage='H_S_prepared',seconds=preparation_seconds)),flush=True)
    # Only whitelisted fit-query labels enter this single source feedback call.
    q=read_scans(CSV,event['query_rows'],True)
    assert not set(map(tuple,support['xy'])).intersection(map(tuple,q['xy']))
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    tick=time.monotonic()
    loss=system.backward_source(q['rss'],q['xy'],scaler)
    backward_seconds=time.monotonic()-tick
    assert np.isfinite(loss)
    scale=float(scaler.get_scale())
    gradients={}
    for name,p in consumer.relation_encoder.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()),name
        gradients[name]=float(p.grad.detach().abs().sum())/scale
    assert gradients['output.weight']>0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert before==state_hash(model) and encoder_before==state_hash(consumer.relation_encoder)
    complete=dict(complete=True,created=now(),backward_contract_pass=True,event=EVENT,canonical_context_id=CID,
        preparation_seconds=preparation_seconds,backward_seconds=backward_seconds,total_seconds=time.monotonic()-started,
        loss=float(loss),loss_scope='Wiring check source loss only; not a reported performance improvement',
        unscaled_gradient_absolute_sums=gradients,gradient_reporting='Stored scaled gradients divided by GradScaler scale; no optimizer/unscale_/step',
        AMP_scale=scale,backbone_gradients_all_None=True,backbone_state_unchanged=True,encoder_state_unchanged=True,
        relation_shape=list(consumer.relation_features.shape),recompute_modules=recompute,
        peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,backward_calls=1,optimizer_steps=0,
        new_GP_fits=0,protected_labels_read=0,target_rows_read=0,downloads=0,
        freeze_sha256=digest(OUT/'freeze.json'),
        limits=['One legal source K1 task only; no parameter update or trained relation benefit established',
                'Zero head makes inner-layer zero gradients expected; only finite nonzero head gradient is required'])
    write(OUT/'complete.json',complete)
    print(json.dumps(complete),flush=True)


if __name__=='__main__':
    main()
