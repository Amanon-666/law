"""Synthetic engineering check of the explicit final-query-only learning scope.

No real data, localization scoring, target adaptation or persistent learned
checkpoint. A single update verifies that the selected matcher receives the
existing full loss while every other weight and pre-final representation stays
unchanged. This is not evidence of source or cross-environment performance.
"""
import hashlib
import json
import socket
import time

import numpy as np
import torch

from tabicl_history import CHECKPOINT, PACKAGE, FrozenHistoryRegressor, features, state_hash
from tabicl_context_matching import ContextMatchingRegressor
from recomposition_trial import ROOT, digest, write, now

OUT = ROOT/'results/tabicl_context_matching_structure_20261105'
SEED = 1150543


def frozen_hash(model):
    selected = set(model.matching_parameter_names)
    h=hashlib.sha256()
    for name,value in model.loaded.model_.state_dict().items():
        if name not in selected:
            h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def last_input_and_output(model, query_rss):
    x,y=model.views(query_rss,0)
    captured=[]
    def capture(module,args,kwargs):
        captured.append((kwargs['q'] if 'q' in kwargs else args[0]).detach().clone())
    block=model.loaded.model_.icl_predictor.tf_icl.blocks[-1]
    handle=block.register_forward_pre_hook(capture,with_kwargs=True)
    model.loaded.model_.train()
    with torch.no_grad():
        result=model.loaded.model_(torch.tensor(x[:1],dtype=torch.float32),
                                  torch.tensor(y[:1],dtype=torch.float32),embed_with_test=False)
    handle.remove()
    model.loaded.model_.eval()
    assert len(captured)==1
    return captured[0],result


def run():
    started=time.monotonic();OUT.mkdir(exist_ok=False)
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    def scans(n):
        rss=rng.integers(-95,-40,size=(n,8),dtype=np.int16)
        rss[rng.random(rss.shape)<.2]=100
        return dict(rss=rss,xy=rng.normal(size=(n,2))*10)
    h={'B0F0':scans(6),'B0F2':scans(6)}
    support=scans(5);query=scans(4)
    xy=np.concatenate([v['xy'] for v in [*h.values(),support]])
    bounds=np.array([xy.min(0),xy.max(0)]);aps=np.arange(8)
    code=['tabicl_context_matching.py','check_tabicl_context_matching.py',
          'tabicl_history.py','tabicl_history_learning.py','recomposition_trial.py']
    write(OUT/'freeze.json',dict(created=now(),seed=SEED,real_data=False,
        code={n:digest(ROOT/n) for n in code},checkpoint_sha256=digest(CHECKPOINT),
        package={str(p.relative_to(PACKAGE)):digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))},
        structural_question='Does only final context query_mlp change while its input and all other weights remain identical?',
        real_training_updates=0,synthetic_updates=1,localization_scores=0,
        resources=dict(cpu=2,gpu=0,wall_seconds=120,downloads=0)))
    for n in code:(OUT/n).write_bytes((ROOT/n).read_bytes())
    base=FrozenHistoryRegressor()
    model=ContextMatchingRegressor()
    assert state_hash(base.loaded.model_)==state_hash(model.loaded.model_)
    base.fit(h,support,'B0F1',aps,bounds,SEED)
    model.fit(h,support,'B0F1',aps,bounds,SEED)
    base_prediction=base.predict(query['rss'])['raw']
    same_prediction=model.predict(query['rss'])['raw']
    views_equal=[]
    for axis in range(2):
        estimator=base.estimators[axis]
        transformed=estimator.ensemble_generator_.transform(
            estimator.X_encoder_.transform(features(query['rss'],base.floor,base.aps)),mode='both')
        reference=tuple(np.concatenate([pair[i] for pair in transformed.values()]) for i in range(2))
        views_equal.append(all(np.array_equal(a,b) for a,b in zip(reference,model.views(query['rss'],axis))))
    comparison=dict(initial_max_gap=float(np.max(np.abs(base_prediction-same_prediction))),
        base_repeat_gap=float(np.max(np.abs(base_prediction-base.predict(query['rss'])['raw']))),
        state_equal=state_hash(base.loaded.model_)==state_hash(model.loaded.model_),
        views_equal=views_equal)
    matching_flags=dict(model.loaded.model_.named_parameters())
    for name,parameter in base.loaded.model_.named_parameters():
        parameter.requires_grad_(matching_flags[name].requires_grad)
    comparable_prediction=base.predict(query['rss'])['raw']
    comparison['same_gradient_flags_max_gap']=float(np.max(np.abs(comparable_prediction-same_prediction)))
    write(OUT/'initial_comparison.json',comparison)
    print(json.dumps(comparison),flush=True)
    assert comparison['state_equal'] and all(views_equal)
    assert comparison['initial_max_gap']<1e-5
    assert np.array_equal(comparable_prediction,same_prediction)
    del base
    parameters=[p for p in model.loaded.model_.parameters() if p.requires_grad]
    count=sum(p.numel() for p in parameters)
    assert count==8320 and len(model.matching_parameter_names)==4
    assert all(n.startswith('icl_predictor.tf_icl.blocks.11.attn.ssmax_layer.query_mlp.')
               for n in model.matching_parameter_names)
    before_frozen=frozen_hash(model)
    before_state=state_hash(model.loaded.model_)
    before_input,before_output=last_input_and_output(model,query['rss'])
    optimizer=torch.optim.AdamW(parameters,lr=1e-3,weight_decay=0.)
    optimizer.zero_grad(set_to_none=True)
    loss=model.backward(query['rss'],query['xy'])
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
    assert all(p.grad is None for p in model.loaded.model_.parameters() if not p.requires_grad)
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
    assert norm>0
    optimizer.step()
    assert frozen_hash(model)==before_frozen
    assert state_hash(model.loaded.model_)!=before_state
    after_input,after_output=last_input_and_output(model,query['rss'])
    assert torch.equal(before_input,after_input)
    output_gap=float((after_output-before_output).abs().max())
    assert output_gap>0
    result=dict(complete=True,created=now(),seconds=time.monotonic()-started,
        trainable_parameters=count,total_parameters=sum(p.numel() for p in model.loaded.model_.parameters()),
        matching_parameter_names=model.matching_parameter_names,initial_comparison=comparison,
        frozen_state_sha256=before_frozen,frozen_weights_unchanged=True,pre_final_representation_unchanged=True,
        synthetic_pinball=loss,synthetic_gradient_norm=norm,synthetic_output_max_change=output_gap,
        real_data=False,real_training_updates=0,synthetic_updates=1,localization_scores=0,
        checkpoint_saved=False)
    write(OUT/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    assert socket.gethostname()=='ray-web'
    torch.set_num_threads(2)
    run()
