"""Deploy the learned ReFi field using only H/S and frozen shared parameters."""
import hashlib
import torch
from predict import adapt
from posterior import proposal
from recomposition import Parameters,prepare,prior,field
from observations import scan_log_prob

def load_parameters(path):
    state=torch.load(path,weights_only=True)
    # Multi-budget checkpoints also retain the optimizer for continued training.
    if "model" in state: state=state["model"]
    model=Parameters(state["center"],state["scale"])
    model.load_state_dict(state)
    return model.eval()

@torch.no_grad()
def refresh_case(model,case,weights,obs,seed):
    """Generate a numerical proposal for the current prior, without query data."""
    p=prior(model,case["blocks"],weights);k=len(p["sm"])
    if not k:return dict(case,proposal=None),dict(sampler="no_support")
    prepared=dict(sm=p["sm"],sk=p["sk"],phi_s=p["sm"].new_zeros((k,0)),**case["stats"])
    numerical,info=proposal(prepared,obs,p["sm"].new_empty(0),seed)
    return dict(case,proposal=(numerical["samples"],numerical["log_proposal"])),info

@torch.no_grad()
def build_field(model,cases,positions,target_minus_source,base,obs,state,grid,event_id):
    """Cache a target field once. True query positions and query RSS are absent.

    positions contains every actual retained H scan location, including
    non-detections; AP-specific history inducing locations are not substitutes.
    state is the original H/S-only adaptation, fixed before recomposition.
    """
    weights=grid.new_tensor(state["weights"])
    means=[];variances=[];diagnostics=[]
    for ap,c in cases.items():
        blocks=prepare(c["histories"],positions,target_minus_source,c["sx"],grid,
                       base,state["sigma"]["intercept_std"])
        seed=int.from_bytes(hashlib.sha256(f"recomposition/{event_id}/{ap}".encode()).digest()[:4],"little")
        case,info=refresh_case(model,dict(blocks=blocks,stats=c["stats"]),weights,obs,seed)
        prediction,ess=field(model,case,weights,obs)
        means.append(prediction["mean"]);variances.append(prediction["variance"])
        diagnostics.append(dict(ap=ap,ESS=ess,sampler=info["sampler"]))
    return dict(APs=list(cases),grid=grid,mean=torch.stack(means),
                variance=torch.stack(variances),obs=obs,diagnostics=diagnostics)

@torch.no_grad()
def locate(cached,query_rss):
    """query_rss columns follow cached APs; returns scores and expected coordinates."""
    scores=cached["grid"].new_zeros((len(query_rss),len(cached["grid"])))
    for j,(mean,var) in enumerate(zip(cached["mean"],cached["variance"])):
        scores+=scan_log_prob(torch.as_tensor(query_rss[:,j],dtype=mean.dtype),
                              mean,var,cached["obs"])
    return dict(scores=scores,predictions=scores.softmax(1)@cached["grid"])

