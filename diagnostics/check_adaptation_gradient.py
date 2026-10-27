"""Analytic Gaussian limit: shared learning must include S adaptation.

This is not a localization experiment or a proposed new model. It falsifies
the claim that treating the fitted support state as constant always gives
the derivative of the deployed predictor.
"""
import hashlib
import json
from pathlib import Path


def prediction(theta,features,labels,query_feature,penalty):
    intercept=sum(y-theta*f for y,f in zip(labels,features))/(len(labels)+penalty)
    return theta*query_feature+intercept


def check():
    rows=[]
    for features,query_feature,penalty in [([1.,1.],1.,0.),([2.,2.],1.,0.),([1.,1.],1.,1.)]:
        labels=[0.,0.];query_label=1.;theta=0.;eps=1e-5
        estimate=prediction(theta,features,labels,query_feature,penalty)
        derivative=query_feature-sum(features)/(len(features)+penalty)
        full_gradient=(estimate-query_label)*derivative
        stopped_gradient=(estimate-query_label)*query_feature
        def loss(value):
            return .5*(prediction(value,features,labels,query_feature,penalty)-query_label)**2
        numerical=(loss(theta+eps)-loss(theta-eps))/(2*eps)
        assert abs(full_gradient-numerical)<1e-9
        rows.append(dict(support_features=features,query_feature=query_feature,penalty=penalty,
            total_gradient=full_gradient,finite_difference=numerical,stopped_adaptation_gradient=stopped_gradient))
    assert rows[0]['total_gradient']==0 and rows[0]['stopped_adaptation_gradient']!=0
    assert rows[1]['total_gradient']*rows[1]['stopped_adaptation_gradient']<0
    return dict(passed=True,scope='Gaussian ridge/intercept analytical counterexample only',
        model='f_theta(x)=theta*phi(x)+b; fit b on S using squared error plus penalty*b^2',
        fitted_intercept='sum(y_S-theta*phi_S)/(K+penalty)',
        total_prediction_derivative='phi_Q-sum(phi_S)/(K+penalty)',cases=rows,
        consequence='Stopping gradients through S adaptation can invent a nonzero learning direction or reverse its sign. A score-level decision gradient check does not validate the full training path.',
        limits='Does not prove that any existing training run used this wrong gradient, or predict real localization benefit. Current frozen comparison performs no shared-parameter learning.',
        implementation_choices='Either differentiate the actual adaptation map, or explicitly define and deploy a distinct fixed-adaptation model. Envelope cancellation applies to the optimized inner objective, not arbitrary outer positioning loss.')


if __name__=='__main__':
    result=check();result['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=Path(__file__).resolve().parents[1]/'results/fusion_candidate_20261027/decision_adaptation_audit.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
