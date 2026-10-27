"""Compare the port with the actual legacy functions without import-name clashes."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import log_ndtr

from external_baselines import tpm_adapter as adapter
from data import group_scans


def legacy_functions(path, names, namespace):
    tree=ast.parse(path.read_text())
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


def run(legacy_root):
    legacy_root=Path(legacy_root)
    source=legacy_root/'scripts/evaluate_signal_calibrated_map.py'
    match_source=legacy_root/'scripts/evaluate_scm_tobit.py'
    namespace=dict(np=np,log_ndtr=log_ndtr,NO_SIGNAL=-110.)
    legacy_functions(source,['dbm','position_map','lookup','gp_mean','build_map'],namespace)
    # The retained server revision has the weights inline in tobit_match.
    legacy_functions(match_source,['tobit_match'],namespace)
    path=legacy_root/'models/cross_floor.py'
    spec=importlib.util.spec_from_file_location('tpm_check_cross',path)
    cross=importlib.util.module_from_spec(spec);spec.loader.exec_module(cross)
    h=dict(rss=np.array([[-60,100],[-62,100],[-70,-80],[-80,-65]]),
           xy=np.array([[0,0],[0,0],[10,0],[20,0]],float))
    support=dict(rss=np.array([[-65,100],[-75,-70]]),xy=np.array([[3,0],[15,0]],float))
    query=np.array([[-64,100],[-73,-71],[100,100]])
    tf=dict(sigma_total=4.,rho=np.array([1.,.8,.9,1.,1.1,1.2]),bbar=np.arange(6)-3.,
            sb2=np.full(6,16.),s_w2=4.,mu0=np.full(6,-100.),vL0=np.full(6,20.),s=4.,h=.9,theta=-85.)
    field,match=(20.,.3),(16.,-80.)
    grouped=group_scans(h['rss'],h['xy'])
    prior=cross.prior_map(dict(xy=grouped['positions'],**grouped['stats']),tf)
    calibrated=namespace['build_map']('scm',prior,support['rss'],support['xy'],field,{'anchor_lookup_neighbors':3})
    expected=namespace['tobit_match'](query,*calibrated,*match)
    actual=adapter.predict([h],support,query,tf,field,match,cross)
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
    second=dict(rss=h['rss']-np.where(h['rss']==100,0,3),xy=h['xy'].copy())
    many=adapter.predict([h,second],support,query,tf,field,match,cross)
    reverse=adapter.predict([second,h],support,query,tf,field,match,cross)
    np.testing.assert_allclose(many,reverse,atol=1e-10,rtol=1e-10)
    empty=dict(rss=np.empty((0,2)),xy=np.empty((0,2)))
    zero=adapter.predict([h],empty,query,tf,field,match,cross)
    np.testing.assert_allclose(zero,namespace['tobit_match'](query,*prior,*match),atol=1e-10,rtol=1e-10)
    one=dict(rss=support['rss'][:1],xy=support['xy'][:1])
    np.testing.assert_allclose(adapter.predict([],one,query,tf,field,match,cross),np.repeat(one['xy'],len(query),axis=0))
    raw=namespace['position_map'](h['rss'],h['xy'])[:2]
    raw_cal=namespace['build_map']('scm',raw,support['rss'],support['xy'],field,{'anchor_lookup_neighbors':3})
    np.testing.assert_allclose(adapter.predict([h],support,query,None,field,match,cross),
                               namespace['tobit_match'](query,*raw_cal,*match),atol=1e-10,rtol=1e-10)
    extra=adapter.predict([h],support,np.vstack((query,query)),tf,field,match,cross)
    np.testing.assert_allclose(extra[:len(query)],actual,atol=1e-10,rtol=1e-10)
    root=Path(__file__).resolve().parents[1]
    out=root/'results/fusion_candidate_20261027'
    result=dict(passed=True,scope='Synthetic bridge consistency only, not a localization comparison',
        original_single_source_max_gap=float(np.max(np.abs(actual-expected))),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,match_source,path]},
        adapter_sha256=hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
        checks=['full single-H TPM pipeline equality','SCM-T identity pipeline equality',
                'source permutation with coincident coordinates','empty S','no H plus one S','query batch invariance'],
        adaptation='Sources transformed separately, exact-coordinate pooling then one shared S calibration; only single-H case is original TPM pipeline.')
    (out/'tpm_adapter_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    run(sys.argv[1])
