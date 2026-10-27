"""Frozen paired development comparison; this does not finish LAW G1-G4."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import scipy
import torch

import fusion_task_data as io
import fusion_predict as prediction
import fusion_prior,fusion_moments,fusion_posterior,fusion_adaptation
import inference,transfer,data,history_bank,multi_source_adaptation,full_ap_study,operator_prior,run as run_module
from external_baselines import tpm_adapter
from external_baselines.tpm_source_calibration import sha,write
from uji_probability_study import load_legacy

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/fusion_study_20261028'
BASE=io.PARAMETERS
LAW=ROOT/'results/tpm_source_calibration_20261027/all_B1.json'
REFERENCE=ROOT/'results/tpm_source_selection_20261027/selection.json'
METHODS=(*prediction.ARMS,'legacy_global_identity','GP_S','TPM_pool','SCMT_pool','WKNN_HS','WKNN_S')


def prepare():
    checks=ROOT/'results/fusion_candidate_20261027'
    for name in ['check.json','moment_reuse_check.json','batch_posterior_check.json','task_rights_check.json','tpm_adapter_check.json','joint_adaptation_check.json']:
        assert io.read(checks/name)['passed'],name
    wiring=io.read(checks/'joint_full_size_wiring_check.json')
    assert not wiring['failures'] and wiring['method_count']==len(METHODS)
    for name,digest in wiring['code_sha256'].items():assert sha(ROOT/name)==digest
    manifest=io.prepare_manifest(128)
    for meta in manifest['events']:
        assert meta['target'] not in meta['histories']
        for k in io.K_VALUES:
            assert meta['support_rows'][str(k)]==meta['support_rows']['20'][:k]
        assert set(meta['support_scan_ids']['20']).isdisjoint(meta['query_scan_ids'])
    OUT.mkdir(exist_ok=False)
    write(OUT/'manifest.json',manifest)
    modules=[io,prediction,fusion_prior,fusion_moments,fusion_posterior,fusion_adaptation,
             inference,transfer,data,history_bank,multi_source_adaptation,full_ap_study,operator_prior,run_module,tpm_adapter,io.data_module()]
    cross=load_legacy('cross_floor')
    paths={Path(m.__file__) for m in modules}|{Path(__file__),Path(cross.__file__),Path(io.data_module().load_building.__code__.co_filename)}
    snapshots={}
    for p in sorted(paths):
        dest=OUT/'code'/f'{sha(p)[:12]}_{p.name}'
        dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest);snapshots[str(p)]=str(dest.relative_to(OUT))
    hf=io.read(io.CACHE/'freeze.json')
    inputs=dict(hf['input_sha256'])
    for p in [BASE,LAW,REFERENCE,io.CACHE/'freeze.json',io.CACHE/'summary.json',OUT/'manifest.json',
              checks/'hypothesis_gate_20261028.json',checks/'source_precision_planning.json',
              checks/'joint_adaptation_check.json',checks/'joint_full_size_wiring_check.json']:
        inputs[str(p)]=sha(p)
    for meta in io.history_index().values():
        directory=io.CACHE/meta['building']/f'F{meta["floor"]}'
        for ap in range(len(meta['AP_ids'])):
            status=directory/f'ap{ap}.json';cache=directory/f'ap{ap}.npz';record=io.read(status)
            assert record['parameters_sha256']==sha(BASE)
            assert sha(cache)==record['cache_sha256']
            inputs[str(status)]=sha(status);inputs[str(cache)]=record['cache_sha256']
    write(OUT/'freeze.json',dict(created_UTC=datetime.now(timezone.utc).isoformat(),methods=METHODS,
        candidate_arms=list(prediction.ARMS),draws=128,K=io.K_VALUES,base_events=len(manifest['events']),
        events_with_K=len(manifest['events'])*len(io.K_VALUES),
        data_rights='Target-floor H cache excluded. H rows unchanged. AP set/domain use H and CURRENT S only. Fixed Q and nested S; one scan per position. All/far share S/Q but have their own H-derived metadata.',
        source_parameters='All shared parameters/source selection use B1, not the four development buildings. Existing development data were previously inspected; this is not independent-building confirmation.',
        hypotheses={'adaptation':'Joint covariance can identify coherent changes discarded by marginal sigma selection; joint-vs-legacy intervention includes continuous sigma and joint weights, not an isolated sigma-grid comparison.',
                    'coverage':'Local precision-inspired fusion may help complementary spatial coverage; precision is not assumed to guarantee relevance.',
                    'affine':'Source-fitted strength-bin transformation is a control, not a required G2 solution or proof of a universal learnable map.'},
        adaptation='Four candidate arms jointly fit shared source weights, continuous sigma in [0,20], and full-Gaussian q on S. Legacy_global_identity keeps two-stage weight fit and marginal sigma grid. GP_S has no H field but retains common H metadata and B1 base/obs.',
        posterior='Common full-Gaussian VI approximation, sum-over-AP objective, maxiter640 and per-AP projected gradient <=1e-3. Joint EB maxiter800, simplex KKT<1e-3, variance gradient<1e-4, per-AP VI gradient<1e-3. No PMC equivalence claim.',
        readout='Full eligible-AP log scores summed, posterior mean coordinates on 2-unit H+current-S grid. TPM/SCMT retain native prototype readout; WKNN retains existing raw-scan k=3 inverse-RMS-distance implementation.',
        inference_limits=['Independent modeled source fields and conditional AP observations are assumptions, not established physical independence.',
                          'Historical all-zero-detection fallback remains None and is not called unmeasured.',
                          'Numerical stationarity does not prove exact posterior, correct model or global optimum.'],
        primary_estimand='For all-H scenario, average query-position MDE within a floor/draw, then equal-floor building mean per draw. Buildings/units never pooled.',
        primary_contrasts='Four candidate arms x [4 buildings x 5 K x 2 references(WKNN_HS,GP_S) + 4 buildings x 4 adjacent-K changes] =224 contrasts. Differences are candidate-reference or laterK-earlierK; negative is better.',
        primary_CI='95% simultaneous paired max-absolute-studentized bootstrap intervals over the 224 means; 20000 bootstrap draws, seed2026102801. Resample whole draw groups independently by building, retaining all floors/methods/K within each group. Pointwise 95% paired intervals also reported. Any zero-sample-variance contrast is explicitly flagged, not silently divided by zero.',
        acceptance='No positive noninferiority margin. A G1 claim requires complete predictions and simultaneous upper bounds <=0 against both references and for the adjacent-K changes on every development building. Raw curves never monotonicized. Wide intervals or missing outcomes do not count as success.',
        planning='128 draws chosen before scoring. B1 selected-model variance plus a normal-theory SD upper sensitivity suggests up to96 draws for224 comparisons at5% effect/80% per-contrast power with4-floor aggregation;128 allows the3-floor building count under the planning independence assumption. This is not a guarantee across model families, far strata or nonnormal distributions;5% is NOT the noninferiority margin.',
        secondary='Separate all/far, each target floor, S-hull inside/outside, complementary H geometry and no H geometric coverage. Geometry uses source base.length as a fixed diagnostic scale. Empty strata remain null. Ratio-of-sums stratum means use each floor query count for equal-floor population weights; bootstrap whole draw groups. Secondary intervals are labeled exploratory, not used to override primary failure.',
        attribution='History content: WKNN_HS vs WKNN_S under common feature catalogue. Mature utilization: candidates/TPM/SCMT vs WKNN_HS, including their source experience. New learned-module increment is absent in this study and cannot be claimed.',
        efficiency='Only measured discrete K matching/relative-error contrasts; no interpolation through nonmonotone curves and no theoretical upper-bound or label-saving claim beyond tested K.',
        failures='Keep all failures and counts. No partial-AP predictions, threshold relaxation, automatic retry or Q-based fallback. Incomplete comparisons cannot pass G1; complete-case descriptives are labeled conditional.',
        monitoring='Operational progress/failures only until planned cases finish; no interim parameter selection or outcome-based model changes.',
        independent_validation_opened=False,
        versions=dict(torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__),
        code_sha256={str(p):sha(p) for p in paths},code_snapshots=snapshots,input_sha256=inputs))
    print(json.dumps(dict(stage='frozen',base_events=len(manifest['events']),K_events=len(manifest['events'])*5)),flush=True)


def verify():
    f=io.read(OUT/'freeze.json')
    for section in ('code_sha256','input_sha256'):
        for path,digest in f[section].items():
            if sha(path)!=digest:raise ValueError(f'Changed frozen file: {path}')
    return f


def run(building):
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    frozen=verify();freeze_sha=sha(OUT/'freeze.json')
    parameters=io.read(BASE);law=io.read(LAW)['transfer']
    law={k:np.asarray(v) if isinstance(v,list) else v for k,v in law.items()}
    cross=load_legacy('cross_floor');selection=io.read(REFERENCE)
    events=[e for e in io.read(OUT/'manifest.json')['events'] if e['building']==building]
    assert events
    done=0;failed_methods=0
    for meta in events:
        dest=OUT/building/meta['id'];dest.mkdir(parents=True,exist_ok=True)
        for k in io.K_VALUES:
            path=dest/f'K{k}.json'
            if path.exists():
                old=io.read(path);assert old['freeze_sha256']==freeze_sha
                assert set(old['results'])|set(old['failures'])==set(METHODS)
                done+=1;failed_methods+=len(old['failures']);continue
            start=time.perf_counter();task=io.adaptation_inputs(meta,k)
            state=prediction.adapt(task,parameters,law,cross,sha(BASE))
            # Q is fetched after S fitting; its coordinates enter only scoring below.
            query=io.query_for_scoring(meta,task['AP_indices'])
            result=prediction.predict(task,query['rss'],state,law,cross,selection)
            assert set(result['results'])|set(result['failures'])==set(METHODS)
            for name,row in result['results'].items():
                pred=np.asarray(row['predictions']);errors=np.linalg.norm(pred-query['xy'],axis=1)
                assert len(errors)==meta['query_positions'] and np.isfinite(errors).all()
                row.update(errors=errors.tolist(),mde=float(errors.mean()),p90=float(np.quantile(errors,.9)))
                if 'radius90' in row:row['covered90']=(errors<=np.asarray(row['radius90'])).tolist()
            record=dict(id=meta['id'],building=building,target=meta['target'],scenario=meta['scenario'],draw=meta['draw'],K=k,
                unit=meta['coordinate_unit'],query_scan_ids=query['scan_ids'].tolist(),query_xy=query['xy'].tolist(),
                AP_indices=task['AP_indices'],grid=task['grid'],omitted_detected_APs=query['omitted_detected_APs'].tolist(),
                strata=io.scoring_strata(task,query['xy'],parameters['base']['length']),
                results=result['results'],failures=result['failures'],seconds=time.perf_counter()-start,freeze_sha256=freeze_sha)
            write(path,record);done+=1;failed_methods+=len(result['failures'])
            print(json.dumps(dict(building=building,id=meta['id'],K=k,completed=done,expected=len(events)*5,
                failed_methods=list(result['failures']),seconds=record['seconds'])),flush=True)
    write(OUT/f'{building}_finished.json',dict(complete=True,building=building,K_events=done,
        failed_method_events=failed_methods,freeze_sha256=freeze_sha))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','verify']);p.add_argument('--building');a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='verify':verify()
    else:
        try:run(a.building)
        except Exception as error:
            write(OUT/f'{a.building}_fatal.json',dict(error=repr(error),UTC=datetime.now(timezone.utc).isoformat()))
            raise
