import os, json, time, hashlib, subprocess, datetime
from pathlib import Path
SRC=Path('/home/panyushuo/projects/panyushuo/claude_fp_wt/current_20261110/src')
REPO=SRC.parent.parent
os.chdir(SRC)
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONPATH='/home/panyushuo/projects/panyushuo/DKT-UJI/env/lib/python3.12/site-packages:/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/python_dependencies:/home/panyushuo/projects/panyushuo/multifloor-check-20261026/env/lib/python3.12/site-packages',VT_DIR=str(SRC/'claude_hurdle_20261110'),LAW_UJI_CSV='/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv')
import sys
sys.path[:0]=[str(SRC)]+os.environ['PYTHONPATH'].split(':')
import joblib
PYTHON='/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python'
state={'driver_pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'folds':[],'review':{'training':False,'AP_axis':'same H/S only','candidates':'same H positions plus S empirical candidates','reader':'same, fieldonly skips S conditioning but still uses S candidates/AP axis','field_difference':'naive constant variance source calibration versus learned s2; complete field generator difference'}}
meta=SRC/'results/wolearn_execution_20261111.json'
def save(): meta.write_text(json.dumps(state,indent=2)+'\n')
start=1794359812.625399
if meta.exists(): state=json.loads(meta.read_text()); state['resumed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat(); state['driver_pid']=os.getpid()
save()
for protocol,buildings in [('legacy54',['B1','B2']),('static-resample-v2',['B0','B1','B2'])]:
 for b in buildings:
  if time.time()-start>3300:
   state['stopped_before_next']='55 minute budget'; save(); sys.exit(0)
  old=SRC/'results'/ (f'legacy54_{b}_claude_20261111' if protocol=='legacy54' else f'bench_uji_{b}_20261111')
  out=SRC/'results'/ (f'legacy54_{b}_wolearn_20261111' if protocol=='legacy54' else f'bench_uji_{b}_wolearn_20261111')
  if (out/'summary.json').exists(): continue
  execution=SRC/'results/wolearn_execution_files_20261111'
  execution.mkdir(exist_ok=True)
  model=old/'source_models/ours_v1.joblib'
  loaded=joblib.load(model)
  identities=[{'class':type(op).__name__,'sup':getattr(op,'sup',False),'sup_attribute_absent':not hasattr(op,'sup')} for op in loaded.model.ops]
  assert identities and all(i['sup'] is False for i in identities),identities
  options=execution/(out.name+'_options.json'); options.write_text(json.dumps({'ours_loaded_fieldonly':{'model_path':str(model)}}))
  cfg=json.loads((old/'freeze.json').read_text())['configuration']
  cmd=[PYTHON,'-u','-m','law_bench.run_claude','--dataset','uji','--building',b,'--protocol',protocol,'--budgets',*map(str,cfg['budgets']),'--repeats',*map(str,cfg['repeats']),'--seed',str(cfg['seed']),'--R',str(cfg['R']),'--methods']
  cmd+=['ours_naive_noM2','ours_loaded_fieldonly'] if protocol=='legacy54' else ['ours_loaded_fieldonly']
  cmd+=['--method-options',str(options),'--output',str(out)]
  if cfg.get('source_buildings'): cmd+=['--source-buildings',*cfg['source_buildings']]
  if cfg.get('query_positions') is not None: cmd+=['--query-positions',str(cfg['query_positions'])]
  if protocol=='legacy54':
   manifest=Path(cfg['manifest']); assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='bb7db6eb8ce070a400f18057af46dc68f734f5db6f60235f4c60189e62d3c496'
   cmd+=['--manifest',str(manifest)]
  f={'building':b,'protocol':protocol,'output':str(out),'command':cmd,'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'operators':identities,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  state['folds'].append(f); t=time.time()
  with (execution/(out.name+'.log')).open('w') as log:
   p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT); f['pid']=p.pid; save(); print(json.dumps(f),flush=True); rc=p.wait()
  f.update(exit_code=rc,elapsed_seconds=time.time()-t,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); save()
  if rc: sys.exit(rc)
state['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat(); save()
