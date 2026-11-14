"""Four-thread resume with bounded History lifetime; same full-H GP mathematics."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

SOURCE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110/src')
sys.path.insert(0, str(SOURCE))

# Load the original runner's import-time two-thread defaults before overriding them.
from law_bench import run
from paper_bench_20261111 import ext_gp_cuda_diagnostic
from paper_bench_20261111.ext_gp_streaming import _NoFieldCache
import torch
import scipy.linalg
import time
from threadpoolctl import threadpool_info, threadpool_limits


original_evaluate = run.evaluate


def verify_model(model, stage):
    expected = ext_gp_cuda_diagnostic.ReFiGPCUDADiagnostic
    if not isinstance(model, expected) or not isinstance(model.gp._fields, _NoFieldCache):
        raise ValueError('Saved execution model is not the CUDA streaming class; '
                         'use a new output directory and reuse the verified history cache')
    print(json.dumps(dict(stage=stage,
        model_type=type(model).__module__ + '.' + type(model).__name__,
        field_cache_type=type(model.gp._fields).__name__, backend='cuda')), flush=True)


def checked_evaluate(args):
    model_path = args.output.resolve() / 'source_models' / 'refi_GP.joblib'
    original_load, original_dump = run.joblib.load, run.joblib.dump

    def checked_load(path, *load_args, **load_kwargs):
        model = original_load(path, *load_args, **load_kwargs)
        if Path(path).resolve() == model_path:
            verify_model(model, 'resumed_execution_model')
        return model

    def checked_dump(model, path, *dump_args, **dump_kwargs):
        if Path(path).resolve() == model_path.with_suffix('.tmp'):
            verify_model(model, 'fitted_execution_model')
        return original_dump(model, path, *dump_args, **dump_kwargs)

    with patch.object(run.joblib, 'load', checked_load), patch.object(run.joblib, 'dump', checked_dump):
        return original_evaluate(args)


def main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    probe = torch.empty(1, device='cuda')
    print(json.dumps(dict(stage='cuda_preflight', device=str(probe.device), name=torch.cuda.get_device_name(0))), flush=True)
    del probe
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        print(json.dumps(dict(resource_wrapper=str(Path(__file__).resolve()),
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            streaming_module_sha256=hashlib.sha256(Path(ext_gp_cuda_diagnostic.__file__).read_bytes()).hexdigest(),
            memory_policy='lazy AP histories; no persistent in-memory field cache',
            torch_num_threads=torch.get_num_threads(), threadpools=threadpool_info(),
            scientific_settings_changed=False)), flush=True)
        if sys.argv[1:] == ['--threads-only']:
            return
        sys.modules['law_bench.paper_ext_gp_cuda_diagnostic'] = ext_gp_cuda_diagnostic
        run.METHODS['refi_GP'] = ('paper_ext_gp_cuda_diagnostic', 'ReFiGPCUDADiagnostic', {})
        with patch.object(run, 'evaluate', checked_evaluate):
            run.main()


if __name__ == '__main__':
    main()
