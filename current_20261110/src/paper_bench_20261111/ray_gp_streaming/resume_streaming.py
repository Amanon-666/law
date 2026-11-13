"""Four-thread resume with bounded History lifetime; same full-H GP mathematics."""
import hashlib
import json
from pathlib import Path
import sys

SOURCE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110/src')
sys.path.insert(0, str(SOURCE))

# Load the original runner's import-time two-thread defaults before overriding them.
from law_bench import run
from paper_bench_20261111 import ext_gp_streaming
import torch
import scipy.linalg
from threadpoolctl import threadpool_info, threadpool_limits


def main():
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        print(json.dumps(dict(resource_wrapper=str(Path(__file__).resolve()),
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            streaming_module_sha256=hashlib.sha256(Path(ext_gp_streaming.__file__).read_bytes()).hexdigest(),
            memory_policy='lazy AP histories; no persistent in-memory field cache',
            torch_num_threads=torch.get_num_threads(), threadpools=threadpool_info(),
            scientific_settings_changed=False)), flush=True)
        if sys.argv[1:] == ['--threads-only']:
            return
        sys.modules['law_bench.paper_ext_gp_streaming'] = ext_gp_streaming
        run.METHODS['refi_GP'] = ('paper_ext_gp_streaming', 'ReFiGPStreaming', {})
        run.main()


if __name__ == '__main__':
    main()
