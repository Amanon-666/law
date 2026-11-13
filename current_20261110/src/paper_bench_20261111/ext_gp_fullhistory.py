"""Original paper ReFi+GP with explicit original full-history cache construction."""
import sys
from pathlib import Path

from law_bench import run
from law_bench.method_retained_gp import RetainedGPMethod
from paper_bench_20261111.ext_gp import ReFiGP, export_gp


class ReFiGPFullHistory(ReFiGP):
    def __init__(self, source_run, gp_options):
        if gp_options.get('fit_missing_history') is not True:
            raise ValueError('Full-history entry requires explicit fit_missing_history=True')
        self.source_run = Path(source_run)
        self.gp = RetainedGPMethod(**gp_options)


if __name__ == '__main__':
    if sys.argv[1:2] == ['export-gp']:
        export_gp(sys.argv[2])
    else:
        from paper_bench_20261111 import ext_gp_fullhistory
        sys.modules['law_bench.paper_ext_gp_fullhistory'] = ext_gp_fullhistory
        run.METHODS['refi_GP'] = ('paper_ext_gp_fullhistory', 'ReFiGPFullHistory', {})
        run.main()
