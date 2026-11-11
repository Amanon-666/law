"""Register the isolated original-A adapter, then use the unmodified law_bench runner."""
import sys
from law_bench import run
from . import refi

sys.modules['law_bench.paper_refi'] = refi
run.METHODS['refi'] = ('paper_refi', 'ReFi', {})

if __name__ == '__main__':
    run.main()
