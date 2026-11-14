"""Original DL runner with one completed-optimizer-step startup marker."""
import json
import time
from unittest.mock import patch

import torch
from law_bench import run_claude

original_step = torch.optim.Adam.step
reported = False


def traced_step(optimizer, *args, **kwargs):
    global reported
    result = original_step(optimizer, *args, **kwargs)
    if not reported:
        reported = True
        print(json.dumps(dict(stage='first_optimizer_step_complete',
            time=time.time(), threads=torch.get_num_threads(), device='cpu')), flush=True)
    return result


if __name__ == '__main__':
    with patch.object(torch.optim.Adam, 'step', traced_step):
        run_claude.R.main()
