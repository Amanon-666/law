"""Release unused L-BFGS output storage before constructing GP posterior matrices."""
from unittest.mock import patch

import history_fit
from paper_bench_20261111.ext_gp_streaming import ReFiGPStreaming


def release_minimize_output(minimize):
    def solve(*args, **kwargs):
        result = minimize(*args, **kwargs)
        # L-BFGS-B's inverse-Hessian views retain its large workspace. The
        # history fitter and acceptance monitor never consume this output.
        del result.hess_inv
        return result
    return solve


class ReFiGPStreamingRelease(ReFiGPStreaming):
    def predict(self, task):
        with patch.object(history_fit, 'minimize',
                          release_minimize_output(history_fit.minimize)):
            return super().predict(task)
