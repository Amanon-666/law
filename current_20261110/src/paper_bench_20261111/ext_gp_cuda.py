"""Original full-history solve on CUDA; CPU optimizer and cache unchanged."""
from unittest.mock import patch
import history_fit
from paper_bench_20261111.history_fit_cuda import infer
from paper_bench_20261111.ext_gp_streaming import ReFiGPStreaming


class ReFiGPCUDA(ReFiGPStreaming):
    def predict(self, task):
        with patch.object(history_fit, 'infer', infer):
            return super().predict(task)
