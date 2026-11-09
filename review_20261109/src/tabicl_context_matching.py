"""Learn only the released model's final context-attention query modulation.

The existing last QASSMax query_mlp is retained at its pretrained weights.
All preceding representations, Q/K/V projections, length scaling, feed-forward
layers and the quantile head are frozen. This is a restricted adaptation of
TabICLv2, not a ReFi spatial operator or a reproduction of DR-TFM's DRO losses.

The learned object is a query-conditioned diagonal modulation before final
attention to H+S context. Earlier layers already depend on H/S, so this is a
conditional evidence-matching rule, not a globally fixed source weight. Freezing
the feature map does not guarantee better OOD predictions or eliminate overfit.
"""
from tabicl_history import CHECKPOINT
from tabicl_history_learning import EpisodicHistoryRegressor


class ContextMatchingRegressor(EpisodicHistoryRegressor):
    def __init__(self, checkpoint=CHECKPOINT, device='cpu'):
        super().__init__(checkpoint, device)
        network = self.loaded.model_
        network.requires_grad_(False)
        network.icl_predictor.tf_icl.blocks[-1].attn.ssmax_layer.query_mlp.requires_grad_(True)
        self.matching_parameter_names = tuple(n for n,p in network.named_parameters() if p.requires_grad)

