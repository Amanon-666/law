"""H/S-only preparation for the existing GP + trained Tab position system.

This connects the AP-bound relation consumer to the actual retained-GP path,
including explicit source feedback for its fixed-half coordinate readout.
It does not choose a fusion weight, step an optimizer, or launch a schedule.
H/S provenance belongs to the caller's task preparation:
cached history fields must have been fitted to exactly the permitted H rows,
and the supplied adaptation state must correspond to this same H/S context.
The wrapper cannot certify a cache's training inputs from numerical arrays.

The historical context and its relation graph are prepared before query scans
are supplied. The same GP posterior pass supplies both the relation graph and
the fixed likelihood fields. Additional queries reuse that preparation without
refitting or resampling S. Query coordinates are absent from fit/predict;
only backward_source accepts labeled SOURCE queries for supervised feedback.
"""
import numpy as np

from context_influence_features import RetainedInfluenceCollector
from position_fusion_gp import prepare_retained_fields, score_retained_fields


class InfluencePositionSystem:
    def __init__(self, consumer):
        """consumer is an InfluenceHistoryRegressor on an explicit trained asset."""
        self.consumer = consumer

    def fit(self, histories, support, target_floor, aps, tab_bounds, seed, *, task, state, event_id, k):
        """Prepare GP and Tab from matched legal H/S; no query argument exists."""
        assert np.array_equal(aps, task['aps'])
        collector = RetainedInfluenceCollector(task, state, histories)
        self.gp_fields = prepare_retained_fields(task, state, event_id, k, relation_collector=collector)
        relations = collector.finish()
        self.consumer.fit(histories, support, target_floor, aps, tab_bounds, seed, relations=relations)
        self.relation_metadata = {key: relations[key] for key in
            ('feature_names', 'moment_names', 'source_names', 'source_offsets', 'aps', 'scale_db', 'length',
             'relation_scope', 'uncertainty_scope')}
        return self

    def predict(self, query_rss, *, enabled=True):
        """Preserve component projections and the existing fixed half-average.

        enabled=False bypasses only the learned relation path, retaining the
        same GP fields, original H/S Tab, and fixed system readout. This is a
        within-system comparison, not a different information condition.
        """
        gp = score_retained_fields(self.gp_fields, query_rss)
        tab = self.consumer.predict(query_rss, enabled=enabled)
        return dict(GP=gp['projected'], Tab=tab['projected'],
                    GP_Tab_half=.5*(gp['projected']+tab['projected']))

    def backward_source(self, query_rss, source_xy, scaler=None):
        """Source-only feedback through the deployed fixed-half system.

        The H/S preparation remains fixed. This only accumulates encoder
        gradients: no optimizer step, checkpoint selection or target updates.
        Protected source evaluation labels are not legal training inputs.
        """
        gp = score_retained_fields(self.gp_fields, query_rss)
        return self.consumer.backward_system(query_rss, source_xy, gp['projected'], scaler)
