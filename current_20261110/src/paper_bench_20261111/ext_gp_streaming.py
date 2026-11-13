"""Same full-H GP computation, with History lifetime limited to active APs.

Only loading is deferred. Original task construction, joint support adaptation,
History arithmetic, sampling, readout and on-disk field cache are unchanged.
The old for-loop locals can briefly retain the previous AP as the next loads.
"""
from collections.abc import Mapping
from unittest.mock import patch

from episodic_tasks import task_inputs as original_task_inputs
from paper_bench_20261111.ext_gp_fullhistory import ReFiGPFullHistory


class _NoFieldCache(dict):
    def __setitem__(self, key, value):
        pass


class _Case(Mapping):
    def __init__(self, template, loader):
        self.template = template
        self.loader = loader
        self.loaded = None

    def __getitem__(self, key):
        if key != 'histories':
            return self.template[key]
        if self.loaded is None:
            self.loaded = [None if pair is None else self.loader(*pair)
                           for pair in self.template['histories']]
        return self.loaded

    def __iter__(self):
        return iter(self.template)

    def __len__(self):
        return len(self.template)


class _Cases(Mapping):
    def __init__(self, templates, loader):
        self.templates = templates
        self.loader = loader

    def __getitem__(self, ap):
        # Never store a loaded case in this mapping: each traversal is fresh.
        return _Case(self.templates[ap], self.loader)

    def __iter__(self):
        return iter(self.templates)

    def __len__(self):
        return len(self.templates)


def lazy_task_inputs(history_scans, support, base, obs, load_history):
    task = original_task_inputs(history_scans, support, base, obs,
                                lambda source, ap: (source, ap))
    task['cases'] = _Cases(task['cases'], load_history)
    return task


class ReFiGPStreaming(ReFiGPFullHistory):
    def __init__(self, source_run, gp_options):
        super().__init__(source_run, gp_options)
        self.gp._fields = _NoFieldCache()

    def predict(self, task):
        import position_fusion_gp
        # Original prepare_from_scans still groups H/S and validates each load.
        with patch.object(position_fusion_gp, 'task_inputs', lazy_task_inputs):
            return super().predict(task)
