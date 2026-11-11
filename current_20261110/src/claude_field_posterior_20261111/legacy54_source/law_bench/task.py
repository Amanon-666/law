"""law_bench interface contract (Claude, 2026-11-11). Shared by all data adapters and methods.

Task: one deployment of a new floor.
  dataset, building : str            e.g. 'uji', 'B1'
  floor            : int             physical floor of the target
  K, repeat        : int             number of calibration positions; repetition seed index
  task_id          : str             f'{dataset}/{building}/F{floor}/K{K}/r{repeat}'
  ap_ids           : list            the building's AP axis (column order of every rss below)
  H                : {floor:int -> dict(rss=float[N,A], xy=float[N,2], acq=int[N])}
                     all OTHER floors of the building, raw training-side scans (no averaging); 100 = not detected
  S                : dict(rss=float[n_s,A], xy=float[n_s,2], acq=int[n_s])   K positions, <= R scans each, nested in K
  Q_rss            : float[n_q,A]    query scans (<= R per query position); query coordinates are NOT in the task
  meta             : dict            R, whether S/Query come from different acquisitions, pooling used for splits, ...
Scorer-only (never passed to methods): Q_xy float[n_q,2].

Source building (for learned methods): dict(dataset, building, floors={floor: dict(rss, xy, acq)}, ap_ids).
Methods never receive buildings of the task's own (dataset, building) as source.

Method:
  name: str
  fit_source(sources: list[source building]) -> None      (no-op for non-learned methods)
  predict(task: Task) -> float[n_q, 2]
Rules: AP axis may be reduced inside a method to APs detected in H or S; it must never be extended or selected with
Query information. Hyper-parameters only from sources.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Task:
    dataset: str
    building: str
    floor: int
    K: int
    repeat: int
    ap_ids: list
    H: dict
    S: dict
    Q_rss: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def task_id(self):
        if self.meta.get('protocol') == 'legacy54':
            return f"{self.dataset}/{self.building}/legacy54/{self.meta['legacy_event_id']}"
        return f'{self.dataset}/{self.building}/F{self.floor}/K{self.K}/r{self.repeat}'


class Method:
    name = 'base'

    def fit_source(self, sources):
        pass

    def predict(self, task):
        raise NotImplementedError
