"""Public or existing source-finetuned TabICLv2 on the shared Task.

variant='public' keeps the original wrapper; 'ordinary' loads source-selected
ordinary_step_30, 'full24' loads the fixed episodic full_step_24. Both saved
checkpoints use the official config/state_dict format and the SAME existing
FrozenHistoryRegressor deployment path; EpisodicHistoryRegressor is needed
only for training, which this adapter never does.

Both source-trained assets use the protected-legal B0 F0/F1/F2 corpus (2861
rows), not full four-floor B0. They cannot evaluate any UJI B0 heldout task.
Their source-development results and old 54-event target results are separate
from this benchmark. The existing eight views, preprocessing, full H/current
S, bounds and wrapper seed+repeat recipe are unchanged for all variants.
"""
from pathlib import Path
import hashlib
import json

import numpy as np


RESULTS = Path('/home/panyushuo/projects/panyushuo/ReFi-UJI-Core-20261031/results')
PUBLIC_SHA256 = '0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a'
SOURCE_ASSETS = {
    'ordinary': ('ordinary_context_trial_20261109', 'ordinary_step_30.ckpt',
                 '39d72ddfa3211797c56b127068b592c45acef52609efec51fb3892af5f653978'),
    'full24': ('tabicl_multitask_full_trial_20261107_checked', 'full_step_24.ckpt',
               'bfc3553049ec88e847411d7a99e6dfe98bf330fce4adb8b5c6bf0d50c9223a59'),
}


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class TabICL:
    name = 'tabicl'

    def __init__(self, checkpoint=None, seed=261026, *, variant='public', source_assets=None):
        if variant not in ('public', *SOURCE_ASSETS):
            raise ValueError('TabICL variant must be public, ordinary or full24')
        self.checkpoint, self.seed = checkpoint, seed
        self.variant = variant
        self.name = 'tabicl' if variant == 'public' else 'tabicl_' + variant
        self.source_assets = None if source_assets is None else Path(source_assets)
        self.allowed_sources = None
        self.provenance = dict(kind='public', project_source_updates=0,
                               public_pretraining='not audited')
        self.regressor = None

    def fit_source(self, sources):
        """Verify an existing asset and this fold's permission; never train."""
        self.allowed_sources = None
        if self.variant != 'public':
            allowed = {(s['dataset'], s['building']) for s in sources}
            if ('uji', 'B0') not in allowed:
                raise ValueError('Non-runnable Tab fold: existing fine-tuned asset requires UJI B0 as source')
            directory, filename, expected_sha = SOURCE_ASSETS[self.variant]
            assets = RESULTS/directory if self.source_assets is None else self.source_assets
            complete = json.loads((assets/'complete.json').read_text())
            freeze_path = assets/'freeze.json'
            freeze = json.loads(freeze_path.read_text())
            contract = freeze.get('contract', freeze)
            recorded_sha = (complete['selected_checkpoint_sha256'] if self.variant == 'ordinary'
                            else complete['checkpoint_sha256']['24'])
            if not (complete['complete'] and _digest(freeze_path) == complete['freeze_sha256']
                    and recorded_sha == expected_sha):
                raise ValueError('Source Tab completion/freeze/checkpoint binding mismatch')
            catalog_path = next(Path(p) for p in contract['inputs'] if p.endswith('/source_episode_catalog_20261105/catalog.json'))
            if _digest(catalog_path) != contract['inputs'][str(catalog_path)]:
                raise ValueError('Source Tab training-catalog checksum mismatch')
            catalog = json.loads(catalog_path.read_text())
            if catalog['source_floors'] != ['B0F0', 'B0F1', 'B0F2']:
                raise ValueError('Source Tab asset does not have the audited three-floor provenance')
            pool = {f: set() for f in catalog['source_floors']}
            for bank in catalog['history_rows'].values():
                for floor, rows in bank.items():
                    pool[floor].update(rows)
            self.checkpoint = assets/filename if self.checkpoint is None else Path(self.checkpoint)
            if _digest(self.checkpoint) != expected_sha:
                raise ValueError('Source Tab checkpoint checksum mismatch')
            self.provenance = dict(kind='source_finetuned', recipe=self.variant,
                training_sources=[dict(dataset='uji', building='B0')],
                source_floors=catalog['source_floors'], training_rows_by_floor={f: len(v) for f, v in pool.items()},
                protected_rows=len(catalog['protected_rows']), four_floor_B0_matched=False,
                catalog_path=str(catalog_path), catalog_sha256=contract['inputs'][str(catalog_path)],
                freeze_path=str(freeze_path), freeze_sha256=complete['freeze_sha256'],
                complete_path=str(assets/'complete.json'), complete_sha256=_digest(assets/'complete.json'),
                checkpoint_sha256=expected_sha, source_updates=30 if self.variant == 'ordinary' else 24,
                selection='source-selected step30' if self.variant == 'ordinary' else 'fixed step24')
            self.allowed_sources = allowed
        return self

    def predict(self, task):
        from tabicl_history import CHECKPOINT, FrozenHistoryRegressor

        if self.variant != 'public':
            if self.allowed_sources is None:
                raise ValueError('Call fit_source to validate this source-trained Tab fold first')
            if (task.dataset, task.building) == ('uji', 'B0') or (task.dataset, task.building) in self.allowed_sources:
                raise ValueError('Non-runnable Tab fold: heldout building appears in source permissions')
        if self.regressor is None:
            path = CHECKPOINT if self.checkpoint is None else self.checkpoint
            expected = PUBLIC_SHA256 if self.variant == 'public' else SOURCE_ASSETS[self.variant][2]
            if _digest(path) != expected:
                raise ValueError('Tab checkpoint does not match its declared public/source-finetuned variant')
            self.regressor = FrozenHistoryRegressor(checkpoint=path, device='cpu')
        histories = {f'B0F{f}': d for f, d in sorted(task.H.items())}
        seen = (task.S['rss'] != 100).any(0)
        for history in histories.values():
            seen |= (history['rss'] != 100).any(0)
        aps = np.flatnonzero(seen)
        xy = np.concatenate([d['xy'] for d in histories.values()] + [task.S['xy']])
        bounds = np.stack([np.floor(xy.min(0) / 2) * 2, np.ceil(xy.max(0) / 2) * 2])
        seed = self.seed + task.repeat
        self.regressor.fit(histories, task.S, f'B0F{task.floor}', aps, bounds, seed)
        result = self.regressor.predict(task.Q_rss)
        self.diagnostics = dict(task_id=task.task_id, checkpoint=str(self.regressor.checkpoint),
            variant=self.variant, provenance=self.provenance,
            source_fit=False, history_scans=sum(len(d['rss']) for d in histories.values()),
            support_scans=len(task.S['rss']), ap_count=len(aps), seed=seed,
            output='projected', bounds=bounds.tolist())
        return result['projected']
