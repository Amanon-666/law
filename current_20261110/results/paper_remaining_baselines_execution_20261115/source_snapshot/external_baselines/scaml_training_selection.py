"""Post-hoc ScaML restart-selection correction; preserve the original run.

BoTorch 0.7.3 returns a fitted MLL in eval mode. The official restart loop
then scores model(train_X), which is a conditioned posterior. Restore train
mode before that score so restart selection uses the same MAP/MLL objective
as fitting. No changes to data, kernel, priors, seeds, restarts or iteration cap.
Source posterior() and target predict() subsequently restore evaluation mode.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import json
import torch
from botorch import fit as botorch_fit
from external_baselines import scaml_run as original
from external_baselines.scaml_adapter import FitTrace


class TrainingSelectionTrace(FitTrace):
    @contextmanager
    def observe(self):
        with super().observe():
            observed = botorch_fit.fit_gpytorch_mll
            def fit_then_train(mll, **kwargs):
                fitted = observed(mll, **kwargs)
                fitted.train()
                return fitted
            botorch_fit.fit_gpytorch_mll = fit_then_train
            try:
                yield self
            finally:
                botorch_fit.fit_gpytorch_mll = observed

    def selected(self, model):
        result = super().selected(model)
        best = result['best_successful_objective']
        chosen = result['selected_optimizer_objective']
        if best is not None and chosen is not None:
            assert chosen <= best + 1e-7, (chosen, best)
        return result


def configure():
    original.OUT = original.ROOT/'results/scaml_training_selection_v1_20261027'
    original.FitTrace = TrainingSelectionTrace


def freeze():
    original.freeze()
    path = original.OUT/'freeze.json'
    record = original.read(path)
    record['code_sha256'][str(Path(__file__).resolve())] = original.digest(__file__)
    record['fitting'] = 'Post-hoc correction: restore train mode after each BoTorch fit, before official restart scoring. Same MAP/MLL objective for fitting and selection, source and target. All other frozen settings unchanged.'
    check=original.ROOT/'results/scaml_compatibility_20261027/training_selection_check.json'
    assert original.read(check)['passed']
    record['inputs_sha256'][str(check)]=original.digest(check)
    record['reverse_audit']='Same H/S fitting and Query-only prediction/scoring path as original. Mode correction executes after each fit and before official restart ranking. No Query read, resampling, gradient update or parameter transformation in the correction. Source posterior() and target predict() restore eval mode. Synthetic 8-model check confirms selected optimizer objective equals recomputed training objective within 1e-7.'
    record['original_run'] = str(original.ROOT/'results/scaml_k_curve_v1_20261027')
    record['reason'] = 'Confirmed train/eval restart-objective mismatch in official code; correction chosen without tuning Query performance. Original results retained.'
    path.write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['freeze','run'])
    p.add_argument('--building')
    args=p.parse_args()
    configure()
    freeze() if args.stage=='freeze' else original.run(args.building)
