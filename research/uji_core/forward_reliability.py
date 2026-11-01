"""Test whether a missed support detection needs a smaller map update.

Runs the same fixed 40U forward map in every arm. A scalar learned on two
buildings scales only the innovations of the affected support observations.
The control learns one multiplier for *all* innovations from the same grid.
"""
import argparse
import json
from pathlib import Path

from research.uji_core.forward import Arm, FIXED_RADIUS, RIDGE, MATCH, measure
from research.uji_core.protocol import BUDGETS, HASHES, SEEDS, read_floors

GAINS = (0., .25, .5, .75, 1.)


def miss(gain):
    return Arm(FIXED_RADIUS, 1., gain)


def common(gain):
    return Arm(FIXED_RADIUS, gain, gain)


def main(directory, output):
    output.mkdir(parents=True, exist_ok=True)
    train = read_floors(directory, 'trainingData.csv')
    buildings = sorted({b for b, _ in train})
    candidates = {f'{name}_{g}': configure(g) for name, configure in [('miss',miss), ('common',common)] for g in GAINS}
    source = measure(train, {b: candidates for b in buildings})
    source.to_csv(output / 'forward_gain_source.csv', index=False)
    risks = source.groupby(['building','floor','method']).mde.mean().groupby(['building','method']).mean().unstack()
    source_risks = {}
    selected = {}
    for b in buildings:
        held_risks = risks.drop(index=b).mean()
        source_risks[b] = held_risks.to_dict()
        selected[b] = {name: min(GAINS, key=lambda g: held_risks[f'{name}_{g}']) for name in ('miss','common')}
    selection = dict(radii=[FIXED_RADIUS], match=MATCH, ridge=RIDGE, grid=GAINS, fixed_half=.5,
                     training_csv_sha256=HASHES['trainingData.csv'], budgets=BUDGETS, seeds=SEEDS,
                     scope='Development follow-up after seeing aggregate radius experiment. No official queries select gains.',
                     families='miss: scalar on undetected support innovations; common: scalar on all innovations; raw support prototypes always appended.',
                     selection='training only, mean MDE by building/floor/seed/K, hold entire target building out, tie to smaller multiplier',
                     source_risks=source_risks, selected=selected)
    freeze = output / 'forward_gain_selection.json'
    freeze.write_text(json.dumps(selection, indent=2, allow_nan=False) + '\n')
    selected = json.loads(freeze.read_text())['selected']
    arms = {b: {'fixed_unit': miss(1.), 'fixed_miss_half': miss(.5), 'fixed_common_half': common(.5),
                'learned_miss': miss(selected[str(b)]['miss']), 'learned_common': common(selected[str(b)]['common'])}
            for b in buildings}
    official = read_floors(directory, 'validationData.csv')
    result = measure(train, arms, official)
    result.to_csv(output / 'forward_gain_official.csv', index=False)
    print(selected, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    main(args.data, args.output)
