"""Plot calibration budgets from a matched comparison summary."""

import argparse
import json
from pathlib import Path

from make_figs import style
import matplotlib.pyplot as plt


PAPER = Path(__file__).resolve().parents[1]
SOURCE = PAPER / "results/paper_tampere_TIE1_F0_comparison_20261114.json"
OUTPUT = PAPER / "generated/tie1_f0_20261114"
METHODS = [
    ("pretrain_dnn", "Supervised pretraining", "#444444", "o", ":"),
    ("finetune_dnn", "Supervised + target FT", "#0072B2", "s", "--"),
    ("maml_dnn", "FOMAML", "#CC79A7", "D", "-."),
    ("tabicl", "TabICL", "#009E73", "v", ":"),
    ("retained_gp", "GP component", "#E69F00", "P", "--"),
    ("refi_GP", "ReFi (complete)", "#D55E00", "^", "-"),
]


def build_figure(source=SOURCE, title="Tampere TIE1, target floor F0"):
    data = json.loads(Path(source).read_text())
    style("en")
    plt.rcParams.update({"savefig.bbox": None, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.70, wspace=0.28)
    budgets = [1, 3, 5]
    for axis, metric, panel_title in zip(
        axes, ["mean", "p90"], ["(a) Mean error", "(b) P90 error"]
    ):
        for method, label, color, marker, line in METHODS:
            values = [data["by_K"][str(k)][method][metric] for k in budgets]
            axis.plot(
                budgets, values, label=label, color=color, marker=marker,
                linestyle=line, linewidth=1.8 if method == "refi_GP" else 1.2,
                markersize=4.5,
            )
        axis.set_title(panel_title, loc="left", pad=8)
        axis.set_xticks(budgets)
        axis.set_xlim(0.7, 5.3)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Calibration locations $K$")
        axis.set_ylabel("Error (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.92),
        ncol=3, frameon=False, columnspacing=1.5, handlelength=2.2,
    )
    fig.suptitle(title, y=0.99, fontsize=9)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--title", default="Tampere TIE1, target floor F0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figure = build_figure(args.source, args.title)
    figure.savefig(args.output / "calibration_budget.pdf")
    figure.savefig(args.output / "calibration_budget.png", dpi=600)
