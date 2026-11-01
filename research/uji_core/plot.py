"""Render the three building panels from the published aggregate CSV."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render(directory):
    data = pd.read_csv(directory/"building_means.csv")
    with plt.rc_context({"font.size": 9, "axes.spines.top":False, "axes.spines.right":False,
                         "svg.fonttype":"none","svg.hashsalt":"uji-core-v1"}):
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1), sharey=True, layout="constrained")
        for b, ax in enumerate(axes):
            rows = data[data.building == b]
            for key, name, col, style in (("joint_tuned","Source-selected WKNN","#707B89","--"),
                                          ("uncorrected_shape","Map update only","#C56A32",":"),
                                          ("core","Smoothed history + map update","#00766D","-")):
                ax.plot(rows.K, rows[key], style, marker="o", ms=3.8, lw=2.1, color=col, label=name)
            ax.set(title=f"Building {b}", xlabel="Calibration locations (K)", xticks=[1,3,5,10,20], ylim=(5,20))
            ax.grid(axis="y", alpha=.18)
        axes[0].set_ylabel("Mean error (UJI coordinate units)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncols=3, loc="outside upper center", frameon=False)
        path = directory / "curves.svg"
        fig.savefig(path, metadata={"Date": None})
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    render(p.parse_args().results)
