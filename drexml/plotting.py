# -*- coding: utf-8 -*-
"""
Plotting module for DREXML.
"""
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_stability(input_path, output_path=None):
    """Plot the stability results."""
    input_path = pathlib.Path(input_path)
    if output_path is None:
        output_path = input_path.parent
    else:
        output_path = pathlib.Path(output_path)

    data = pd.read_csv(input_path, sep="\t")
    data["stability_error"] = data["stability_upper_95ci"] - data["stability"]

    data = data.sort_values(by="stability", ascending=False)
    data = data.replace({"circuit_name": {"map": "Disease Map"}})
    data = data.reset_index(drop=True)
    query = data["circuit_name"] == "Disease Map"
    iquery = data.index[query]
    ilow = data.index[iquery - 1]
    low = data.loc[ilow, "circuit_name"]
    iup = data.index[iquery + 1]
    up = data.loc[iup, "circuit_name"]

    data["r2_error"] = 1.96 * data.r2_std / 2

    font_scale = 0.4
    scale = 2
    this_figsize = (5 / scale, 30 / scale)
    label_font_size = 10
    scatter_size = 10
    lw = 1

    x = data.stability
    y = data.circuit_name
    xerr = data.stability_error
    yerr = None
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=font_scale)
    fig, ax = plt.subplots(1, 1, figsize=this_figsize)
    ax_right = ax.twiny()
    sns.scatterplot(
        x,
        y,
        s=scatter_size,
        ax=ax,
        color="k",
        marker="+",
        label="Stability",
        legend=False,
        alpha=0.5,
    )
    ax.grid(axis="x")
    ax.errorbar(
        x, y, yerr=yerr, xerr=xerr, ls="", color="k", lw=lw, label="CI", alpha=0.5
    )
    # Get the first two and last y-tick positions.
    miny, nexty, *_, maxy = ax.get_yticks()

    # Compute half the y-tick interval (for example).
    eps = (nexty - miny) / 2  # <-- Your choice.

    # Adjust the limits.
    ax.set_ylim(maxy + eps, miny - eps)
    # ax.axvline(0.4, color="r", linestyle="--")
    # ax.axvline(0.75, color="b", linestyle="--")
    # ax.axhspan("Disease Map", color="k", linestyle="--")
    ax.axhspan(ilow + 0.5, iup - 0.5, color="gray", alpha=0.2)

    ax.axvspan(0, 0.4, color="red", alpha=0.2)
    ax.axvspan(0.4, 0.75, color="y", alpha=0.2)
    ax.axvspan(0.75, 1.0, color="g", alpha=0.2)
    ax.set_xticks([0, 0.4, 0.75, 1])

    sns.despine(left=True, right=True, top=True, bottom=True)
    axis_font = {"size": f"{label_font_size}"}
    ax.set_xlabel("Nogueria Stability Stimate with 95% CI", **axis_font)
    ax.set_ylabel("Circuit Name", **axis_font)
    lh1, l1 = ax.get_legend_handles_labels()

    x = data["r2_mean"]
    xerr = data["r2_error"]
    sns.scatterplot(
        x,
        y,
        s=scatter_size,
        ax=ax_right,
        color="b",
        marker="x",
        label=r"$R^2$",
        legend=False,
        alpha=0.5,
    )
    ax_right.errorbar(
        x, y, yerr=yerr, xerr=xerr, ls="", color="b", lw=lw, label="CI", alpha=0.5
    )
    ax_right.set_xlabel(r"$R^2$ score mean and 95% CI", **axis_font)
    ax_right.grid(axis="x")

    lh2, l2 = ax_right.get_legend_handles_labels()

    # added these three lines
    # ask matplotlib for the plotted objects and their labels
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax_right.get_legend_handles_labels()
    # ax_right.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.legend(
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(-0.2, 1.035),
        bbox_transform=ax.transAxes,
        fontsize="large",
    )

    # plt.tight_layout()
    fig.set_size_inches(8.27 / 2, 11.69)
    plt.savefig(
        output_path.joinpath(f"{input_path.stem}.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_path.joinpath(f"{input_path.stem}.pdf"), bbox_inches="tight")
