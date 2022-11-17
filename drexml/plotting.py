# -*- coding: utf-8 -*-
"""
Plotting module for DREXML.
"""
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def preprocess_data(input_path):
    """Read and preprocess results."""

    data = pd.read_csv(input_path, sep="\t")
    if "stability_upper_95ci" not in data.columns:
        data["stability_upper_95ci"] = data["upper"]
    data["stability_error"] = data["stability_upper_95ci"] - data["stability"]

    data = data.sort_values(by="stability", ascending=False)
    data = data.replace({"circuit_name": {"map": "Disease Map"}})
    data = data.reset_index(drop=True)
    query = data["circuit_name"] == "Disease Map"
    iquery = data.index[query]
    ilow = data.index[iquery - 1]
    iup = data.index[iquery + 1]

    data["r2_error"] = 1.96 * data.r2_std / 2

    return data, ilow, iup


def plot_stability_ax(ax, data, ilow, iup):
    """Plot stability axis."""
    label_font_size = 10
    axis_font = {"size": f"{label_font_size}"}
    scatter_size = 10
    lw = 1

    x = data.stability
    y = data.circuit_name
    xerr = data.stability_error
    yerr = None

    # pylint: disable=E1121
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
    # pylint: enable=E1121
    ax.grid(axis="x")
    ax.errorbar(
        x, y, yerr=yerr, xerr=xerr, ls="", color="k", lw=lw, label="CI", alpha=0.5
    )
    # Get the first two and last y-tick positions.
    if ilow == iup:
        miny, nexty = ax.get_yticks()
        maxy = nexty
    else:
        miny, nexty, *_, maxy = ax.get_yticks()

    # Compute half the y-tick interval (for example).
    eps = (nexty - miny) / 2  # <-- Your choice.

    # Adjust the limits.
    ax.set_ylim(maxy + eps, miny - eps)
    ax.set_xlim(-0.05, 1.05)
    # ax.axvline(0.4, color="r", linestyle="--")
    # ax.axvline(0.75, color="b", linestyle="--")
    # ax.axhspan("Disease Map", color="k", linestyle="--")
    if ilow != iup:
        ax.axhspan(ilow + 0.5, iup - 0.5, color="gray", alpha=0.2)

    ax.axvspan(0, 0.4, color="red", alpha=0.2)
    ax.axvspan(0.4, 0.75, color="y", alpha=0.2)
    ax.axvspan(0.75, 1.0, color="g", alpha=0.2)
    ax.set_xticks([0, 0.4, 0.75, 1])

    sns.despine(left=True, right=True, top=True, bottom=True)
    ax.set_xlabel("Nogueria Stability Stimate with 95% CI", **axis_font)
    ax.set_ylabel("Circuit Name", **axis_font)

    return ax


def plot_r2_ax(ax_right, data):
    """Plot R2 axis."""

    label_font_size = 10
    axis_font = {"size": f"{label_font_size}"}
    lw = 1
    scatter_size = 10

    x = data["r2_mean"]
    xerr = data["r2_error"]
    y = data.circuit_name
    yerr = None

    # pylint: disable=E1121
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
    # pylint: enable=E1121
    ax_right.errorbar(
        x, y, yerr=yerr, xerr=xerr, ls="", color="b", lw=lw, label="CI", alpha=0.5
    )
    ax_right.set_xlabel(r"$R^2$ score mean and 95% CI", **axis_font)
    ax_right.grid(axis="x")
    ax_right.set_xlim(-0.05, 1.05)

    return ax_right


def plot_stability(input_path, output_path=None):
    """Plot the stability results."""
    print(input_path)
    print(output_path)
    input_path = pathlib.Path(input_path)
    if output_path is None:
        output_path = input_path.parent
    else:
        output_path = pathlib.Path(output_path)
    print(output_path)

    data, ilow, iup = preprocess_data(input_path)
    print(data)
    print(f"{ilow=}")
    print(f"{iup=}")

    font_scale = 0.4
    this_figsize = (5 / 2.0, 30 / 2.0)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=font_scale)
    fig, ax = plt.subplots(1, 1, figsize=this_figsize)
    ax = plot_stability_ax(ax, data, ilow, iup)

    ax_right = ax.twiny()
    ax_right = plot_r2_ax(ax_right, data)

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
