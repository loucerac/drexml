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
        bbox_to_anchor=(-0.3, 1.035),
        bbox_transform=ax.transAxes,
        fontsize="large",
    )

    # plt.tight_layout()
    fig.set_size_inches(8.27 / 2, 11.69)
    plt.savefig(
        output_path.joinpath(f"{input_path.stem}.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_path.joinpath(f"{input_path.stem}.pdf"), bbox_inches="tight")


def plot_metrics(input_path, output_folder=None, width=2.735):
    """Plot stability versus R^2 with 95% CI"""

    input_path = pathlib.Path(input_path).absolute()
    if output_folder is None:
        output_folder = input_path.parent
    else:
        output_folder = pathlib.Path(output_folder)

    results_df = pd.read_csv(input_path, sep="\t").fillna(0)
    results_df["r2_error"] = 1.96 * results_df.r2_std / 2
    results_df = results_df.sort_values(by="stability", ascending=True).reset_index()

    custom_params = {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }
    sns.set_theme(
        style="whitegrid", palette="colorblind", context="paper", rc=custom_params
    )

    fig, ax = plt.subplots(1, 1)

    plt.scatter(results_df.stability, results_df.r2_mean, s=1)
    plt.errorbar(
        results_df.stability,
        results_df.r2_mean,
        yerr=results_df.r2_error,
        linestyle="None",
        linewidth=1,
        label=r"$R^2$ mean 95% CI",
    )
    plt.errorbar(
        results_df.stability,
        results_df.r2_mean,
        xerr=results_df.stability - results_df.stability_lower_95ci,
        linestyle="None",
        linewidth=1,
        label="Stability 95% CI",
    )

    ax.axvspan(0, 0.4, color="red", alpha=0.1)
    ax.axvspan(0.4, 0.75, color="y", alpha=0.1)
    ax.axvspan(0.75, 1.0, color="g", alpha=0.1)
    ax.set_xticks([0, 0.4, 0.75, 1])

    plt.xlabel("Stability")
    plt.ylabel(r"$R^2$ score")
    plt.legend()

    fig.tight_layout()
    fig.set_size_inches(width, (width * 3) / 4)

    fname = "stability-vs-r2_by-circuit"
    plt.savefig(output_folder.joinpath(f"{fname}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(output_folder.joinpath(f"{fname}.pdf"), bbox_inches="tight")
