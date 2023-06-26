# -*- coding: utf-8 -*-
"""
Plotting module for DREXML.
"""
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metrics(input_path, output_folder=None, width=2.735):
    """
    Read the drexml results TSV file and plot it. The R^2 confidence interval for the
    mean go to y-axis, whereas the x-axis shows the 95% interval for the Nogueiras's
    stability stimate.

    Parameters
    ----------
    input_path : str
        Path to the input file.
    output_folder : str, optional
        Path to the output folder. If None, the output folder is the same
        as the input folder.
    width : float, optional
        Width of the plot.

    """

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

    fname = input_path.stem
    plt.savefig(output_folder.joinpath(f"{fname}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(output_folder.joinpath(f"{fname}.pdf"), bbox_inches="tight")
