# -*- coding: utf-8 -*-
"""
Plotting module for DREXML.
"""
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import maxabs_scale


@dataclass(frozen=False)
class RepurposingResult:
    """
    Class for storing the results of the DREXML analysis.
    """

    sel_mat: "pd.DataFrame | pathlib.Path | str" = pd.DataFrame
    score_mat: "pd.DataFrame | pathlib.Path | str" = pd.DataFrame
    stab_mat: "pd.DataFrame | pathlib.Path | str" = pd.DataFrame

    # stable_circuits: list = field(init=False)

    def __post_init__(self):
        if isinstance(self.sel_mat, (str, pathlib.Path)):
            self.sel_mat = pd.read_csv(self.sel_mat, index_col=0, sep="\t")

        if isinstance(self.score_mat, (str, pathlib.Path)):
            self.score_mat = pd.read_csv(self.score_mat, index_col=0, sep="\t")

        if isinstance(self.stab_mat, (str, pathlib.Path)):
            self.stab_mat = pd.read_csv(self.stab_mat, index_col=0, sep="\t")

        self.stable_circuits = self.get_stable_circuits()

    def get_stable_circuits(self):
        """
        Get the stable circuits.

        Returns
        -------
        stable_circuits : list
            List of stable circuits.

        """

        stable_circuits = self.stab_mat[
            (self.stab_mat["stability_lower_95ci"] > 0.4)
            & (self.stab_mat["stability_upper_95ci"] > 0.75)
        ].index.tolist()

        return stable_circuits

    def filter_scores(self, remove_unstable=True):
        """
        Filter the scores to only the selected genes and stable circuits.

        Parameters
        ----------
        remove_unstable : bool, optional
            Remove unstable circuits, by default True

        Returns
        -------
        scores_filt : pandas.DataFrame
            Filtered scores.
        """
        scores_filt = self.score_mat.multiply(self.sel_mat)
        scores_filt = scores_filt.transform(maxabs_scale)
        if remove_unstable:
            scores_filt = scores_filt.loc[
                scores_filt.index.intersection(self.stable_circuits)
            ]
        scores_filt = scores_filt.loc[scores_filt.abs().sum(axis=1) > 0]
        scores_filt = scores_filt.loc[:, scores_filt.abs().sum(axis=0) > 0]

        return scores_filt

    def plot_relevance_heatmap(
        self, remove_unstable=True, output_folder=None
    ):  # pragma: no cover
        """
        Plot the relevance heatmap of the scores.

        Parameters
        ----------
        remove_unstable : bool, optional
            Remove unstable circuits, by default True
        output_folder : str, optional
            Output folder, by default None

        Returns
        -------
        None.

        """
        sns.set_context("paper", font_scale=0.6)
        scores_filt = self.filter_scores(remove_unstable=remove_unstable)

        cbar_specs = {"orientation": "horizontal", "label": "Relevance"}
        sns.clustermap(
            scores_filt,
            cmap=sns.color_palette("vlag", as_cmap=True),
            cbar_kws=cbar_specs,
            cbar_pos=(0.02, 0.89, 0.1, 0.025),
            row_cluster=True,
            vmin=-1,
            vmax=1,
            yticklabels=True,
            xticklabels=True,
        )

        if output_folder is not None:
            output_folder = pathlib.Path(output_folder)
            if remove_unstable:
                fname = "relevance_heatmap_filtered"
            else:
                fname = "relevance_heatmap"

            plt.savefig(
                output_folder.joinpath(f"{fname}.png"), dpi=300, bbox_inches="tight"
            )
            plt.savefig(output_folder.joinpath(f"{fname}.pdf"), bbox_inches="tight")
        else:
            plt.show()

    def plot_gene_profile(self, gene: str, output_folder=None):
        """
        Plot the gene profile.

        Parameters
        ----------
        gene : str
            Gene name.
        output_folder : str, optional
            Output folder, by default None

        Returns
        -------
        None.

        """

        scores = self.filter_scores(remove_unstable=False)
        if gene not in scores.columns:
            raise KeyError
        signal = scores[gene]
        signal = signal[signal.abs() > 0].sort_values(ascending=False)

        sns.set_context("paper", font_scale=1)
        width = 1
        height = (signal.size * 1.1) / 4
        plt.figure(figsize=(width, height))
        ax = sns.heatmap(
            signal.to_frame(),
            cbar_kws={"label": "Relevance"},
            cmap=sns.color_palette("vlag", as_cmap=True),
            vmin=-1,
            vmax=1,
            yticklabels=True,
            xticklabels=True,
        )
        ax.set_ylabel("")

        for tick_label in ax.get_yticklabels():
            if tick_label.get_text() not in self.stable_circuits:
                tick_label.set_color("grey")

        if output_folder is not None:
            output_folder = pathlib.Path(output_folder)
            fname = f"profile_{gene.lower()}"
            plt.savefig(
                output_folder.joinpath(f"{fname}.png"), dpi=300, bbox_inches="tight"
            )
            plt.savefig(output_folder.joinpath(f"{fname}.pdf"), bbox_inches="tight")
        else:  # pragma: no cover
            plt.show()

    def plot_metrics(self, width=2.735, output_folder=None):
        """
        Read the drexml results TSV file and plot it. The R^2 confidence interval for the
        mean go to y-axis, whereas the x-axis shows the 95% interval for the Nogueiras's
        stability stimate.

        Parameters
        ----------
        width : float, optional
            Width of the plot.

        output_folder : str, optional
            Path to the output folder. If None, the output folder is the same
            as the input folder.

        Returns
        -------
        None.
        """

        results_df = self.stab_mat.copy()
        results_df["r2_error"] = 1.96 * results_df.r2_std / 2
        results_df = results_df.sort_values(
            by="stability", ascending=True
        ).reset_index()

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

        if output_folder is not None:
            output_folder = pathlib.Path(output_folder)
            fname = "metrics"
            plt.savefig(
                output_folder.joinpath(f"{fname}.png"), dpi=300, bbox_inches="tight"
            )
            plt.savefig(output_folder.joinpath(f"{fname}.pdf"), bbox_inches="tight")

        else:  # pragma: no cover
            plt.show()
