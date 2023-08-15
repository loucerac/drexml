# -*- coding: utf-8 -*-
"""
SHAP scores analysis on histograms for DREXML.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_histograms(
    data_folder, filtered_file_name="shap_filtered_stability_symbol.tsv"
):
    """
    This function reads the shap scores filtered data, creates histograms from the SHAP scores of each drug-target,
    and saves the histograms to a composed .pdf file.

    Args:
    data_folder (str): The path to the data folder.
    filtered_file_name (str): The name of the  shap scores filtered file.

    Returns:
    The histograms are saved as a .pdf file.
    """
    # Load the previous filtered table
    data = pd.read_csv(f"{data_folder}/{filtered_file_name}", sep="\t", index_col=0)

    # Number of columns in the data
    num_cols = len(data.columns[1:])

    # Calculate the number of rows required
    num_rows = int(np.ceil(num_cols / 4.0))

    # Set up the figure and axes
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 3 * num_rows))

    # Flatten the axes
    axes = axes.flatten()

    # Remove the extra subplots
    for i in range(num_cols, num_rows * 4):
        fig.delaxes(axes[i])

    # Create histograms for all score columns
    for i, col in enumerate(data.columns[1:]):
        sns.histplot(data[col], ax=axes[i], kde=True)
        axes[i].set_title(col)
        axes[i].set_xlabel("")  # remove the x-label

    # Improve layout
    plt.tight_layout()
    plt.savefig(f"{data_folder}/relevantDrugTargets_scores.pdf", dpi=300)
    plt.show()
