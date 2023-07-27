# -*- coding: utf-8 -*-
"""
SHAP scores analysis on histograms for DREXML.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_histograms(data_folder, filtered_file_name):
    """
    This function reads the shap scores filtered data, creates histograms from the SHAP scores of each drug-target, 
    and saves the histograms to a composed .pdf file.

    Args:
    data_folder (str): The path to the data folder.
    filtered_file_name (str): The name of the  shap scores filtered file.

    Returns:
    None. The histograms are saved as a .pdf file.
    """
    # Load the previous filtered table
    data = pd.read_csv(f"{data_folder}/{filtered_file_name}", sep='\t', index_col=0)

    # Set up the figure and axes
    fig, axes = plt.subplots(10, 4, figsize=(20, 30))

    # Flatten the axes
    axes = axes.flatten()

    # Remove the extra subplots
    for i in range(len(data.columns)-1, 40):
        fig.delaxes(axes[i])

    # Create histograms for all score columns
    for i, col in enumerate(data.columns[1:]):
        sns.histplot(data[col], ax=axes[i], kde=True)
        axes[i].set_title(col)

    # Improve layout
    plt.tight_layout()
    plt.savefig(f"{data_folder}/relevantDrugTargets_scores.pdf", dpi=300)
    plt.show()