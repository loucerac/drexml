# -*- coding: utf-8 -*-
"""
SHAP scores analysis on boxplots for DREXML.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_boxplots(data_folder, filtered_file_name):
    """
    This function reads the filtered data, creates boxplots for all score columns, and saves the boxplots to a .pdf file.

    Args:
    data_folder (str): The path to the data folder.
    filtered_file_name (str): The name of the shap scores filtered file.

    Returns:
    The boxplots are saved as a .pdf file.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load the previous filtered table
    data = pd.read_csv(f"{data_folder}/{filtered_file_name}", sep='\t', index_col=0)

    # Melt the DataFrame to make it suitable for boxplots
    data.reset_index(inplace=True) ## Reset the index back to a column
    melted_data = data.melt(id_vars = "circuit_name")

    # Compute median of each column
    medians = melted_data.groupby("variable")["value"].median().sort_values(ascending=False)

    # Order the data based on medians
    melted_data["variable"] = pd.Categorical(melted_data["variable"], categories=medians.index, ordered=True)

    # Create the boxplots
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.figure(figsize=(20, 12), facecolor='white')
    sns.set_style("whitegrid")  # set Seaborn style to white grid
    sns.boxplot(x="variable", y="value", data = melted_data)
    plt.xticks(rotation=40, ha = "right")
    plt.title("Boxplots of SHAP-Scores")
    plt.ylabel('SHAP Scores')
    plt.xlabel('Drug-target')
    plt.tight_layout()
    plt.savefig(f"{data_folder}/relevant_drugTargetscores_boxplots.pdf", dpi=300)
    plt.show()