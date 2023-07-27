# -*- coding: utf-8 -*-
"""
SHAP scores analysis on Heatmap for DREXML.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_heatmap_KDTsCir_annot(data_folder, assets_folder, filtered_file_name):
    """
    This function reads the filtered shap_relevant_table data generated from the process_data() function  and the annotations 
    of most common drug-effect over the drug-targets. Then, it creates a heatmap with the drug-targets (annotated with the most
    common drug effect on top) as columns and the circuits as rows, filled  with the SHAP scores obtained. 
    The heatmap is saved to a .pdf file on the same data_folder.

    Args:
    data_folder (str): The path to the data folder.
    assets_folder (str): The path to the assets folder.
    filtered_file_name (str): The name of the shap scores filtered file.

    Returns:
    None. The heatmap is saved as a .pdf file.
    """
    # Load the previous filtered table
    filtered_df_stable = pd.read_csv(f"{data_folder}/{filtered_file_name}", sep='\t', index_col=0)

    # Load annotations of the most common drug-effects over the drug-targets
    annots = pd.read_csv(f"{assets_folder}/drugbank-v050110_mostdrugeffects_KDTs.tsv", sep='\t', index_col=0)

    # Subset the relevant drug-targets
    annots = annots.loc[annots.index.intersection(filtered_df_stable.columns),]

    drug_action_colors = {
        "Inhibitor": "magenta",
        "Activator": "cyan",
        "Ligand": "yellow",
        "other": "grey",
    }
    drug_action_colors_list = [drug_action_colors[action] for action in annots["drug_action"]]

    sns.set(font_scale= 1.5) # Adjust the font size if necessary

    # Create the clustermap
    cluster_map = sns.clustermap(filtered_df_stable, cmap="RdBu_r", col_colors=[drug_action_colors_list], linewidths=0.5, cbar_kws={'label': 'SHAP Value'}, figsize=(20,20))

    # Add titles to the x and y axes
    cluster_map.ax_heatmap.set_xlabel('Drug-target')
    cluster_map.ax_heatmap.set_ylabel('Circuit name')

    # Add a title to the color legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=action, markersize=15)
        for action, color in drug_action_colors.items()
    ]
    legend = plt.legend(handles=legend_elements, title="Drug Action", loc='upper left', bbox_to_anchor=(15, 1))
    legend.get_frame().set_facecolor('white')

    # Add a title to the entire figure
    plt.suptitle('Heatmap of circuits and genes', fontsize=30, y=1.05)

    # Save the plot
    plt.savefig(f"{data_folder}/heatmap_circuits_KDT_drugeff.pdf", dpi=300)
    plt.show()
