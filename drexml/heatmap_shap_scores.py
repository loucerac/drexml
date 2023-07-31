# -*- coding: utf-8 -*-
"""
SHAP scores analysis on Heatmap for DREXML.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_heatmap_KDTsCir_annot(data_folder, assets_folder, filtered_file_name):
    """
    This function reads the filtered shap_relevant_table data generated from the process_data() function  and the annotations
    of the most common drug-effect over the drug-targets. Then, it creates a heatmap with the drug-targets (annotated with the most
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
    filtered_df_stable = pd.read_csv(
        f"{data_folder}/{filtered_file_name}", sep="\t", index_col=0
    )

    # Load annotations of the most common drug-effects over the drug-targets
    annots = pd.read_csv(
        f"{assets_folder}/drugbank-v050110_mostdrugeffects_KDTs.tsv",
        sep="\t",
        index_col=0,
    )

    # Subset the relevant drug-targets
    annots = annots.loc[
        annots.index.intersection(filtered_df_stable.columns),
    ]

    drug_action_colors = {
        "Inhibitor": "magenta",
        "Activator": "cyan",
        "Ligand": "yellow",
        "other": "grey",
        "Modulator": "orange",
    }
    drug_action_colors_list = [
        drug_action_colors[action] for action in annots["drug_action"]
    ]

    # If the number of circuits is greater than 50, group by pathways
    if filtered_df_stable.shape[0] > 50:
        # Split the circuit names by ": " to get the pathways
        filtered_df_stable.index = filtered_df_stable.index.str.split(": ").str[0]

        # Aggregate by pathway
        filtered_df_stable = filtered_df_stable.groupby(level=0).mean()

        # Calculate the figure size and the font size based on the number of rows and columns
        num_rows, num_cols = filtered_df_stable.shape
        scale_factor = max(num_rows, num_cols) ** 0.5  # adjust this as needed

        fig_width = max(50, num_cols / scale_factor)  # adjust this as needed
        fig_height = max(30, num_rows / scale_factor)  # adjust this as needed
        # font_size = min(5, 300 / max(num_rows, num_cols))  # adjust this as needed

        # Create the clustermap with adjusted size
        cluster_map = sns.clustermap(
            filtered_df_stable,
            cmap="RdBu_r",
            col_colors=[drug_action_colors_list],
            linewidths=0.5,
            cbar_kws={"label": "SHAP Value"},
            figsize=(fig_width, fig_height),
        )
        # cluster_map.ax_heatmap.tick_params(axis='both', which='major', labelsize= 0.8)
        # Add titles to the x and y axes
        cluster_map.ax_heatmap.set_xlabel("Drug-target")
        cluster_map.ax_heatmap.set_ylabel("Pathway name")

        print(
            "The number of circuits is greater than 50. The heatmap will be created using pathways."
        )
    else:
        print(
            "The number of circuits is less than or equal to 50. The heatmap will be created using circuits."
        )

        sns.set(font_scale=1.5)  # Adjust the font size if necessary

        # Create the clustermap with fixed size
        cluster_map = sns.clustermap(
            filtered_df_stable,
            cmap="RdBu_r",
            col_colors=[drug_action_colors_list],
            linewidths=0.5,
            cbar_kws={"label": "SHAP Value"},
            figsize=(20, 20),
        )
        # Add titles to the x and y axes
        cluster_map.ax_heatmap.set_xlabel("Drug-target")
        cluster_map.ax_heatmap.set_ylabel("Circuit name")

    # Add a title to the color legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            label=action,
            markersize=15,
        )
        for action, color in drug_action_colors.items()
    ]
    legend = plt.legend(
        handles=legend_elements,
        title="Drug Action",
        loc="upper left",
        bbox_to_anchor=(15, 1),
    )
    legend.get_frame().set_facecolor("white")

    # Add a title to the entire figure
    plt.suptitle(
        "Heatmap of Disease Map and relevant Drug-targets", fontsize=30, y=1.05
    )

    # Save the plot
    plt.savefig(f"{data_folder}/heatmap_circuits_KDT_drugeff.pdf", dpi=300)
    plt.show()
