# -*- coding: utf-8 -*-
"""
Analysis of drugs from the top 10 drug-targets on a Heatmap over disease circuits or pathways depending on size.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_heatmap_drugsCircuit_top30KDTs(
    data_folder, filtered_file_name, assets_folder
):
    """
    This function reads the filtered matrix of shap scores (columns=drug-targets, rows = disease-circuits) and the drug-target-effects data,
    filters the top 30 best-scored drug-targets from SHAP and creates a heatmap with the drugs from the top 10 scored targets and the circuits.
    If the number of circuits is greater than 50, the circuits are grouped by their pathways (the first part of the circuit name before ": ").

    Args:
    data_folder (str): The path to the data folder where all results from the DREXML model are.
    filtered_file_name (str): The name of the shap scores filtered data file.
    assets_folder (str): The path to the assets folder where the drug-target interactions file is located.

    Returns:
    A heatmap is displayed and saved in the data folder.
    """

    # Read the genes data
    df = pd.read_csv(f"{data_folder}/{filtered_file_name}", sep="\t", index_col=0)

    # Calculate the mean of absolute values for each gene
    mean_scores = df.abs().mean()

    # Get the top 10 genes by mean score
    top_10_genes = mean_scores.nlargest(30)

    # Read the drugs data
    df_drugs = pd.read_csv(
        f"{assets_folder}/drugbank-v050110_alldrugbyaction.tsv",
        sep="\t",
        skiprows=1,
        names=["Drug", "Action", "KDT", "Gene", "Drug_effect", "drugKDT"],
    )

    # Filter the drugs that target the top 10 genes
    df_drugs_top_10_genes = df_drugs[df_drugs["Gene"].isin(top_10_genes.index)]

    # Create a new dataframe where each row corresponds to a drug and a circuit, and the value is the mean score of the top gene targeted by the drug in that circuit
    df_heatmap_10 = pd.DataFrame(
        index=df.index.unique(), columns=df_drugs_top_10_genes["Drug"].unique()
    )

    # Populate the dataframe with mean scores
    for circuit in df_heatmap_10.index:
        for drug in df_heatmap_10.columns:
            # Get the genes targeted by the drug
            genes = df_drugs_top_10_genes.loc[
                df_drugs_top_10_genes["Drug"] == drug, "Gene"
            ].values

            # Calculate the mean score of the genes in the circuit
            df_heatmap_10.loc[circuit, drug] = df.loc[circuit, genes].abs().mean()

    # Convert the scores to floats
    df_heatmap_10 = df_heatmap_10.astype(float)

    # If the number of circuits is greater than 50, group by pathways
    if df_heatmap_10.shape[0] > 50:
        # Split the circuit names by ": " to get the pathways
        df_heatmap_10.index = df_heatmap_10.index.str.split(": ").str[0]

        # Aggregate by pathway
        df_heatmap_10 = df_heatmap_10.groupby(level=0).mean()

        # Calculate the figure size and the font size based on the number of rows and columns
        num_rows, num_cols = df_heatmap_10.shape
        scale_factor = max(num_rows, num_cols) ** 0.5  # adjust this as needed

        fig_width = max(20, num_cols / scale_factor)  # adjust this as needed
        fig_height = max(20, num_rows / scale_factor)  # adjust this as needed

        font_size = 10.0 / scale_factor  # adjust this as needed

        sns.set(font_scale=font_size)

        plt.figure(figsize=(fig_width, fig_height))
        plt.ylabel("Pathway")

        print(
            "The number of circuits is greater than 50. The heatmap will be created using pathways."
        )
    else:
        print(
            "The number of circuits is less than or equal to 50. The heatmap will be created using circuits."
        )
        plt.figure(figsize=(40, 20))
        sns.set(font_scale=2)  # Adjust the font size if necessary
        plt.ylabel("Circuit")

    # Plot a heatmap
    sns.heatmap(df_heatmap_10, cmap="Blues", center=0)
    plt.title(
        "Mean scores of the drugs from the top 10 best-scored drug-targets over the Disease Map "
    )
    plt.xlabel("Drug")
    plt.savefig(f"{data_folder}/heatmap_top10KDTdrugs_circuit.pdf", dpi=300)
    plt.tight_layout()
    plt.show()
